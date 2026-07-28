"""
Thin wrapper around yfinance calls: hard timeout, jittered exponential backoff,
per-call exception filter. yfinance has no native per-call timeout; Yahoo's
default libcurl timeout is 30s, and during rate-limiting it can hang longer.

Usage:
    from ._yf_safe import safe_call, _staggered_submit

    tk = safe_call(yf.Ticker, ticker)
    expirations = safe_call(lambda t: t.options, tk)
    spot = safe_call(lambda t: t.fast_info.get("lastPrice"), tk)
"""

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import requests

logger = logging.getLogger("scylla.yf_safe")

DEFAULT_EXC = (
    KeyError,
    ValueError,
    ConnectionError,
    TimeoutError,
    requests.exceptions.RequestException,
    requests.exceptions.ChunkedEncodingError,
    IndexError,
)

_NEVER_SWALLOW = (KeyboardInterrupt, SystemExit, asyncio.CancelledError)


def safe_call(fn, *args, timeout: float = 12, retries: int = 2, base_delay: float = 0.8,
              retry_on=DEFAULT_EXC, **kwargs):
    """Run a yfinance-bound callable with a hard timeout and jittered backoff retries."""
    attempt = 0
    while True:
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(fn, *args, **kwargs)
        try:
            result = fut.result(timeout=timeout)
            pool.shutdown(wait=False)
            return result
        except _NEVER_SWALLOW:
            pool.shutdown(wait=False)
            raise
        except FutureTimeoutError:
            pool.shutdown(wait=False)
            logger.warning(f"yfinance call {getattr(fn, '__name__', repr(fn))} timed out after {timeout}s")
            raise TimeoutError(f"yfinance call timed out after {timeout}s")
        except retry_on as e:
            pool.shutdown(wait=False)
            if attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            logger.debug(
                f"yfinance retry {attempt + 1}/{retries} for "
                f"{getattr(fn, '__name__', repr(fn))}: {e}; sleeping {delay:.2f}s"
            )
            time.sleep(delay)
            attempt += 1


def _staggered_submit(executor, fn, *args, delay_fn=None, **kwargs):
    """Submit a future after a small randomized delay to avoid burst-detection on Yahoo."""
    if delay_fn is None:
        delay_fn = lambda: random.uniform(0.05, 0.25)
    time.sleep(delay_fn())
    return executor.submit(fn, *args, **kwargs)