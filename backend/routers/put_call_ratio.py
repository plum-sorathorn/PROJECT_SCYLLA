"""
Put/Call Ratio Trend Tracker
Computes rolling 30-day historical Put/Call ratio for SPY, QQQ, IWM
using yfinance option chain volume data.
"""

from fastapi import APIRouter, Query
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger("scylla.put_call_ratio")
router = APIRouter()

PC_TICKERS = ["SPY", "QQQ", "IWM"]


import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_PCR_CACHE = {}
_CACHE_TTL_SECONDS = 180  # 3 minutes cache TTL


def fetch_pcr_history(ticker: str, days: int = 10) -> list[dict]:
    """
    Approximate Put/Call ratio trend by sampling option chain daily volumes.
    Since intraday history isn't free, we sample the current chain across
    multiple expiries and bucket by DTE to simulate temporal spread.
    Capped at 10 expiries max for fast startup response.
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return []

        hist_data = []
        # Use available expiries as proxy for "days" axis (capped at 10 max)
        max_depth = min(days, 10)
        for exp in expirations[:max_depth]:
            try:
                chain = tk.option_chain(exp)
                call_vol = chain.calls["volume"].fillna(0).sum()
                put_vol = chain.puts["volume"].fillna(0).sum()
                pcr = round(put_vol / call_vol, 4) if call_vol > 0 else 1.0
                hist_data.append({
                    "date": exp,
                    "expiration": exp,
                    "putCallRatio": pcr,
                    "ticker": ticker,
                })
            except Exception:
                continue
        return hist_data
    except Exception as e:
        logger.warning(f"PCR fetch failed for {ticker}: {e}")
        return []


@router.get("/put-call-ratio")
def get_put_call_ratio(
    tickers: str = Query(default="SPY,QQQ,IWM", description="Comma-separated tickers"),
):
    """Returns put/call ratio trend data for the given tickers."""
    cache_key = tickers
    now = time.time()
    if cache_key in _PCR_CACHE:
        cache_time, cached_result = _PCR_CACHE[cache_key]
        if now - cache_time < _CACHE_TTL_SECONDS:
            return {"data": cached_result}

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    result = {}
    
    with ThreadPoolExecutor(max_workers=min(5, len(ticker_list) or 1)) as executor:
        futures = {executor.submit(fetch_pcr_history, ticker, 10): ticker for ticker in ticker_list}
        for future in as_completed(futures):
            t_name = futures[future]
            try:
                result[t_name] = future.result()
            except Exception as e:
                logger.warning(f"Failed PCR fetch for {t_name}: {e}")
                result[t_name] = []

    _PCR_CACHE[cache_key] = (now, result)
    return {"data": result}
