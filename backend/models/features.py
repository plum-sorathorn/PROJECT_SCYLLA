"""Advanced feature engineering for option trades.

Extracted from ml_model.py: functions that compute engineered features
(moneyness, IV/HV ratio, VIX level, log premium, DTE bucket) from raw
trade columns, plus market-data fetchers with caching.
"""
import datetime
import numpy as np
import pandas as pd
import logging

try:
    from ..routers._yf_safe import safe_call
except ImportError:
    from routers._yf_safe import safe_call
import yfinance as yf

logger = logging.getLogger("scylla.ml_model")

# ── In-memory caches for expensive market data lookups ───────
_hv_cache = {}   # ticker -> annualized HV
_vix_cache = {"value": None, "ts": None}  # cached VIX level


def _fetch_historical_volatility(ticker: str) -> float:
    """Fetch 30-day annualized historical volatility for a ticker (cached)."""
    global _hv_cache
    if ticker in _hv_cache:
        return _hv_cache[ticker]
    try:
        tk = safe_call(yf.Ticker, ticker, retries=1)
        hist = safe_call(lambda t: t.history(period="3mo"), tk)
        if hist.empty or len(hist) < 5:
            _hv_cache[ticker] = 0.0
            return 0.0
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        # Annualized std of log returns (252 trading days)
        hv = float(log_returns.std() * np.sqrt(252) * 100)  # as percentage
        _hv_cache[ticker] = hv
        return hv
    except Exception as ex:
        logger.warning(f"HV fetch failed for {ticker}: {ex}")
        _hv_cache[ticker] = 0.0
        return 0.0


def _fetch_vix_level() -> float:
    """Fetch current VIX level (cached for 5 minutes)."""
    global _vix_cache
    now = datetime.datetime.now()
    if _vix_cache["value"] is not None and _vix_cache["ts"] is not None:
        elapsed = (now - _vix_cache["ts"]).total_seconds()
        if elapsed < 300:  # 5-minute cache
            return _vix_cache["value"]
    try:
        vix = safe_call(yf.Ticker, "^VIX", retries=1)
        hist = safe_call(lambda t: t.history(period="5d"), vix)
        if hist.empty:
            _vix_cache["value"] = 20.0
            _vix_cache["ts"] = now
            return 20.0
        val = float(hist['Close'].iloc[-1])
        _vix_cache["value"] = val
        _vix_cache["ts"] = now
        return val
    except Exception as ex:
        logger.warning(f"VIX fetch failed: {ex}")
        _vix_cache["value"] = 20.0
        _vix_cache["ts"] = now
        return 20.0


def _dte_to_bucket(dte: int) -> str:
    """Categorize DTE into labeled buckets."""
    if dte <= 7:
        return "weekly"
    elif dte <= 30:
        return "short"
    elif dte <= 60:
        return "medium"
    else:
        return "long"


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features from raw trade columns.
    Adds: moneyness, iv_hv_ratio, vix_level, log_premium, dte_bucket.
    Prevents lookahead leakage by avoiding live yfinance calls on historical datasets.
    """
    df = df.copy()

    # 1. Moneyness: percentage distance from spot
    df['moneyness'] = (df['strike'] - df['underlier_price']) / df['underlier_price']

    # 2. IV / HV ratio per ticker
    if 'iv_hv_ratio' not in df.columns:
        if len(df) > 10:
            # PHASE A (5.2): DEFERRED — would require _fetch_hv_at(ticker, ts) per unique
            # ticker as of median timestamp, then iv/median_hv. Yfinance historical
            # fetches are rate-limited (~1-3s per ticker × 50 tickers = 1-3 min per
            # training run), and the v2_settlement labels are noisy enough that the
            # IV/HV-ratio feature is a secondary signal — the strategy filter rules
            # in api_backtest carry more weight. Leaving baseline constant (25.0) for
            # now. TODO(phase_b): implement _fetch_hv_at with disk cache.
            df['iv_hv_ratio'] = df['implied_vol'] / 25.0
        else:
            # Live inference mode: fetch current HV
            tickers = df['ticker'].unique() if 'ticker' in df.columns else []
            for t in tickers:
                _fetch_historical_volatility(t)

            def _iv_hv_ratio(row):
                hv = _hv_cache.get(row.get('ticker', ''), 0.0)
                iv = row.get('implied_vol', 0.0)
                return iv / hv if hv > 0 else 1.0

            df['iv_hv_ratio'] = df.apply(_iv_hv_ratio, axis=1) if 'ticker' in df.columns else 1.0

    # 3. VIX level (global market regime)
    if 'vix_level' not in df.columns:
        if len(df) > 10:
            # PHASE A (5.2): DEFERRED — same reason as iv_hv_ratio above.
            df['vix_level'] = 20.0
        else:
            df['vix_level'] = _fetch_vix_level()

    # 4. Log-scaled premium
    df['log_premium'] = np.log1p(df['premium'].clip(lower=0))

    # 5. DTE bucket
    df['dte_bucket'] = df['dte'].apply(_dte_to_bucket)

    return df


def get_hv_cache():
    """Return the global HV cache dictionary (for read-only access by callers)."""
    return _hv_cache


def get_vix_cache():
    """Return the global VIX cache dict (for read-only access by callers)."""
    return _vix_cache
