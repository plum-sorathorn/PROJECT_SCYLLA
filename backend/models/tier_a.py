"""Tier A and profitable ticker filtering.

Extracted from ml_model.py: the TIER_A_TICKERS computation (data-driven
profitable-ticker detection) and the hardcoded TIER_PROFITABLE_TICKERS
v1 universe.
"""
import sqlite3
import threading
import logging
import pandas as pd

try:
    from ..config.constants import DB_PATH
except ImportError:
    from config.constants import DB_PATH

logger = logging.getLogger("scylla.ml_model")

# PHASE A (5.4): TIER_A_TICKERS — universe filter for new strategies.
# Lazy-initialized: computed on first call from past-year labeled real trades
# with >= 30% win rate (where win = observed_return >= profit_threshold) and
# at least 10 trades. This empirically restricts the strategy to tickers where
# the v2_settlement labeling actually has positive signal, rather than
# "scanning" the whole market and diluting with noise tickers.
_TIER_A_TICKERS_CACHE = None
_TIER_A_TICKERS_LOCK = threading.Lock()


def _compute_tier_a_tickers() -> set:
    """Return the set of tickers that meet TIER_A criteria.

    Cached at module level for the lifetime of the process. If the DB is
    unavailable or has < 1 year of data, returns an empty set (which means
    NO trade will pass the new strategy filters — a safe default for the
    pre-Phase-C cutover).
    """
    global _TIER_A_TICKERS_CACHE
    if _TIER_A_TICKERS_CACHE is not None:
        return _TIER_A_TICKERS_CACHE
    with _TIER_A_TICKERS_LOCK:
        if _TIER_A_TICKERS_CACHE is not None:
            return _TIER_A_TICKERS_CACHE
        try:
            target_pct = 0.03
            conn = sqlite3.connect(DB_PATH)
            max_ts_row = pd.read_sql_query(
                "SELECT MAX(timestamp) as max_ts FROM options_trades WHERE labeled = 1",
                conn,
            )
            conn.close()
            max_ts = max_ts_row["max_ts"].iloc[0]
            if not max_ts:
                logger.warning("TIER_A_TICKERS: no labeled trades in DB")
                _TIER_A_TICKERS_CACHE = set()
                return _TIER_A_TICKERS_CACHE
            cutoff = (pd.to_datetime(max_ts) - pd.Timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("""
                SELECT ticker,
                       AVG(CASE WHEN observed_return >= ? THEN 1.0 ELSE 0.0 END) as wr,
                       COUNT(*) as n
                FROM options_trades
                WHERE labeled = 1 AND is_synthetic = 0
                  AND timestamp >= ?
                GROUP BY ticker
                HAVING n >= 10 AND wr >= 0.30
                ORDER BY wr DESC
            """, conn, params=[target_pct, cutoff])
            conn.close()
            _TIER_A_TICKERS_CACHE = set(df['ticker'].tolist())
            logger.info(
                f"TIER_A_TICKERS computed: {len(_TIER_A_TICKERS_CACHE)} tickers meet criteria "
                f"(cutoff={cutoff}, target_pct={target_pct}, top 5: {list(_TIER_A_TICKERS_CACHE)[:5]})"
            )
            return _TIER_A_TICKERS_CACHE
        except Exception as ex:
            logger.warning(f"Failed to compute TIER_A_TICKERS: {ex}")
            _TIER_A_TICKERS_CACHE = set()
            return _TIER_A_TICKERS_CACHE


# TIER_PROFITABLE_TICKERS: v1 hardcoded universe identified by the Phase A
# per-ticker diagnostic. These underlyings produced positive PnL across
# multiple strategies; the rest (TSLA, NVDA, AMD, META, MSFT, AMZN, GOOGL, ARKK)
# were perennially unprofitable. Next retrain should re-evaluate via
# walkforward on a rolling 6-month lookback.
TIER_PROFITABLE_TICKERS = {
    "IWM", "JPM", "BAC", "GS", "CVX", "AAPL", "BABA"
}


def _compute_tier_profitable_tickers() -> set:
    """Return the Phase A profitable-ticker universe (v1 hardcoded).

    Identified by the per-ticker diagnostic in the previous iteration: these
    underlyings produced positive PnL across multiple strategies. The
    remainder of the in-DB ticker list (TSLA, NVDA, AMD, META, MSFT, AMZN,
    GOOGL, ARKK) was perennially unprofitable.

    NOTE: this is a v1 universe, not data-driven. The next retrain should
    re-evaluate via walkforward on a rolling 6-month lookback and replace
    this hardcoded set with a freshly-computed one.
    """
    return set(TIER_PROFITABLE_TICKERS)
