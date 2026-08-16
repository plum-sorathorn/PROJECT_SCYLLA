"""
Volume Concentration by Expiration
Returns aggregated call/put volume bucketed by expiration cycle for stacked bar chart.
"""

from fastapi import APIRouter, Query
import yfinance as yf
import pandas as pd
import logging

from ._yf_safe import safe_call

logger = logging.getLogger("scylla.volume_concentration")
router = APIRouter()


@router.get("/volume-concentration")
def get_volume_concentration(
    ticker: str = Query(default="SPY", description="Ticker symbol"),
    max_expiries: int = Query(default=8, description="Number of expiry cycles to return"),
):
    """Returns aggregated option volume by expiration, split by Call and Put."""
    try:
        tk = safe_call(yf.Ticker, ticker, retries=1)
        all_exps = safe_call(lambda t: list(t.options) if t.options else [], tk)
        expirations = all_exps[:max_expiries]
        rows = []

        for exp in expirations:
            try:
                chain = safe_call(lambda t, e: t.option_chain(e), tk, exp)
                call_vol = int(chain.calls["volume"].fillna(0).sum())
                put_vol = int(chain.puts["volume"].fillna(0).sum())
                rows.append({
                    "expiration": exp,
                    "callVolume": call_vol,
                    "putVolume": put_vol,
                    "totalVolume": call_vol + put_vol,
                })
            except Exception:
                continue

        return {"ticker": ticker, "data": rows}
    except Exception as e:
        logger.warning(f"Vol concentration fetch failed: {e}")
        return {"ticker": ticker, "data": [], "error": str(e)}
