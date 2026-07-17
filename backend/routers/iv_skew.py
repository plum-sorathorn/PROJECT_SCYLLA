"""
IV Skew & IV Rank / Percentile Calculator
Fetches option chain implied volatility data for a ticker and computes:
  - IV Rank (30-day)
  - IV Percentile (30-day)
  - IV Smile (Strike vs IV) for front 2 expiration cycles
Uses yfinance for free historical HV data as IV proxy baseline.
"""

from fastapi import APIRouter, Query
import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("scylla.iv_skew")
router = APIRouter()


def compute_historical_vol(ticker: str, window: int = 30) -> tuple[float, float, float]:
    """
    Compute annualized HV, and use the chain IV distribution to estimate
    IV Rank and IV Percentile over the last `window` trading days.
    Returns (current_iv, iv_rank, iv_percentile).
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < window + 5:
            return 0.0, 0.0, 0.0

        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        # Rolling 30-day annualized vol as HV proxy
        rolling_hv = log_ret.rolling(window).std() * np.sqrt(252) * 100
        rolling_hv = rolling_hv.dropna()

        current_iv = float(rolling_hv.iloc[-1])
        iv_min = float(rolling_hv.min())
        iv_max = float(rolling_hv.max())
        iv_rank = round(((current_iv - iv_min) / (iv_max - iv_min)) * 100, 1) if iv_max != iv_min else 0.0
        iv_pct = round((rolling_hv <= current_iv).mean() * 100, 1)
        return round(current_iv, 2), iv_rank, iv_pct
    except Exception as e:
        logger.warning(f"HV compute failed for {ticker}: {e}")
        return 0.0, 0.0, 0.0


def fetch_iv_smile(ticker: str, num_expiries: int = 2) -> list[dict]:
    """Returns strike vs IV data for the front N expiry cycles."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options[:num_expiries]
        smile_data = []

        for exp in expirations:
            chain = tk.option_chain(exp)
            calls = chain.calls[["strike", "impliedVolatility"]].copy()
            calls["optionType"] = "Call"
            calls["expiration"] = exp
            puts = chain.puts[["strike", "impliedVolatility"]].copy()
            puts["optionType"] = "Put"
            puts["expiration"] = exp

            for _, row in pd.concat([calls, puts]).iterrows():
                iv_val = float(row["impliedVolatility"])
                if 0 < iv_val < 5:  # filter nonsense values
                    smile_data.append({
                        "expiration": exp,
                        "strike": float(row["strike"]),
                        "iv": round(iv_val * 100, 2),
                        "optionType": row["optionType"],
                    })

        return smile_data
    except Exception as e:
        logger.warning(f"IV smile fetch failed for {ticker}: {e}")
        return []


@router.get("/iv-skew")
def get_iv_skew(
    ticker: str = Query(default="SPY", description="Ticker symbol"),
):
    """Returns IV Rank, IV Percentile, and the Volatility Smile for the front 2 expiry cycles."""
    current_iv, iv_rank, iv_pct = compute_historical_vol(ticker)
    smile = fetch_iv_smile(ticker)
    return {
        "ticker": ticker,
        "currentIV": current_iv,
        "ivRank": iv_rank,
        "ivPercentile": iv_pct,
        "smileData": smile,
    }
