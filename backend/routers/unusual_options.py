"""
Unusual Options Volume Scanner
Fetches EOD unusual options activity using OpenBB + yfinance.
Returns ticker, expiry, strike, type, volume, OI, vol/OI ratio, underlyer price, SMA flags, expected move.
"""

from fastapi import APIRouter, Query
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger("scylla.unusual_options")

router = APIRouter()

# Curated list of high-liquidity tickers for whale scanning
SCAN_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META",
    "AMZN", "GOOGL", "NFLX", "BAC", "GS", "JPM", "XOM", "CVX",
    "IWM", "DIA", "ARKK", "BABA"
]


def fetch_option_chain(ticker: str) -> pd.DataFrame:
    """Fetch full option chain for a ticker via yfinance."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return pd.DataFrame()

        spot = tk.fast_info.get("lastPrice", None)
        if spot is None or spot == 0:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

        # Fetch front 3 expiration cycles for volume depth
        frames = []
        for exp in expirations[:4]:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["optionType"] = "Call"
                puts["optionType"] = "Put"
                combined = pd.concat([calls, puts], ignore_index=True)
                combined["expiration"] = exp
                combined["ticker"] = ticker
                combined["underlierPrice"] = round(spot, 2)
                frames.append(combined)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch chain for {ticker}: {e}")
        return pd.DataFrame()


def compute_sma_flags(ticker: str) -> dict:
    """Compute 50d and 200d SMA alignment flags."""
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty or len(hist) < 50:
            return {"above50dSMA": None, "above200dSMA": None}
        close = hist["Close"]
        price = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        return {
            "above50dSMA": price > sma50,
            "above200dSMA": (price > sma200) if sma200 is not None else None,
        }
    except Exception:
        return {"above50dSMA": None, "above200dSMA": None}


def compute_expected_move(ticker: str) -> "float | None":
    """ATM straddle price = call_price + put_price for front-month near ATM."""
    try:
        tk = yf.Ticker(ticker)
        spot = tk.fast_info.get("lastPrice", 0.0)
        expirations = tk.options
        if not expirations or spot == 0:
            return None

        exp = expirations[0]
        chain = tk.option_chain(exp)
        calls = chain.calls
        puts = chain.puts

        # Find ATM strike
        atm_strike = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].values
        if len(atm_strike) == 0:
            return None
        atm = atm_strike[0]

        call_row = calls[calls["strike"] == atm]
        put_row = puts[puts["strike"] == atm]
        if call_row.empty or put_row.empty:
            return None

        call_mid = (call_row["bid"].values[0] + call_row["ask"].values[0]) / 2
        put_mid = (put_row["bid"].values[0] + put_row["ask"].values[0]) / 2
        straddle = call_mid + put_mid
        return round(straddle, 2)
    except Exception:
        return None


@router.get("/unusual-options")
def get_unusual_options(
    tickers: str = Query(default=",".join(SCAN_TICKERS), description="Comma-separated ticker list"),
    min_vol_oi: float = Query(default=2.0, description="Minimum Vol/OI ratio filter"),
    limit: int = Query(default=100, description="Max rows returned"),
):
    """
    Returns unusual options flow sorted by Vol/OI ratio descending.
    Includes SMA alignment flags and Expected Move per ticker.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    all_rows = []
    sma_cache = {}
    em_cache = {}

    for ticker in ticker_list:
        df = fetch_option_chain(ticker)
        if df.empty:
            continue

        required_cols = ["volume", "openInterest", "strike", "expiration", "optionType",
                         "ticker", "underlierPrice", "impliedVolatility", "lastPrice", "bid", "ask"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)

        df = df[df["volume"] > 0]
        df["volOiRatio"] = df.apply(
            lambda r: round(r["volume"] / r["openInterest"], 2) if r["openInterest"] > 0 else float("inf"),
            axis=1
        )
        df = df[df["volOiRatio"] >= min_vol_oi]

        sma_flags = compute_sma_flags(ticker)
        sma_cache[ticker] = sma_flags
        em = compute_expected_move(ticker)
        em_cache[ticker] = em

        for _, row in df.iterrows():
            all_rows.append({
                "ticker": row["ticker"],
                "expiration": str(row["expiration"]),
                "strike": float(row["strike"]),
                "optionType": row["optionType"],
                "volume": int(row["volume"]),
                "openInterest": int(row["openInterest"]),
                "volOiRatio": row["volOiRatio"],
                "impliedVolatility": round(float(row["impliedVolatility"]) * 100, 2),
                "underlierPrice": float(row["underlierPrice"]),
                "above50dSMA": sma_flags["above50dSMA"],
                "above200dSMA": sma_flags["above200dSMA"],
                "expectedMove": em,
            })

    all_rows.sort(key=lambda x: x["volOiRatio"] if x["volOiRatio"] != float("inf") else 9999, reverse=True)
    return {"data": all_rows[:limit], "count": len(all_rows[:limit])}
