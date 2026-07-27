"""
Unusual Options Volume Scanner
Fetches EOD unusual options activity using OpenBB + yfinance.
Returns ticker, expiry, strike, type, volume, OI, vol/OI ratio, underlyer price, SMA flags, expected move.
"""

from fastapi import APIRouter, Query, BackgroundTasks
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger("scylla.unusual_options")

router = APIRouter()

# Curated list of top 50 market cap stocks (excluding indices) for whale scanning
SCAN_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "WMT", "LLY",
    "JPM", "UNH", "V", "XOM", "MA", "ORCL", "COST", "HD", "PG", "NFLX",
    "BAC", "JNJ", "ABBV", "CRM", "CVX", "AMD", "KO", "PEP", "MRK", "TMO",
    "PLTR", "DIS", "WFC", "ABT", "CSCO", "GE", "ACN", "IBM", "MCD", "NOW",
    "INTU", "QCOM", "TXN", "GS", "AMAT", "CAT", "MS", "AMGN", "INTC", "UBER"
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

        # Target expirations between 0 and 90 Days to Expiration (DTE), capping at a maximum of 6
        import datetime
        today = datetime.date.today()
        selected_exps = []
        for exp_str in expirations:
            try:
                exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if 0 <= dte <= 90:
                    selected_exps.append(exp_str)
            except Exception:
                continue

        if not selected_exps and expirations:
            selected_exps = list(expirations[:4])

        # Cap at maximum of 6 expirations
        selected_exps = selected_exps[:6]

        # Fallback to front 4 if no expirations match the 14-90 DTE window
        if not selected_exps:
            selected_exps = expirations[:4]

        frames = []
        for exp in selected_exps:
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


import math

def _sanitize_float_values(obj):
    """Recursively converts NaN, Infinity, -Infinity floats to None for standard JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_float_values(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


def compute_expected_move(ticker: str) -> "float | None":
    """ATM straddle price = call_price + put_price for front-month near ATM."""
    try:
        tk = yf.Ticker(ticker)
        spot = tk.fast_info.get("lastPrice", 0.0)
        expirations = tk.options
        if not expirations or spot == 0:
            return 0.0

        exp = expirations[0]
        chain = tk.option_chain(exp)
        calls = chain.calls
        puts = chain.puts

        # Find ATM strike
        atm_strike = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].values
        if len(atm_strike) == 0:
            return 0.0
        atm = atm_strike[0]

        call_row = calls[calls["strike"] == atm]
        put_row = puts[puts["strike"] == atm]
        if call_row.empty or put_row.empty:
            return 0.0

        call_bid = call_row["bid"].values[0]
        call_ask = call_row["ask"].values[0]
        put_bid = put_row["bid"].values[0]
        put_ask = put_row["ask"].values[0]

        if pd.isna(call_bid) or pd.isna(call_ask) or pd.isna(put_bid) or pd.isna(put_ask):
            return 0.0

        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2
        straddle = call_mid + put_mid
        if pd.isna(straddle) or math.isnan(straddle) or math.isinf(straddle):
            return 0.0

        return round(float(straddle), 2)
    except Exception:
        return 0.0


import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCANNER_CACHE = {}
_CACHE_TTL_SECONDS = 180  # 3 minutes cache TTL


def _process_single_ticker(ticker: str, min_vol_oi: float) -> list[dict]:
    """Process a single ticker's option chain and compute unusual options data."""
    try:
        df = fetch_option_chain(ticker)
        if df.empty:
            return []

        required_cols = ["volume", "openInterest", "strike", "expiration", "optionType",
                         "ticker", "underlierPrice", "impliedVolatility", "lastPrice", "bid", "ask", "lastTradeDate"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)

        # Off-hours fallback: if total volume across chain is 0, use openInterest as synthetic volume for scanning
        if (df["volume"] > 0).sum() == 0 and (df["openInterest"] > 0).sum() > 0:
            df["volume"] = df["openInterest"]

        df = df[df["volume"] > 0]
        if df.empty:
            return []

        df["volOiRatio"] = df.apply(
            lambda r: round(r["volume"] / r["openInterest"], 2) if r["openInterest"] > 0 else 0.0,
            axis=1
        )
        df = df[df["volOiRatio"] >= min_vol_oi]
        if df.empty:
            return []

        sma_flags = compute_sma_flags(ticker)
        em = compute_expected_move(ticker)

        trend_alignment = "NEUTRAL"
        if sma_flags.get("above50dSMA") and sma_flags.get("above200dSMA"):
            trend_alignment = "BULL_ALIGNED"
        elif not sma_flags.get("above50dSMA") and not sma_flags.get("above200dSMA"):
            trend_alignment = "BEAR_ALIGNED"

        rows = []
        for _, row in df.iterrows():
            last_trade_date_str = None
            if not pd.isna(row["lastTradeDate"]):
                if hasattr(row["lastTradeDate"], "tz_convert"):
                    try:
                        if row["lastTradeDate"].tz is None:
                            ny_time = row["lastTradeDate"].tz_localize("UTC").tz_convert("America/New_York")
                        else:
                            ny_time = row["lastTradeDate"].tz_convert("America/New_York")
                        last_trade_date_str = ny_time.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        last_trade_date_str = row["lastTradeDate"].strftime("%Y-%m-%d %H:%M:%S")
                elif hasattr(row["lastTradeDate"], "strftime"):
                    last_trade_date_str = row["lastTradeDate"].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    last_trade_date_str = str(row["lastTradeDate"])

            bid = row["bid"]
            ask = row["ask"]
            last_price = row["lastPrice"]
            side = "MID"

            if pd.notna(bid) and pd.notna(ask) and pd.notna(last_price) and bid is not None and ask is not None and last_price is not None:
                if bid != ask:
                    mid = (bid + ask) / 2
                    if last_price >= ask:
                        side = "BUY"
                    elif last_price <= bid:
                        side = "SELL"
                    elif last_price > mid:
                        side = "BUY"
                    elif last_price < mid:
                        side = "SELL"
                    else:
                        side = "MID"

            # Calculate Days to Expiration (DTE)
            import datetime
            try:
                exp_date = datetime.datetime.strptime(str(row["expiration"]), "%Y-%m-%d").date()
                dte = (exp_date - datetime.date.today()).days
            except Exception:
                dte = 0

            # Calculate if option is weekly or monthly
            is_weekly = True
            try:
                if exp_date.weekday() == 4 and 15 <= exp_date.day <= 21:
                    is_weekly = False
            except Exception:
                pass

            last_price_val = float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0.0
            volume_val = int(row["volume"])
            premium = round(volume_val * last_price_val * 100.0, 2)

            rows.append({
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
                "trendAlignment": trend_alignment,
                "expectedMove": em,
                "lastTradeDate": last_trade_date_str,
                "side": side,
                "dte": dte,
                "premium": premium,
                "isWeekly": is_weekly,
            })
        return rows
    except Exception as e:
        logger.warning(f"Error processing ticker {ticker}: {e}")
        return []


def scan_raw_options(tickers: str, min_vol_oi: float, limit: int) -> list[dict]:
    """Helper function to fetch and calculate raw unusual options data concurrently with caching."""
    cache_key = (tickers, min_vol_oi)
    now = time.time()
    if cache_key in _SCANNER_CACHE:
        cache_time, cached_rows = _SCANNER_CACHE[cache_key]
        if now - cache_time < _CACHE_TTL_SECONDS:
            return cached_rows[:limit]

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return []

    all_rows = []
    # Fetch option chains in parallel across max 20 worker threads
    workers = min(20, len(ticker_list))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_ticker, ticker, min_vol_oi): ticker for ticker in ticker_list}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_rows.extend(res)
            except Exception as e:
                t_name = futures[future]
                logger.warning(f"Failed parallel scan for ticker {t_name}: {e}")

    all_rows.sort(key=lambda x: x["volOiRatio"], reverse=True)
    results = all_rows[:limit]
    _SCANNER_CACHE[cache_key] = (now, results)
    return results


@router.get("/unusual-options")
def get_unusual_options(
    tickers: str = Query(default=",".join(SCAN_TICKERS), description="Comma-separated ticker list"),
    min_vol_oi: float = Query(default=8.0, description="Minimum Vol/OI ratio filter"),
    limit: int = Query(default=100, description="Max rows returned"),
):
    """
    Returns unusual options flow sorted by Vol/OI ratio descending.
    Includes SMA alignment flags and Expected Move per ticker.
    """
    data = scan_raw_options(tickers, min_vol_oi, limit)
    return {"data": data, "count": len(data)}


@router.get("/scanner")
def get_scanner(
    tickers: str = Query(default=",".join(SCAN_TICKERS), description="Comma-separated ticker list"),
    min_vol_oi: float = Query(default=8.0, description="Minimum Vol/OI ratio filter"),
    limit: int = Query(default=100, description="Max rows returned"),
    _probe: int = Query(default=0, description="Fast health probe indicator"),
    background_tasks: BackgroundTasks = None,
):
    """
    C++ Compatibility endpoint: returns processed options flow with whale signals,
    log-normalized Vol/OI ratios, trend alignment classifications, expected move upper/lower ranges,
    and a summary metrics block. Used in DEV MODE directly by the frontend.
    """
    if _probe == 1:
        tickers = "SPY"
    raw_data = scan_raw_options(tickers, min_vol_oi, limit)
    import math

    processed = []
    whales = 0
    sum_ratio = 0.0
    max_ratio = 0.0
    call_vol = 0
    put_vol = 0

    for r in raw_data:
        # Enrich row to match C++ Metrics Engine logic
        is_whale = r["volOiRatio"] >= 5.0
        normalized_vol_oi = round(math.log1p(r["volOiRatio"]), 4)

        em_val = r.get("expectedMove")
        if em_val is None or pd.isna(em_val) or (isinstance(em_val, float) and (math.isnan(em_val) or math.isinf(em_val))):
            expected_move = 0.0
        else:
            expected_move = float(em_val)

        u_price_val = r.get("underlierPrice", 0.0)
        if u_price_val is None or pd.isna(u_price_val) or (isinstance(u_price_val, float) and (math.isnan(u_price_val) or math.isinf(u_price_val))):
            u_price = 0.0
        else:
            u_price = float(u_price_val)

        expected_move_upper = round(u_price + expected_move, 2)
        expected_move_lower = round(u_price - expected_move, 2)

        above50_val = 1 if r["above50dSMA"] is True else (0 if r["above50dSMA"] is False else -1)
        above200_val = 1 if r["above200dSMA"] is True else (0 if r["above200dSMA"] is False else -1)

        bullish_trend = (above50_val == 1 and above200_val != 0)
        bearish_trend = (above50_val == 0 or above200_val == 0)
        is_call = r["optionType"] == "Call"

        if above50_val == -1:
            trend_alignment = "UNKNOWN"
        elif bullish_trend and is_call:
            trend_alignment = "BULL_ALIGNED"
        elif bearish_trend and not is_call:
            trend_alignment = "BEAR_ALIGNED"
        elif bullish_trend and not is_call:
            trend_alignment = "BULL_CONTRARIAN"
        else:
            trend_alignment = "NEUTRAL"

        processed.append({
            "ticker": r["ticker"],
            "expiration": r["expiration"],
            "strike": r["strike"],
            "optionType": r["optionType"],
            "volume": r["volume"],
            "openInterest": r["openInterest"],
            "volOiRatio": r["volOiRatio"],
            "impliedVolatility": r["impliedVolatility"],
            "underlierPrice": r["underlierPrice"],
            "above50dSMA": above50_val,
            "above200dSMA": above200_val,
            "expectedMove": expected_move,
            "expectedMoveUpper": expected_move_upper,
            "expectedMoveLower": expected_move_lower,
            "isWhaleSignal": is_whale,
            "trendAlignment": trend_alignment,
            "normalizedVolOI": normalized_vol_oi,
            "lastTradeDate": r["lastTradeDate"],
            "side": r["side"],
            "dte": r["dte"],
            "premium": r["premium"],
            "isWeekly": r["isWeekly"],
        })

    ratio_list = [r["volOiRatio"] for r in raw_data if r.get("volOiRatio") is not None and pd.notna(r["volOiRatio"])]
    sum_ratio = sum(ratio_list)
    max_ratio = max(ratio_list) if ratio_list else 0.0

    for r in raw_data:
        is_whale = r["volOiRatio"] >= 5.0
        if r["optionType"] == "Call":
            call_vol += r["volume"]
        else:
            put_vol += r["volume"]
        if is_whale:
            whales += 1

    avg_ratio = round(sum_ratio / len(raw_data), 2) if raw_data else 0.0
    aggregate_pcr = round(put_vol / call_vol, 4) if call_vol > 0 else 1.0

    summary = {
        "avgVolOI": avg_ratio,
        "maxVolOI": max_ratio,
        "totalCallVolume": call_vol,
        "totalPutVolume": put_vol,
        "aggregatePCR": aggregate_pcr,
        "whaleSignalCount": whales,
    }

    # Automatically log whale signals to ML Database in background
    if background_tasks:
        try:
            from routers.ml_model import TradeSchema, api_log_trade
            for row in processed:
                if row.get("isWhaleSignal") or row.get("volOiRatio", 0) >= 5.0:
                    trade_schema_data = TradeSchema(
                        ticker=row["ticker"],
                        expiration=row["expiration"],
                        strike=row["strike"],
                        optionType=row["optionType"],
                        volume=row["volume"],
                        openInterest=row["openInterest"],
                        volOiRatio=row["volOiRatio"],
                        impliedVolatility=row["impliedVolatility"],
                        underlierPrice=row["underlierPrice"],
                        premium=row["premium"],
                        side=row["side"],
                        dte=row["dte"],
                        isWeekly=row["isWeekly"],
                        trendAlignment=row["trendAlignment"]
                    )
                    background_tasks.add_task(api_log_trade, trade_schema_data)
        except Exception as e:
            logger.warning(f"Failed to enqueue trade logging: {e}")

    return _sanitize_float_values({
        "data": processed,
        "summary": summary,
        "count": len(processed)
    })
