"""
Technical Swing Alignment Overlay
Returns SMA 50d / 200d alignment for a list of tickers.
Used to validate if unusual option flow aligns with macro price trend.
"""

from fastapi import APIRouter, Query
import yfinance as yf
import numpy as np
import logging

logger = logging.getLogger("scylla.technicals")
router = APIRouter()


@router.get("/technicals")
def get_technicals(
    tickers: str = Query(default="SPY,QQQ,AAPL,NVDA,TSLA", description="Comma-separated ticker list"),
):
    """Returns last price, 50d SMA, 200d SMA, and alignment flags per ticker."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []

    for ticker in ticker_list:
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty or len(hist) < 50:
                continue
            close = hist["Close"]
            price = round(float(close.iloc[-1]), 2)
            sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
            sma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None
            results.append({
                "ticker": ticker,
                "price": price,
                "sma50": sma50,
                "sma200": sma200,
                "above50dSMA": price > sma50,
                "above200dSMA": (price > sma200) if sma200 is not None else None,
                "trend": "BULLISH" if (price > sma50 and (sma200 is None or price > sma200)) else
                         "MIXED" if (price > sma50 or (sma200 is not None and price > sma200)) else "BEARISH",
            })
        except Exception as e:
            logger.warning(f"Technicals failed for {ticker}: {e}")

    return {"data": results}
