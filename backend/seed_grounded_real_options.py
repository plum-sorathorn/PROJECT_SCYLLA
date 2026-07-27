import sqlite3
import datetime
import random
import os
import math
import sys
import argparse
import numpy as np
import yfinance as yf
import pandas as pd
import joblib

# Import the 50-ticker universe from the live scanner router so this dataset
# stays in sync with production. Falls back to a hard-coded list if the
# import fails (e.g. running outside the backend venv).
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from routers.unusual_options import SCAN_TICKERS
    TICKERS = list(SCAN_TICKERS)
except Exception:
    TICKERS = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "WMT", "LLY",
        "JPM", "UNH", "V", "XOM", "MA", "ORCL", "COST", "HD", "PG", "NFLX",
        "BAC", "JNJ", "ABBV", "CRM", "CVX", "AMD", "KO", "PEP", "MRK", "TMO",
        "PLTR", "DIS", "WFC", "ABT", "CSCO", "GE", "ACN", "IBM", "MCD", "NOW",
        "INTU", "QCOM", "TXN", "GS", "AMAT", "CAT", "MS", "AMGN", "INTC", "UBER",
    ]

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scylla_ml.db"))
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
CACHE_FILE = os.path.join(CACHE_DIR, "cache_predictions_walkforward.pkl")
MODEL_PATH = os.path.join(CACHE_DIR, "scylla_predictor.pkl")

# Synthetic success threshold: an option return of >=5% over a ~10-day horizon is
# a realistic bar for a long premium/short premium edge in equity options.
# (Stock-only returns of 5% over 10 days are trivially easy; option P&L is
#  convex in vol/gamma so 5% on the option premium is a meaningful cutoff.)
SUCCESS_THRESHOLD = 0.05

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def bs_call(S, K, T, r, sigma):
    if T <= 0: return max(0.0, S - K)
    if S <= 0 or K <= 0 or sigma <= 0: return max(0.0, S - K)
    return S * norm_cdf(d1(S, K, T, r, sigma)) - K * math.exp(-r * T) * norm_cdf(d2(S, K, T, r, sigma))

def bs_put(S, K, T, r, sigma):
    if T <= 0: return max(0.0, K - S)
    if S <= 0 or K <= 0 or sigma <= 0: return max(0.0, K - S)
    return K * math.exp(-r * T) * norm_cdf(-d2(S, K, T, r, sigma)) - S * norm_cdf(-d1(S, K, T, r, sigma))

def compute_garman_klass_vol(df, window=20):
    """
    Garman-Klass volatility estimate derived from Open, High, Low, Close.
    Gives a much more realistic volatility proxy than close-to-close std.
    """
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * math.log(2) - 1) * (log_co ** 2)
    gk_vol = np.sqrt(np.maximum(gk_var.rolling(window=window, min_periods=5).mean(), 1e-6)) * math.sqrt(252)
    return gk_vol.fillna(0.25)

def seed_grounded_options(wipe: bool = False):
    print(f"Seeding grounded options dataset into: {DB_PATH}")
    print(f"Universe: {len(TICKERS)} tickers from SCAN_TICKERS.")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # DESTRUCTIVE: only allowed with explicit --wipe flag. Default preserves
    # the 44,320 real trades already in options_trades.
    if wipe:
        cursor.execute("DELETE FROM options_trades")
        conn.commit()
        print("[wipe] Cleared existing trades from options_trades table.")
    else:
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0")
        real_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 1")
        synth_count = cursor.fetchone()[0]
        print(f"Preserving existing data: {real_count} real trades, {synth_count} existing synthetic trades.")

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=3650) # ~10 years

    print("Fetching historical VIX benchmark index...")
    try:
        vix_df = yf.download("^VIX", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_closes = vix_df['Close'].iloc[:, 0].dropna()
        else:
            vix_closes = vix_df['Close'].dropna()
    except Exception as e:
        print(f"Warning: Could not fetch ^VIX: {e}. Defaulting to 20.0 VIX baseline.")
        vix_closes = pd.Series(20.0, index=pd.date_range(start_date, end_date))

    total_inserted = 0
    rng = np.random.RandomState(42)

    for ticker in TICKERS:
        print(f"Processing ticker {ticker}...")
        try:
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            if df.empty or len(df) < 100:
                continue

            # Handle MultiIndex columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df_clean = pd.DataFrame({
                    'Open': df['Open'].iloc[:, 0],
                    'High': df['High'].iloc[:, 0],
                    'Low': df['Low'].iloc[:, 0],
                    'Close': df['Close'].iloc[:, 0],
                    'Volume': df['Volume'].iloc[:, 0]
                }, index=df.index).dropna()
            else:
                df_clean = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

            if len(df_clean) < 100:
                continue

            # Technical indicators
            gk_vol = compute_garman_klass_vol(df_clean, window=20)
            sma20 = df_clean['Close'].rolling(20).mean()
            sma50 = df_clean['Close'].rolling(50).mean()

            dates = df_clean.index
            n_rows = len(df_clean)

            # Sample trading dates (approx every 3-5 business days per ticker)
            step_size = 3
            ticker_trades = 0

            for i in range(50, n_rows - 20, step_size):
                trade_date = dates[i]
                timestamp_str = trade_date.strftime("%Y-%m-%d 10:00:00")
                S_t = float(df_clean['Close'].iloc[i])
                if S_t <= 0:
                    continue

                # Align VIX level
                date_key = trade_date.strftime("%Y-%m-%d")
                vix_val = float(vix_closes.loc[date_key]) if date_key in vix_closes.index else 20.0
                if math.isnan(vix_val) or vix_val <= 0:
                    vix_val = 20.0

                base_vol = float(gk_vol.iloc[i])
                if math.isnan(base_vol) or base_vol <= 0:
                    base_vol = 0.25

                # Scale IV by VIX regime vs historical average (VIX 20)
                vix_scale = math.sqrt(vix_val / 20.0)
                atm_iv = base_vol * (0.7 + 0.3 * vix_scale)

                # Trend alignment
                s20 = float(sma20.iloc[i]) if not math.isnan(sma20.iloc[i]) else S_t
                s50 = float(sma50.iloc[i]) if not math.isnan(sma50.iloc[i]) else S_t
                if S_t > s20 and s20 > s50:
                    trend = "BULL_ALIGNED"
                elif S_t < s20 and s20 < s50:
                    trend = "BEAR_ALIGNED"
                else:
                    trend = "NEUTRAL"

                # Generate 2-3 option contracts per date
                for opt_type in ["Call", "Put"]:
                    # Select strike relative to current price
                    strike_mults = [0.95, 1.0, 1.05] if opt_type == "Call" else [0.90, 0.95, 1.0]
                    strike_mult = rng.choice(strike_mults)
                    
                    # Round strike to standard strike increments ($0.50, $1.00, $2.50, $5.00)
                    raw_strike = S_t * strike_mult
                    if S_t > 200:
                        strike = round(raw_strike / 5.0) * 5.0
                    elif S_t > 50:
                        strike = round(raw_strike / 2.5) * 2.5
                    else:
                        strike = round(raw_strike / 1.0) * 1.0
                    if strike <= 0: continue

                    dte = int(rng.choice([15, 30, 45, 60]))
                    r = 0.045 # Risk-free rate

                    # Volatility Skew & Smile: parabolic based on moneyness m = ln(K/S)
                    m = math.log(strike / S_t)
                    skew_adj = 0.15 * (m ** 2) - 0.10 * m * math.sqrt(30.0 / dte)
                    
                    # Stochastic vol-of-vol shock (OU process noise)
                    vov_noise = float(rng.normal(0, 0.03))
                    
                    contract_iv = max(0.10, atm_iv + skew_adj + vov_noise)
                    
                    T_years = dte / 365.0
                    if opt_type == "Call":
                        bs_price = bs_call(S_t, strike, T_years, r, contract_iv)
                    else:
                        bs_price = bs_put(S_t, strike, T_years, r, contract_iv)

                    premium_dollars = bs_price * 100.0
                    if premium_dollars < 15.0 or math.isnan(premium_dollars):
                        continue

                    # Dynamic Bid-Ask Spread Friction (wider for OTM & short DTE)
                    spread_pct = 0.015 + 0.04 * abs(m) + (0.02 / math.sqrt(dte / 30.0))
                    half_spread = (premium_dollars * spread_pct) / 2.0
                    
                    side = "BUY" if (trend == "BULL_ALIGNED" and opt_type == "Call") or (trend == "BEAR_ALIGNED" and opt_type == "Put") or rng.rand() > 0.4 else "SELL"
                    
                    if side == "BUY":
                        entry_price = premium_dollars + half_spread
                    else:
                        entry_price = premium_dollars - half_spread

                    # Real daily path evaluation over next 10 business days (or DTE)
                    eval_days = min(10, n_rows - i - 1)
                    if eval_days <= 1:
                        continue

                    daily_mark_to_market = []
                    for d in range(1, eval_days + 1):
                        curr_idx = i + d
                        S_curr = float(df_clean['Close'].iloc[curr_idx])
                        remaining_dte = max(1, dte - d)
                        rem_T = remaining_dte / 365.0

                        # Updated IV along path with random walk noise
                        curr_gk_vol = float(gk_vol.iloc[curr_idx]) if not math.isnan(gk_vol.iloc[curr_idx]) else base_vol
                        curr_contract_iv = max(0.10, curr_gk_vol * (0.7 + 0.3 * vix_scale) + skew_adj + float(rng.normal(0, 0.02)))

                        if opt_type == "Call":
                            curr_bs = bs_call(S_curr, strike, rem_T, r, curr_contract_iv) * 100.0
                        else:
                            curr_bs = bs_put(S_curr, strike, rem_T, r, curr_contract_iv) * 100.0

                        # Exit bid/ask value
                        if side == "BUY":
                            exit_val = curr_bs - half_spread
                            pnl_pct = (exit_val - entry_price) / entry_price
                        else:
                            exit_val = curr_bs + half_spread
                            pnl_pct = (entry_price - exit_val) / entry_price

                        daily_mark_to_market.append(pnl_pct)

                    # Calculate exact MAE (Max Adverse Excursion) and final observed return
                    mae = float(np.min(daily_mark_to_market)) if daily_mark_to_market else 0.0
                    final_return = float(daily_mark_to_market[-1]) if daily_mark_to_market else 0.0

                    # Cap extreme outliers to realistic market ranges (-100% to +300%)
                    final_return = max(-1.0, min(3.0, final_return))
                    mae = max(-1.0, min(0.0, mae))

                    success = 1 if final_return >= SUCCESS_THRESHOLD else 0

                    # Volume & Open Interest correlated with stock volume
                    avg_vol = int(df_clean['Volume'].iloc[i]) if not math.isnan(df_clean['Volume'].iloc[i]) else 1000000
                    base_opt_vol = int(max(50, (avg_vol / 5000) * (1.0 / (1.0 + abs(m) * 5))))
                    volume = int(rng.randint(int(base_opt_vol * 0.5), int(base_opt_vol * 2.0)))
                    open_interest = int(rng.randint(int(base_opt_vol * 2.0), int(base_opt_vol * 10.0)))
                    vol_oi_ratio = round(volume / open_interest, 2) if open_interest > 0 else 0.0

                    iv_pct = round(contract_iv * 100.0, 2)
                    eval_date_str = dates[i + eval_days].strftime("%Y-%m-%d")
                    expiration_str = (trade_date + datetime.timedelta(days=dte)).strftime("%Y-%m-%d")

                    # Idempotency: skip if (timestamp, ticker, strike, option_type) already exists.
                    cursor.execute(
                        "SELECT 1 FROM options_trades WHERE timestamp = ? AND ticker = ? AND strike = ? AND option_type = ? LIMIT 1",
                        (timestamp_str, ticker, round(strike, 2), opt_type),
                    )
                    if cursor.fetchone() is not None:
                        continue

                    cursor.execute('''
                        INSERT INTO options_trades (
                            timestamp, ticker, expiration, strike, option_type, volume,
                            open_interest, vol_oi_ratio, implied_vol, underlier_price,
                            premium, side, dte, is_weekly, trend_alignment, labeled, label_success, observed_return, evaluation_date, max_adverse_return, is_synthetic
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?, ?, 1)
                    ''', (
                        timestamp_str, ticker, expiration_str, round(strike, 2), opt_type, volume,
                        open_interest, vol_oi_ratio, iv_pct, round(S_t, 2), round(entry_price, 2), side, dte,
                        trend, success, round(final_return, 4), eval_date_str, round(mae, 4)
                    ))

                    ticker_trades += 1
                    total_inserted += 1

            conn.commit()
            print(f"Successfully inserted {ticker_trades} trades for {ticker}.")
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    conn.close()

    # Invalidate stale prediction caches so model & backtester retrain cleanly
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            print("Cleared stale walkforward prediction cache.")
        except Exception:
            pass

    print(f"Finished seeding grounded options dataset! Total trades inserted: {total_inserted}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the SCYLLA synthetic options dataset grounded in real yfinance OHLCV.")
    parser.add_argument("--wipe", action="store_true", help="DESTRUCTIVE: delete all existing options_trades before seeding.")
    args = parser.parse_args()
    seed_grounded_options(wipe=args.wipe)
