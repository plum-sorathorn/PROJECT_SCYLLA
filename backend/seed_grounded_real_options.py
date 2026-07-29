import sqlite3
import datetime
import random
import os
import math
import sys
import argparse
import bisect
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
COMMISSION_PER_SIDE = 1.10

FOMC_DATES = [
    "2016-01-26", "2016-03-15", "2016-04-26", "2016-06-14", "2016-07-26", "2016-09-20", "2016-11-01", "2016-12-13",
    "2017-01-31", "2017-03-14", "2017-05-02", "2017-06-13", "2017-07-25", "2017-09-19", "2017-10-31", "2017-12-12",
    "2018-01-30", "2018-03-20", "2018-05-01", "2018-06-12", "2018-07-31", "2018-09-25", "2018-11-07", "2018-12-18",
    "2019-01-29", "2019-03-19", "2019-04-30", "2019-06-18", "2019-07-30", "2019-09-17", "2019-10-29", "2019-12-10",
    "2020-01-28", "2020-03-15", "2020-04-28", "2020-06-09", "2020-07-28", "2020-09-15", "2020-11-04", "2020-12-15",
    "2021-01-26", "2021-03-16", "2021-04-27", "2021-06-15", "2021-07-27", "2021-09-21", "2021-11-02", "2021-12-14",
    "2022-01-25", "2022-03-15", "2022-05-03", "2022-06-14", "2022-07-26", "2022-09-20", "2022-11-01", "2022-12-13",
    "2023-01-31", "2023-03-21", "2023-05-02", "2023-06-13", "2023-07-25", "2023-09-19", "2023-10-31", "2023-12-12",
    "2024-01-30", "2024-03-19", "2024-04-30", "2024-06-11", "2024-07-30", "2024-09-17", "2024-11-06", "2024-12-17",
    "2025-01-28", "2025-03-18", "2025-04-29", "2025-06-17", "2025-07-29", "2025-09-16", "2025-10-28", "2025-12-09",
    "2026-01-27", "2026-03-17", "2026-04-28", "2026-06-16", "2026-07-28", "2026-09-15", "2026-10-27", "2026-12-15",
]
FOMC_DATES_PARSED = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES]

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

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def greek_delta(opt_type, S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6 or S <= 0:
        return 1.0 if (opt_type == "Call" and S > K) else (-1.0 if (opt_type == "Put" and S < K) else 0.0)
    _d1 = d1(S, K, T, r, sigma)
    return norm_cdf(_d1) if opt_type == "Call" else norm_cdf(_d1) - 1.0

def greek_gamma(opt_type, S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6 or S <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return norm_pdf(_d1) / (S * sigma * math.sqrt(T))

def greek_vega(opt_type, S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6 or S <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return S * norm_pdf(_d1) * math.sqrt(T)

def greek_theta(opt_type, S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6 or S <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    term1 = -S * norm_pdf(_d1) * sigma / (2.0 * math.sqrt(T))
    if opt_type == "Call":
        term2 = -r * K * math.exp(-r * T) * norm_cdf(_d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm_cdf(-_d2)
    return term1 + term2

def greek_rho(opt_type, S, K, T, r, sigma):
    if T < 1e-6 or sigma < 1e-6 or S <= 0:
        return 0.0
    _d2 = d2(S, K, T, r, sigma)
    if opt_type == "Call":
        return K * T * math.exp(-r * T) * norm_cdf(_d2)
    return -K * T * math.exp(-r * T) * norm_cdf(-_d2)

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

def seed_grounded_options(wipe: bool = False, wipe_all: bool = False, seed: int = 42):
    print(f"Seeding grounded options dataset into: {DB_PATH}")
    print(f"Universe: {len(TICKERS)} tickers from SCAN_TICKERS.")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if wipe_all:
        cursor.execute("DELETE FROM options_trades")
        conn.commit()
        print("[wipe-all] DESTRUCTIVE: cleared ALL rows including real data from options_trades.")
    elif wipe:
        cursor.execute("DELETE FROM options_trades WHERE is_synthetic = 1")
        conn.commit()
        print("[wipe] Cleared existing SYNTHETIC trades from options_trades (real data preserved).")
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
    rng = np.random.RandomState(seed)

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

            earnings_dates_by_ticker = []
            try:
                _ticker_obj = yf.Ticker(ticker)
                _ed = _ticker_obj.earnings_dates
                if _ed is not None and len(_ed) > 0:
                    earnings_dates_by_ticker = sorted([idx.date() if hasattr(idx, "date") else idx for idx in _ed.index])
            except Exception:
                earnings_dates_by_ticker = []

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

                trade_date_only = trade_date.date() if hasattr(trade_date, "date") else trade_date

                if earnings_dates_by_ticker:
                    _pos_e = bisect.bisect_left(earnings_dates_by_ticker, trade_date_only)
                    if _pos_e < len(earnings_dates_by_ticker):
                        days_to_earnings = (earnings_dates_by_ticker[_pos_e] - trade_date_only).days
                    elif _pos_e > 0:
                        days_to_earnings = -(trade_date_only - earnings_dates_by_ticker[_pos_e - 1]).days
                    else:
                        days_to_earnings = None
                else:
                    days_to_earnings = None
                is_earnings_window = 1 if (days_to_earnings is not None and abs(days_to_earnings) <= 3) else 0

                _pos_f = bisect.bisect_left(FOMC_DATES_PARSED, trade_date_only)
                if _pos_f < len(FOMC_DATES_PARSED):
                    days_to_fomc = (FOMC_DATES_PARSED[_pos_f] - trade_date_only).days
                elif _pos_f > 0:
                    days_to_fomc = -(trade_date_only - FOMC_DATES_PARSED[_pos_f - 1]).days
                else:
                    days_to_fomc = None
                is_fomc_day = 1 if (days_to_fomc is not None and abs(days_to_fomc) <= 1) else 0

                if vix_val < 15.0:
                    vix_regime = "low"
                elif vix_val < 22.0:
                    vix_regime = "med"
                elif vix_val < 30.0:
                    vix_regime = "high"
                else:
                    vix_regime = "extreme"

                if i < 200:
                    market_regime = None
                else:
                    _ret_200d = (S_t / float(df_clean['Close'].iloc[i - 200])) - 1.0
                    _trend_up = _ret_200d > 0.0
                    _prior_20d_vol = float(gk_vol.iloc[i - 20])
                    if math.isnan(_prior_20d_vol) or _prior_20d_vol <= 0:
                        _prior_20d_vol = base_vol
                    _vol_expansion = base_vol > _prior_20d_vol
                    if _trend_up and _vol_expansion:
                        market_regime = "trend_up_vol_expansion"
                    elif _trend_up and not _vol_expansion:
                        market_regime = "trend_up_vol_contraction"
                    elif not _trend_up and _vol_expansion:
                        market_regime = "trend_down_vol_expansion"
                    else:
                        market_regime = "trend_down_vol_contraction"

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

                    ret_5d_pre = (S_t / df_clean['Close'].iloc[i - 5]) - 1.0 if i >= 5 else 0.0
                    weekly_gate = (vix_val > 25.0) or (abs(ret_5d_pre) > 0.06)
                    if weekly_gate:
                        dte = int(rng.choice([7, 14, 30, 45, 60], p=[0.35, 0.25, 0.20, 0.12, 0.08]))
                    else:
                        dte = int(rng.choice([15, 30, 45, 60], p=[0.15, 0.30, 0.30, 0.25]))
                    is_weekly = 1 if dte <= 7 else 0
                    intrinsic = max(0.0, S_t - strike) if opt_type == "Call" else max(0.0, strike - S_t)
                    early_exercise_risk = 1 if (dte <= 7 and intrinsic > 0.0) else 0
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

                    delta_entry = greek_delta(opt_type, S_t, strike, T_years, r, contract_iv)
                    gamma_entry = greek_gamma(opt_type, S_t, strike, T_years, r, contract_iv)
                    vega_entry = greek_vega(opt_type, S_t, strike, T_years, r, contract_iv)
                    theta_entry = greek_theta(opt_type, S_t, strike, T_years, r, contract_iv)
                    rho_entry = greek_rho(opt_type, S_t, strike, T_years, r, contract_iv)

                    premium_dollars = bs_price * 100.0
                    if premium_dollars < 15.0 or math.isnan(premium_dollars):
                        continue

                    # Dynamic Bid-Ask Spread Friction (wider for OTM & short DTE)
                    spread_pct = 0.015 + 0.04 * abs(m) + (0.02 / math.sqrt(dte / 30.0))
                    half_spread = (premium_dollars * spread_pct) / 2.0
                    bid_ask_spread_pct = round(spread_pct, 4)
                    commission_per_contract = round(COMMISSION_PER_SIDE, 2)
                    
                    if trend == "BULL_ALIGNED":
                        side = rng.choice(["BUY", "SELL", "MID"], p=[0.65, 0.30, 0.05])
                    elif trend == "BEAR_ALIGNED":
                        side = rng.choice(["BUY", "SELL", "MID"], p=[0.40, 0.55, 0.05])
                    else:
                        side = rng.choice(["BUY", "SELL", "MID"], p=[0.55, 0.40, 0.05])
                    
                    if side == "BUY":
                        entry_price = premium_dollars + half_spread
                    elif side == "SELL":
                        entry_price = premium_dollars - half_spread
                    else:
                        entry_price = premium_dollars

                    # Real daily path evaluation over next 10 business days (or DTE)
                    eval_days = min(10, n_rows - i - 1)
                    if eval_days <= 1:
                        continue

                    daily_mark_to_market = []
                    delta_exit = gamma_exit = vega_exit = theta_exit = rho_exit = 0.0
                    spot_exit = S_t
                    for d in range(1, eval_days + 1):
                        curr_idx = i + d
                        S_curr = float(df_clean['Close'].iloc[curr_idx])
                        remaining_dte = max(1, dte - d)
                        rem_T = remaining_dte / 365.0

                        curr_gk_vol = float(gk_vol.iloc[curr_idx]) if not math.isnan(gk_vol.iloc[curr_idx]) else base_vol
                        curr_contract_iv = max(0.10, curr_gk_vol * (0.7 + 0.3 * vix_scale) + skew_adj + float(rng.normal(0, 0.02)))

                        if opt_type == "Call":
                            curr_bs = bs_call(S_curr, strike, rem_T, r, curr_contract_iv) * 100.0
                        else:
                            curr_bs = bs_put(S_curr, strike, rem_T, r, curr_contract_iv) * 100.0

                        if d == eval_days:
                            spot_exit = S_curr
                            delta_exit = greek_delta(opt_type, S_curr, strike, rem_T, r, curr_contract_iv)
                            gamma_exit = greek_gamma(opt_type, S_curr, strike, rem_T, r, curr_contract_iv)
                            vega_exit = greek_vega(opt_type, S_curr, strike, rem_T, r, curr_contract_iv)
                            theta_exit = greek_theta(opt_type, S_curr, strike, rem_T, r, curr_contract_iv)
                            rho_exit = greek_rho(opt_type, S_curr, strike, rem_T, r, curr_contract_iv)

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

                    if S_t > 0:
                        delta_hedged_return = round(final_return - delta_entry * (spot_exit / S_t - 1.0), 4)
                    else:
                        delta_hedged_return = None

                    success = 1 if final_return >= SUCCESS_THRESHOLD else 0

                    # Volume & Open Interest correlated with stock volume
                    avg_vol = int(df_clean['Volume'].iloc[i]) if not math.isnan(df_clean['Volume'].iloc[i]) else 1000000
                    base_opt_vol = int(max(50, (avg_vol / 5000) * (1.0 / (1.0 + abs(m) * 5))))
                    ret_5d = (S_t / df_clean['Close'].iloc[i - 5]) - 1.0 if i >= 5 else 0.0
                    whale_gate = (vix_val > 22.0) or (abs(ret_5d) > 0.04)
                    p_whale = 0.18 if whale_gate else 0.02
                    is_whale = rng.rand() < p_whale
                    if is_whale:
                        vol_oi_ratio = round(min(10.0, rng.lognormal(mean=math.log(5.0), sigma=0.35)), 2)
                        open_interest = int(max(100, base_opt_vol * rng.randint(2, 10)))
                        volume = int(max(1, round(open_interest * vol_oi_ratio)))
                    else:
                        volume = int(rng.randint(int(base_opt_vol * 0.5), int(base_opt_vol * 2.0)))
                        open_interest = int(rng.randint(int(base_opt_vol * 2.0), int(base_opt_vol * 10.0)))
                        vol_oi_ratio = round(volume / open_interest, 2) if open_interest > 0 else 0.0

                    iv_pct = round(contract_iv * 100.0, 2)
                    eval_date_str = dates[i + eval_days].strftime("%Y-%m-%d")
                    expiration_str = (trade_date + datetime.timedelta(days=dte)).strftime("%Y-%m-%d")

                    # Idempotency: skip if (timestamp, ticker, strike, option_type) already exists.
                    cursor.execute(
                        "SELECT 1 FROM options_trades WHERE timestamp = ? AND ticker = ? AND strike = ? AND option_type = ? AND ensemble_id = ? LIMIT 1",
                        (timestamp_str, ticker, round(strike, 2), opt_type, seed),
                    )
                    if cursor.fetchone() is not None:
                        continue

                    cursor.execute('''
                        INSERT INTO options_trades (
                            timestamp, ticker, expiration, strike, option_type, volume,
                            open_interest, vol_oi_ratio, implied_vol, underlier_price,
                            premium, side, dte, is_weekly, trend_alignment, labeled, label_success, observed_return, evaluation_date, max_adverse_return, is_synthetic,
                            delta_entry, gamma_entry, vega_entry, theta_entry, rho_entry,
                            delta_exit, gamma_exit, vega_exit, theta_exit, rho_exit,
                            ensemble_id,
                            commission_per_contract, bid_ask_spread_pct, early_exercise_risk, delta_hedged_return,
                            days_to_earnings, is_earnings_window, days_to_fomc, is_fomc_day, vix_regime, market_regime
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp_str, ticker, expiration_str, round(strike, 2), opt_type, volume,
                        open_interest, vol_oi_ratio, iv_pct, round(S_t, 2), round(entry_price, 2), side, dte, is_weekly,
                        trend, success, round(final_return, 4), eval_date_str, round(mae, 4),
                        round(delta_entry, 6), round(gamma_entry, 6), round(vega_entry, 6), round(theta_entry, 6), round(rho_entry, 6),
                        round(delta_exit, 6), round(gamma_exit, 6), round(vega_exit, 6), round(theta_exit, 6), round(rho_exit, 6),
                        seed,
                        commission_per_contract, bid_ask_spread_pct, early_exercise_risk, delta_hedged_return,
                        days_to_earnings, is_earnings_window, days_to_fomc, is_fomc_day, vix_regime, market_regime
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
    parser.add_argument("--wipe", action="store_true", help="DESTRUCTIVE: delete existing SYNTHETIC options_trades rows before seeding. Real data is preserved.")
    parser.add_argument("--wipe-all", action="store_true", help="ULTRA-DESTRUCTIVE: delete ALL options_trades rows INCLUDING real data. Use only if you understand the consequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible synthetic generation. Default 42.")
    args = parser.parse_args()
    seed_grounded_options(wipe=args.wipe, wipe_all=args.wipe_all, seed=args.seed)
