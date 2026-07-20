import sqlite3
import datetime
import random
import os
import math
import numpy as np
import yfinance as yf
import pandas as pd

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scylla_ml.db"))

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

def bs_call(S, K, T, r, sigma):
    if T <= 0: return max(0.0, S - K)
    return S * norm_cdf(d1(S, K, T, r, sigma)) - K * math.exp(-r * T) * norm_cdf(d2(S, K, T, r, sigma))

def bs_put(S, K, T, r, sigma):
    if T <= 0: return max(0.0, K - S)
    return K * math.exp(-r * T) * norm_cdf(-d2(S, K, T, r, sigma)) - S * norm_cdf(-d1(S, K, T, r, sigma))

def generate_bridge(S0, ST, T, sigma, steps):
    """Generate Geometric Brownian Bridge from S0 to ST over `steps`."""
    if steps <= 1:
        return [S0, ST]
    
    log_S0 = math.log(S0)
    log_ST = math.log(ST)
    
    dt = T / steps
    paths = [log_S0]
    
    for i in range(1, steps):
        rem_steps = steps - i + 1
        mean = paths[-1] + (log_ST - paths[-1]) / rem_steps
        var = (sigma**2) * dt * (rem_steps - 1) / rem_steps
        Z = np.random.normal(0, 1)
        next_log_S = mean + math.sqrt(max(var, 0)) * Z
        paths.append(next_log_S)
        
    paths.append(log_ST)
    return [math.exp(x) for x in paths]

def seed():
    print(f"Seeding database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing real trades to keep database clean
    cursor.execute("DELETE FROM options_trades WHERE is_synthetic = 0")
    conn.commit()
    print("Cleared existing trades from database.")
    
    tickers = ["AAPL", "NVDA", "TSLA"] # Kept small for speed
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730) # 2 years
    
    inserted_count = 0
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        try:
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                # Sometimes yfinance returns MultiIndex, just take the 'Close' at level 0
                closes = df['Close'].iloc[:, 0].dropna().values if isinstance(df['Close'], pd.DataFrame) else df['Close'].dropna().values
            else:
                closes = df['Close'].dropna().values
            
            dates = df.index
            
            if len(closes) < 30:
                continue
                
            returns = np.diff(np.log(closes))
            hist_vol = np.std(returns) * math.sqrt(252)
            if hist_vol == 0 or math.isnan(hist_vol):
                hist_vol = 0.3
                
            bridge_interval = 14
            
            full_price_path = []
            full_date_path = []
            
            for i in range(0, len(closes) - bridge_interval, bridge_interval):
                S0 = float(closes[i])
                ST = float(closes[i + bridge_interval])
                
                T = bridge_interval / 252.0
                bridge_prices = generate_bridge(S0, ST, T, hist_vol, bridge_interval)
                
                full_price_path.extend(bridge_prices[:-1])
                
                segment_dates = pd.date_range(start=dates[i], periods=bridge_interval, freq='B')
                full_date_path.extend(segment_dates.tolist())
                
            last_idx = len(closes) - 1
            full_price_path.append(float(closes[-1]))
            full_date_path.append(dates[-1])
            
            for i in range(len(full_price_path) - 15):
                S_t = full_price_path[i]
                trade_time = full_date_path[i]
                timestamp_str = trade_time.strftime("%Y-%m-%d %H:%M:%S")
                
                for opt_type in ["Call", "Put"]:
                    for strike_mult in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        strike = round(S_t * strike_mult, 2)
                        dte = 30
                        
                        r = 0.045
                        iv = hist_vol + abs(strike_mult - 1.0) * 0.5
                        
                        if opt_type == "Call":
                            premium = bs_call(S_t, strike, dte/365.0, r, iv)
                        else:
                            premium = bs_put(S_t, strike, dte/365.0, r, iv)
                            
                        premium_dollars = premium * 100
                        if premium_dollars < 1.0 or math.isnan(premium_dollars):
                            continue
                            
                        vol_oi = round(random.uniform(1.1, 15.0), 2)
                        iv_pct = round(iv * 100, 2)
                        side = random.choice(["BUY", "SELL"])
                        trend = random.choice(["BULL_ALIGNED", "BEAR_ALIGNED", "NEUTRAL"])
                        
                        eval_idx = i + 10
                        S_eval = full_price_path[eval_idx]
                        eval_date = full_date_path[eval_idx].strftime("%Y-%m-%d")
                        
                        if opt_type == "Call":
                            eval_premium = bs_call(S_eval, strike, (dte-10)/365.0, r, iv) * 100
                        else:
                            eval_premium = bs_put(S_eval, strike, (dte-10)/365.0, r, iv) * 100
                            
                        if premium_dollars > 0:
                            if side == "BUY":
                                observed_ret = (eval_premium - premium_dollars) / premium_dollars
                            else:
                                observed_ret = (premium_dollars - eval_premium) / premium_dollars
                        else:
                            observed_ret = 0
                            
                        success = 1 if observed_ret > 0.05 else 0
                        
                        max_adv = -0.05 if success else -0.20
                        
                        cursor.execute('''
                            INSERT INTO options_trades (
                                timestamp, ticker, expiration, strike, option_type, volume, 
                                open_interest, vol_oi_ratio, implied_vol, underlier_price, 
                                premium, side, dte, is_weekly, trend_alignment, labeled, label_success, observed_return, max_adverse_return, evaluation_date, is_synthetic
                            ) VALUES (?, ?, '2026-08-20', ?, ?, 2000, 400, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?, ?, 0)
                        ''', (
                            timestamp_str, ticker, strike, opt_type, vol_oi, iv_pct, round(S_t, 2), round(premium_dollars, 2), side, dte, trend, success, round(observed_ret, 4), round(max_adv, 4), eval_date
                        ))
                        inserted_count += 1
                        
            conn.commit()
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    conn.close()
    print(f"Successfully seeded {inserted_count} trades using Geometric Brownian Bridge.")

if __name__ == "__main__":
    seed()
