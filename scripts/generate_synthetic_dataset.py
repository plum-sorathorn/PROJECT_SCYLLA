"""
Synthetic Options Dataset Generator
Creates realistic options trade data from actual stock prices using Black-Scholes pricing.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import os
import datetime
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed

TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "GOOGL", "AMD",
    "NFLX", "GS", "JPM", "BAC", "CVX", "XOM", "IWM", "DIA", "ARKK", "BABA"
]

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
HORIZON_DAYS = 10
TRADES_PER_DAY_PER_TICKER = 2
RISK_FREE_RATE = 0.05
VOL_RISK_PREMIUM = 1.15

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def compute_iv_skew(moneyness, base_iv):
    skew = 0.0
    if moneyness < -0.05:
        skew = 0.15
    elif moneyness < -0.02:
        skew = 0.08
    elif moneyness < 0.02:
        skew = 0.0
    elif moneyness < 0.05:
        skew = -0.03
    else:
        skew = -0.05
    return base_iv * (1 + skew)

def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        df['ticker'] = ticker
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['returns'] = df['Close'].pct_change()
        df['realized_vol_30d'] = df['returns'].rolling(30).std() * np.sqrt(252) * 100
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        df = df.dropna(subset=['realized_vol_30d', 'sma_50', 'sma_200'])
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def generate_option_trades(stock_df, ticker):
    trades = []
    rng = np.random.default_rng(hash(ticker) % 2**32)
    
    for i in range(len(stock_df)):
        row = stock_df.iloc[i]
        date = row['Date']
        spot = float(row['Close'])
        realized_vol = float(row['realized_vol_30d'])
        sma_50 = float(row['sma_50'])
        sma_200 = float(row['sma_200'])
        
        if pd.isna(realized_vol) or realized_vol <= 0:
            continue
        
        trend = "NEUTRAL"
        if spot > sma_50 and spot > sma_200:
            trend = "BULL_ALIGNED"
        elif spot < sma_50 and spot < sma_200:
            trend = "BEAR_ALIGNED"
        
        for _ in range(TRADES_PER_DAY_PER_TICKER):
            dte = int(rng.choice([14, 21, 30, 45, 60], p=[0.15, 0.25, 0.30, 0.20, 0.10]))
            moneyness = rng.normal(0, 0.03)
            moneyness = np.clip(moneyness, -0.10, 0.10)
            strike = round(spot * (1 + moneyness), 2)
            
            option_type = rng.choice(["Call", "Put"], p=[0.55, 0.45])
            
            base_iv = realized_vol * VOL_RISK_PREMIUM
            iv = compute_iv_skew(moneyness, base_iv)
            iv = max(iv, 5.0)
            iv = min(iv, 150.0)
            
            T = dte / 365.0
            if option_type == "Call":
                premium = black_scholes_call(spot, strike, T, RISK_FREE_RATE, iv / 100)
            else:
                premium = black_scholes_put(spot, strike, T, RISK_FREE_RATE, iv / 100)
            
            premium = max(premium, 0.01)
            
            open_interest = int(rng.lognormal(8, 1.5))
            open_interest = max(open_interest, 100)
            
            vol_oi_ratio = rng.lognormal(0, 0.5)
            vol_oi_ratio = np.clip(vol_oi_ratio, 0.05, 3.0)
            volume = int(open_interest * vol_oi_ratio)
            
            start_idx = max(0, i - 5)
            recent_returns = stock_df.iloc[start_idx:i+1]['returns'].sum()
            if option_type == "Call" and recent_returns > 0:
                side = "BUY"
            elif option_type == "Put" and recent_returns < 0:
                side = "BUY"
            else:
                side = rng.choice(["BUY", "SELL"], p=[0.6, 0.4])
            
            is_weekly = 1 if dte <= 21 else 0
            
            trades.append({
                'timestamp': date.strftime("%Y-%m-%d 10:00:00"),
                'ticker': ticker,
                'expiration': (date + datetime.timedelta(days=dte)).strftime("%Y-%m-%d"),
                'strike': strike,
                'option_type': option_type,
                'volume': volume,
                'open_interest': open_interest,
                'vol_oi_ratio': round(vol_oi_ratio, 2),
                'implied_vol': round(iv, 2),
                'underlier_price': round(spot, 2),
                'premium': round(premium * 100, 2),
                'side': side,
                'dte': dte,
                'is_weekly': is_weekly,
                'trend_alignment': trend,
                'entry_date': date,
                'entry_spot': spot,
                'entry_iv': iv,
                'entry_dte': dte,
            })
    
    return trades

def compute_outcomes(trades, all_stock_data):
    results = []
    
    for trade in trades:
        ticker = trade['ticker']
        entry_date = trade['entry_date']
        entry_spot = trade['entry_spot']
        entry_iv = trade['entry_iv']
        entry_dte = trade['entry_dte']
        strike = trade['strike']
        option_type = trade['option_type']
        side = trade['side']
        
        stock_df = all_stock_data[ticker]
        date_mask = stock_df['Date'] == entry_date
        if not date_mask.any():
            continue
        entry_idx = stock_df[date_mask].index[0]
        
        exit_idx = min(entry_idx + HORIZON_DAYS, len(stock_df) - 1)
        if exit_idx <= entry_idx:
            continue
        
        exit_date = stock_df.iloc[exit_idx]['Date']
        exit_spot = float(stock_df.iloc[exit_idx]['Close'])
        exit_dte = max(entry_dte - HORIZON_DAYS, 0)
        
        slice_start = entry_idx
        slice_end = min(exit_idx + 1, len(stock_df))
        exit_returns = stock_df.iloc[slice_start+1:slice_end]['returns']
        min_spot = float(stock_df.iloc[slice_start:slice_end]['Low'].min())
        max_spot = float(stock_df.iloc[slice_start:slice_end]['High'].max())
        
        exit_vol = stock_df.iloc[slice_start:slice_end]['realized_vol_30d'].mean()
        if pd.isna(exit_vol) or exit_vol <= 0:
            exit_vol = entry_iv
        exit_iv = exit_vol * VOL_RISK_PREMIUM
        
        T_entry = entry_dte / 365.0
        T_exit = exit_dte / 365.0
        
        if option_type == "Call":
            entry_price = black_scholes_call(entry_spot, strike, T_entry, RISK_FREE_RATE, entry_iv / 100)
            exit_price = black_scholes_call(exit_spot, strike, T_exit, RISK_FREE_RATE, exit_iv / 100)
            worst_price = black_scholes_call(min_spot, strike, T_exit, RISK_FREE_RATE, exit_iv / 100)
        else:
            entry_price = black_scholes_put(entry_spot, strike, T_entry, RISK_FREE_RATE, entry_iv / 100)
            exit_price = black_scholes_put(exit_spot, strike, T_exit, RISK_FREE_RATE, exit_iv / 100)
            worst_price = black_scholes_put(max_spot, strike, T_exit, RISK_FREE_RATE, exit_iv / 100)
        
        entry_price = max(entry_price, 0.01)
        exit_price = max(exit_price, 0.01)
        worst_price = max(worst_price, 0.01)
        
        if side == "BUY":
            observed_return = (exit_price - entry_price) / entry_price
            max_adverse_return = (worst_price - entry_price) / entry_price
        else:
            observed_return = (entry_price - exit_price) / entry_price
            max_adverse_return = (entry_price - worst_price) / entry_price
        
        observed_return = np.clip(observed_return, -1.0, 5.0)
        max_adverse_return = np.clip(max_adverse_return, -1.0, 0.0)
        
        trade['evaluation_date'] = exit_date.strftime("%Y-%m-%d")
        trade['observed_return'] = round(observed_return, 4)
        trade['max_adverse_return'] = round(max_adverse_return, 4)
        trade['labeled'] = 1
        trade['label_success'] = 1 if observed_return >= 0.50 else 0
        trade['is_synthetic'] = 0
        
        results.append(trade)
    
    return results

def main():
    print("Fetching stock data...")
    all_stock_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, ticker): ticker for ticker in TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if df is not None:
                    all_stock_data[ticker] = df
                    print(f"  {ticker}: {len(df)} days")
            except Exception as e:
                print(f"  {ticker}: ERROR - {e}")
    
    print(f"\nGenerating option trades for {len(all_stock_data)} tickers...")
    all_trades = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_option_trades, df, ticker): ticker 
                   for ticker, df in all_stock_data.items()}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                trades = future.result()
                all_trades.extend(trades)
                print(f"  {ticker}: {len(trades)} trades")
            except Exception as e:
                print(f"  {ticker}: ERROR - {e}")
    
    print(f"\nTotal trades generated: {len(all_trades)}")
    print("Computing outcomes...")
    results = compute_outcomes(all_trades, all_stock_data)
    print(f"Trades with outcomes: {len(results)}")
    
    db_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'scylla_ml_test.db')
    print(f"\nSaving to {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS options_trades")
    cursor.execute("""
        CREATE TABLE options_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            option_type TEXT NOT NULL,
            volume INTEGER NOT NULL,
            open_interest INTEGER NOT NULL,
            vol_oi_ratio REAL NOT NULL,
            implied_vol REAL NOT NULL,
            underlier_price REAL NOT NULL,
            premium REAL NOT NULL,
            side TEXT NOT NULL,
            dte INTEGER NOT NULL,
            is_weekly INTEGER NOT NULL,
            trend_alignment TEXT NOT NULL,
            labeled INTEGER DEFAULT 0,
            label_success INTEGER DEFAULT NULL,
            observed_return REAL DEFAULT NULL,
            max_adverse_return REAL DEFAULT NULL,
            evaluation_date TEXT DEFAULT NULL,
            is_synthetic INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("DROP TABLE IF EXISTS ml_settings")
    cursor.execute("""
        CREATE TABLE ml_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT OR REPLACE INTO ml_settings (key, value) VALUES ('horizon_days', '10')")
    cursor.execute("INSERT OR REPLACE INTO ml_settings (key, value) VALUES ('profit_threshold', '0.50')")
    
    for trade in results:
        cursor.execute("""
            INSERT INTO options_trades (
                timestamp, ticker, expiration, strike, option_type, volume,
                open_interest, vol_oi_ratio, implied_vol, underlier_price,
                premium, side, dte, is_weekly, trend_alignment,
                labeled, label_success, observed_return, max_adverse_return,
                evaluation_date, is_synthetic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['timestamp'], trade['ticker'], trade['expiration'], trade['strike'],
            trade['option_type'], trade['volume'], trade['open_interest'], trade['vol_oi_ratio'],
            trade['implied_vol'], trade['underlier_price'], trade['premium'], trade['side'],
            trade['dte'], trade['is_weekly'], trade['trend_alignment'],
            trade['labeled'], trade['label_success'], trade['observed_return'],
            trade['max_adverse_return'], trade['evaluation_date'], trade['is_synthetic']
        ))
    
    conn.commit()
    conn.close()
    
    print("\nDataset summary:")
    if not results:
        print("  No trades generated!")
        return
    df = pd.DataFrame(results)
    print(f"  Total trades: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"\n  Return distribution:")
    print(f"    Mean: {df['observed_return'].mean():.4f}")
    print(f"    Median: {df['observed_return'].median():.4f}")
    print(f"    Std: {df['observed_return'].std():.4f}")
    print(f"    Win rate (>0): {(df['observed_return'] > 0).mean()*100:.1f}%")
    print(f"    Win rate (>=50%): {(df['observed_return'] >= 0.50).mean()*100:.1f}%")
    print(f"\n  By option type:")
    for opt_type in ['Call', 'Put']:
        subset = df[df['option_type'] == opt_type]
        print(f"    {opt_type}: n={len(subset)}, avg_ret={subset['observed_return'].mean():.4f}, win%={(subset['observed_return']>0).mean()*100:.1f}%")
    print(f"\n  By side:")
    for side in ['BUY', 'SELL']:
        subset = df[df['side'] == side]
        print(f"    {side}: n={len(subset)}, avg_ret={subset['observed_return'].mean():.4f}, win%={(subset['observed_return']>0).mean()*100:.1f}%")
    print(f"\n  By trend:")
    for trend in ['BULL_ALIGNED', 'NEUTRAL', 'BEAR_ALIGNED']:
        subset = df[df['trend_alignment'] == trend]
        print(f"    {trend}: n={len(subset)}, avg_ret={subset['observed_return'].mean():.4f}, win%={(subset['observed_return']>0).mean()*100:.1f}%")
    
    print(f"\n[OK] Dataset saved to {db_path}")

if __name__ == "__main__":
    main()
