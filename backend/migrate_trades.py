import sqlite3
import datetime
import os
import numpy as np
import yfinance as yf

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "scylla_ml.db"))

def migrate():
    print(f"Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure max_adverse_return column exists
    try:
        cursor.execute("ALTER TABLE options_trades ADD COLUMN max_adverse_return REAL DEFAULT NULL")
        print("Added column max_adverse_return to options_trades table.")
    except sqlite3.OperationalError:
        print("Column max_adverse_return already exists in options_trades table.")

    # Get ml_settings
    cursor.execute("SELECT value FROM ml_settings WHERE key = 'profit_threshold'")
    row = cursor.fetchone()
    profit_threshold = float(row[0]) if row else 0.03

    cursor.execute("SELECT value FROM ml_settings WHERE key = 'horizon_days'")
    row = cursor.fetchone()
    horizon_days = int(row[0]) if row else 10

    print(f"Settings: profit_threshold={profit_threshold}, horizon_days={horizon_days}")

    # Fetch labeled trades
    cursor.execute("""
        SELECT id, timestamp, ticker, option_type, underlier_price, side, evaluation_date 
        FROM options_trades 
        WHERE labeled = 1
    """)
    labeled_trades = cursor.fetchall()
    print(f"Found {len(labeled_trades)} labeled trades to migrate.")

    updated_count = 0
    for row in labeled_trades:
        trade_id, timestamp_str, ticker, option_type, start_price, side, evaluation_date_str = row
        
        trade_date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").date()
        if evaluation_date_str:
            end_date = datetime.datetime.strptime(evaluation_date_str, "%Y-%m-%d").date()
        else:
            end_date = trade_date + datetime.timedelta(days=horizon_days)

        try:
            print(f"Fetching history for {ticker} from {trade_date} to {end_date}...")
            tk = yf.Ticker(ticker)
            hist = tk.history(start=trade_date.strftime("%Y-%m-%d"), 
                              end=(end_date + datetime.timedelta(days=2)).strftime("%Y-%m-%d"))
            
            if hist.empty:
                print(f"No history found for {ticker}")
                continue
                
            hist = hist.loc[trade_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
            if hist.empty:
                print(f"No sliced history found for {ticker} between {trade_date} and {end_date}")
                continue
                
            prices = hist['Close'].values
            is_bullish = (option_type == "Call" and side == "BUY") or (option_type == "Put" and side == "SELL")
            
            if len(prices) > 0:
                max_price = np.max(prices)
                min_price = np.min(prices)
                
                if is_bullish:
                    continuous_favorable_return = (max_price - start_price) / start_price
                    max_adverse_return = (min_price - start_price) / start_price
                else:
                    continuous_favorable_return = (start_price - min_price) / start_price
                    max_adverse_return = (start_price - max_price) / start_price
                
                success = 1 if continuous_favorable_return >= profit_threshold else 0
                
                cursor.execute("""
                    UPDATE options_trades
                    SET observed_return = ?,
                        max_adverse_return = ?,
                        label_success = ?
                    WHERE id = ?
                """, (round(float(continuous_favorable_return), 4), 
                      round(float(max_adverse_return), 4), 
                      success, 
                      trade_id))
                conn.commit()
                updated_count += 1
                print(f"Updated trade {trade_id} ({ticker}): fav_ret={continuous_favorable_return:.4f}, adv_ret={max_adverse_return:.4f}, success={success}")
        except Exception as e:
            print(f"Failed to update trade {trade_id}: {e}")
            
    conn.close()
    print(f"Migration completed. Updated {updated_count} trades.")

if __name__ == "__main__":
    migrate()
