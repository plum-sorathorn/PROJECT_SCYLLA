import sys
import os
import sqlite3

# Add backend to path
sys.path.append(r"C:\Users\plum\Documents\FIN Works\PROJECT_SCYLLA\backend")

from routers.ml_model import api_backtest, BacktestRequestSchema, init_db

def test_backtest():
    print("Initializing Database...")
    init_db()
    
    print("\nRunning Walk-forward Backtest...")
    req = BacktestRequestSchema(
        mode="walkforward",
        initial_capital=100000.0,
        prob_threshold=0.55,  # lower threshold to trigger trades in small dataset
        kelly_multiplier=0.5,
        kelly_cap=0.25,
        stop_lambda=1.2,
        max_risk_pct_per_trade=0.02,
        walkforward_train_window=50,
        walkforward_test_increment=10,
        confirm_direct_dev=False
    )
    try:
        res = api_backtest(req)
        print("Walk-forward Backtest Successful!")
        print(f"Summary: {res['summary']}")
        print(f"Transactions count: {len(res['transactions'])}")
        print(f"Equity curve points: {len(res['equity_curve'])}")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\nRunning Direct Dev (Dev) Backtest...")
    req_dev = BacktestRequestSchema(
        mode="direct_dev",
        initial_capital=100000.0,
        prob_threshold=0.55,
        kelly_multiplier=0.5,
        kelly_cap=0.25,
        stop_lambda=1.2,
        max_risk_pct_per_trade=0.02,
        confirm_direct_dev=True
    )
    try:
        res_dev = api_backtest(req_dev)
        print("Direct Dev Backtest Successful!")
        print(f"Summary: {res_dev['summary']}")
        print(f"Transactions count: {len(res_dev['transactions'])}")
        print(f"Equity curve points: {len(res_dev['equity_curve'])}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest()
