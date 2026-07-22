import sqlite3
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.routers.ml_model import api_backtest, BacktestRequestSchema

def run_backtest(params):
    req = BacktestRequestSchema(**params)
    res = api_backtest(req)
    return res

strategies = [
    {
        "name": "standard_portfolio",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "standard",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 5
        },
        "grid": [
            {"prob_threshold": 0.6, "kelly_multiplier": 0.3, "stop_lambda": 1.0},
            {"prob_threshold": 0.65, "kelly_multiplier": 0.5, "stop_lambda": 1.5},
            {"prob_threshold": 0.7, "kelly_multiplier": 0.2, "stop_lambda": 1.0}
        ]
    },
    {
        "name": "aggressive_kelly",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "standard",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 10,
            "kelly_cap": 0.5
        },
        "grid": [
            {"prob_threshold": 0.55, "kelly_multiplier": 1.0, "stop_lambda": 2.0},
            {"prob_threshold": 0.6, "kelly_multiplier": 0.8, "stop_lambda": 2.5},
            {"prob_threshold": 0.5, "kelly_multiplier": 1.0, "stop_lambda": 1.5}
        ]
    }
]

results = []

for strat in strategies:
    for g in strat["grid"]:
        params = strat["base_params"].copy()
        params.update(g)
        print(f"Running {strat['name']} with {g}...")
        try:
            res = run_backtest(params)
            if hasattr(res, "summary"):
                metrics = res.summary
            elif isinstance(res, dict) and "summary" in res:
                metrics = res["summary"]
            else:
                print(f"API Error: {res}")
                continue
            
            if isinstance(metrics, dict):
                sharpe = metrics.get('sharpe', 0)
                sortino = metrics.get('sortino', 0)
                win_rate = metrics.get('win_rate_pct', 0)
                max_drawdown = metrics.get('max_drawdown_pct', 0)
                total_return = metrics.get('cumulative_pnl_pct', 0)
                profit_factor = metrics.get('profit_factor', 0)
                total_trades = metrics.get('trades_triggered', 0)
            else:
                sharpe = getattr(metrics, 'sharpe', 0)
                sortino = getattr(metrics, 'sortino', 0)
                win_rate = getattr(metrics, 'win_rate_pct', 0)
                max_drawdown = getattr(metrics, 'max_drawdown_pct', 0)
                total_return = getattr(metrics, 'cumulative_pnl_pct', 0)
                profit_factor = getattr(metrics, 'profit_factor', 0)
                total_trades = getattr(metrics, 'trades_triggered', 0)

            print(f"Done. Sharpe: {sharpe}")
            results.append({
                "strategy": strat["name"],
                "params": g,
                "sharpe": sharpe,
                "sortino": sortino,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "profit_factor": profit_factor,
                "total_trades": total_trades
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error on {strat['name']} {g}: {e}")

df = pd.DataFrame(results)
if not df.empty:
    print("\n=== STRATEGY RANKING (by Sharpe Ratio) ===")
    df = df.sort_values(by="sharpe", ascending=False)
    print(df.to_string())

    df.to_csv(os.path.join(os.path.dirname(__file__), "strategy_results.csv"), index=False)
else:
    print("No results to display.")
