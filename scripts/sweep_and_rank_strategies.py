import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.routers.ml_model import api_backtest, BacktestRequestSchema

# 8 Strategy Specs (4 New + 4 Existing)
strategies = [
    # ── 4 NEW STRATEGIES ──
    {
        "name": "quantile_spread",
        "description": "Trades only when quantile uncertainty spread is tight (P90-P10 <= max_quantile_spread).",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "quantile_spread",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 5,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.60, "max_quantile_spread": 0.35, "kelly_multiplier": 0.5, "stop_lambda": 1.2},
            {"prob_threshold": 0.65, "max_quantile_spread": 0.30, "kelly_multiplier": 0.4, "stop_lambda": 1.5},
            {"prob_threshold": 0.70, "max_quantile_spread": 0.40, "kelly_multiplier": 0.5, "stop_lambda": 1.0}
        ]
    },
    {
        "name": "directional_quantile_shift",
        "description": "Triggers on median return shift confirmed by trend alignment.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "directional_quantile_shift",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 5,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.60, "min_median_return": 0.03, "kelly_multiplier": 0.5, "stop_lambda": 1.2},
            {"prob_threshold": 0.65, "min_median_return": 0.04, "kelly_multiplier": 0.5, "stop_lambda": 1.5},
            {"prob_threshold": 0.55, "min_median_return": 0.02, "kelly_multiplier": 0.6, "stop_lambda": 1.0}
        ]
    },
    {
        "name": "mean_reversion_overlay",
        "description": "Fades extreme quantile predictions when combined with high Vol/OI ratio.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "mean_reversion_overlay",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 3,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.55, "kelly_multiplier": 0.3, "hard_stop_loss": 15.0},
            {"prob_threshold": 0.60, "kelly_multiplier": 0.4, "hard_stop_loss": 20.0},
            {"prob_threshold": 0.50, "kelly_multiplier": 0.3, "hard_stop_loss": 10.0}
        ]
    },
    {
        "name": "volatility_regime_adaptive",
        "description": "Adapts position caps and confidence thresholds based on implied volatility regime.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "volatility_regime_adaptive",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 5,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.60, "kelly_multiplier": 0.5, "kelly_cap": 0.25, "stop_lambda": 1.2},
            {"prob_threshold": 0.65, "kelly_multiplier": 0.4, "kelly_cap": 0.20, "stop_lambda": 1.5},
            {"prob_threshold": 0.70, "kelly_multiplier": 0.5, "kelly_cap": 0.30, "stop_lambda": 1.0}
        ]
    },

    # ── 4 EXISTING STRATEGIES ──
    {
        "name": "highest_prob_scan",
        "description": "Intraday scanner executing highest probability alert prior to cut-off time.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "highest_prob_scan",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 1,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.60, "kelly_multiplier": 0.5, "scan_time": "10:00:00"},
            {"prob_threshold": 0.65, "kelly_multiplier": 0.5, "scan_time": "10:30:00"},
            {"prob_threshold": 0.70, "kelly_multiplier": 0.4, "scan_time": "11:00:00"}
        ]
    },
    {
        "name": "standard_portfolio",
        "description": "Standard multi-position portfolio allocation based on model probability threshold.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "standard",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 5,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.60, "kelly_multiplier": 0.3, "stop_lambda": 1.0},
            {"prob_threshold": 0.65, "kelly_multiplier": 0.5, "stop_lambda": 1.5},
            {"prob_threshold": 0.70, "kelly_multiplier": 0.2, "stop_lambda": 1.0}
        ]
    },
    {
        "name": "conservative_tight_stops",
        "description": "Conservative strategy demanding high probability cutoff and tight hard stops.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "standard",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 3,
            "kelly_cap": 0.10,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.75, "kelly_multiplier": 0.2, "hard_stop_loss": 20.0},
            {"prob_threshold": 0.80, "kelly_multiplier": 0.25, "hard_stop_loss": 10.0},
            {"prob_threshold": 0.70, "kelly_multiplier": 0.15, "hard_stop_loss": 15.0}
        ]
    },
    {
        "name": "aggressive_kelly",
        "description": "Aggressive strategy targeting max total return with higher Kelly cap and wider stop bounds.",
        "base_params": {
            "mode": "walkforward",
            "strategy_type": "standard",
            "walkforward_train_window": 500,
            "walkforward_test_increment": 100,
            "initial_capital": 100000.0,
            "max_concurrent_trades": 10,
            "kelly_cap": 0.50,
            "slippage_pct": 0.005
        },
        "grid": [
            {"prob_threshold": 0.55, "kelly_multiplier": 1.0, "stop_lambda": 2.0},
            {"prob_threshold": 0.60, "kelly_multiplier": 0.8, "stop_lambda": 2.5},
            {"prob_threshold": 0.50, "kelly_multiplier": 1.0, "stop_lambda": 1.5}
        ]
    }
]

results = []

print("=== STARTING STRATEGY SWEEP & RANKING ON WALK-FORWARD DATA ===")

for strat in strategies:
    for g in strat["grid"]:
        params = strat["base_params"].copy()
        params.update(g)
        print(f"Testing {strat['name']} with params {g}...")
        try:
            req = BacktestRequestSchema(**params)
            res = api_backtest(req)
            
            if hasattr(res, "summary"):
                metrics = res.summary
            elif isinstance(res, dict) and "summary" in res:
                metrics = res["summary"]
            else:
                print(f"API Error response: {res}")
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

            calmar = (total_return / max_drawdown) if max_drawdown > 0 else 0.0

            print(f"Done -> Sharpe: {sharpe:.4f}, Return: {total_return:.2f}%, MaxDD: {max_drawdown:.2f}%, Trades: {total_trades}")
            
            results.append({
                "strategy": strat["name"],
                "params": str(g),
                "opt_params": g,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "profit_factor": profit_factor,
                "total_trades": total_trades,
                "res": res
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error on {strat['name']} {g}: {e}")

df_res = pd.DataFrame(results)

if not df_res.empty:
    print("\n" + "="*80)
    print("=== STRATEGY RANKING REPORT ===")
    print("="*80)
    
    # Sort by composite score (Sharpe * 0.4 + Sortino * 0.3 + Calmar * 0.3)
    df_res['composite_score'] = df_res['sharpe'] * 0.4 + df_res['sortino'] * 0.3 + df_res['calmar'] * 0.3
    df_sorted = df_res.sort_values(by="composite_score", ascending=False)
    
    print(df_sorted[['strategy', 'sharpe', 'sortino', 'calmar', 'win_rate', 'max_drawdown', 'total_return', 'total_trades']].to_string())

    # Get best configuration per strategy
    best_per_strat = {}
    for strat_name, group in df_res.groupby("strategy"):
        best_row = group.sort_values(by="composite_score", ascending=False).iloc[0]
        best_per_strat[strat_name] = best_row

    print("\n=== TOP PARAMETERS PER STRATEGY ===")
    for strat_name, best_row in best_per_strat.items():
        print(f"\nStrategy: {strat_name}")
        print(f"Optimal Params: {best_row['opt_params']}")
        print(f"Sharpe: {best_row['sharpe']:.4f} | Sortino: {best_row['sortino']:.4f} | WinRate: {best_row['win_rate']:.2f}% | MaxDD: {best_row['max_drawdown']:.2f}% | Return: {best_row['total_return']:.2f}% | Trades: {best_row['total_trades']}")

    # Output top 4 strategies
    top_4 = sorted(best_per_strat.values(), key=lambda x: x['composite_score'], reverse=True)[:4]
    print("\n" + "="*80)
    print("=== TOP 4 SELECTED STRATEGIES FOR INTEGRATION ===")
    print("="*80)
    for idx, s in enumerate(top_4, 1):
        print(f"{idx}. {s['strategy']} (Composite Score: {s['composite_score']:.4f})")
        print(f"   Optimal Params: {s['opt_params']}")
        print(f"   Sharpe: {s['sharpe']:.4f}, Sortino: {s['sortino']:.4f}, MaxDD: {s['max_drawdown']:.2f}%, Return: {s['total_return']:.2f}%, Trades: {s['total_trades']}")
