"""
QUICK PARAMETER SWEEP - Reduced grid for faster results.
Focuses on the most impactful parameters with fewer combinations.
"""
import argparse
import os
import sys
import csv
import time
import json
import itertools
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.routers.ml_model import api_backtest, BacktestRequestSchema

RESULTS_CSV = os.path.join(SCRIPT_DIR, "sweep_quick_results.csv")
OPTIMAL_JSON = os.path.join(SCRIPT_DIR, "sweep_optimal.json")
PROGRESS_LOG = os.path.join(SCRIPT_DIR, "sweep_quick_progress.log")
START_TIME = time.time()

CANONICAL = {
    "mode": "walkforward",
    "initial_capital": 100000.0,
    "walkforward_train_window": 500,
    "walkforward_test_increment": 100,
    "profit_threshold": 1.0,
    "min_kelly_fraction": 0.01,
    "slippage_pct": 0.02,
    "lookback_days": None,
    "min_open_interest": 0,
    "scan_time": "10:00:00",
    "max_risk_pct_per_trade": 0.03,
}

def log(msg):
    line = f"[{int(time.time() - START_TIME):>5}s] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _to_float_pct(value, pct=False):
    if pct:
        return value / 100.0
    return value

def _count_total_trades():
    db_path = os.path.join(PROJECT_ROOT, "backend", "scylla_ml.db")
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0 AND labeled = 1"
        ).fetchone()
        if row and row[0]:
            log(f"found {row[0]} real labeled trades at {db_path}")
            return int(row[0]), db_path
    finally:
        conn.close()
    raise RuntimeError("Could not locate scylla_ml.db with labeled real trades.")

def build_quick_grid():
    """Reduced grid focusing on most impactful parameters with realistic risk/reward."""
    return {
        "standard": {
            "strategy_type": "standard",
            "grid_axes": {
                "prob_threshold": [0.30, 0.40, 0.50],
                "kelly_multiplier": [0.40, 0.60, 0.80],
                "kelly_cap": [0.20, 0.30],
                "profit_threshold": [0.05, 0.06, 0.07],
                "hard_stop_loss": [0.025, 0.03, 0.035],
                "max_concurrent_trades": [5, 10],
            },
        },
        "quantile_spread": {
            "strategy_type": "quantile_spread",
            "grid_axes": {
                "prob_threshold": [0.30, 0.40, 0.50],
                "kelly_multiplier": [0.40, 0.60, 0.80],
                "kelly_cap": [0.20, 0.30],
                "profit_threshold": [0.05, 0.06, 0.07],
                "hard_stop_loss": [0.025, 0.03, 0.035],
                "max_quantile_spread": [0.10, 0.15],
                "max_concurrent_trades": [5, 10],
            },
        },
        "confluence_sniper": {
            "strategy_type": "confluence_sniper",
            "grid_axes": {
                "prob_threshold": [0.30, 0.40, 0.50],
                "kelly_multiplier": [0.40, 0.60, 0.80],
                "kelly_cap": [0.20, 0.30],
                "profit_threshold": [0.05, 0.06, 0.07],
                "hard_stop_loss": [0.025, 0.03, 0.035],
                "max_quantile_spread": [0.10, 0.15],
                "max_iv": [55.0, 70.0],
                "max_concurrent_trades": [5, 10],
            },
        },
    }

def expand_grid(grid_axes):
    keys = list(grid_axes.keys())
    for combo in itertools.product(*[grid_axes[k] for k in keys]):
        yield dict(zip(keys, combo))

def total_combos(grids):
    return sum(1 for s in grids.values() for _ in expand_grid(s["grid_axes"]))

def _run_one_payload(strat_name, params, data_start_idx, data_end_idx):
    payload = dict(CANONICAL)
    payload.update(params)
    if "strategy_type" not in payload:
        payload["strategy_type"] = strat_name
    payload["hard_stop_loss"] = _to_float_pct(payload.get("hard_stop_loss", 0.0))
    payload["profit_threshold"] = _to_float_pct(payload.get("profit_threshold", 1.0))
    if data_start_idx is not None:
        payload["data_start_idx"] = data_start_idx
    if data_end_idx is not None:
        payload["data_end_idx"] = data_end_idx
    return payload

def run_one(args):
    strat_name, params, data_start_idx, data_end_idx = args
    payload = _run_one_payload(strat_name, params, data_start_idx, data_end_idx)
    try:
        req = BacktestRequestSchema(**payload)
        res = api_backtest(req)
        summary = res.get("summary", {})
        return {
            "strategy": strat_name,
            "params": params,
            "sharpe": summary.get("sharpe", 0.0),
            "sortino": summary.get("sortino", 0.0),
            "cumulative_pnl_pct": summary.get("cumulative_pnl_pct", 0.0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
            "win_rate_pct": summary.get("win_rate_pct", 0.0),
            "trades_triggered": summary.get("trades_triggered", 0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "status": "ok",
        }
    except Exception as ex:
        return {
            "strategy": strat_name,
            "params": params,
            "status": "error",
            "error": str(ex),
        }

def main():
    log("=== QUICK SWEEP START ===")
    
    N_total, db_path = _count_total_trades()
    holdout_frac = 0.2
    sweep_end = int(N_total * (1 - holdout_frac))
    
    grids = build_quick_grid()
    n_combos = total_combos(grids)
    log(f"N_total={N_total}  sweep_slice=[0:{sweep_end}]  combos={n_combos}")
    
    results = []
    tasks = []
    for strat_name, strat_config in grids.items():
        for params in expand_grid(strat_config["grid_axes"]):
            tasks.append((strat_name, params, 0, sweep_end))
    
    log(f"Starting {len(tasks)} tasks with 8 workers...")
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)
            
            if result["status"] == "error":
                failed += 1
            
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - START_TIME
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                log(f"progress: {completed}/{len(tasks)} ({rate:.2f}/s) | failed={failed} | ETA {eta_min:.1f} min")
    
    # Write results to CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    log(f"Results written to {RESULTS_CSV}")
    
    # Find optimal params per strategy
    optimal = {}
    df = pd.DataFrame(results)
    df_ok = df[df["status"] == "ok"].copy()
    
    for strat_name in df_ok["strategy"].unique():
        strat_df = df_ok[df_ok["strategy"] == strat_name].copy()
        
        # Filter: at least 10 trades triggered
        strat_df = strat_df[strat_df["trades_triggered"] >= 10]
        if strat_df.empty:
            log(f"{strat_name}: no results with >= 10 trades")
            continue
        
        # Rank by Sharpe
        strat_df = strat_df.sort_values("sharpe", ascending=False)
        best = strat_df.iloc[0]
        
        optimal[strat_name] = {
            "params": best["params"],
            "metrics": {
                "sharpe": float(best["sharpe"]),
                "sortino": float(best["sortino"]),
                "cumulative_pnl_pct": float(best["cumulative_pnl_pct"]),
                "max_drawdown_pct": float(best["max_drawdown_pct"]),
                "win_rate_pct": float(best["win_rate_pct"]),
                "trades_triggered": int(best["trades_triggered"]),
                "profit_factor": float(best["profit_factor"]),
            },
            "evaluation": "quick-sweep",
        }
        
        log(f"{strat_name}: best Sharpe={best['sharpe']:.3f} | trades={int(best['trades_triggered'])} | PnL={best['cumulative_pnl_pct']:.2f}%")
    
    # Write optimal params
    with open(OPTIMAL_JSON, "w", encoding="utf-8") as f:
        json.dump(optimal, f, indent=2)
    
    log(f"Optimal params written to {OPTIMAL_JSON}")
    log("=== QUICK SWEEP COMPLETE ===")

if __name__ == "__main__":
    main()
