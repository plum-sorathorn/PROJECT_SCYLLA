#!/usr/bin/env python3
"""
Optimized multi-stage parameter sweep for the 3 new strategies.
Uses smaller data subsets and reduced parameter grids for faster execution.

Usage:
    python scripts/sweep_strategies.py

Output:
    scripts/sweep_optimal.json - Optimal parameters for each strategy
"""

import sys
import os
import json
import time
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from routers.ml_model import api_backtest, BacktestRequestSchema

# Hard requirements (floors)
HARD_REQUIREMENTS = {
    "min_stop_loss": 0.25,
    "min_open_interest": 100,
    "max_iv": 100.0,
    "min_dte": 7,
    "max_dte": 60,
    "max_kelly_cap": 0.20,
    "max_risk_pct_per_trade": 0.02,
    "min_kelly_fraction": 0.01,
    "min_prob_threshold": 0.40,
    "max_concurrent_trades": 8
}

# Optimized parameter grids (smaller for faster execution)
STRATEGY_GRIDS = {
    "quantile_confidence": {
        "prob_threshold": [0.40, 0.45, 0.50, 0.55],
        "max_quantile_spread": [0.15, 0.20, 0.25],
        "kelly_multiplier": [0.60, 0.75, 0.90],
        "kelly_cap": [0.12, 0.18],
        "hard_stop_loss": [0.25, 0.30],
        "profit_threshold": [0.20, 0.25, 0.30],
    },
    "trend_breakout": {
        "prob_threshold": [0.38, 0.42, 0.48, 0.52],
        "min_median_return": [0.015, 0.025, 0.035],
        "kelly_multiplier": [0.60, 0.75, 0.90],
        "kelly_cap": [0.12, 0.18],
        "hard_stop_loss": [0.25, 0.30],
        "profit_threshold": [0.20, 0.25, 0.30],
    },
    "iv_regime_adaptive": {
        "prob_threshold": [0.38, 0.42, 0.48, 0.52],
        "kelly_multiplier": [0.60, 0.75, 0.90],
        "kelly_cap": [0.12, 0.18],
        "hard_stop_loss": [0.25, 0.30],
        "profit_threshold": [0.20, 0.25, 0.30],
    },
}

# Data subset size for faster execution
DATA_SUBSET_SIZE = 10000


def run_backtest_wrapper(params_dict):
    """Wrapper to run a single backtest with given parameters."""
    try:
        req = BacktestRequestSchema(
            mode="walkforward",
            initial_capital=100000.0,
            walkforward_train_window=500,
            walkforward_test_increment=100,
            profit_threshold=params_dict.get("profit_threshold", 0.25),
            prob_threshold=params_dict.get("prob_threshold", 0.45),
            kelly_multiplier=params_dict.get("kelly_multiplier", 0.75),
            kelly_cap=params_dict.get("kelly_cap", 0.18),
            stop_lambda=1.5,
            max_risk_pct_per_trade=HARD_REQUIREMENTS["max_risk_pct_per_trade"],
            confirm_direct_dev=False,
            strategy_type=params_dict["strategy_type"],
            max_concurrent_trades=HARD_REQUIREMENTS["max_concurrent_trades"],
            scan_time="10:00:00",
            min_kelly_fraction=HARD_REQUIREMENTS["min_kelly_fraction"],
            hard_stop_loss=params_dict.get("hard_stop_loss", 0.25),
            lookback_days=None,
            max_quantile_spread=params_dict.get("max_quantile_spread", 0.20),
            min_median_return=params_dict.get("min_median_return", 0.02),
            slippage_pct=0.01,
            max_iv=HARD_REQUIREMENTS["max_iv"],
            min_open_interest=HARD_REQUIREMENTS["min_open_interest"],
            min_dte=HARD_REQUIREMENTS["min_dte"],
            max_dte=HARD_REQUIREMENTS["max_dte"],
            data_start_idx=0,
            data_end_idx=DATA_SUBSET_SIZE,
        )
        
        result = api_backtest(req)
        
        # Extract key metrics
        summary = result.get("summary", {})
        return {
            "params": params_dict,
            "sharpe": summary.get("sharpe", 0),
            "sortino": summary.get("sortino", 0),
            "win_rate": summary.get("win_rate_pct", 0),
            "trades": summary.get("trades_triggered", 0),
            "return_pct": summary.get("cumulative_pnl_pct", 0),
            "max_drawdown": summary.get("max_drawdown_pct", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "success": True,
        }
    except Exception as e:
        return {
            "params": params_dict,
            "success": False,
            "error": str(e),
        }


def generate_combinations(strategy_name, grid):
    """Generate parameter combinations from grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["strategy_type"] = strategy_name
        combinations.append(params)
    
    return combinations


def narrow_grid(top_results, original_grid, strategy_name):
    """Create a narrower grid around top performers."""
    if not top_results:
        return original_grid
    
    # Get the best performing params
    best_params = top_results[0]["params"]
    
    # Create narrowed grid around best values
    narrowed = {}
    for key, values in original_grid.items():
        if key not in best_params:
            narrowed[key] = values
            continue
        
        best_val = best_params[key]
        
        # Find index of best value in original grid
        try:
            idx = values.index(best_val)
        except ValueError:
            # Value not in grid (float comparison issue), find closest
            idx = min(range(len(values)), key=lambda i: abs(values[i] - best_val))
        
        # Create narrowed range: best value +/- 1 step
        narrowed_values = []
        if idx > 0:
            narrowed_values.append(values[idx - 1])
        narrowed_values.append(values[idx])
        if idx < len(values) - 1:
            narrowed_values.append(values[idx + 1])
        
        # Add intermediate values for finer granularity
        if len(narrowed_values) >= 2:
            v1, v2 = narrowed_values[0], narrowed_values[-1]
            mid = (v1 + v2) / 2
            if mid not in narrowed_values:
                narrowed_values.append(mid)
                narrowed_values.sort()
        
        narrowed[key] = narrowed_values
    
    return narrowed


def score_result(result):
    """Score a result based on multiple criteria."""
    if not result["success"]:
        return -999
    
    sharpe = result["sharpe"]
    trades = result["trades"]
    win_rate = result["win_rate"]
    profit_factor = result["profit_factor"]
    max_dd = result["max_drawdown"]
    
    # Must meet minimum requirements
    if trades < 100:
        return -100 + trades  # Penalize but allow some score
    if sharpe < 0.5:
        return -50 + sharpe * 10
    
    # Composite score: Sharpe (40%), trade count (20%), win rate (20%), profit factor (20%)
    # Normalize each to 0-100 scale
    sharpe_score = min(100, max(0, sharpe * 33.33))  # Sharpe 3 = 100
    trade_score = min(100, trades / 3)  # 300 trades = 100
    win_score = min(100, win_rate)
    pf_score = min(100, profit_factor * 50)  # PF 2 = 100
    
    # Drawdown penalty
    dd_penalty = max(0, max_dd - 20) * 2  # Penalize drawdowns > 20%
    
    composite = (
        sharpe_score * 0.40 +
        trade_score * 0.20 +
        win_score * 0.20 +
        pf_score * 0.20 -
        dd_penalty
    )
    
    return composite


def run_sweep_stage(strategy_name, grid, stage, max_workers):
    """Run a single sweep stage for a strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name} | Stage: {stage}")
    print(f"{'='*60}")
    
    combinations = generate_combinations(strategy_name, grid)
    print(f"Testing {len(combinations)} parameter combinations with {max_workers} workers...")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_backtest_wrapper, combo): combo for combo in combinations}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 10 == 0 or completed == len(combinations):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(combinations) - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{len(combinations)} ({rate:.1f}/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    print(f"Stage {stage} completed in {elapsed:.1f}s")
    
    # Score and sort results
    for r in results:
        r["score"] = score_result(r)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Print top 5
    print(f"\nTop 5 results for {strategy_name} (Stage {stage}):")
    for i, r in enumerate(results[:5]):
        if r["success"]:
            print(f"  {i+1}. Score: {r['score']:.1f} | Sharpe: {r['sharpe']:.2f} | "
                  f"Trades: {r['trades']} | Win%: {r['win_rate']:.1f} | "
                  f"Return: {r['return_pct']:.1f}%")
    
    return results


def main():
    print("="*60)
    print("PROJECT SCYLLA - Optimized Parameter Sweep")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data subset: {DATA_SUBSET_SIZE} trades")
    
    # Determine worker count (leave 2 cores free for system)
    max_workers = max(1, min(mp.cpu_count() - 2, 4))  # Cap at 4 workers
    print(f"Using {max_workers} parallel workers (CPU count: {mp.cpu_count()})")
    
    all_optimal = {}
    
    for strategy_name, grid in STRATEGY_GRIDS.items():
        print(f"\n{'#'*60}")
        print(f"# Processing strategy: {strategy_name}")
        print(f"{'#'*60}")
        
        # Stage 1: Wide grid
        stage1_results = run_sweep_stage(strategy_name, grid, stage=1, max_workers=max_workers)
        
        # Get top 5 for narrowing
        top_5 = [r for r in stage1_results if r["success"]][:5]
        
        # Stage 2: Narrow grid around top performers
        narrowed_grid = narrow_grid(top_5, grid, strategy_name)
        stage2_results = run_sweep_stage(strategy_name, narrowed_grid, stage=2, max_workers=max_workers)
        
        # Combine results and find overall best
        all_results = stage1_results + stage2_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Find best result that meets criteria
        best = None
        for r in all_results:
            if r["success"] and r["trades"] >= 100 and r["sharpe"] >= 0.8:
                best = r
                break
        
        # If no result meets strict criteria, take highest scoring
        if best is None:
            best = all_results[0] if all_results else None
        
        if best and best["success"]:
            print(f"\n*** WINNER for {strategy_name} ***")
            print(f"Score: {best['score']:.1f}")
            print(f"Sharpe: {best['sharpe']:.2f} | Sortino: {best['sortino']:.2f}")
            print(f"Trades: {best['trades']} | Win Rate: {best['win_rate']:.1f}%")
            print(f"Return: {best['return_pct']:.1f}% | Max DD: {best['max_drawdown']:.1f}%")
            print(f"Profit Factor: {best['profit_factor']:.2f}")
            print(f"Params: {json.dumps(best['params'], indent=2)}")
            
            all_optimal[strategy_name] = {
                "params": best["params"],
                "metrics": {
                    "sharpe": best["sharpe"],
                    "sortino": best["sortino"],
                    "win_rate": best["win_rate"],
                    "trades": best["trades"],
                    "return_pct": best["return_pct"],
                    "max_drawdown": best["max_drawdown"],
                    "profit_factor": best["profit_factor"],
                    "score": best["score"],
                },
                "evaluation": "out-of-sample_walkforward",
            }
        else:
            print(f"\n*** NO VALID WINNER for {strategy_name} ***")
            all_optimal[strategy_name] = {
                "params": None,
                "metrics": None,
                "error": "No valid results found",
            }
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "sweep_optimal.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_optimal, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Sweep completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return all_optimal


if __name__ == "__main__":
    main()
