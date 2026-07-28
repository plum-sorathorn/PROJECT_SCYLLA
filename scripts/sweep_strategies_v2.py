#!/usr/bin/env python3
"""
Multi-stage parameter sweep for SCYLLA's 3 active strategies.
Uses random sampling + refinement for efficient exploration of the parameter space.

Usage:
    python scripts/sweep_strategies_v2.py

Output:
    backend/cache/sweep_optimal_v2.json - Optimal parameters for each strategy

Methodology:
    1. Coarse grid: 30 random combinations per strategy (within realistic ranges)
    2. Refinement: top 3 from coarse, 15 combos each with smaller steps around best
    3. Uses walkforward backtest (train 500 / test 100) on first 15k trades
    4. ProcessPoolExecutor(max_workers=5) for CPU parallelism
    5. Keeps walkforward_label_threshold=0.5 to preserve 96 MB predictions cache
"""

import sys
import os
import json
import time
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from routers.ml_model import api_backtest, BacktestRequestSchema

# Realistic parameter ranges (HARD CONSTRAINTS from options trading domain)
PARAM_RANGES = {
    "prob_threshold": (0.30, 0.50),
    "kelly_multiplier": (0.5, 1.0),
    "kelly_cap": (0.05, 0.20),
    "hard_stop_loss": (0.15, 0.40),
    "profit_threshold": (0.15, 0.50),
    "max_quantile_spread": (0.10, 0.40),
    "min_median_return": (0.0, 0.05),
    "max_iv": (50.0, 150.0),
    "max_concurrent_trades": (4, 12),
}

# Strategy-specific parameter grids (which params to sweep for each strategy)
STRATEGY_PARAMS = {
    "whale_quality": [
        "prob_threshold", "kelly_multiplier", "kelly_cap", "hard_stop_loss",
        "profit_threshold", "max_quantile_spread", "min_median_return", "max_iv",
        "max_concurrent_trades"
    ],
    "contrarian_trend": [
        "prob_threshold", "kelly_multiplier", "kelly_cap", "hard_stop_loss",
        "profit_threshold", "max_iv", "max_concurrent_trades"
    ],
    "vol_regime": [
        "prob_threshold", "kelly_multiplier", "kelly_cap", "hard_stop_loss",
        "profit_threshold", "max_quantile_spread", "max_iv", "max_concurrent_trades"
    ],
}

# Data subset for faster execution (15k trades = ~145 walkforward steps)
DATA_SUBSET_SIZE = 15000


def sample_random_params(strategy_name, n_samples):
    """Generate n random parameter combinations within realistic ranges."""
    param_names = STRATEGY_PARAMS[strategy_name]
    samples = []
    
    for _ in range(n_samples):
        params = {"strategy_type": strategy_name}
        for param in param_names:
            low, high = PARAM_RANGES[param]
            if param == "max_concurrent_trades":
                params[param] = random.randint(int(low), int(high))
            else:
                params[param] = round(random.uniform(low, high), 3)
        samples.append(params)
    
    return samples


def sample_refined_params(base_params, n_samples):
    """Generate refined samples around a base parameter set (smaller steps)."""
    param_names = [k for k in base_params.keys() if k != "strategy_type"]
    samples = []
    
    for _ in range(n_samples):
        params = {"strategy_type": base_params["strategy_type"]}
        for param in param_names:
            if param not in PARAM_RANGES:
                params[param] = base_params[param]
                continue
            
            low, high = PARAM_RANGES[param]
            base_val = base_params[param]
            
            # Refinement: +/- 20% of the range around the base value
            range_size = high - low
            step = range_size * 0.20
            
            if param == "max_concurrent_trades":
                new_low = max(int(low), int(base_val - step))
                new_high = min(int(high), int(base_val + step))
                params[param] = random.randint(new_low, new_high)
            else:
                new_low = max(low, base_val - step)
                new_high = min(high, base_val + step)
                params[param] = round(random.uniform(new_low, new_high), 3)
        
        samples.append(params)
    
    return samples


def run_backtest_wrapper(params_dict):
    """Wrapper to run a single backtest with given parameters."""
    try:
        req = BacktestRequestSchema(
            mode="walkforward",
            initial_capital=100000.0,
            walkforward_train_window=500,
            walkforward_test_increment=100,
            walkforward_label_threshold=0.5,  # Preserve 96 MB cache
            calibration_target_pct=0.025,
            profit_threshold=params_dict.get("profit_threshold", 0.25),
            prob_threshold=params_dict.get("prob_threshold", 0.40),
            kelly_multiplier=params_dict.get("kelly_multiplier", 0.75),
            kelly_cap=params_dict.get("kelly_cap", 0.15),
            stop_lambda=1.5,
            max_risk_pct_per_trade=0.02,
            confirm_direct_dev=False,
            strategy_type=params_dict["strategy_type"],
            max_concurrent_trades=params_dict.get("max_concurrent_trades", 8),
            scan_time="10:00:00",
            min_kelly_fraction=0.01,
            hard_stop_loss=params_dict.get("hard_stop_loss", 0.25),
            lookback_days=None,
            max_quantile_spread=params_dict.get("max_quantile_spread", 0.25),
            min_median_return=params_dict.get("min_median_return", 0.0),
            slippage_pct=0.01,
            max_iv=params_dict.get("max_iv", 100.0),
            min_open_interest=100,
            min_dte=14,
            max_dte=60,
            data_start_idx=0,
            data_end_idx=DATA_SUBSET_SIZE,
            use_synthetic=False,
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


def score_result(result):
    """Score a result based on multiple criteria (Sharpe-heavy with trade count floor)."""
    if not result["success"]:
        return -999
    
    sharpe = result["sharpe"]
    trades = result["trades"]
    win_rate = result["win_rate"]
    profit_factor = result["profit_factor"]
    max_dd = result["max_drawdown"]
    return_pct = result["return_pct"]
    
    # Must have minimum trades to be statistically meaningful
    if trades < 20:
        return -100 + trades
    
    # Negative returns are heavily penalized
    if return_pct < 0:
        return -50 + return_pct * 0.1
    
    # Composite score: Sharpe (50%), return (25%), trade count (15%), win rate (10%)
    sharpe_score = min(100, max(0, sharpe * 40))  # Sharpe 2.5 = 100
    return_score = min(100, max(0, return_pct * 2))  # 50% return = 100
    trade_score = min(100, trades / 2)  # 200 trades = 100
    win_score = min(100, win_rate)
    
    # Drawdown penalty
    dd_penalty = max(0, abs(max_dd) - 15) * 1.5  # Penalize drawdowns > 15%
    
    composite = (
        sharpe_score * 0.50 +
        return_score * 0.25 +
        trade_score * 0.15 +
        win_score * 0.10 -
        dd_penalty
    )
    
    return composite


def run_sweep_stage(strategy_name, param_sets, stage, max_workers):
    """Run a single sweep stage for a strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name} | Stage: {stage}")
    print(f"{'='*60}")
    
    print(f"Testing {len(param_sets)} parameter combinations with {max_workers} workers...")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_backtest_wrapper, params): params for params in param_sets}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 5 == 0 or completed == len(param_sets):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(param_sets) - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{len(param_sets)} ({rate:.2f}/s, ETA: {eta:.0f}s)")
    
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
    print("PROJECT SCYLLA - Strategy Sweep v2")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data subset: {DATA_SUBSET_SIZE} trades")
    print(f"Walkforward: train=500, test_increment=100, label_threshold=0.5")
    
    # Determine worker count (leave 2 cores free for system)
    max_workers = max(1, min(mp.cpu_count() - 2, 5))  # Cap at 5 workers
    print(f"Using {max_workers} parallel workers (CPU count: {mp.cpu_count()})")
    
    all_optimal = {}
    
    for strategy_name in STRATEGY_PARAMS.keys():
        print(f"\n{'#'*60}")
        print(f"# Processing strategy: {strategy_name}")
        print(f"{'#'*60}")
        
        # Stage 1: Coarse random sampling (30 combos)
        coarse_params = sample_random_params(strategy_name, n_samples=30)
        coarse_results = run_sweep_stage(strategy_name, coarse_params, stage="coarse", max_workers=max_workers)
        
        # Get top 3 for refinement
        top_3 = [r for r in coarse_results if r["success"] and r["trades"] >= 10][:3]
        
        if not top_3:
            print(f"\n*** WARNING: No valid results in coarse sweep for {strategy_name} ***")
            # Use the best result even if it has few trades
            top_3 = [r for r in coarse_results if r["success"]][:3]
        
        # Stage 2: Refinement around top performers (15 combos each)
        refined_params = []
        for top_result in top_3:
            refined = sample_refined_params(top_result["params"], n_samples=15)
            refined_params.extend(refined)
        
        if refined_params:
            refined_results = run_sweep_stage(strategy_name, refined_params, stage="refined", max_workers=max_workers)
        else:
            refined_results = []
        
        # Combine results and find overall best
        all_results = coarse_results + refined_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Find best result that meets minimum criteria
        best = None
        for r in all_results:
            if r["success"] and r["trades"] >= 20 and r["return_pct"] > 0:
                best = r
                break
        
        # If no result meets strict criteria, take highest scoring valid result
        if best is None:
            for r in all_results:
                if r["success"] and r["trades"] >= 10:
                    best = r
                    break
        
        # Last resort: take the top result
        if best is None and all_results:
            best = all_results[0]
        
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
    
    # Save results to backend/cache/sweep_optimal_v2.json
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'cache')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sweep_optimal_v2.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_optimal, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Sweep completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for strategy_name, result in all_optimal.items():
        if result.get("metrics"):
            m = result["metrics"]
            print(f"{strategy_name:20s} -> {m['return_pct']:7.2f}% return, {m['trades']:3d} trades, Sharpe {m['sharpe']:.2f}")
        else:
            print(f"{strategy_name:20s} -> NO VALID RESULT")
    
    return all_optimal


if __name__ == "__main__":
    main()
