"""
VOLUME-OPTIMIZED PARAMETER SWEEP for PROJECT SCYLLA.

Goal: Maximize trade count * PnL across all strategies, accepting lower win rates.

Key differences from sweep_oos.py:
  - Composite score weights TRADE COUNT (30%) alongside PnL (40%), Sharpe (20%), MaxDD (10%)
  - Wider, more permissive parameter grids to encourage more entries
  - 2-phase: coarse grid -> refine top candidates with perturbation
  - No aggressive_kelly strategy
  - Outputs to scripts/sweep_volume_optimal.json (coexists with sweep_optimal.json)

Run:
    python scripts/sweep_volume.py
    python scripts/sweep_volume.py --workers 8
    python scripts/sweep_volume.py --holdout 0.2 --top-k 5
"""

import argparse
import os
import sys
import csv
import time
import json
import itertools
import sqlite3
import numpy as np
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

from backend.routers.ml_model import api_backtest, BacktestRequestSchema  # noqa: E402

RESULTS_CSV = os.path.join(SCRIPT_DIR, "sweep_volume_results.csv")
OPTIMAL_JSON = os.path.join(SCRIPT_DIR, "sweep_optimal.json")
OPTIMAL_JSON_BACKUP = os.path.join(SCRIPT_DIR, "sweep_volume_optimal.json")
PROGRESS_LOG = os.path.join(SCRIPT_DIR, "sweep_volume_progress.log")
START_TIME = time.time()

CANONICAL = {
    "mode": "walkforward",
    "initial_capital": 100000.0,
    "walkforward_train_window": 500,
    "walkforward_test_increment": 100,
    "profit_threshold": 1.0,
    "min_kelly_fraction": 0.01,
    "slippage_pct": 0.005,
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
    db_path_candidates = [
        os.path.join(PROJECT_ROOT, "backend", "scylla_ml.db"),
        os.path.join(PROJECT_ROOT, "scylla_ml.db"),
        os.path.join(PROJECT_ROOT, "backend", "scylla.db"),
        os.path.join(PROJECT_ROOT, "backend", "data", "scylla.db"),
        os.path.join(PROJECT_ROOT, "data", "scylla.db"),
        os.path.join(PROJECT_ROOT, "scylla.db"),
    ]
    for p in db_path_candidates:
        if os.path.exists(p):
            conn = sqlite3.connect(p)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0 AND labeled = 1"
                ).fetchone()
                if row and row[0]:
                    log(f"found {row[0]} real labeled trades at {p}")
                    return int(row[0]), p
            finally:
                conn.close()
    raise RuntimeError(
        "Could not locate scylla.db with labeled real trades. Looked at: "
        + ", ".join(db_path_candidates)
    )


def build_volume_grid():
    """Wider, more permissive grids designed to maximize trade count.

    Key changes vs sweep_oos.py:
      - prob_threshold starts at 0.25 (vs 0.40)
      - hard_stop_loss uses fewer values (stops work intraday now)
      - min_median_return relaxed to 0.00-0.02 (vs 0.03-0.07)
      - max_concurrent_trades starts at 10 (vs 5)
      - kelly_multiplier biased higher
      - max_iv allows 999 (effectively no cap)
      - max_quantile_spread wider (up to 7.0)
    """
    # Balanced grid: 3 values for key filters, 2 for secondary axes
    # Keeps confluence_sniper manageable (9 axes) while covering the search space
    common_prob = [0.25, 0.35, 0.45]
    common_kelly = [0.60, 0.90]
    common_kelly_cap = [0.15, 0.25, 0.35]
    common_hard_stop = [0.0, 0.15, 0.30]
    common_stop_lambda = [1.0, 1.5, 2.0]
    common_concurrent = [10, 20]
    common_med_ret = [0.00, 0.02]
    common_max_qs = [4.0, 5.5, 7.0]

    return {
        "standard": {
            "strategy_type": "standard",
            "grid_axes": {
                "prob_threshold": common_prob,
                "kelly_multiplier": common_kelly,
                "kelly_cap": common_kelly_cap,
                "hard_stop_loss": common_hard_stop,
                "stop_lambda": common_stop_lambda,
                "max_concurrent_trades": common_concurrent,
            },
        },
        "volatility_regime_adaptive": {
            "strategy_type": "volatility_regime_adaptive",
            "grid_axes": {
                "prob_threshold": common_prob,
                "kelly_multiplier": common_kelly,
                "kelly_cap": common_kelly_cap,
                "hard_stop_loss": common_hard_stop,
                "max_concurrent_trades": common_concurrent,
            },
        },
        "quantile_spread": {
            "strategy_type": "quantile_spread",
            "grid_axes": {
                "prob_threshold": common_prob,
                "kelly_multiplier": common_kelly,
                "kelly_cap": common_kelly_cap,
                "hard_stop_loss": common_hard_stop,
                "max_quantile_spread": common_max_qs,
                "min_median_return": common_med_ret,
                "max_concurrent_trades": common_concurrent,
            },
        },
        "directional_quantile_shift": {
            "strategy_type": "directional_quantile_shift",
            "grid_axes": {
                "prob_threshold": common_prob,
                "kelly_multiplier": common_kelly,
                "kelly_cap": common_kelly_cap,
                "hard_stop_loss": common_hard_stop,
                "min_median_return": common_med_ret,
                "max_concurrent_trades": common_concurrent,
            },
        },
        "confluence_sniper": {
            "strategy_type": "confluence_sniper",
            "grid_axes": {
                "prob_threshold": [0.25, 0.35, 0.45],
                "kelly_multiplier": [0.60, 0.90],
                "kelly_cap": [0.20, 0.35],
                "hard_stop_loss": [0.0, 0.15, 0.30],
                "stop_lambda": [1.0, 2.0],
                "max_quantile_spread": [5.5, 7.0],
                "min_median_return": [0.00, 0.02],
                "max_iv": [80.0, 999.0],
                "max_concurrent_trades": [10, 20],
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
    try:
        payload = _run_one_payload(strat_name, params, data_start_idx, data_end_idx)
        req = BacktestRequestSchema(**payload)
        res = api_backtest(req)
    except Exception as e:
        return {
            "strategy": strat_name, "params": params, "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }
    summary = res.get("summary") if isinstance(res, dict) else getattr(res, "summary", None)
    if not isinstance(summary, dict):
        return {
            "strategy": strat_name, "params": params, "ok": False, "error": "no summary",
        }
    return {
        "strategy": strat_name, "params": params, "ok": True,
        "sharpe": float(summary.get("sharpe", 0.0) or 0.0),
        "sortino": float(summary.get("sortino", 0.0) or 0.0),
        "win_rate": float(summary.get("win_rate_pct", 0.0) or 0.0),
        "max_drawdown": float(summary.get("max_drawdown_pct", 0.0) or 0.0),
        "total_return": float(summary.get("cumulative_pnl_pct", 0.0) or 0.0),
        "cagr": float(summary.get("cagr_pct", 0.0) or 0.0),
        "profit_factor": float(summary.get("profit_factor", 0.0) or 0.0),
        "total_trades": int(summary.get("trades_triggered", 0) or 0),
        "trade_days": int(summary.get("trade_days_used_for_sharpe", 0) or 0),
    }


def _run_wrapper(args):
    return run_one(args)


def _append_csv(row, path):
    write_header = not os.path.exists(path)
    fieldnames = [
        "phase", "strategy", "params", "ok", "error",
        "sharpe", "sortino", "win_rate", "max_drawdown",
        "total_return", "cagr", "profit_factor", "total_trades",
        "trade_days", "elapsed_sec",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        row["params"] = json.dumps(row["params"], sort_keys=True)
        row["elapsed_sec"] = round(time.time() - START_TIME, 1)
        w.writerow(row)


def run_phase(phase_name, grids, data_start_idx, data_end_idx, max_workers, csv_path=None):
    tasks = []
    for strat_name, defn in grids.items():
        for combo in expand_grid(defn["grid_axes"]):
            tasks.append((strat_name, combo, data_start_idx, data_end_idx))

    log(f"=== {phase_name} START | {len(tasks)} tasks | slice=[{data_start_idx}:{data_end_idx}] | workers={max_workers} ===")
    if csv_path is None:
        csv_path = RESULTS_CSV

    done = 0
    failed = 0
    t_phase = time.time()
    last_status = t_phase
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_wrapper, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                failed += 1
                log(f"  worker exception: {e}")
                continue
            row["phase"] = phase_name
            rows.append(row)
            if not row.get("ok"):
                failed += 1
                log(f"  FAIL {row.get('strategy')} {row.get('params')}: {row.get('error')}")
            else:
                _append_csv(row, csv_path)
            done += 1
            now = time.time()
            if now - last_status >= 30.0:
                rate = done / max(now - t_phase, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                log(f"  {phase_name} progress: {done}/{len(tasks)} ({rate:.2f}/s) | failed={failed} | ETA {eta/60:.1f} min")
                last_status = now
    log(f"=== {phase_name} END | {done} done in {(time.time()-t_phase)/60:.1f} min | failed={failed} ===")
    return rows


def load_phase_from_csv(phase_name, csv_path=None):
    if csv_path is None:
        csv_path = RESULTS_CSV
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[(df["ok"] == True) & (df["phase"] == phase_name)].copy()  # noqa: E712
    df["params"] = df["params"].apply(json.loads)
    return df


def volume_composite_score(df, w_pnl=0.2, w_trades=0.5, w_sharpe=0.2, w_dd=0.1):
    """Composite score that HEAVILY rewards trade count.

    Components:
      - total_return (PnL %):  log-scale to compress outliers, min-max normalized
      - total_trades:          log-scale absolute (200 trades = 1.0, PRIMARY driver)
      - sharpe:                min-max normalized
      - max_drawdown:          min-max normalized (penalty)
    """
    df = df.copy()

    def _norm(series):
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    # Log-scale PnL to compress extreme outliers (2405% vs 802%)
    df["n_pnl"] = _norm(np.log1p(df["total_return"].fillna(0.0).clip(lower=0)))
    # Absolute log-scale trade count: 200 trades = 1.0 cap, PRIMARY scoring driver
    df["n_trades"] = np.minimum(np.log1p(df["total_trades"].fillna(0).astype(float)) / np.log1p(200), 1.0)
    df["n_sharpe"] = _norm(df["sharpe"].fillna(0.0))
    df["n_dd"] = _norm(df["max_drawdown"].fillna(0.0))

    df["score"] = (
        w_pnl * df["n_pnl"]
        + w_trades * df["n_trades"]
        + w_sharpe * df["n_sharpe"]
        - w_dd * df["n_dd"]
    )
    return df


def select_top_per_strategy(df, top_k=5, min_trades=5):
    """Select top-K per strategy by volume-weighted composite score.

    Score is computed GLOBALLY (across all strategies) so trade count
    comparisons are meaningful — a strategy with 50 trades scores higher
    on the trade component than one with 10 trades.
    """
    if df.empty:
        return {}
    # Score globally first
    df = df.copy()
    df = df[df["total_trades"] >= min_trades]
    if df.empty:
        df = df  # fall back to all
    df = volume_composite_score(df)

    out = {}
    for strat, sub in df.groupby("strategy"):
        if sub.empty:
            continue
        top = sub.sort_values("score", ascending=False).head(top_k)
        out[strat] = [
            {k: float(v) if isinstance(v, (int, float)) and k != "total_trades" else (int(v) if k == "total_trades" else v)
             for k, v in r["params"].items()}
            for _, r in top.iterrows()
        ]
    return out


def build_refined_grid(phase1_top, original_grids, perturbation=0.20):
    """Phase 2: take Phase 1 winners and create a refined grid around them.

    For each numeric parameter, generate values at -20%, center, +20% (clamped to valid ranges).
    """
    refined = {}
    for strat_name, top_params_list in phase1_top.items():
        if strat_name not in original_grids:
            continue
        orig_axes = original_grids[strat_name]["grid_axes"]
        refined_axes = defaultdict(list)

        for params in top_params_list:
            for key, val in params.items():
                if key not in orig_axes:
                    continue
                if isinstance(val, (int, float)):
                    # Generate perturbation around this value
                    lo = val * (1.0 - perturbation)
                    hi = val * (1.0 + perturbation)
                    # Clamp to original grid range
                    orig_vals = orig_axes[key]
                    orig_min = min(orig_vals)
                    orig_max = max(orig_vals)
                    lo = max(lo, orig_min)
                    hi = min(hi, orig_max)
                    # Generate 3 points
                    candidates = [lo, val, hi]
                    # Deduplicate and round
                    for c in candidates:
                        c_rounded = round(c, 4) if isinstance(val, float) else int(round(c))
                        if c_rounded not in refined_axes[key]:
                            refined_axes[key].append(c_rounded)
                else:
                    if val not in refined_axes[key]:
                        refined_axes[key].append(val)

        # Ensure each axis has at least 2 values
        for key in orig_axes:
            if key in refined_axes and len(refined_axes[key]) < 2:
                refined_axes[key] = list(orig_axes[key])

        refined[strat_name] = {
            "strategy_type": original_grids[strat_name]["strategy_type"],
            "grid_axes": dict(refined_axes),
        }
    return refined


def main():
    ap = argparse.ArgumentParser(description="Volume-optimized parameter sweep for PROJECT SCYLLA.")
    ap.add_argument("--holdout", type=float, default=0.20,
                    help="Fraction held out for OOS evaluation (default 0.20).")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Top-K params per strategy to refine in Phase 2 (default 5).")
    ap.add_argument("--min-trades", type=int, default=5,
                    help="Min trades for a result to be considered (default 5).")
    ap.add_argument("--workers", type=int, default=None,
                    help="ProcessPool worker count (default min(8, cpu-2)).")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Skip Phase 1; reuse existing CSV.")
    ap.add_argument("--phase1-csv", type=str, default=None,
                    help="CSV file to reuse for Phase 1 results.")
    args = ap.parse_args()

    if os.path.exists(PROGRESS_LOG):
        os.remove(PROGRESS_LOG)
    if os.path.exists(RESULTS_CSV):
        os.rename(RESULTS_CSV, RESULTS_CSV + f".bak.{int(time.time())}")
    log("=== sweep_volume START ===")
    log(f"holdout_frac={args.holdout}  top_k={args.top_k}  min_trades={args.min_trades}")

    n_total, db_path = _count_total_trades()
    sweep_end = int(n_total * (1.0 - args.holdout))
    oos_start = sweep_end
    oos_end = n_total
    log(f"N_total={n_total}  sweep_slice=[0:{sweep_end}] ({sweep_end/n_total*100:.0f}%)  oos_slice=[{oos_start}:{oos_end}] ({args.holdout*100:.0f}%)")

    if sweep_end < 500 + 100:
        raise RuntimeError(f"holdout too aggressive: sweep slice is only {sweep_end} trades, need >= 600")
    if (oos_end - oos_start) < 500 + 100:
        raise RuntimeError(f"holdout too small: OOS slice is only {oos_end - oos_start} trades, need >= 600")

    cpu = os.cpu_count() or 8
    max_workers = args.workers if args.workers else max(4, min(8, cpu - 2))
    log(f"using {max_workers} workers")

    # ── Phase 1: Coarse sweep on 80% ──
    coarse_grids = build_volume_grid()
    log(f"Phase 1 coarse grid total combos: {total_combos(coarse_grids)}")

    if not args.skip_phase1:
        run_phase("phase1_coarse", coarse_grids, 0, sweep_end, max_workers)
    else:
        if not args.phase1_csv:
            raise RuntimeError("--skip-phase1 requires --phase1-csv")
        log(f"reusing phase1 results from {args.phase1_csv}")
        import shutil
        shutil.copy(args.phase1_csv, RESULTS_CSV)

    phase1_df = load_phase_from_csv("phase1_coarse")
    if phase1_df.empty:
        raise RuntimeError("Phase 1 produced no results")
    log(f"Phase 1 ok rows: {len(phase1_df)}")

    phase1_top = select_top_per_strategy(phase1_df, top_k=args.top_k, min_trades=args.min_trades)
    log("Phase 1 top per strategy (by volume composite score):")
    for s, lst in phase1_top.items():
        log(f"  {s}: {len(lst)} winners")

    # ── Phase 2: Refined sweep around winners on remaining 20% ──
    phase2_grids = build_refined_grid(phase1_top, coarse_grids, perturbation=0.20)
    p2_total = total_combos(phase2_grids)
    log(f"Phase 2 refined grid total combos: {p2_total}")
    run_phase("phase2_refined_oos", phase2_grids, oos_start, oos_end, max_workers)

    phase2_df = load_phase_from_csv("phase2_refined_oos")
    if phase2_df.empty:
        raise RuntimeError("Phase 2 produced no results")

    # ── Final selection: volume composite score on OOS results (GLOBAL scoring) ──
    final_best = {}
    phase2_scored = phase2_df.copy()
    phase2_scored = phase2_scored[phase2_scored["total_trades"] >= args.min_trades]
    if phase2_scored.empty:
        phase2_scored = phase2_df.copy()
    phase2_scored = volume_composite_score(phase2_scored)

    for strat, sub in phase2_scored.groupby("strategy"):
        if sub.empty:
            continue
        best = sub.sort_values("score", ascending=False).iloc[0]
        final_best[strat] = {
            "params": {k: (float(v) if isinstance(v, (int, float)) and k != "total_trades" else (int(v) if k == "total_trades" else v))
                       for k, v in best["params"].items()},
            "metrics": {
                "sharpe": float(best["sharpe"]),
                "sortino": float(best["sortino"]),
                "cagr": float(best["cagr"]),
                "total_return": float(best["total_return"]),
                "win_rate": float(best["win_rate"]),
                "max_drawdown": float(best["max_drawdown"]),
                "profit_factor": float(best["profit_factor"]),
                "total_trades": int(best["total_trades"]),
                "trade_days": int(best["trade_days"]),
            },
            "oos_window": {"start": oos_start, "end": oos_end, "holdout_frac": args.holdout},
            "evaluation": "out-of-sample-volume-optimized",
        }

    log("=== final_best (volume-optimized, OOS-validated) ===")
    for s, b in final_best.items():
        log(f"  {s}:")
        log(f"    params:   {b['params']}")
        log(f"    metrics:  {b['metrics']}")

    with open(OPTIMAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_best, f, indent=2)
    log(f"wrote {OPTIMAL_JSON}")

    import shutil
    shutil.copy2(OPTIMAL_JSON, OPTIMAL_JSON_BACKUP)
    log(f"backup wrote {OPTIMAL_JSON_BACKUP}")
    log("=== sweep_volume DONE ===")


if __name__ == "__main__":
    main()
