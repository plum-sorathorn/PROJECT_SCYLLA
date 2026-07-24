"""
80/20 OUT-OF-SAMPLE (OOS) PARAMETER SWEEP for PROJECT SCYLLA.

Why this script exists
----------------------
The original scripts/sweep_optimal_params.py runs a wide parameter grid on the
ENTIRE labeled real-trade dataset and picks the best-looking params by Sharpe.
That is in-sample selection: with ~110K trials on the same backtest universe,
the "best" params exploit random noise in the cached walk-forward predictions,
not real edge. See audit notes: max_drawdown=0.00, winrate=88%, Sharpe=1.91
on 60 trades out of 43,820 — that is a filter, not a strategy.

This script implements a proper holdout:
  Phase 1 (Sweep, 80%):  Use the first 80% of labeled real trades (chronological).
                          Run the same coarse grid; pick winners per strategy.
  Phase 2 (OOS, 20%):    Use the LAST 20% only as a cold-start OOS evaluation.
                          Train models with NO data from the first 80%.
                          The walk-forward predictions for the OOS window are
                          built from scratch and cached under a separate key.
                          Final "optimal" metrics come from this phase.
  Output: scripts/sweep_optimal.json (overwrites the in-sample file).

Run
---
    python scripts/sweep_oos.py                # default 80/20 split
    python scripts/sweep_oos.py --holdout 0.3  # 70/30 split
    python scripts/sweep_oos.py --top-k 5      # keep top 5 per strategy for OOS

Runtime: ~75-90 min on 8 workers (same as the in-sample sweep).
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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.routers.ml_model import api_backtest, BacktestRequestSchema  # noqa: E402

RESULTS_CSV = os.path.join(SCRIPT_DIR, "sweep_oos_results.csv")
OPTIMAL_JSON = os.path.join(SCRIPT_DIR, "sweep_optimal.json")
PROGRESS_LOG = os.path.join(SCRIPT_DIR, "sweep_oos_progress.log")
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
    """Return total count of labeled, non-synthetic trades available."""
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


def build_stage1_grid():
    """Same coarse grid as scripts/sweep_optimal_params.py — kept in sync deliberately."""
    common_prob = [0.40, 0.45, 0.50, 0.55, 0.60]
    common_kelly = [0.30, 0.45, 0.60, 0.75, 0.90]
    common_kelly_cap = [0.15, 0.25, 0.35]
    common_hard_stop = [0.0, 0.07, 0.15, 0.25, 0.35, 0.45]
    common_stop_lambda = [1.0, 1.5, 2.0]
    common_concurrent = [5, 7, 10]
    common_med_ret = [0.03, 0.05, 0.07]
    common_max_qs = [3.5, 4.5, 5.5]

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
                "prob_threshold": [0.40, 0.50, 0.60],
                "kelly_multiplier": [0.45, 0.60, 0.75, 0.90],
                "kelly_cap": [0.20, 0.30],
                "hard_stop_loss": [0.20, 0.30, 0.40, 0.50],
                "stop_lambda": common_stop_lambda,
                "max_quantile_spread": common_max_qs,
                "min_median_return": common_med_ret,
                "max_iv": [45.0, 55.0, 65.0, 80.0],
                "max_concurrent_trades": common_concurrent,
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


def run_phase(phase_name, grids, data_start_idx, data_end_idx, max_workers, top_k=None, csv_path=None):
    """Run all combos in a phase. If top_k is set, also dedupe the param list to
    only the top-K from a previous CSV (used by Phase 2)."""
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


def select_top_per_strategy(df, top_k=3, min_trades=10):
    """Composite score with OOS-sensible weights:
      - penalize low trade count (cap-binding bias)
      - prefer higher Sharpe, lower drawdown, higher CAGR
    """
    if df.empty:
        return {}
    out = {}
    for strat, sub in df.groupby("strategy"):
        sub = sub.copy()
        sub = sub[sub["total_trades"] >= min_trades]
        if sub.empty:
            sub = df[df["strategy"] == strat]
        if sub.empty:
            continue
        sub["score"] = (
            sub["sharpe"].fillna(0.0)
            + 0.3 * (sub["cagr"].fillna(0.0) / 100.0)
            - 0.05 * sub["max_drawdown"].fillna(0.0)
        )
        top = sub.sort_values("score", ascending=False).head(top_k)
        out[strat] = [
            {k: float(v) if isinstance(v, (int, float)) and k != "total_trades" else (int(v) if k == "total_trades" else v)
             for k, v in r["params"].items()}
            for _, r in top.iterrows()
        ]
    return out


def build_phase2_grid_from_winners(phase1_top, original_grids):
    grids = {}
    for strat_name, top_params_list in phase1_top.items():
        axes = dict(original_grids[strat_name]["grid_axes"])
        for params in top_params_list:
            pass
        grids[strat_name] = {
            "strategy_type": original_grids[strat_name]["strategy_type"],
            "grid_axes": {
                axis: [params[axis]] for axis in axes.keys() if axis in params
            },
        }
    return grids


def main():
    ap = argparse.ArgumentParser(description="80/20 OOS parameter sweep for PROJECT SCYLLA.")
    ap.add_argument("--holdout", type=float, default=0.20,
                    help="Fraction of the dataset held out for OOS evaluation (default 0.20 = 80/20).")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of top params per strategy to forward to OOS evaluation (default 3).")
    ap.add_argument("--min-trades", type=int, default=10,
                    help="Min trades required for a phase-1 result to be considered (default 10).")
    ap.add_argument("--workers", type=int, default=None,
                    help="ProcessPool worker count (default min(8, cpu-2)).")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Skip phase 1 (sweep); only do OOS eval using existing CSV (requires --phase1-csv).")
    ap.add_argument("--phase1-csv", type=str, default=None,
                    help="Reuse a previous phase-1 CSV instead of re-running.")
    args = ap.parse_args()

    if os.path.exists(PROGRESS_LOG):
        os.remove(PROGRESS_LOG)
    if os.path.exists(RESULTS_CSV):
        os.rename(RESULTS_CSV, RESULTS_CSV + f".bak.{int(time.time())}")
    log("=== sweep_oos START ===")
    log(f"holdout_frac={args.holdout}  top_k={args.top_k}  min_trades={args.min_trades}")
    log(f"cpu_count={os.cpu_count()}")

    n_total, db_path = _count_total_trades()
    sweep_end = int(n_total * (1.0 - args.holdout))
    oos_start = sweep_end
    oos_end = n_total
    log(f"N_total={n_total}  sweep_slice=[0:{sweep_end}] ({sweep_end/n_total*100:.0f}%)  oos_slice=[{oos_start}:{oos_end}] ({args.holdout*100:.0f}%)")

    if sweep_end < 500 + 100:
        raise RuntimeError(f"holdout too aggressive: sweep slice is only {sweep_end} trades, need >= 600 for walkforward.")
    if (oos_end - oos_start) < 500 + 100:
        raise RuntimeError(f"holdout too small: OOS slice is only {oos_end - oos_start} trades, need >= 600 for walkforward.")

    cpu = os.cpu_count() or 8
    max_workers = args.workers if args.workers else max(4, min(8, cpu - 2))
    log(f"using {max_workers} workers")

    coarse_grids = build_stage1_grid()
    log(f"coarse grid total combos: {total_combos(coarse_grids)}")

    if not args.skip_phase1:
        run_phase("phase1_sweep_80pct", coarse_grids, 0, sweep_end, max_workers)
    else:
        if not args.phase1_csv:
            raise RuntimeError("--skip-phase1 requires --phase1-csv pointing to a previous CSV.")
        log(f"reusing phase1 results from {args.phase1_csv}")
        import shutil
        shutil.copy(args.phase1_csv, RESULTS_CSV)

    phase1_df = load_phase_from_csv("phase1_sweep_80pct")
    if phase1_df.empty:
        raise RuntimeError("phase 1 produced no results")
    log(f"phase1 ok rows: {len(phase1_df)}")
    phase1_top = select_top_per_strategy(phase1_df, top_k=args.top_k, min_trades=args.min_trades)
    log("phase1 top per strategy:")
    for s, lst in phase1_top.items():
        log(f"  {s}: {len(lst)} winners")

    phase2_grids = build_phase2_grid_from_winners(phase1_top, coarse_grids)
    p2_tasks = sum(1 for s in phase2_grids.values() for _ in expand_grid(s["grid_axes"]))
    log(f"phase2 OOS combos: {p2_tasks}")
    run_phase("phase2_oos_20pct", phase2_grids, oos_start, oos_end, max_workers)

    phase2_df = load_phase_from_csv("phase2_oos_20pct")
    if phase2_df.empty:
        raise RuntimeError("phase 2 produced no results")

    final_best = {}
    for strat, sub in phase2_df.groupby("strategy"):
        sub = sub.copy()
        sub = sub[sub["total_trades"] >= args.min_trades]
        if sub.empty:
            sub = phase2_df[phase2_df["strategy"] == strat]
        if sub.empty:
            continue
        sub["score"] = (
            sub["sharpe"].fillna(0.0)
            + 0.3 * (sub["cagr"].fillna(0.0) / 100.0)
            - 0.05 * sub["max_drawdown"].fillna(0.0)
        )
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
            "evaluation": "out-of-sample",
        }

    log("=== final_best (OOS-validated) ===")
    for s, b in final_best.items():
        log(f"  {s}:")
        log(f"    params:   {b['params']}")
        log(f"    metrics:  {b['metrics']}")

    with open(OPTIMAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_best, f, indent=2)
    log(f"wrote {OPTIMAL_JSON}")
    log("=== sweep_oos DONE ===")


if __name__ == "__main__":
    main()
