"""
Filter profile for the labeled options_trades database.

Counts how many candidate trades pass each filter combination
to identify the "narrow waist" of the new strategy rules.
Also prints per-ticker breakdown of win rate and trade count.

Usage:
    backend/.venv/Scripts/python.exe scripts/profile_filters.py
"""

import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
DB_PATH = os.path.join(BACKEND_DIR, "scylla_ml.db")
CACHE_DIR = os.path.join(BACKEND_DIR, "cache")
MODEL_PATH = os.path.join(CACHE_DIR, "scylla_predictor.pkl")
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

import pandas as pd
import numpy as np
import sqlite3
import joblib


def line(label, n):
    """Print a filter row."""
    print(f"  {label:<55} {n:>10,}")


def main() -> int:
    conn = sqlite3.connect(DB_PATH)

    # Step 0: total labeled real trades
    df_all = pd.read_sql_query(
        "SELECT * FROM options_trades WHERE labeled = 1 AND is_synthetic = 0",
        conn,
    )
    n_total = len(df_all)
    print(f"=== Filter profile over the {n_total:,} real labeled trades ===\n")
    line("No filter", n_total)

    # Step 1: data clean
    df_clean = df_all[
        (df_all['vol_oi_ratio'] >= 0.5) & (df_all['vol_oi_ratio'] <= 100) &
        (df_all['implied_vol'] >= 5) & (df_all['implied_vol'] <= 200) &
        (df_all['premium'] >= 100) & (df_all['premium'] <= 5_000_000) &
        (df_all['underlier_price'] > 5) &
        (df_all['observed_return'] >= -1.0) & (df_all['observed_return'] <= 5.0)
    ].copy()
    n_clean = len(df_clean)
    line("After data clean (vol_oi 0.5-100, IV 5-200, premium 100-5M)", n_clean)

    # Step 2: vol_oi thresholds
    for thresh in [5, 3, 2]:
        line(f"After data clean + vol_oi >= {thresh}", len(df_clean[df_clean['vol_oi_ratio'] >= thresh]))

    # Step 3: DTE windows
    for lo, hi in [(7, 60), (14, 30), (14, 45)]:
        line(f"After data clean + dte in [{lo}, {hi}]", len(df_clean[(df_clean['dte'] >= lo) & (df_clean['dte'] <= hi)]))

    # Step 4: IV range
    line("After data clean + IV in [15, 150]", len(df_clean[(df_clean['implied_vol'] >= 15) & (df_clean['implied_vol'] <= 150)]))

    # Step 5: TIER_A and TIER_B from past year
    max_ts = df_clean['timestamp'].max()
    cutoff = (pd.to_datetime(max_ts) - pd.Timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")
    df_year = df_clean[df_clean['timestamp'] >= cutoff]
    print()
    print(f"  TIER cutoff: {cutoff}  (max_ts={max_ts}, year-trades={len(df_year):,})")

    agg = df_year.groupby('ticker').agg(
        wr=('observed_return', lambda s: (s >= 0.03).mean()),
        n=('observed_return', 'count')
    )
    tier_a = set(agg[(agg['wr'] >= 0.30) & (agg['n'] >= 10)].index)
    tier_b = set(agg[(agg['wr'] >= 0.25) & (agg['n'] >= 10)].index)
    print(f"  TIER_A tickers (WR>=30%, n>=10): {len(tier_a)}")
    print(f"  TIER_B tickers (WR>=25%, n>=10): {len(tier_b)}")

    line("After data clean + ticker in TIER_A", len(df_clean[df_clean['ticker'].isin(tier_a)]))
    line("After data clean + ticker in TIER_B", len(df_clean[df_clean['ticker'].isin(tier_b)]))

    # Step 6: load model (or train) for strategy-filter counts
    print()
    print("Loading model (or training)...")
    t0 = time.time()
    models = None
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            if isinstance(m, dict) and all(q in m for q in [0.1, 0.25, 0.5, 0.75, 0.9]):
                models = m
                print(f"  Loaded existing model from {MODEL_PATH} ({time.time()-t0:.1f}s)")
        except Exception as ex:
            print(f"  Failed to load model: {ex}")
    if models is None:
        print("  No model found — training a new one (this can take ~30-60s)...")
        from routers.ml_model import api_train_model
        api_train_model()
        m = joblib.load(MODEL_PATH)
        if isinstance(m, dict) and all(q in m for q in [0.1, 0.25, 0.5, 0.75, 0.9]):
            models = m
        print(f"  Training done ({time.time()-t0:.1f}s)")

    # Step 7: predict on cleaned data
    print("  Computing model predictions on cleaned data...")
    t0 = time.time()
    from routers.ml_model import compute_advanced_features, enforce_monotonic_quantiles
    from routers.ml_derivations import compute_calibrated_p_success

    df_clean_feat = compute_advanced_features(df_clean)
    numeric_features = [
        'strike', 'volume', 'open_interest', 'vol_oi_ratio', 'implied_vol',
        'underlier_price', 'premium', 'dte',
        'moneyness', 'iv_hv_ratio', 'vix_level', 'log_premium'
    ]
    categorical_features = ['option_type', 'side', 'trend_alignment', 'dte_bucket']
    X = df_clean_feat[['ticker'] + numeric_features + categorical_features]

    p10 = models[0.1].predict(X)
    p25 = models[0.25].predict(X)
    p50 = models[0.5].predict(X)
    p75 = models[0.75].predict(X)
    p90 = models[0.9].predict(X)

    iqr_arr = np.zeros(len(df_clean_feat))
    p_succ_arr = np.zeros(len(df_clean_feat))
    for i in range(len(df_clean_feat)):
        q = enforce_monotonic_quantiles({
            "p10": float(p10[i]), "p25": float(p25[i]),
            "p50": float(p50[i]), "p75": float(p75[i]), "p90": float(p90[i])
        })
        iqr_arr[i] = q["p90"] - q["p10"]
        # Use the SAME calibration_target_pct as the backtest (0.025) so the
        # p_success value is consistent with the strategy filter in ml_model.py.
        p_succ_arr[i] = compute_calibrated_p_success(
            0.025, q["p10"], q["p25"], q["p50"], q["p75"], q["p90"]
        )

    df_clean_feat = df_clean_feat.assign(
        _p10=p10, _p25=p25, _p50=p50, _p75=p75, _p90=p90,
        _iqr=iqr_arr, _p_success=p_succ_arr
    )
    print(f"  Predictions computed ({time.time()-t0:.1f}s)")

    # IMPORTANT: the in-DB vol_oi_ratio is CAPPED at 3.0 (not 5+ as the spec
    # assumed). The "whale" filter must use a data-driven threshold instead.
    # See profile output: q95 of vol_oi_ratio = 2.27, max = 3.0. We use
    # vol_oi >= 2.0 (top 10%) for whale_quality, >= 1.0 (median) for the
    # other two strategies.

    # Strategy filters (CURRENT, before loosening)
    print()
    print("--- CURRENT strategy filters (before loosening) ---")
    cur_wq = (
        df_clean_feat['ticker'].isin(tier_a)
        & df_clean_feat['vol_oi_ratio'].between(3, 50)
        & df_clean_feat['implied_vol'].between(15, 150)
        & df_clean_feat['dte'].between(14, 30)
        & (df_clean_feat['_p_success'] >= 0.55)
        & (df_clean_feat['_iqr'] <= 0.20)
        & (df_clean_feat['_p50'] >= 0.04)
    )
    line("whale_quality (current)", int(cur_wq.sum()))

    cur_ct = (
        df_clean_feat['ticker'].isin(tier_a)
        & df_clean_feat['vol_oi_ratio'].between(3, 50)
        & df_clean_feat['implied_vol'].between(15, 150)
        & df_clean_feat['dte'].between(14, 30)
        & (
            ((df_clean_feat['trend_alignment'] == 'BULL_ALIGNED') & (df_clean_feat['option_type'] == 'Put'))
            | ((df_clean_feat['trend_alignment'] == 'BEAR_ALIGNED') & (df_clean_feat['option_type'] == 'Call'))
        )
        & (df_clean_feat['_p_success'] >= 0.50)
        & (df_clean_feat['_p50'] >= 0.03)
    )
    line("contrarian_trend (current)", int(cur_ct.sum()))

    cur_vr = (
        df_clean_feat['ticker'].isin(tier_a)
        & df_clean_feat['vol_oi_ratio'].between(3, 50)
        & df_clean_feat['dte'].between(14, 30)
        & (
            (df_clean_feat['implied_vol'] < 30) & (df_clean_feat['_p_success'] >= 0.60)
            | (df_clean_feat['implied_vol'] >= 30) & (df_clean_feat['_iqr'] <= 0.30) & (df_clean_feat['_p_success'] >= 0.50)
        )
        & (df_clean_feat['_p50'] >= 0.04)
    )
    line("vol_regime (current)", int(cur_vr.sum()))

    # Strategy filters (PROPOSED loosening, DATA-DRIVEN thresholds)
    # vol_oi ratios are capped at 3.0 in the data, so the "5x whale" spec
    # threshold is unreachable. We use top-decile (vol_oi >= 2.0) for the
    # high-conviction whale strategy, and median (vol_oi >= 1.0) for the
    # broader regime/fade strategies. dte is bounded to [14, 60] in the
    # data; we use the full valid range. IV is bounded to [~7.8, 150].
    print()
    print("--- PROPOSED loosened strategy filters (data-driven thresholds) ---")
    prop_wq = (
        (df_clean_feat['ticker'].isin(tier_a))
        & (df_clean_feat['vol_oi_ratio'] >= 2.0)
        & (df_clean_feat['dte'].between(14, 60))
        & (df_clean_feat['implied_vol'].between(15, 150))
        & (df_clean_feat['_p_success'] >= 0.35)
        & (df_clean_feat['_iqr'] <= 0.25)
        & (df_clean_feat['_p50'] >= 0.001)
    )
    line("whale_quality (proposed)", int(prop_wq.sum()))

    prop_ct = (
        (df_clean_feat['ticker'].isin(tier_a))
        & (df_clean_feat['vol_oi_ratio'] >= 1.0)
        & (df_clean_feat['dte'].between(14, 60))
        & (df_clean_feat['implied_vol'].between(15, 150))
        & (
            ((df_clean_feat['trend_alignment'] == 'BULL_ALIGNED') & (df_clean_feat['option_type'] == 'Put'))
            | ((df_clean_feat['trend_alignment'] == 'BEAR_ALIGNED') & (df_clean_feat['option_type'] == 'Call'))
        )
        & (df_clean_feat['_p_success'] >= 0.35)
    )
    line("contrarian_trend (proposed)", int(prop_ct.sum()))

    prop_vr = (
        (df_clean_feat['ticker'].isin(tier_a))
        & (df_clean_feat['vol_oi_ratio'] >= 1.0)
        & (df_clean_feat['dte'].between(14, 60))
        & (df_clean_feat['implied_vol'].between(15, 150))
        & (
            (df_clean_feat['implied_vol'] < 30) & (df_clean_feat['_p_success'] >= 0.40)
            | (df_clean_feat['implied_vol'] >= 30) & (df_clean_feat['_iqr'] <= 0.35) & (df_clean_feat['_p_success'] >= 0.35)
        )
    )
    line("vol_regime (proposed)", int(prop_vr.sum()))

    # Per-ticker breakdown
    print()
    print("=== Per-ticker breakdown (post data-clean, all years) ===")
    print(f"  {'Ticker':<8} | {'Trades':>7} | {'Win Rate':>8} | {'Avg Return':>11}")
    print("  " + "-" * 50)
    per_tk = df_clean.groupby('ticker').agg(
        trades=('observed_return', 'count'),
        wr=('observed_return', lambda s: (s >= 0.03).mean()),
        avg_ret=('observed_return', 'mean')
    ).sort_values('trades', ascending=False)
    for ticker, row in per_tk.head(40).iterrows():
        print(f"  {ticker:<8} | {int(row['trades']):>7,} | {row['wr']*100:>7.1f}% | {row['avg_ret']*100:>+10.2f}%")

    # TIER_A only
    print()
    print("=== Per-ticker breakdown (TIER_A only) ===")
    print(f"  {'Ticker':<8} | {'Trades':>7} | {'Win Rate':>8} | {'Avg Return':>11}")
    print("  " + "-" * 50)
    df_tier_a = df_clean[df_clean['ticker'].isin(tier_a)]
    per_tk_a = df_tier_a.groupby('ticker').agg(
        trades=('observed_return', 'count'),
        wr=('observed_return', lambda s: (s >= 0.03).mean()),
        avg_ret=('observed_return', 'mean')
    ).sort_values('trades', ascending=False)
    for ticker, row in per_tk_a.head(40).iterrows():
        print(f"  {ticker:<8} | {int(row['trades']):>7,} | {row['wr']*100:>7.1f}% | {row['avg_ret']*100:>+10.2f}%")

    # Drill-down: distribution of model-derived features on the
    # TIER_A + vol_oi >= 2 universe (the candidate pool for the loosened
    # whale strategy). Tells us which filter is the narrow waist.
    print()
    print("=== Drill-down: model features on TIER_A + vol_oi >= 2 universe ===")
    drill = df_clean_feat[(df_clean_feat['ticker'].isin(tier_a)) & (df_clean_feat['vol_oi_ratio'] >= 2.0)]
    print(f"  Universe size: {len(drill):,}")
    for col in ['_p_success', '_p50', '_iqr']:
        q = np.quantile(drill[col], [0.25, 0.5, 0.75, 0.9, 0.95])
        print(f"  {col}: q25={q[0]:.4f}, q50={q[1]:.4f}, q75={q[2]:.4f}, q90={q[3]:.4f}, q95={q[4]:.4f}")
    print()
    print("  Cumulative pass-through on TIER_A + vol_oi >= 2 (n=" + str(len(drill)) + "):")
    base = drill
    print(f"    + dte in [14, 60]:                    {int(((base['dte'] >= 14) & (base['dte'] <= 60)).sum()):>6,}")
    print(f"    + IV in [15, 150]:                    {int(((base['implied_vol'] >= 15) & (base['implied_vol'] <= 150)).sum()):>6,}")
    print(f"    + _p50 >= 0.03:                       {int((base['_p50'] >= 0.03).sum()):>6,}")
    print(f"    + _p_success >= 0.50:                 {int((base['_p_success'] >= 0.50).sum()):>6,}")
    print(f"    + _iqr <= 0.20:                       {int((base['_iqr'] <= 0.20).sum()):>6,}")
    print(f"    + _iqr <= 0.30:                       {int((base['_iqr'] <= 0.30).sum()):>6,}")
    print(f"    + _iqr <= 0.40:                       {int((base['_iqr'] <= 0.40).sum()):>6,}")
    print(f"    + _iqr <= 0.60:                       {int((base['_iqr'] <= 0.60).sum()):>6,}")
    print(f"    + _iqr <= 1.0:                        {int((base['_iqr'] <= 1.0).sum()):>6,}")
    print()
    print("  All 5 (TIER_A + vol>=2 + dte+IV + _p50>=0.03 + _ps>=0.50 + _iqr<=X):")
    for iqr_cap in [0.20, 0.30, 0.40, 0.60, 1.0]:
        n = int(((base['_p50'] >= 0.03) & (base['_p_success'] >= 0.50) & (base['_iqr'] <= iqr_cap)).sum())
        print(f"    _iqr <= {iqr_cap}: {n:>6,}")

    # Per-ticker breakdown of the CANDIDATE universe (post-data-clean +
    # TIER_A gate) for each loosened strategy. Shows the underlying WR
    # structure that the backtest is filtering against.
    print()
    print("=== Per-ticker breakdown on TIER_A + vol_oi>=2 (whale universe) ===")
    print(f"  {'Ticker':<8} | {'Trades':>7} | {'WR (3%)':>9} | {'WR (8%)':>9} | {'Avg Ret':>9} | {'WR (cap=0.06)':>14}")
    print("  " + "-" * 72)
    whale_uni = df_clean_feat[(df_clean_feat['ticker'].isin(tier_a)) & (df_clean_feat['vol_oi_ratio'] >= 2.0)]
    per_tk_w = whale_uni.groupby('ticker').agg(
        trades=('observed_return', 'count'),
        wr3=('observed_return', lambda s: (s >= 0.03).mean()),
        wr8=('observed_return', lambda s: (s >= 0.08).mean()),
        wr6=('observed_return', lambda s: (s >= 0.06).mean()),
        avg_ret=('observed_return', 'mean')
    ).sort_values('trades', ascending=False)
    for ticker, row in per_tk_w.iterrows():
        print(f"  {ticker:<8} | {int(row['trades']):>7,} | {row['wr3']*100:>8.1f}% | {row['wr8']*100:>8.1f}% | {row['avg_ret']*100:>+8.3f}% | {row['wr6']*100:>13.1f}%")

    print()
    print("=== Per-ticker breakdown on TIER_A + vol_oi>=1 + is_fade (contrarian universe) ===")
    print(f"  {'Ticker':<8} | {'Trades':>7} | {'WR (3%)':>9} | {'WR (8%)':>9} | {'WR (6%)':>9}")
    print("  " + "-" * 55)
    fade_uni = df_clean_feat[
        (df_clean_feat['ticker'].isin(tier_a))
        & (df_clean_feat['vol_oi_ratio'] >= 1.0)
        & (
            ((df_clean_feat['trend_alignment'] == 'BULL_ALIGNED') & (df_clean_feat['option_type'] == 'Put'))
            | ((df_clean_feat['trend_alignment'] == 'BEAR_ALIGNED') & (df_clean_feat['option_type'] == 'Call'))
        )
    ]
    per_tk_f = fade_uni.groupby('ticker').agg(
        trades=('observed_return', 'count'),
        wr3=('observed_return', lambda s: (s >= 0.03).mean()),
        wr8=('observed_return', lambda s: (s >= 0.08).mean()),
        wr6=('observed_return', lambda s: (s >= 0.06).mean())
    ).sort_values('trades', ascending=False)
    for ticker, row in per_tk_f.iterrows():
        print(f"  {ticker:<8} | {int(row['trades']):>7,} | {row['wr3']*100:>8.1f}% | {row['wr8']*100:>8.1f}% | {row['wr6']*100:>8.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
