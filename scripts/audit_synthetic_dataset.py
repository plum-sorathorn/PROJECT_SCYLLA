"""
Audit distributional properties of the synthetic options dataset
(`is_synthetic=1`) against real data (`is_synthetic=0`) in scylla_ml.db.
Prints per-ticker and per-feature stats, whale-threshold crossing counts,
and label rates.

Run:
    python scripts/audit_synthetic_dataset.py
"""

import os
import sqlite3
import statistics
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DB_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "backend", "scylla_ml.db"))

FEATURES = [
    "vol_oi_ratio", "implied_vol", "dte", "premium",
    "commission_per_contract", "bid_ask_spread_pct", "delta_hedged_return",
    "days_to_earnings", "days_to_fomc",
]
CATEGORICAL_FEATURES = [
    "early_exercise_risk",
    "is_earnings_window", "is_fomc_day",
    "vix_regime", "market_regime",
]
LABEL_ONLY_FEATURES = ["observed_return", "max_adverse_return"]
GREEK_FEATURES = [
    "delta_entry", "gamma_entry", "vega_entry", "theta_entry", "rho_entry",
    "delta_exit", "gamma_exit", "vega_exit", "theta_exit", "rho_exit",
]
WHALE_THRESHOLDS = [2.0, 5.0, 8.0]
DTE_BUCKETS = [
    ("<=7",   lambda d: d is not None and d <= 7),
    ("8-21",  lambda d: d is not None and 8 <= d <= 21),
    ("22-45", lambda d: d is not None and 22 <= d <= 45),
    ("46-90", lambda d: d is not None and 46 <= d <= 90),
    (">90",   lambda d: d is not None and d > 90),
]


def _percentile(sorted_vals, q):
    """Linear-interpolation percentile on a pre-sorted list. q in [0,1]."""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _fmt(v, width=10, decimals=4):
    if v is None:
        return f"{'N/A':>{width}}"
    return f"{v:>{width}.{decimals}f}"


def _stats_block(values):
    """Return dict of count/min/p10/.../max/mean/stdev for a list of floats."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0}
    s = sorted(clean)
    out = {
        "count": len(s),
        "min": s[0],
        "p10": _percentile(s, 0.10),
        "p25": _percentile(s, 0.25),
        "p50": _percentile(s, 0.50),
        "p75": _percentile(s, 0.75),
        "p90": _percentile(s, 0.90),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }
    return out


def _print_stats_row(label, syn_stats, real_stats):
    """Print one feature row: label | syn stats | real stats side by side."""
    keys = ["count", "min", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean", "stdev"]
    syn_parts = []
    real_parts = []
    for k in keys:
        v = syn_stats.get(k)
        if k == "count":
            syn_parts.append(f"{v if v is not None else 0:>7}")
        else:
            syn_parts.append(_fmt(v, width=11, decimals=4))
        v = real_stats.get(k)
        if k == "count":
            real_parts.append(f"{v if v is not None else 0:>7}")
        else:
            real_parts.append(_fmt(v, width=11, decimals=4))
    print(f"  {label:<22} | {'SYN':^60} | {'REAL':^60}")
    print(f"  {'':22} | cnt  min       p10       p25       p50       p75       p90       p99       max       mean      stdev    "
          f" | cnt  min       p10       p25       p50       p75       p90       p99       max       mean      stdev")
    print(f"  {'':22} | {''.join(f'{p:>11}' if i > 0 else f'{p:>7}' for i, p in enumerate(syn_parts))}"
          f" | {''.join(f'{p:>11}' if i > 0 else f'{p:>7}' for i, p in enumerate(real_parts))}")


def main() -> int:
    # Ensure UTF-8 output on Windows (for → and ⚠ characters)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 90)
    print("=== SCYLLA Synthetic Dataset Audit ===")
    print("=" * 90)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        return 1

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Single-pass fetch of all rows we need ───────────────────────────
    cols = [
        "id", "timestamp", "ticker", "expiration", "strike", "option_type",
        "volume", "open_interest", "vol_oi_ratio", "implied_vol",
        "underlier_price", "premium", "side", "dte", "is_weekly",
        "trend_alignment", "labeled", "label_success", "observed_return",
        "max_adverse_return", "is_synthetic",
        "delta_entry", "gamma_entry", "vega_entry", "theta_entry", "rho_entry",
        "delta_exit", "gamma_exit", "vega_exit", "theta_exit", "rho_exit",
        "commission_per_contract", "bid_ask_spread_pct", "early_exercise_risk",
        "delta_hedged_return",
        "days_to_earnings", "is_earnings_window", "days_to_fomc", "is_fomc_day",
        "vix_regime", "market_regime",
    ]
    cur.execute(f"SELECT {', '.join(cols)} FROM options_trades")
    rows = cur.fetchall()
    conn.close()

    # Index columns
    ci = {name: i for i, name in enumerate(cols)}

    syn_rows = [r for r in rows if r[ci["is_synthetic"]] == 1]
    real_rows = [r for r in rows if r[ci["is_synthetic"]] == 0]

    if not syn_rows:
        print("\nERROR: zero synthetic rows (is_synthetic=1) in options_trades.")
        print("Nothing to audit.")
        return 1

    # ── Section 1: Overview ─────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("1. OVERVIEW")
    print("-" * 70)
    syn_ts = [r[ci["timestamp"]] for r in syn_rows if r[ci["timestamp"]]]
    real_ts = [r[ci["timestamp"]] for r in real_rows if r[ci["timestamp"]]]
    syn_tickers = sorted({r[ci["ticker"]] for r in syn_rows if r[ci["ticker"]]})
    real_tickers = sorted({r[ci["ticker"]] for r in real_rows if r[ci["ticker"]]})

    print(f"  Synthetic rows : {len(syn_rows):>8,}")
    print(f"  Real rows      : {len(real_rows):>8,}")
    if syn_ts:
        print(f"  Synthetic dates: {min(syn_ts)}  \u2192  {max(syn_ts)}")
    else:
        print("  Synthetic dates: N/A")
    if real_ts:
        print(f"  Real dates     : {min(real_ts)}  \u2192  {max(real_ts)}")
    else:
        print("  Real dates     : N/A")
    print(f"  Synthetic tickers: {len(syn_tickers)} distinct")
    print(f"  Real tickers     : {len(real_tickers)} distinct")

    # ── Section 2: Per-ticker breakdown (synthetic) ─────────────────────
    print("\n" + "-" * 70)
    print("2. PER-TICKER BREAKDOWN (synthetic)")
    print("-" * 70)

    # Group by ticker in Python
    by_ticker = {}
    for r in syn_rows:
        t = r[ci["ticker"]] or "?"
        by_ticker.setdefault(t, []).append(r)

    # Pre-compute per-ticker stats
    ticker_stats = []
    for t, trs in by_ticker.items():
        ts = sorted(x[ci["timestamp"]] for x in trs if x[ci["timestamp"]])
        voi = sorted(x[ci["vol_oi_ratio"]] for x in trs if x[ci["vol_oi_ratio"]] is not None)
        iv = sorted(x[ci["implied_vol"]] for x in trs if x[ci["implied_vol"]] is not None)
        labeled_rows = [x for x in trs if x[ci["labeled"]] == 1]
        success_count = sum(1 for x in labeled_rows if x[ci["label_success"]] == 1)
        label_rate = (success_count / len(labeled_rows) * 100.0) if labeled_rows else None
        buys = sum(1 for x in trs if (x[ci["side"]] or "").upper() == "BUY")
        sells = sum(1 for x in trs if (x[ci["side"]] or "").upper() == "SELL")
        total_side = buys + sells
        buy_pct = (buys / total_side * 100.0) if total_side else 0.0
        sell_pct = (sells / total_side * 100.0) if total_side else 0.0
        ticker_stats.append({
            "ticker": t,
            "count": len(trs),
            "date_min": min(ts) if ts else "N/A",
            "date_max": max(ts) if ts else "N/A",
            "voi_p50": _percentile(voi, 0.50),
            "voi_p99": _percentile(voi, 0.99),
            "voi_max": max(voi) if voi else None,
            "iv_p50": _percentile(iv, 0.50),
            "iv_p99": _percentile(iv, 0.99),
            "label_rate": label_rate,
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
        })
    ticker_stats.sort(key=lambda x: x["count"], reverse=True)

    hdr = (f"  {'Ticker':<8} | {'Rows':>7} | {'Date Range':<23} | "
           f"{'voi_p50':>8} {'voi_p99':>8} {'voi_max':>8} | "
           f"{'iv_p50':>8} {'iv_p99':>8} | "
           f"{'lbl%':>6} | {'BUY%':>6} {'SELL%':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in ticker_stats:
        dr = f"{s['date_min']}..{s['date_max']}"
        if len(dr) > 23:
            dr = dr[:23]
        lbl = f"{s['label_rate']:.1f}" if s["label_rate"] is not None else "N/A"
        print(f"  {s['ticker']:<8} | {s['count']:>7,} | {dr:<23} | "
              f"{_fmt(s['voi_p50'], 8, 3)} {_fmt(s['voi_p99'], 8, 3)} {_fmt(s['voi_max'], 8, 3)} | "
              f"{_fmt(s['iv_p50'], 8, 4)} {_fmt(s['iv_p99'], 8, 4)} | "
              f"{lbl:>6} | {s['buy_pct']:>5.1f}% {s['sell_pct']:>5.1f}%")

    # ── Section 3: Feature distributions (synthetic vs real) ────────────
    print("\n" + "-" * 70)
    print("3. FEATURE DISTRIBUTIONS (synthetic vs real)")
    print("-" * 70)

    def _collect(rows, col_name, labeled_only=False):
        idx = ci[col_name]
        if labeled_only:
            lab_idx = ci["labeled"]
            return [r[idx] for r in rows if r[lab_idx] == 1 and r[idx] is not None]
        return [r[idx] for r in rows if r[idx] is not None]

    for feat in FEATURES:
        syn_vals = _collect(syn_rows, feat)
        real_vals = _collect(real_rows, feat)
        syn_s = _stats_block(syn_vals)
        real_s = _stats_block(real_vals)
        print()
        _print_stats_row(feat, syn_s, real_s)

    for feat in LABEL_ONLY_FEATURES:
        syn_vals = _collect(syn_rows, feat, labeled_only=True)
        real_vals = _collect(real_rows, feat, labeled_only=True)
        syn_s = _stats_block(syn_vals)
        real_s = _stats_block(real_vals)
        print()
        _print_stats_row(feat + " (lbl=1)", syn_s, real_s)

    print("\n  --- BS Greeks (synthetic rows only; real rows are NULL) ---")
    for feat in GREEK_FEATURES:
        syn_vals = _collect(syn_rows, feat)
        real_vals = _collect(real_rows, feat)
        syn_s = _stats_block(syn_vals)
        real_s = _stats_block(real_vals)
        print()
        _print_stats_row(feat, syn_s, real_s)

    print("\n  --- Categorical / binary features ---")
    for feat in CATEGORICAL_FEATURES:
        syn_vals = _collect(syn_rows, feat)
        real_vals = _collect(real_rows, feat)
        syn_unique = {}
        for v in syn_vals:
            syn_unique[v] = syn_unique.get(v, 0) + 1
        real_unique = {}
        for v in real_vals:
            real_unique[v] = real_unique.get(v, 0) + 1
        all_vals = sorted(set(syn_unique.keys()) | set(real_unique.keys()), key=lambda x: (x is None, x))
        print(f"\n  {feat}:")
        print(f"    {'Value':<10} | {'SYN cnt':>9} {'SYN %':>8} | {'REAL cnt':>9} {'REAL %':>8}")
        print("    " + "-" * 55)
        for v in all_vals:
            sc_ = syn_unique.get(v, 0)
            rc_ = real_unique.get(v, 0)
            sp_ = (sc_ / len(syn_rows) * 100.0) if syn_rows else 0.0
            rp_ = (rc_ / len(real_rows) * 100.0) if real_rows else 0.0
            lbl = str(v) if v is not None else "NULL"
            print(f"    {lbl:<10} | {sc_:>9,} {sp_:>7.2f}% | {rc_:>9,} {rp_:>7.2f}%")

    # ── Section 4: Label summary ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("4. LABEL SUMMARY")
    print("-" * 70)

    def _label_stats(rows):
        labeled = [r for r in rows if r[ci["labeled"]] == 1]
        unlabeled = [r for r in rows if r[ci["labeled"]] != 1]
        success = sum(1 for r in labeled if r[ci["label_success"]] == 1)
        rate = (success / len(labeled) * 100.0) if labeled else None
        return len(labeled), len(unlabeled), success, rate

    syn_labeled, syn_unlabeled, syn_success, syn_rate = _label_stats(syn_rows)
    real_labeled, real_unlabeled, real_success, real_rate = _label_stats(real_rows)

    print(f"  Synthetic: labeled={syn_labeled:,}, unlabeled={syn_unlabeled:,}, "
          f"success={syn_success:,}, rate={syn_rate:.2f}%" if syn_rate is not None else
          f"  Synthetic: labeled={syn_labeled:,}, unlabeled={syn_unlabeled:,}, success={syn_success:,}, rate=N/A")
    print(f"  Real     : labeled={real_labeled:,}, unlabeled={real_unlabeled:,}, "
          f"success={real_success:,}, rate={real_rate:.2f}%" if real_rate is not None else
          f"  Real     : labeled={real_labeled:,}, unlabeled={real_unlabeled:,}, success={real_success:,}, rate=N/A")

    # ── Section 5: Side balance ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("5. SIDE BALANCE")
    print("-" * 70)

    def _side_stats(rows):
        buys = sum(1 for r in rows if (r[ci["side"]] or "").upper() == "BUY")
        sells = sum(1 for r in rows if (r[ci["side"]] or "").upper() == "SELL")
        total = buys + sells
        return buys, sells, (buys / total * 100.0 if total else 0.0), (sells / total * 100.0 if total else 0.0)

    sb, ss, sbp, ssp = _side_stats(syn_rows)
    rb, rs, rbp, rsp = _side_stats(real_rows)
    print(f"  Synthetic: BUY={sb:,} ({sbp:.1f}%)  SELL={ss:,} ({ssp:.1f}%)")
    print(f"  Real     : BUY={rb:,} ({rbp:.1f}%)  SELL={rs:,} ({rsp:.1f}%)")

    # ── Section 6: Whale threshold crossing ─────────────────────────────
    print("\n" + "-" * 70)
    print("6. WHALE THRESHOLD CROSSING (vol_oi_ratio >= threshold)")
    print("-" * 70)

    syn_voi = [r[ci["vol_oi_ratio"]] for r in syn_rows if r[ci["vol_oi_ratio"]] is not None]
    real_voi = [r[ci["vol_oi_ratio"]] for r in real_rows if r[ci["vol_oi_ratio"]] is not None]

    hdr2 = f"  {'Threshold':>10} | {'SYN cnt':>9} {'SYN %':>8} | {'REAL cnt':>9} {'REAL %':>8}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for thr in WHALE_THRESHOLDS:
        syn_cnt = sum(1 for v in syn_voi if v >= thr)
        real_cnt = sum(1 for v in real_voi if v >= thr)
        syn_pct = (syn_cnt / len(syn_rows) * 100.0) if syn_rows else 0.0
        real_pct = (real_cnt / len(real_rows) * 100.0) if real_rows else 0.0
        print(f"  {thr:>10.1f} | {syn_cnt:>9,} {syn_pct:>7.2f}% | {real_cnt:>9,} {real_pct:>7.2f}%")
        if syn_cnt == 0 and thr >= 5.0:
            print(f"           \u26a0 WARNING: zero synthetic whales at threshold {thr}")

    # ── Section 7: Option type balance ──────────────────────────────────
    print("\n" + "-" * 70)
    print("7. OPTION TYPE BALANCE")
    print("-" * 70)

    def _opt_stats(rows):
        calls = sum(1 for r in rows if (r[ci["option_type"]] or "").upper().startswith("C"))
        puts = sum(1 for r in rows if (r[ci["option_type"]] or "").upper().startswith("P"))
        total = calls + puts
        return calls, puts, (calls / total * 100.0 if total else 0.0)

    sc, sp, scp = _opt_stats(syn_rows)
    rc, rp, rcp = _opt_stats(real_rows)
    print(f"  Synthetic: Call={sc:,}  Put={sp:,}  %Call={scp:.1f}%")
    print(f"  Real     : Call={rc:,}  Put={rp:,}  %Call={rcp:.1f}%")

    # ── Section 8: DTE distribution ─────────────────────────────────────
    print("\n" + "-" * 70)
    print("8. DTE DISTRIBUTION")
    print("-" * 70)

    def _dte_hist(rows):
        dtes = [r[ci["dte"]] for r in rows]
        counts = {}
        nulls = 0
        for name, pred in DTE_BUCKETS:
            c = sum(1 for d in dtes if pred(d))
            counts[name] = c
        nulls = sum(1 for d in dtes if d is None)
        return counts, nulls

    syn_dte, syn_null = _dte_hist(syn_rows)
    real_dte, real_null = _dte_hist(real_rows)

    hdr3 = f"  {'Bucket':<8} | {'SYN cnt':>9} {'SYN %':>8} | {'REAL cnt':>9} {'REAL %':>8}"
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))
    for name, _ in DTE_BUCKETS:
        sc_ = syn_dte.get(name, 0)
        rc_ = real_dte.get(name, 0)
        sp_ = (sc_ / len(syn_rows) * 100.0) if syn_rows else 0.0
        rp_ = (rc_ / len(real_rows) * 100.0) if real_rows else 0.0
        print(f"  {name:<8} | {sc_:>9,} {sp_:>7.2f}% | {rc_:>9,} {rp_:>7.2f}%")
    print(f"  {'NULL':<8} | {syn_null:>9,} {'':>8} | {real_null:>9,}")

    # ── Section 9: is_weekly / trend_alignment ──────────────────────────
    print("\n" + "-" * 70)
    print("9. IS_WEEKLY / TREND_ALIGNMENT CONTINGENCY")
    print("-" * 70)

    def _contingency(rows):
        wk_idx = ci["is_weekly"]
        ta_idx = ci["trend_alignment"]
        counts = {}
        for r in rows:
            wk = r[wk_idx]
            ta = r[ta_idx]
            key = (wk, ta)
            counts[key] = counts.get(key, 0) + 1
        return counts

    syn_ct = _contingency(syn_rows)
    real_ct = _contingency(real_rows)

    all_keys = sorted(set(syn_ct.keys()) | set(real_ct.keys()),
                      key=lambda k: (str(k[0]), str(k[1])))
    hdr4 = f"  {'is_weekly':<12} {'trend_align':<18} | {'SYN':>8} | {'REAL':>8}"
    print(hdr4)
    print("  " + "-" * (len(hdr4) - 2))
    for k in all_keys:
        wk_label = str(k[0]) if k[0] is not None else "NULL"
        ta_label = str(k[1]) if k[1] is not None else "NULL"
        print(f"  {wk_label:<12} {ta_label:<18} | {syn_ct.get(k, 0):>8,} | {real_ct.get(k, 0):>8,}")

    # ── Final scrapeable line ───────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"SYN_ROWS={len(syn_rows)} REAL_ROWS={len(real_rows)}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
