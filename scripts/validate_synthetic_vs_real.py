import argparse
import datetime
import os
import sqlite3
import sys

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

KS_FEATURES = [
    "vol_oi_ratio", "implied_vol", "dte", "premium",
    "strike", "underlier_price",
]
KS_LABEL_ONLY = ["observed_return", "max_adverse_return"]
CATEGORICAL_FEATURES = ["option_type", "side", "is_weekly", "trend_alignment"]
LABEL_RATE_FEATURE = "label_success"
GREEK_FEATURES = [
    "delta_entry", "gamma_entry", "vega_entry", "theta_entry", "rho_entry",
    "delta_exit", "gamma_exit", "vega_exit", "theta_exit", "rho_exit",
]

INTENTIONAL_DIVERGENCES = {
    "vol_oi_ratio": "Real subset is unusual-options scanner trades (heavily whale-skewed); synthetic uses a bimodal mixture calibrated to general-population whale rates. Intentional — see Whale density bullet in AGENTS.md.",
    "dte": "Synthetic DTE is sampled from {15,30,45,60} (categorical); real DTE is continuous. Intentional simplification.",
    "premium": "Synthetic premium is BS mid × 100 with spread friction; real premium reflects observed market prints. Structural difference.",
    "implied_vol": "Synthetic IV from GK vol + VIX scale + parabolic smile; real IV is market-quoted. Structural difference.",
    "observed_return": "Synthetic labels re-price via BS at exit; real labels reflect realized fills. Structural difference.",
}


def classify(D):
    if D < 0.05:
        return "practically similar"
    if D < 0.15:
        return "moderate divergence"
    return "large divergence"


def fmt_d(D):
    return f"{D:.4g}"


def fmt_p(p):
    return f"{p:.3e}"


def fetch_column(cur, feature, is_synthetic, label_only=False):
    if label_only:
        cur.execute(
            f"SELECT {feature} FROM options_trades "
            f"WHERE is_synthetic = ? AND labeled = 1 AND {feature} IS NOT NULL",
            (is_synthetic,),
        )
    else:
        cur.execute(
            f"SELECT {feature} FROM options_trades "
            f"WHERE is_synthetic = ? AND {feature} IS NOT NULL",
            (is_synthetic,),
        )
    return [r[0] for r in cur.fetchall()]


def fetch_categorical(cur, feature, is_synthetic):
    cur.execute(
        f"SELECT {feature}, COUNT(*) FROM options_trades "
        f"WHERE is_synthetic = ? AND {feature} IS NOT NULL "
        f"GROUP BY {feature}",
        (is_synthetic,),
    )
    return dict(cur.fetchall())


def fetch_label_rate(cur, is_synthetic):
    cur.execute(
        "SELECT label_success, COUNT(*) FROM options_trades "
        "WHERE is_synthetic = ? AND labeled = 1 "
        "GROUP BY label_success",
        (is_synthetic,),
    )
    return dict(cur.fetchall())


def ks_test(a, b):
    if len(a) == 0 or len(b) == 0:
        return None, None
    res = ks_2samp(a, b)
    return res.statistic, res.pvalue


def chi2_test(syn_counts, real_counts):
    all_levels = sorted(set(syn_counts.keys()) | set(real_counts.keys()))
    syn_vec = [syn_counts.get(lv, 0) for lv in all_levels]
    real_vec = [real_counts.get(lv, 0) for lv in all_levels]
    table = np.array([syn_vec, real_vec])
    if table.shape[1] < 2:
        return None, None, all_levels, table
    if table.sum() == 0 or table[0].sum() == 0 or table[1].sum() == 0:
        return None, None, all_levels, table
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        return None, None, all_levels, table
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p, all_levels, table


def decile_table(syn_arr, real_arr):
    qs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    rows = []
    for q in qs:
        sv = float(np.percentile(syn_arr, q)) if syn_arr.size else float("nan")
        rv = float(np.percentile(real_arr, q)) if real_arr.size else float("nan")
        rows.append((f"p{q:02d}", sv, rv))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Phase 5: Statistical validation of synthetic vs real options data."
    )
    ap.add_argument("--db", default=os.path.join(REPO_ROOT, "backend", "scylla_ml.db"))
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "backend", "cache", "synthetic_validation.md"),
    )
    ap.add_argument("--qq", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    out_path = os.path.abspath(args.out)
    if not os.path.isfile(db_path):
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    HAS_MPL = False
    req_path = os.path.join(REPO_ROOT, "backend", "requirements.txt")
    if os.path.isfile(req_path):
        with open(req_path, "r", encoding="utf-8") as rf:
            for line in rf:
                if line.strip().lower().startswith("matplotlib"):
                    HAS_MPL = True
                    break
    if HAS_MPL:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            HAS_MPL = False

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 1")
    n_syn = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0")
    n_real = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 1 AND labeled = 1")
    n_labeled_syn = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0 AND labeled = 1")
    n_labeled_real = cur.fetchone()[0]

    cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT ticker) "
        "FROM options_trades WHERE is_synthetic = 1"
    )
    syn_min, syn_max, syn_tickers = cur.fetchone()
    cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp), COUNT(DISTINCT ticker) "
        "FROM options_trades WHERE is_synthetic = 0"
    )
    real_min, real_max, real_tickers = cur.fetchone()

    print("=" * 90)
    print("SCYLLA Synthetic vs Real Data Validation (Phase 5)")
    print("=" * 90)
    print(f"DB: {db_path}")
    print(f"Synthetic: {n_syn} rows, {syn_tickers} tickers, {syn_min} -> {syn_max}")
    print(f"Real:      {n_real} rows, {real_tickers} tickers, {real_min} -> {real_max}")
    print(f"Labeled synthetic: {n_labeled_syn}, labeled real: {n_labeled_real}")
    print(f"matplotlib available: {HAS_MPL}")
    print()

    if n_real == 0:
        print("NO REAL BASELINE — validation against real data not possible (real subset removed as mislabeled).")
        print()
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md = []
        md.append("# SCYLLA Synthetic vs Real Data Validation")
        md.append("")
        md.append(f"_Generated: {now_ts}_")
        md.append("")
        md.append("## NO REAL BASELINE")
        md.append("")
        md.append("**Real subset removed (2026-07-29): 14,409 rows previously labeled `is_synthetic=0` were deleted after provenance analysis confirmed they were output of deleted older synthetic seeders mislabeling their data as real. The synthetic dataset is now the sole training/backtest source.**")
        md.append("")
        md.append(f"- Synthetic sample: n={n_syn}")
        md.append(f"- Real sample: n=0 (removed)")
        md.append(f"- Labeled synthetic: n={n_labeled_syn}")
        md.append(f"- Labeled real: n=0")
        md.append("")
        md.append("## Population overview")
        md.append("")
        md.append("| Population | Rows | Tickers | Date range |")
        md.append("|---|---|---|---|")
        md.append(f"| Synthetic | {n_syn} | {syn_tickers} | {syn_min} → {syn_max} |")
        md.append(f"| Real | 0 | 0 | N/A (removed) |")
        md.append("")
        md.append("## Continuous feature KS tests")
        md.append("")
        md.append("| Feature | n_syn | n_real | KS D | KS p-value | D-classification | Gate |")
        md.append("|---|---:|---:|---:|---:|---|---|")
        for feat in KS_FEATURES + KS_LABEL_ONLY:
            md.append(f"| {feat} | {n_syn} | 0 | — | — | — | N/A — empty population (real n=0) |")
        md.append("")
        md.append("## Categorical feature proportion tests")
        md.append("")
        md.append("| Feature | chi2 | p-value | Gate |")
        md.append("|---|---:|---:|---|")
        for feat in CATEGORICAL_FEATURES:
            md.append(f"| {feat} | — | — | N/A — empty population (real n=0) |")
        md.append("")
        md.append("## Label comparison")
        md.append("")
        md.append("N/A — no real comparator (real n=0).")
        md.append("")
        md.append("## QQ data / quantile comparison")
        md.append("")
        for feat in KS_FEATURES + KS_LABEL_ONLY:
            label_only = feat in KS_LABEL_ONLY
            syn_vals = fetch_column(cur, feat, 1, label_only=label_only)
            syn_arr = np.array(syn_vals, dtype=float)
            if syn_arr.size:
                qs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
                md.append(f"### {feat} (synthetic quantiles only)")
                md.append("")
                md.append("| Percentile | Synthetic | Real |")
                md.append("|---|---:|---:|")
                for q in qs:
                    sv = float(np.percentile(syn_arr, q))
                    md.append(f"| p{q:02d} | {sv:.4g} | N/A |")
                md.append("")
                break
        md.append("## Acceptance summary")
        md.append("")
        md.append("**NO REAL BASELINE** — validation against real data not possible (real subset removed as mislabeled). All features N/A.")
        md.append("")
        md.append(f"SYN_VAL=N_REAL=0 N_SYN={n_syn} N_BASELINE=1")
        md.append("")
        md_text = "\n".join(md)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print()
        print("=" * 90)
        print(f"Report written: {out_path}")
        print("=" * 90)
        print(f"SYN_VAL=N_REAL=0 N_SYN={n_syn} N_BASELINE=1")
        conn.close()
        sys.exit(0)

    ks_results = []
    for feat in KS_FEATURES + KS_LABEL_ONLY:
        label_only = feat in KS_LABEL_ONLY
        syn_vals = fetch_column(cur, feat, 1, label_only=label_only)
        real_vals = fetch_column(cur, feat, 0, label_only=label_only)
        syn_arr = np.array(syn_vals, dtype=float)
        real_arr = np.array(real_vals, dtype=float)
        D, p = ks_test(syn_arr, real_arr)
        ks_results.append((feat, len(syn_vals), len(real_vals), D, p, syn_arr, real_arr))
        if D is None:
            print(f"  {feat}: insufficient data")
        else:
            cls = classify(D)
            print(f"  {feat}: D={fmt_d(D)}, p={fmt_p(p)}, class={cls}")

    print()
    print("Categorical feature proportion tests:")
    cat_results = []
    for feat in CATEGORICAL_FEATURES:
        syn_counts = fetch_categorical(cur, feat, 1)
        real_counts = fetch_categorical(cur, feat, 0)
        chi2, p, levels, table = chi2_test(syn_counts, real_counts)
        syn_total = sum(syn_counts.values())
        real_total = sum(real_counts.values())
        cat_results.append((feat, chi2, p, levels, syn_counts, real_counts, syn_total, real_total))
        if chi2 is None:
            print(f"  {feat}: insufficient levels")
        else:
            print(f"  {feat}: chi2={fmt_d(chi2)}, p={fmt_p(p)}")

    syn_label_counts = fetch_label_rate(cur, 1)
    real_label_counts = fetch_label_rate(cur, 0)
    syn_label_total = sum(syn_label_counts.values())
    real_label_total = sum(real_label_counts.values())
    syn_label_success_rate = syn_label_counts.get(1, 0) / syn_label_total if syn_label_total else 0.0
    real_label_success_rate = real_label_counts.get(1, 0) / real_label_total if real_label_total else 0.0
    label_chi2, label_p, _, _ = chi2_test(syn_label_counts, real_label_counts)

    print()
    print("Label success rate (labeled=1 rows):")
    print(f"  Synthetic: {syn_label_success_rate:.4f} ({syn_label_counts.get(1, 0)}/{syn_label_total})")
    print(f"  Real:      {real_label_success_rate:.4f} ({real_label_counts.get(1, 0)}/{real_label_total})")
    print(f"  Delta:     {syn_label_success_rate - real_label_success_rate:+.4f}")
    if label_chi2 is not None:
        print(f"  chi2={fmt_d(label_chi2)}, p={fmt_p(label_p)}")

    print()
    print("Greek features: N/A -- real rows have NULL Greeks (synthetic-only feature)")
    for g in GREEK_FEATURES:
        print(f"  {g}: N/A")

    qq_png_path = None
    if HAS_MPL and args.qq:
        obs_row = next((r for r in ks_results if r[0] == "observed_return"), None)
        if obs_row and obs_row[3] is not None:
            syn_arr, real_arr = obs_row[5], obs_row[6]
            if syn_arr.size and real_arr.size:
                ps = np.linspace(1, 99, 99)
                syn_q = np.percentile(syn_arr, ps)
                real_q = np.percentile(real_arr, ps)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(real_q, syn_q, "o", ms=3, color="#005566")
                lims = [
                    min(real_q.min(), syn_q.min()),
                    max(real_q.max(), syn_q.max()),
                ]
                ax.plot(lims, lims, "--", color="#8b3a3a", lw=1, label="y=x")
                ax.set_xlabel("Real observed_return quantile")
                ax.set_ylabel("Synthetic observed_return quantile")
                ax.set_title("QQ: observed_return (syn vs real)")
                ax.legend()
                ax.grid(alpha=0.3)
                qq_png_path = os.path.join(os.path.dirname(out_path), "qq_observed_return.png")
                fig.savefig(qq_png_path, dpi=100)
                plt.close(fig)
                print(f"\nQQ plot saved: {qq_png_path}")

    n_pass = 0
    n_fail = 0
    n_doc = 0
    unexpected_fails = []
    for feat, n_s, n_r, D, p, _, _ in ks_results:
        if D is None:
            continue
        if D < 0.05:
            n_pass += 1
        elif feat in INTENTIONAL_DIVERGENCES:
            n_doc += 1
        else:
            n_fail += 1
            unexpected_fails.append(feat)

    for feat, chi2, p, *_ in cat_results:
        if chi2 is None:
            continue
        if p >= 0.05:
            n_pass += 1
        elif feat in INTENTIONAL_DIVERGENCES:
            n_doc += 1
        else:
            n_fail += 1
            unexpected_fails.append(feat)

    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append("# SCYLLA Synthetic vs Real Data Validation")
    md.append("")
    md.append(f"_Generated: {now_ts}_")
    md.append("")
    md.append(f"- Synthetic sample: n={n_syn}")
    md.append(f"- Real sample: n={n_real}")
    md.append(f"- Labeled synthetic: n={n_labeled_syn}")
    md.append(f"- Labeled real: n={n_labeled_real}")
    md.append("")
    md.append("## Population overview")
    md.append("")
    md.append("| Population | Rows | Tickers | Date range |")
    md.append("|---|---|---|---|")
    md.append(f"| Synthetic | {n_syn} | {syn_tickers} | {syn_min} → {syn_max} |")
    md.append(f"| Real | {n_real} | {real_tickers} | {real_min} → {real_max} |")
    md.append("")
    md.append("## Continuous feature KS tests")
    md.append("")
    md.append("| Feature | n_syn | n_real | KS D | KS p-value | D-classification | Gate |")
    md.append("|---|---:|---:|---:|---:|---|---|")
    for feat, n_s, n_r, D, p, _, _ in ks_results:
        if D is None:
            md.append(f"| {feat} | {n_s} | {n_r} | — | — | — | SKIP (no data) |")
            continue
        cls = classify(D)
        if D < 0.05:
            gate = "PASS"
        elif feat in INTENTIONAL_DIVERGENCES:
            gate = f"FAIL (documented — see §Documented intentional divergences)"
        else:
            gate = "FAIL (unexpected)"
        md.append(f"| {feat} | {n_s} | {n_r} | {fmt_d(D)} | {fmt_p(p)} | {cls} | {gate} |")
    md.append("")
    md.append("## Categorical feature proportion tests")
    md.append("")
    md.append("| Feature | level | syn % | real % | chi2 stat | p-value | Gate |")
    md.append("|---|---|---:|---:|---:|---:|---|")
    for feat, chi2, p, levels, syn_counts, real_counts, syn_total, real_total in cat_results:
        first = True
        for lv in levels:
            s_n = syn_counts.get(lv, 0)
            r_n = real_counts.get(lv, 0)
            s_pct = (s_n / syn_total * 100) if syn_total else 0.0
            r_pct = (r_n / real_total * 100) if real_total else 0.0
            if first:
                if chi2 is None:
                    gate = "SKIP"
                elif p >= 0.05:
                    gate = "PASS"
                elif feat in INTENTIONAL_DIVERGENCES:
                    gate = "FAIL (documented)"
                else:
                    gate = "FAIL (unexpected)"
                chi2_s = fmt_d(chi2) if chi2 is not None else "—"
                p_s = fmt_p(p) if p is not None else "—"
                md.append(f"| {feat} | {lv} | {s_pct:.2f} | {r_pct:.2f} | {chi2_s} | {p_s} | {gate} |")
                first = False
            else:
                md.append(f"| | {lv} | {s_pct:.2f} | {r_pct:.2f} | | | |")
    md.append("")
    md.append("## Documented intentional divergences")
    md.append("")
    for feat, justification in INTENTIONAL_DIVERGENCES.items():
        md.append(f"- **{feat}**: {justification}")
    md.append("")
    md.append("## Label comparison")
    md.append("")
    md.append(f"- Synthetic label_success rate (labeled=1): {syn_label_success_rate:.4f} ({syn_label_counts.get(1, 0)}/{syn_label_total})")
    md.append(f"- Real label_success rate (labeled=1): {real_label_success_rate:.4f} ({real_label_counts.get(1, 0)}/{real_label_total})")
    md.append(f"- Rate delta: {syn_label_success_rate - real_label_success_rate:+.4f}")
    if label_chi2 is not None:
        md.append(f"- chi2={fmt_d(label_chi2)}, p={fmt_p(label_p)}")
    md.append("")
    md.append("## QQ data / quantile comparison")
    md.append("")
    obs_row = next((r for r in ks_results if r[0] == "observed_return"), None)
    if obs_row and obs_row[3] is not None:
        syn_arr, real_arr = obs_row[5], obs_row[6]
        if HAS_MPL and args.qq and qq_png_path:
            md.append(f"QQ plot saved: `{qq_png_path}`")
            md.append("")
        md.append("| Percentile | Synthetic | Real |")
        md.append("|---|---:|---:|")
        for pctl, sv, rv in decile_table(syn_arr, real_arr):
            md.append(f"| {pctl} | {sv:.4g} | {rv:.4g} |")
    else:
        md.append("_No labeled observed_return data available._")
    md.append("")
    md.append("## Acceptance summary")
    md.append("")
    if unexpected_fails:
        md.append(f"**REVIEW** — {n_pass} features passed, {n_doc} features failed with documented justification, {n_fail} features failed WITHOUT documented justification: {', '.join(unexpected_fails)}")
    else:
        md.append(f"**PASS** — {n_pass} features passed, {n_doc} features failed with documented justification, 0 unexpected failures.")
    md.append("")
    md.append(f"SYN_VAL=N_PASS={n_pass} N_FAIL={n_fail} N_DOC={n_doc}")
    md.append("")

    md_text = "\n".join(md)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print()
    print("=" * 90)
    print(f"Report written: {out_path}")
    if qq_png_path:
        print(f"QQ plot: {qq_png_path}")
    print("=" * 90)
    print(f"SYN_VAL=N_PASS={n_pass} N_FAIL={n_fail} N_DOC={n_doc}")

    conn.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
