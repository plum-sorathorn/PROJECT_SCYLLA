import argparse
import os
import sqlite3
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
sys.path.insert(0, BACKEND_DIR)

DB_PATH = os.path.join(BACKEND_DIR, "scylla_ml.db")

from seed_grounded_real_options import seed_grounded_options


def main():
    ap = argparse.ArgumentParser(
        description="Phase 6: Generate multiple decorrelated synthetic ensembles."
    )
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--no-wipe", action="store_true")
    args = ap.parse_args()

    seeds = [args.base_seed + k * args.step for k in range(args.n_seeds)]

    print("=" * 90)
    print("SCYLLA Phase 6: Synthetic Ensemble Generation")
    print("=" * 90)
    print(f"DB: {DB_PATH}")
    print(f"Seeds: {seeds}")
    print(f"Step: {args.step}, Base: {args.base_seed}, N: {args.n_seeds}")
    print()

    if not args.no_wipe:
        print("[wipe-all] Clearing all existing options_trades rows before ensemble generation...")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM options_trades")
        conn.commit()
        conn.close()
        print("[wipe-all] Done.")
        print()

    for idx, seed in enumerate(seeds):
        print(f"Generating ensemble seed={seed} ({idx + 1}/{args.n_seeds})...")
        seed_grounded_options(wipe=False, wipe_all=False, seed=seed)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM options_trades WHERE ensemble_id = ?", (seed,))
        n_this = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM options_trades")
        n_total = cur.fetchone()[0]
        conn.close()
        print(f"  ensemble_id={seed}: inserted {n_this} trades (total now {n_total})")
        print()

    print("=" * 90)
    print("Ensemble summary:")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ensemble_id, COUNT(*) FROM options_trades GROUP BY ensemble_id ORDER BY ensemble_id")
    rows = cur.fetchall()
    for eid, cnt in rows:
        print(f"  ensemble_id={eid}: {cnt} rows")
    cur.execute("SELECT COUNT(*) FROM options_trades")
    total = cur.fetchone()[0]
    conn.close()
    print(f"  TOTAL: {total} rows across {len(rows)} ensembles")
    print("=" * 90)
    sys.exit(0)


if __name__ == "__main__":
    main()
