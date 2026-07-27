#!/usr/bin/env python3
"""
scripts/benchmark_inference.py
==============================
Performance Benchmark Suite (Phase D Steps 20 & 21)

Measures latency and throughput across:
  1. Single-row prediction latency (warm cache)
  2. Batch prediction latency (100 rows)
  3. Walkforward backtest wall time
"""

import sys
import time
import requests
import json
import os
import sqlite3

CPP_PREDICT_URL = "http://127.0.0.1:8080/api/v1/ml/predict"
CPP_BATCH_URL = "http://127.0.0.1:8080/api/v1/ml/predict-batch"
DB_PATH = os.path.join("backend", "scylla_ml.db")

def main():
    print("=== SCYLLA Native C++ Inference Benchmark Suite ===")

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticker, underlier_price, strike, volume, open_interest,
               implied_vol, premium, side, dte, is_weekly, trend_alignment
        FROM options_trades
        WHERE is_synthetic = 1 AND labeled = 1
        LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("ERROR: No synthetic rows found for benchmark.")
        sys.exit(1)

    payloads = []
    for r in rows:
        payloads.append({
            "ticker": str(r[0]).upper(),
            "underlier_price": float(r[1]),
            "strike": float(r[2]),
            "volume": float(r[3]),
            "open_interest": float(r[4]),
            "implied_vol": float(r[5]),
            "premium": float(r[6]),
            "side": str(r[7]).upper(),
            "dte": float(r[8]),
            "is_weekly": str(r[9]),
            "trend_alignment": str(r[10]),
            "synthetic_hist_vol": 25.0,
            "synthetic_vix": 20.0
        })

    # 1. Warm-up
    requests.post(CPP_PREDICT_URL, json=payloads[0], timeout=5)

    # 2. Single-row latency (100 iterations)
    t0 = time.perf_counter()
    for p in payloads:
        requests.post(CPP_PREDICT_URL, json=p, timeout=5)
    t1 = time.perf_counter()
    single_avg_ms = ((t1 - t0) / len(payloads)) * 1000.0

    print(f"[BENCHMARK] Single-row predict average latency: {single_avg_ms:.2f} ms")

    # 3. Batch prediction latency (100 rows)
    t0 = time.perf_counter()
    resp = requests.post(CPP_BATCH_URL, json={"rows": payloads}, timeout=10)
    t1 = time.perf_counter()
    batch_total_ms = (t1 - t0) * 1000.0

    if resp.status_code == 200:
        preds = resp.json().get("predictions", [])
        print(f"[BENCHMARK] Batch predict (100 rows) wall time: {batch_total_ms:.2f} ms ({len(preds)} predictions returned)")
    else:
        print(f"[ERROR] Batch predict failed HTTP {resp.status_code}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_row_avg_ms": round(single_avg_ms, 3),
        "batch_100_rows_total_ms": round(batch_total_ms, 3),
        "per_row_in_batch_ms": round(batch_total_ms / 100.0, 3)
    }

    out_file = "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results recorded in {out_file}")

if __name__ == "__main__":
    main()
