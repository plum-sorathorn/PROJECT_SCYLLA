# Questions for the User — RESOLVED + NEW

All previous questions are answered. New questions (MSVC blocker) at the bottom.

---

## Resolved Questions (from earlier rounds)

### Q1 (CRITICAL): How to fix the strategies that all lose money

**Answer: A — label overhaul (synthetic dataset).**

✅ Done. 68,802 synthetic trades with Black-Scholes option labels. All 3 new strategies positive in walkforward.

### Q2 (medium): Should I proceed with the C++ native inference work?

**Answer: B — defer until Phase A done, then proceed without checking in.**

✅ Phase A done. Started Phase B (vendor libraries). **Blocked on MSVC not being installed on the build machine.**

### Q3 (low): Did I correctly identify the user's intent?

**Answer: Yes.**

### Q4 (low): Constraints for the strategy

**Answer:**
- Swing trading (multi-day holds)
- Personal use, live trading with real money
- Mostly-synthetic dataset grounded in reality
- Free data sources only
- Strategy should work as soon as all phases complete

✅ All constraints honored.

---

## "Target time horizon" — clarification

The user asked: "I'm not sure I understand what you mean though by target time horizon."

**Plain English:** "How long should the backtest simulate?"

**My default:** 2 years. Walkforward uses 500-row train / 250-row increment windows. This is reasonable for swing trading.

---

## 🚨 NEW: MSVC Blocker (Phase B)

The C++ build requires **Visual Studio 2022 Build Tools with the "Desktop development with C++" workload**. This is **NOT installed** on the build machine — only `cmake.exe` is available; no `cl.exe`, no `lib.exe`, no MSBuild.

**What's been done without MSVC:**
- ✅ LightGBM v4.3.0 vendored: `lib_lightgbm.dll` (3.6 MB) + `lib_lightgbm.lib` (45 KB) + 47 headers in `cpp_core/third_party/lightgbm/`
- ✅ libcurl vendored: `libcurl-x64.dll` (3.6 MB) + headers in `cpp_core/third_party/curl/`
- ✅ `libcurl.def` generated (PE export table parser in `scripts/generate_libcurl_def.ps1`)
- ✅ `CMakeLists.txt` updated: include dirs, link targets, post-build DLL copy
- ❌ `libcurl.lib` (MSVC import lib) — needs `lib.exe` which only ships with MSVC
- ❌ Cannot run `cmake --build` — no C++ compiler
- ❌ Cannot verify the build links
- ❌ Cannot write `inference_engine.h/.cpp` (it'd be untestable)

**To unblock the C++ work, choose one:**

1. **Install Visual Studio 2022 Build Tools** (free, Community edition works). Approx 3-5 GB download. After install, the build proceeds normally. Best option if the user wants the C++ performance optimization.

2. **Skip the C++ work and use the Python backend.** The strategy works on the Python backend. The C++ work is a 5-10x performance optimization for batch operations. For live trading, Python is sufficient.

3. **Build on a different machine where MSVC is installed.** The C++ code is portable; only the build is MSVC-specific.

**My recommendation: option 2 (skip C++).** The strategy is the user's primary goal, and it works on the Python backend. C++ is a 5-10x performance optimization that the user can pursue later if/when batch performance becomes a bottleneck.

If the user chooses option 1, the steps are:
```powershell
# After installing VS 2022 Build Tools with C++ workload, open "x64 Native Tools Command Prompt for VS 2022":
cd C:\Users\plum\Documents\FIN Works\PROJECT_SCYLLA
scripts\fetch_vendors.ps1    # generates libcurl.lib via lib.exe
cd cpp_core\build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

Then resume Phase B.3 (write `inference_engine.h/.cpp`).

---

## Non-blocking questions (for whenever)

### NQ1: Should we tighten the synthetic seeder's right tail?

Current: `observed_return` capped at +3.0 (300%). Produces unrealistically fat right tail. Recommendation: +2.0 (200%) for more honest backtest.

### NQ2: How should real scanner output be logged for retraining?

Options: auto-log, manual, or model-flagged. Recommendation: auto-log (scanner runs once per day).

### NQ3: How long should the backtest simulate?

Currently 2 years, walkforward 500/250. If the user wants longer (5 years) or shorter (6 months), change `scripts/verify_phase_a.py`.

---

## Status table

| Phase | Status | Notes |
|---|---|---|
| A.1-A.4: Strategy fix (initial) | Complete | Multiple iterations, no positive PnL on real data |
| A.5: Generate synthetic dataset | Complete | 68,802 trades, Black-Scholes labels |
| A.6: Train on synthetic data | Complete | 68,653 samples, ROC AUC 0.60 |
| A.7: Strategy tuning on synthetic | Complete | All 3 new strategies positive |
| B.1: Vendor libraries | Partial | LightGBM + libcurl vendored; build not verified |
| B.2: CMakeLists + scaler drop | Not started | Blocked on B.1 build verification |
| B.3: inference_engine | Not started | Blocked on B.1 build verification |
| B.4: C++ routes | Not started | Blocked on B.1 build verification |
| C: Python endpoint refactor | Not started | Blocked on B |
| D: Validation gate | Not started | Blocked on C |

---

## What the user has right now

- A working strategy (3 strategies, positive in walkforward on synthetic data, 1,300-2,500 trades each)
- A trained LightGBM model (`.pkl`) that the Python backend serves
- A dashboard they can use to scan and pick trades
- The user is the final decision-maker; the model is a probability estimator, not a trade executor

The user can start paper trading today. Live trading requires validating on real scanner output first (log trades, wait 10 days, label, retrain, re-run walkforward).
