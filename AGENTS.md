# AGENTS.md — PROJECT: SCYLLA

High-performance options whale scanner. Architecture: HTML/JS frontend → C++ Crow (:8080, native LightGBM) → Python FastAPI / yfinance (:6900). Windows-only. No paid API keys.

## Quick reference

| Action | Command |
|---|---|
| One-click launch | `.\LAUNCH_SCYLLA.ps1` — patches `app.js` API_BASE, builds C++ if missing, opens browser |
| Dev mode (C++ optional) | `.\scripts\start_dev.ps1` |
| Full build + launch | `.\scripts\deploy.ps1` |
| C++ build only | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` |
| Python backend | `backend\.venv\Scripts\Activate.ps1` → `uvicorn main:app --host 127.0.0.1 --port 6900 --reload` (from `backend\`) |
| C++ health | `curl http://127.0.0.1:8080/health` → `{inference_engine_loaded: true}` |
| Python health | `curl http://127.0.0.1:6900/health` → `{status, service, port}` |
| Strategy defaults (source of truth) | `curl http://127.0.0.1:6900/api/ml/strategy-defaults` |
| Run sweep | `python scripts\sweep_strategies_v2.py` |
| Backtest smoke test | `curl -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"vol_regime"}'` |
| Synthetic ensemble generation | `python scripts\generate_synthetic_ensemble.py --n-seeds 5 --base-seed 42 --step 100` |

Ports 8080 (C++) and 6900 (Python) are hardcoded in `cpp_core\src\data_fetcher.cpp` (WinHttpConnect to 127.0.0.1:6900) and `frontend\app.js`. The launcher patches app.js in-place; they must move together.

## Architecture

Services form a loop: Python → C++ for inference, C++ → Python for data.

```
frontend → C++ Crow :8080 (native LightGBM)
                 │ WinHTTP → 127.0.0.1:6900 (data)
                 ▼
          Python FastAPI :6900 (CORS=*, all methods)
                 │ yfinance, live inference → C++ :8080 (8s fallback to local LGBM)
                 ▼
          data providers
```

C++ outbound HTTP only in `cpp_core\src\data_fetcher.cpp`. **`PredictRowInput`** in `cpp_core\include\inference_engine.h:27` is the C++↔Python schema — keep in lock-step.

## Module layout

```
backend/
  config/constants.py          Shared paths, LABELING_VERSION, SWEEP_OPTIMAL_PATHS
  config/_strategy_loader.py   Single source of truth for strategy params (get_strategy_params, get_common_params)
  config/strategy_defaults.json  Per-strategy params (whale_quality, contrarian_trend, vol_regime)
  db/schema.py                 init_db(), CREATE TABLE/INDEX, WAL mode
  db/queries.py                get_real_trades, _execute_with_retry, get_dataset_stats
  models/features.py           compute_advanced_features, HV/VIX fetch, feature engineering
  models/tier_a.py             TIER_PROFITABLE_TICKERS, _compute_tier_a_tickers
  models/predict.py            _cpp_batch_predict (8s → C++, fallback local LGBM), prediction cache
  models/train.py              _fit_one_quantile_train, _serialize_cpp_inference_artifacts
  backtest/walkforward.py      _wf_worker_init, _process_walkforward_step
  routers/ml_model.py          All @router endpoints, api_backtest simulation loop (2227 lines)
  routers/ml_derivations.py    Pure functions: P(success), classify_strategy, Kelly — NOT a router
  routers/_yf_safe.py          yfinance timeout+retry wrapper (safe_call) — wrap EVERY yfinance call
  routers/{unusual_options,put_call_ratio,volume_concentration,iv_skew,technicals}.py
  main.py                      FastAPI, CORS, router mounts, tactical_bundle, static frontend
  seed_grounded_real_options.py  Black-Scholes synthetic data generator

frontend/
  js/state.js, utils.js, api.js, scanner.js, ml.js, backtest.js, dashboard.js
  css/base.css, layout.css, components.css, pages.css, responsive.css
  index.html         1071 lines, sidebar: Main Dashboard #tactical, Live Signals #dashboard, Backtester #backtest
  app.js             366 lines, entry point — API_BASE patched by launcher scripts (ephemeral)
  style.css          1370 lines (legacy, unreferenced — delete when safe)

cpp_core/
  src/main.cpp, api_handlers.cpp, inference_engine.cpp, metrics_engine.cpp, data_fetcher.cpp
  include/inference_engine.h (PredictRowInput:27), data_fetcher.h (minVolOI=2.0:60), metrics_engine.h, api_handlers.h
  third_party/       Vendored (Crow, Asio, nlohmann/json, LightGBM, libcurl) — manage via scripts/fetch_vendors.ps1
```

## Critical gotchas

- **Three strategy vocabularies, one per layer.** Per-trade signal: `VOL_EXPANSION`, `SIDEWAYS`, `BULLISH_BREAKOUT`, `BEARISH_BREAKDOWN` in `ml_derivations.classify_strategy()`. Portfolio strategy: `whale_quality`, `contrarian_trend`, `vol_regime` in `strategy_defaults.json`. UI filter labels mirror portfolio names. **Do not assume a name from one layer exists in another.** This is the #1 agent trap.
- **`profit_threshold` has two meanings.** Backtest take-profit cap (per-strategy in `strategy_defaults.json`, range 0.386–0.486) vs ML labeling floor (`ml_settings` DB default, 0.03). Decoupled by design. Do not conflate.
- **Whale thresholds differ by layer.** C++ filter default: `minVolOI=2.0` (`data_fetcher.h:60`). Python router default: `8.0` (`/unusual-options`). Scanner whale label: `>= 5.0` in `get_scanner`. Be explicit.
- **yfinance calls MUST use `_yf_safe.safe_call()`** (12s timeout, 2 retries, jittered exp backoff). All 5 data routers do. Every new yfinance call must too. Without it: 30s curl timeout + Yahoo rate limiting = broken pages.
- **C++ is optional.** Dev mode runs Python-only on :6900 with no C++ build. `deploy.ps1` falls back to `python -m http.server 8080` if C++ build fails.
- **C++→Python HTTP has no timeout** (~120s OS default hang). Reverse hop (Python→C++ inference) has 8s timeout via `_cpp_batch_predict` — falls back to local LGBM.
- **`walkforward_test_increment` controls prediction cache lifetime.** Config default: 500 (`strategy_defaults.json`). Sweep script used 100. Changing it forces full retrain (~30 min cold). Cache key includes all 27 strategy-discriminating params in `models/predict.py`.
- **Ensemble count is in the cache key.** `SELECT COUNT(DISTINCT ensemble_id)` gates prediction cache. Adding/removing ensembles invalidates it (~5-10 min cold).
- **No real dataset.** All ~340k rows across 5 ensembles (ensemble_id ∈ {42,142,242,342,442}) are synthetic (Black-Scholes seeded, `is_synthetic=1`). 14,409 mislabeled "real" rows deleted 2026-07-29. Any backtest assumption of real data is wrong.
- **No tests, no lint, no typecheck, no CI.** No `pytest.ini`, `mypy.ini`, `ruff.toml`, `.eslintrc`, `Makefile`, `.github/`. Do not invent a framework — ask before adding.
- **`docs/` directory removed.** Inline refs to `docs/*.md` are dead links. README link to `docs/DEPLOYMENT.md` is stale.
- **LAUNCH_SCYLLA.ps1 patches `app.js` API_BASE** in-place (swaps :6900↔:8080). On-disk value is ephemeral. If you change defaults, update the patching logic in both launchers.
- **Module boundaries:** DB code → `db/`, model code → `models/`, backtest code → `backtest/`. Not `ml_model.py`. New JS features → new `js/` module, not `app.js`.

## Strategy config source of truth

`backend/config/strategy_defaults.json` is the single source of truth. All consumers reference it:
- Frontend: `loadOptimalParams()` fetches `/api/ml/strategy-defaults`.
- Backend: `_resolve_strategy_defaults(req)` in `ml_model.py` fills None fields from per-strategy or common values.
- HTML form defaults mirror `vol_regime` block (last resort — JS overwrites on load).
- C++: NOT connected — has its own `data_fetcher.h` constants.

Sweep output: `backend/cache/sweep_optimal_v2.json` (gitignore exception, last rebuilt 2026-07-29 on synthetic-only dataset, cost-adjusted P&L). Legacy names `quantile_confidence`, `trend_breakout`, `iv_regime_adaptive` are `_legacy_*` types in `ml_model.py` but not in `strategy_defaults.json` or UI.

## Synthetic data essentials

- Generator: `backend/seed_grounded_real_options.py` (Black-Scholes seeded). Ensembles via `scripts/generate_synthetic_ensemble.py`.
- 5 ensembles (seeds 42, 142, 242, 342, 442). ~9.7yr span, ~340k rows. 10 Greek columns (delta/gamma/vega/theta/rho at entry+exit).
- Bimodal whale distribution: ~4.4% rows vol_oi ≥ 5.0, ~8.8% ≥ 2.0, ~0.8% ≥ 8.0, gated on VIX>22 or |5d return|>4%.
- Validation: `scripts/validate_synthetic_vs_real.py` (KS test, gate D<0.05). Audit: `scripts/audit_synthetic_dataset.py`.

## C++ build notes

Vendored in `cpp_core/third_party/` (Crow, Asio standalone, nlohmann/json, LightGBM, libcurl) — managed by `scripts/fetch_vendors.ps1`, do not hand-edit. MSVC: prefer `"Visual Studio 17 2022"`. `LAUNCH_SCYLLA.ps1` tries VS 18 2026 first then falls back; `deploy.ps1:80` hardcodes VS 17.
