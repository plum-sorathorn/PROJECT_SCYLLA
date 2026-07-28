# AGENTS.md — PROJECT: SCYLLA // TERMINAL

High-performance options whale scanner. Hybrid: cyberpunk HTML/JS frontend → C++ Crow (`:8080`, native LightGBM inference) → Python FastAPI / yfinance (`:6900`). Windows-only. No paid API keys.

## Quick reference

| Action | Command |
|---|---|
| One-click launch (user) | `.\LAUNCH_SCYLLA.ps1` — patches `app.js` API_BASE, auto-builds C++ if missing, opens browser |
| Dev mode (Python only, no C++ build) | `.\scripts\start_dev.ps1` |
| Full production build + launch | `.\scripts\deploy.ps1` |
| C++ build only | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` |
| Python backend | `backend\.venv\Scripts\Activate.ps1` → `uvicorn main:app --host 127.0.0.1 --port 6900 --reload` (from `backend\`) |
| C++ health | `curl http://127.0.0.1:8080/health` → `{inference_engine_loaded: true}` |
| Python health | `curl http://127.0.0.1:6900/health` → `{status, service, port}` |
| Strategy config (single source of truth) | `curl http://127.0.0.1:6900/api/ml/strategy-defaults` |
| Run sweep | `python scripts\sweep_strategies_v2.py` (writes `backend\cache\sweep_optimal_v2.json`) |
| Backtest smoke test | `curl -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"vol_regime","use_synthetic":false}'` |

Ports **8080** (C++) and **6900** (Python) are hardcoded in `cpp_core\src\data_fetcher.cpp:4` and `frontend\app.js` (the launcher patches `app.js` in place). They must move together.

## Architecture

```
frontend/index.html ─HTTP─> C++ Crow :8080
                          │ WinHTTP → 127.0.0.1:6900 (data)
                          │ Native LightGBM on /api/v1/ml/predict* (no Python hop)
                          ▼
                   Python FastAPI :6900 (CORS=*, all methods)
                          │ yfinance / openbb==4.3.2 / CBOE
                          │ Live inference → C++ :8080 via _cpp_batch_predict (8s timeout, falls back to local LGBM)
                          ▼
                   data providers
```

The services form a loop: Python → C++ for inference, C++ → Python for data. C++ outbound HTTP lives only in `cpp_core\src\data_fetcher.cpp`. **`PredictRowInput`** in `cpp_core\include\inference_engine.h:27` is the C++↔Python schema — keep in lock-step with the Python pass-through.

## C++ engine (`cpp_core\`)

- `src\main.cpp` — Crow `SimpleApp` on :8080.
- `src\api_handlers.cpp` — `/health`, `POST /api/v1/ml/predict`, `POST /api/v1/ml/predict-batch` (parallel via `std::async`), proxies for scanner/PCR/volcon/IV, static SPA fallback.
- `src\inference_engine.cpp` — loads 5 LightGBM boosters from `backend\cache\cpp_inference\`, predicts quantiles {0.10, 0.25, 0.50, 0.75, 0.90}, derives strategy. `derive_strategy` rejects contrarian `BULL_ALIGNED+Put` and `BEAR_ALIGNED+Call`.
- `src\metrics_engine.cpp` — multi-threaded only at N ≥ 1000 rows.
- `include\data_fetcher.h:60` — `fetchUnusualOptions(double minVolOI = 2.0)` is the filter default; the whale label is `volOiRatio >= 5.0` at the metrics layer (no constant for it).
- Vendored in `cpp_core\third_party\` (gitignored): Crow, Asio standalone, nlohmann/json, LightGBM, libcurl. Populate via `scripts\fetch_vendors.ps1` — do not hand-edit.
- MSVC generator: prefer `"Visual Studio 17 2022"`. `LAUNCH_SCYLLA.ps1` tries `"Visual Studio 18 2026"` first then falls back; `deploy.ps1:80` hardcodes VS 17. Use VS 17 in any new build script.

## Python backend (`backend\`)

- `main.py` — FastAPI, `allow_origins=["*"]` + `allow_credentials=True` (local-dev only; browsers reject this combo in real cross-origin). Mounts 6 routers under `/api/v1` and `/api`, serves `frontend\` as static at `/`. **`tactical_bundle` (L47-69)** calls handler functions as plain functions via `_call_with_resolved_defaults(func, **overrides)` — this helper uses `inspect.signature` to walk the handler's declared parameters and resolve any FastAPI `Param` (Query/Path/Body/etc.) default to its underlying value, so handlers can be invoked directly without going through the HTTP boundary. If you add a new endpoint to `tactical_bundle`, pass overrides explicitly; the handler signature is the source of truth for its own defaults.
- `routers\` — 6 real routers: `unusual_options.py`, `put_call_ratio.py`, `volume_concentration.py`, `iv_skew.py`, `technicals.py`, `ml_model.py`. Plus `ml_derivations.py` (pure functions: P(success), `classify_strategy`, Kelly — **not** a router, don't add `include_router`) and `utils.py`.
- `routers\_yf_safe.py` — yfinance timeout + retry wrapper. `safe_call(fn, *args, timeout=12, retries=2, base_delay=0.8)` runs `fn` in a 1-thread `ThreadPoolExecutor` with a hard `future.result(timeout=...)` cap, jittered exp backoff on retryable exceptions. Default exceptions: `KeyError` (the `'exchangeTimezoneName'` Yahoo stub), `ValueError`, `ConnectionError`, `TimeoutError`, `IndexError`, requests errors. `_staggered_submit` adds 0.05–0.25s jitter before each `executor.submit` to defeat Yahoo burst-detection. **All 5 data routers use this for every yfinance call** — if you add a new yfinance call, wrap it.
- `routers\ml_model.py` is the only stateful piece. SQLite at `backend\scylla_ml.db` (gitignored, with `.bak` next to it), `PRAGMA journal_mode=WAL` at line 227. Model pickle at `backend\cache\scylla_predictor.pkl`; C++ boosters at `backend\cache\cpp_inference\`. LightGBM quantile regression, `LABELING_VERSION = "v2_settlement"`, defaults: `horizon_days=10`, `ml_settings.profit_threshold=0.03` (ML label floor — intentionally separate from backtest `profit_threshold`), `prob_threshold=0.55` (live) / 0.40 (backtest). `_execute_with_retry` at line 704 (5 retries, exp backoff). Parallelization uses `ProcessPoolExecutor(max_workers=5)` for CPU, `ThreadPoolExecutor` for I/O.
- **`_cpp_batch_predict()` (line 146)** is the only live-inference path — 8s timeout to `127.0.0.1:8080/api/v1/ml/predict-batch`, falls back to local LightGBM on failure.
- **`BacktestRequestSchema` (line 1989)** — per-strategy fields default to `None`. **`_resolve_strategy_defaults(req)` (line 1950)** resolves them from `config\strategy_defaults.json` at request-handling time. Explicit request values still win. The schema-level `Optional[float] = None` is required so the resolver can detect "user didn't override" vs "user set to a real value".
- `config\_strategy_loader.py` — `get_strategy_params(strategy_type)`, `get_common_params()`, `load_defaults()`. Single source of truth for strategy params. Imported by `ml_model.py` and the `/api/ml/strategy-defaults` endpoint (line ~2945).

## Three layers, three strategy vocabularies

**Do not assume a strategy name from one layer exists in another.** The single biggest source of confusion.

| Layer | Location | Names | Purpose |
|---|---|---|---|
| Per-trade signal | `ml_derivations.classify_strategy()` (line 27) | `VOL_EXPANSION`, `SIDEWAYS`, `BULLISH_BREAKOUT`, `BEARISH_BREAKDOWN` | What the model thinks this trade IS |
| Portfolio strategy | `config\strategy_defaults.json` → `strategies.<name>` | `whale_quality`, `contrarian_trend`, `vol_regime` | Walk-forward optimized backtest runs (single source of truth) |
| UI filter labels | `frontend\index.html` strategy `<select>` (~L544) | Same names as portfolio strategy | Frontend dropdown / table filter labels |

The **legacy names** `quantile_confidence`, `trend_breakout`, `iv_regime_adaptive` (from older sweeps) are still present as `_legacy_*` strategy types in `ml_model.py` (L2692-2711) for A/B comparison, but they are NOT in `strategy_defaults.json` and the frontend does not expose them.

## Strategy config consolidation (the "single source of truth" pattern)

`backend\config\strategy_defaults.json` holds per-strategy params. All consumers reference it:
- **Frontend** (`app.js:1706`): `loadOptimalParams()` fetches `/api/ml/strategy-defaults` and builds `state.optimalParams` from it. If `/api/ml/optimal-params` is unavailable (always currently, since `scripts\sweep_optimal.json` doesn't exist), it falls back to `state.strategyDefaults` (the same JSON, served via a different endpoint).
- **Backend** (`ml_model.py:1950`): `_resolve_strategy_defaults(req)` resolves `None` fields in the backtest request to per-strategy or common values.
- **HTML form defaults** (`index.html:557-625`): mirror the `vol_regime` block (the default-selected strategy). Last-resort fallback only — JS overwrites them on page load.
- **C++**: NOT connected. The C++ engine has its own `data_fetcher.h` constants and does not read `strategy_defaults.json`.

**Sweep output**: `backend\cache\sweep_optimal_v2.json` is the v2 sweep result (gitignore exception on line 120). Re-running the sweep overwrites it.

## Synthetic data flow

- `scripts\generate_synthetic_dataset.py` — Black-Scholes synthetic options (manual data-seeding tool, not called by any other script).
- `backend\seed_grounded_real_options.py` — Black-Scholes-seeded "real" options for market-shaped backtests.
- UI toggle `#bt-use-synthetic` (`frontend\index.html:720`, default checked) is read in `app.js` and passed as `use_synthetic` query param to the backtest API.
- `is_synthetic INTEGER` on `options_trades` distinguishes the two at query time.
- **Dataset span (as of HEAD)**: real data = 5.78 years (2020-10-15 → 2026-07-28, ~46k rows); synthetic data = 9.71 years (2016-10-10 → 2026-06-25, ~69k rows). The v2 sweep ran on real data.

## Frontend (`frontend\`)

- `index.html` — three sidebar views: Main Dashboard `#tactical` (default), Live Signals `#dashboard`, Backtester `#backtest`. Backtest strategy `<select>` (~L544) shows real OOS PnL labels (e.g. "Vol Regime (OOS: +17.1% / 5.8yr, Sharpe 1.16, 213 trades)"). **The old +2,042% labels were fake and have been replaced** with values from `sweep_optimal_v2.json`.
- `app.js` — `API_BASE` is **patched in-place by `LAUNCH_SCYLLA.ps1`** and `scripts\start_dev.ps1` to swap 6900↔8080 depending on whether `scylla_core.exe` exists. Treat the on-disk value as ephemeral; if you change defaults, update the patch logic too. **`FALLBACK_OPT_PARAMS` was deleted** — form defaults come from `state.optimalParams` (built from `/api/ml/strategy-defaults`).
- `style.css` — chart legend is disabled in `chartDefaults()`; colors must read from data series. Brand palette: teal `#005566`, green `#2d7a4a`, rust `#8b3a3a` — saturated, never pale.

## Scripts (`scripts\`)

Active: `deploy.ps1`, `start_dev.ps1`, `fetch_vendors.ps1`, `generate_libcurl_def.ps1`, `verify_phase_a.py`, `sweep_strategies_v2.py` (2-stage walkforward sweep, writes `backend\cache\sweep_optimal_v2.json`). Manual seeder: `generate_synthetic_dataset.py`. **Do not recreate** the old sweep/backtest/profile scripts (`sweep_volume.py`, `sweep_oos.py`, `sweep_quick.py`, `sweep_strategies.py`, `backtest_strategies.py`, `profile_filters.py`, `verify_synthetic_dataset.py`, `prime_default_backtest_caches.py`, `verify_inference_equivalence.py`, `benchmark_inference.py`) — they are gone.

## No-go / fragile files

- `frontend\app.js` — rewritten by launchers (API_BASE patched in place).
- `backend\scylla_ml.db`, `backend\scylla_ml.db.bak` — runtime state, gitignored.
- `backend\cache\scylla_predictor.pkl` (1.1 MB), `backend\cache\cpp_inference\` (5 quantile boosters + preprocessor JSON) — runtime state, gitignored. **The walkforward predictions cache was deleted** — the first backtest after a fresh clone will trigger a full prediction computation (~5-10 min cold, then cached).
- `cpp_core\third_party\` — vendored, managed by `fetch_vendors.ps1`.
- `cpp_core\build\`, `backend\.venv\`, `**\__pycache__\`, `graphify-out\`, `.env`, `scylla_odp_launch.bat`, `scylla_cpp_launch.bat` — gitignored build/cache/output.
- `.env` (repo root) — gitignored runtime config; never commit equivalents.
- `backend\config\_strategy_loader.py` — single source of truth loader; do not add field-specific accessors, use `get_strategy_params(strategy_type)`.
- `backend\routers\_yf_safe.py` — yfinance timeout wrapper; every yfinance call in the 5 data routers MUST go through `safe_call`.

## Gotchas

- **No tests, no lint, no typecheck, no CI.** No `pytest.ini`, no `mypy.ini`/`ruff.toml`/`.flake8`, no `.eslintrc`/`.prettierrc`, no `tsconfig.json`, no `Makefile`, no `.github\`. Do not invent a framework mid-task — ask before adding one.
- **C++ exe is optional.** Dev mode runs the full app (Python serves frontend on :6900) without ever compiling C++. `deploy.ps1` auto-falls-back to `python -m http.server 8080` if the C++ build fails.
- **C++ bridge has no HTTP timeouts.** A hung Python backend will hang the C++ request for ~120s (OS default). Not a current bug; just expected. The reverse hop (Python → C++ inference) DOES have an 8s timeout via `_cpp_batch_predict`.
- **yfinance calls have timeouts via `_yf_safe.py`** (12s cap, 2 retries with jittered exp backoff). Without this, the curl 30s timeout + Yahoo rate limiting cascades into 30+ second page loads. If you add a new yfinance call, wrap it in `safe_call`.
- **Whale threshold values differ by layer**: `min_vol_oi=2.0` (C++ filter default, `data_fetcher.h:60`), `8.0` (Python router default for `/unusual-options`), `>= 5.0` (scanner whale label in `get_scanner`). Be explicit when changing any of the three.
- **The `docs/` directory was removed.** Inline references in source comments to `docs/PARALLELIZATION_PLAN.md` / `docs/QUESTIONS_FOR_USER.md` / `docs/SESSION_NOTES.md` point to nothing — treat as historical markers. The README still links to `docs/DEPLOYMENT.md` (stale); ignore the link.
- **Three strategy vocabularies** — see table above. This is the #1 thing agents get wrong.
- **`profit_threshold` has two meanings** that are intentionally decoupled: the backtest take-profit cap (per-strategy in `strategy_defaults.json`, currently 0.418–0.456) and the ML labeling floor (`ml_settings.profit_threshold` DB default, 0.03). See comment at `ml_model.py:2163-2166`. **Do not conflate.**
- **`walkforward_test_increment` controls prediction cache lifetime**: changing it invalidates the cache. The current config uses 100 (matches the v2 sweep). Changing to 250 (PHASE A pass default) would force a full retrain (~30 min cold). The cache key is at `ml_model.py:2150-2157` and includes all 27 strategy-discriminating params.
- **Real data is shorter than you think**: 5.78 years (2020-10 → 2026-07), not 9+ years. The synthetic dataset is 9.71 years but has `vol_oi` capped at p99≈0.92 (no real whale signals). Strategies calibrated on real data may under-trade on synthetic.

## Where to look first

1. `README.md` — architecture + quick start.
2. `LAUNCH_SCYLLA.ps1` — what the user actually runs.
3. `scripts\deploy.ps1` — full build pipeline, dev/prod branching.
4. `cpp_core\src\main.cpp` + `api_handlers.cpp` + `inference_engine.cpp` — C++↔Python contract and native inference.
5. `backend\main.py` + `routers\ml_model.py` — Python entry, router mounts, CORS, ML state, the `_resolve_strategy_defaults` pattern.
6. `backend\config\strategy_defaults.json` + `routers\_strategy_loader.py` — single source of truth for strategy params.
7. `graphify query <topic>` — fastest path to relevant code in this repo (see `.opencode\rules\graphify.md`).
