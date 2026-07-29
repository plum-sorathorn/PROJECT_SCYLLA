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
| Ensemble generation | `python scripts\generate_synthetic_ensemble.py --n-seeds 5 --base-seed 42 --step 100` (writes 5 ensemble_id partitions to `options_trades`) |
| Synthetic dataset regeneration runbook | See `backend/seed_grounded_real_options.py` and `scripts/generate_synthetic_ensemble.py` |

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

## Module layout (refactored 2026-07-30)

```
backend/
  config/
    constants.py         ← Shared constants (DB_PATH, CACHE_DIR, MODEL_PATH, etc.) — 27 lines
    _strategy_loader.py  ← Strategy defaults loader
  db/
    schema.py            ← init_db(), CREATE TABLE/INDEX statements — 178 lines
    queries.py           ← get_real_trades, _execute_with_retry, get_dataset_stats — 94 lines
  models/
    features.py          ← compute_advanced_features, HV/VIX fetch, feature engineering — 119 lines
    tier_a.py            ← _compute_tier_a_tickers, TIER_PROFITABLE_TICKERS — 91 lines
    predict.py           ← _cpp_batch_predict, prediction cache helpers — 92 lines
    train.py             ← _fit_one_quantile_train, _serialize_cpp_inference_artifacts — 72 lines
  backtest/
    walkforward.py       ← _wf_worker_init, _process_walkforward_step — 146 lines
  routers/
    ml_model.py          ← All @router endpoints + api_backtest simulation loop — 2227 lines
    unusual_options.py   ← Scanner / unusual options endpoint
    put_call_ratio.py    ← PCR & ticker lookup
    volume_concentration.py ← Volume concentration & top-ticker aggregation
    iv_skew.py           ← IV skew & term structure
    technicals.py        ← Technical indicators
    ml_derivations.py    ← Pure functions: P(success), classify_strategy, Kelly (NOT a router)
    utils.py             ← Shared helpers
    _yf_safe.py          ← yfinance timeout + retry wrapper
  main.py                ← FastAPI, CORS, router mounts, tactical_bundle, static frontend serving
  seed_grounded_real_options.py ← Black-Scholes-seeded synthetic data generator
```

```
frontend/
  css/
    base.css             ← CSS variables, reset, typography, keyframes — 153 lines
    layout.css           ← App shell, sidebar, header, footer — 232 lines
    components.css       ← Buttons, cards, forms, tables, badges, modals — 451 lines
    pages.css            ← View-specific styles (scanner, backtest, dashboard, ML) — 475 lines
    responsive.css       ← Media queries — 73 lines
  js/
    state.js             ← state object + STRATEGY_DEFAULT_PARAMS — 38 lines
    utils.js             ← fmt, $, showView, handleRouting, chartDefaults, etc. — 257 lines
    api.js               ← All fetch/API functions (checkHealth, fetchScanner, etc.) — 482 lines
    scanner.js           ← Scanner view: renderScannerTable, renderPCRChart, etc. — 325 lines
    ml.js                ← ML cockpit: renderLedgerTable, renderModelRunsChart, etc. — 264 lines
    backtest.js          ← Backtest: runBacktestSimulation, renderBacktestCharts, etc. — 590 lines
    dashboard.js         ← Live signals: renderOpenTradesTable, getStrategyParams, etc. — 178 lines
  index.html             ← 1071 lines; links to css/ and js/ sub-modules
  app.js                 ← Entry point (366 lines): API_BASE, setupEventListeners, boot IIFE
  style.css              ← 1370 lines (legacy, no longer referenced — consider deleting)
```

## C++ engine (`cpp_core\`)

- `src\main.cpp` — Crow `SimpleApp` on :8080.
- `src\api_handlers.cpp` — `/health`, `POST /api/v1/ml/predict`, `POST /api/v1/ml/predict-batch` (parallel via `std::async`), proxies for scanner/PCR/volcon/IV, static SPA fallback.
- `src\inference_engine.cpp` — loads 5 LightGBM boosters from `backend\cache\cpp_inference\`, predicts quantiles {0.10, 0.25, 0.50, 0.75, 0.90}, derives strategy. `derive_strategy` rejects contrarian `BULL_ALIGNED+Put` and `BEAR_ALIGNED+Call`.
- `src\metrics_engine.cpp` — multi-threaded only at N ≥ 1000 rows.
- `include\data_fetcher.h:60` — `fetchUnusualOptions(double minVolOI = 2.0)` is the filter default; the whale label is `volOiRatio >= 5.0` at the metrics layer (no constant for it).
- Vendored in `cpp_core\third_party\` (gitignored): Crow, Asio standalone, nlohmann/json, LightGBM, libcurl. Populate via `scripts\fetch_vendors.ps1` — do not hand-edit.
- MSVC generator: prefer `"Visual Studio 17 2022"`. `LAUNCH_SCYLLA.ps1` tries `"Visual Studio 18 2026"` first then falls back; `deploy.ps1:80` hardcodes VS 17. Use VS 17 in any new build script.

## Python backend

### Entry point: `main.py`
FastAPI, `allow_origins=["*"]` + `allow_credentials=True` (local-dev only; browsers reject this combo in real cross-origin). Mounts 6 routers under `/api/v1` and `/api`, serves `frontend\` as static at `/`. **`tactical_bundle`** calls handler functions as plain functions via `_call_with_resolved_defaults(func, **overrides)` — this helper uses `inspect.signature` to walk the handler's declared parameters and resolve any FastAPI `Param` (Query/Path/Body/etc.) default to its underlying value, so handlers can be invoked directly without going through the HTTP boundary. If you add a new endpoint to `tactical_bundle`, pass overrides explicitly; the handler signature is the source of truth for its own defaults.

### Shared config: `config/`
- `constants.py` — 27 lines. Shared constants: `DB_PATH`, `CACHE_DIR`, `MODEL_PATH`, `CPP_INFERENCE_DIR`, etc. Import this when you need a path; do not hardcode.
- `_strategy_loader.py` — `get_strategy_params(strategy_type)`, `get_common_params()`, `load_defaults()`. Single source of truth for strategy params. Imported by `ml_model.py` and the `/api/ml/strategy-defaults` endpoint.

### Database: `db/`
- `schema.py` — 178 lines. `init_db()` with all `CREATE TABLE`/`INDEX` statements. SQLite at `backend\scylla_ml.db` (gitignored), `PRAGMA journal_mode=WAL`.
- `queries.py` — 94 lines. `get_real_trades()`, `_execute_with_retry` (5 retries, exp backoff), `get_dataset_stats()`. All DB access through this module.

### ML models: `models/`
- `features.py` — 119 lines. `compute_advanced_features()`, HV/VIX fetch, all feature engineering. Imported by `ml_model.py` for both live and backtest feature computation.
- `tier_a.py` — 91 lines. `_compute_tier_a_tickers()`, `TIER_PROFITABLE_TICKERS` constant. The ticker universe for ML.
- `predict.py` — 92 lines. **`_cpp_batch_predict()`** is the only live-inference path — 8s timeout to `127.0.0.1:8080/api/v1/ml/predict-batch`, falls back to local LightGBM on failure. Also contains prediction cache helpers.
- `train.py` — 72 lines. `_fit_one_quantile_train()`, `_serialize_cpp_inference_artifacts()`. Model training and C++ artifact export.

### Backtest: `backtest/`
- `walkforward.py` — 146 lines. `_wf_worker_init()`, `_process_walkforward_step()`. Walkforward backtest worker initialization and step processing. Imported by the backtest API endpoint in `ml_model.py`.

### Routers: `routers/`
- `ml_model.py` — 2227 lines (was ~3230 before extraction of db/models/backtest modules). All `@router` endpoints, the `api_backtest` simulation loop, `BacktestRequestSchema`, `_resolve_strategy_defaults`. LightGBM quantile regression, `LABELING_VERSION = "v2_settlement"`, defaults: `horizon_days=10`, `ml_settings.profit_threshold=0.03` (ML label floor — intentionally separate from backtest `profit_threshold`), `prob_threshold=0.55` (live) / 0.40 (backtest). Parallelization uses `ProcessPoolExecutor(max_workers=5)` for CPU, `ThreadPoolExecutor` for I/O. Model pickle at `backend\cache\scylla_predictor.pkl`; C++ boosters at `backend\cache\cpp_inference\`.
- `_yf_safe.py` — yfinance timeout + retry wrapper. `safe_call(fn, *args, timeout=12, retries=2, base_delay=0.8)` runs `fn` in a 1-thread `ThreadPoolExecutor` with a hard `future.result(timeout=...)` cap, jittered exp backoff on retryable exceptions. Default exceptions: `KeyError` (the `'exchangeTimezoneName'` Yahoo stub), `ValueError`, `ConnectionError`, `TimeoutError`, `IndexError`, requests errors. `_staggered_submit` adds 0.05–0.25s jitter before each `executor.submit` to defeat Yahoo burst-detection. **All 5 data routers use this for every yfinance call** — if you add a new yfinance call, wrap it.
- `ml_derivations.py` — pure functions: P(success), `classify_strategy`, Kelly — **not** a router, don't add `include_router`.
- Five data routers: `unusual_options.py`, `put_call_ratio.py`, `volume_concentration.py`, `iv_skew.py`, `technicals.py`. All yfinance calls go through `_yf_safe.py`.

## Three layers, three strategy vocabularies

**Do not assume a strategy name from one layer exists in another.** The single biggest source of confusion.

| Layer | Location | Names | Purpose |
|---|---|---|---|
| Per-trade signal | `ml_derivations.classify_strategy()` (line 27) | `VOL_EXPANSION`, `SIDEWAYS`, `BULLISH_BREAKOUT`, `BEARISH_BREAKDOWN` | What the model thinks this trade IS |
| Portfolio strategy | `config\strategy_defaults.json` → `strategies.<name>` | `whale_quality`, `contrarian_trend`, `vol_regime` | Walk-forward optimized backtest runs (single source of truth) |
| UI filter labels | `frontend\index.html` strategy `<select>` | Same names as portfolio strategy | Frontend dropdown / table filter labels |

The **legacy names** `quantile_confidence`, `trend_breakout`, `iv_regime_adaptive` (from older sweeps) are still present as `_legacy_*` strategy types in `ml_model.py` for A/B comparison, but they are NOT in `strategy_defaults.json` and the frontend does not expose them.

## Strategy config consolidation (the "single source of truth" pattern)

`backend\config\strategy_defaults.json` holds per-strategy params. All consumers reference it:
- **Frontend** (`js/api.js`): `loadOptimalParams()` fetches `/api/ml/strategy-defaults` and builds `state.optimalParams` from it. If `/api/ml/optimal-params` is unavailable (always currently, since `scripts\sweep_optimal.json` doesn't exist), it falls back to `state.strategyDefaults` (the same JSON, served via a different endpoint).
- **Backend** (`ml_model.py`): `_resolve_strategy_defaults(req)` resolves `None` fields in the backtest request to per-strategy or common values.
- **HTML form defaults** (`index.html`): mirror the `vol_regime` block (the default-selected strategy). Last-resort fallback only — JS overwrites them on page load.
- **C++**: NOT connected. The C++ engine has its own `data_fetcher.h` constants and does not read `strategy_defaults.json`.

**Sweep output**: `backend\cache\sweep_optimal_v2.json` is the v2 sweep result (gitignore exception). Re-running the sweep overwrites it. **Last regenerated 2026-07-29 on the synthetic-only dataset with `use_costs=True` (cost-adjusted P&amp;L).**

## Synthetic data flow

- **Legacy generator removed**: `scripts\generate_synthetic_dataset.py` was deleted — it wrote to a separate `scylla_ml_test.db` with an incorrect `is_synthetic=0` flag and was never consumed by the backtest. Use `backend\seed_grounded_real_options.py` for all synthetic data.
- `backend\seed_grounded_real_options.py` — Black-Scholes-seeded "real" options for market-shaped backtests.
- UI toggle `#bt-use-synthetic` (`frontend\index.html`, default checked) is read in `js/backtest.js` and passed as `use_synthetic` query param to the backtest API.
- `is_synthetic INTEGER` on `options_trades` distinguishes synthetic from any future real data at query time (currently all rows are is_synthetic=1).
- **Dataset span (as of HEAD)**: synthetic data = ~9.7 years (2016-10-11 → 2026-06-26, ~340k rows across 5 ensembles) is the ONLY dataset; "real" data was deleted 2026-07-29 after provenance audit revealed 14,409 mislabeled rows were output of deleted older synthetic generators (`generate_synthetic_dataset.py` / `seed_labeled_real_trades.py` / `seed_hybrid_brownian_options.py`) — timestamps at 00:00:00 UTC, ticker list match, BULL_CONTRARIAN trend only produced by deleted scripts. The v2 `sweep_optimal_v2.json` results are calibrated on the current synthetic-only dataset (use_costs=True, cost-adjusted P&L). **Sweep last regenerated 2026-07-29.**
- **Whale density**: synthetic data includes a bimodal vol_oi mixture (~4.41% of rows ≥ 5.0, ~8.78% ≥ 2.0, ~0.79% ≥ 8.0) gated on VIX>22 or |5d return|>4%.
- **Ensemble generation**: `scripts\generate_synthetic_ensemble.py --n-seeds 5 --base-seed 42 --step 100` generates 5 decorrelated synthetic realizations (ensemble_id ∈ {42, 142, 242, 342, 442}). Each ensemble is a fresh `seed_grounded_real_options.py --seed N` run; ensemble_id equals the seed. Walkforward prediction caches are keyed by ensemble count (`SELECT COUNT(DISTINCT ensemble_id) FROM options_trades WHERE is_synthetic=1`) so adding/removing ensembles invalidates the cache. Baseline audit data is across all ensembles combined.
- **"Real" data removal (2026-07-29)**: 14,409 rows previously labeled is_synthetic=0 were deleted after provenance analysis confirmed they were output of deleted older synthetic seeders mislabeling their data as real. The synthetic dataset is now the sole training/backtest source. Side distribution ~58/37/5 (BUY/SELL/MID) and DTE sampling includes 7-day weeklies on high-vol days (VIX>25 OR |5d ret|>6%).
- **Statistical validation**: `scripts\validate_synthetic_vs_real.py` runs two-sample KS / chi-square tests per feature (synthetic vs real). Acceptance gate: KS D < 0.05 OR documented justification.
- Audit & validation: `scripts\audit_synthetic_dataset.py` prints synthetic-vs-real distributional stats.
- **Greeks**: the seeded dataset stores 10 BS Greek columns (delta/gamma/vega/theta/rho at entry AND exit) per trade. All rows are synthetic (is_synthetic=1) with populated Greeks. Per-year theta, raw vega (per 1.00 sigma), raw rho (per 1.00 rate).

## Frontend

### Entry point and shell
- `index.html` — 1071 lines. Three sidebar views: Main Dashboard `#tactical` (default), Live Signals `#dashboard`, Backtester `#backtest`. Backtest strategy `<select>` shows real OOS PnL labels (e.g. "Vol Regime (OOS: +17.2% / 9.7yr, Sharpe 1.57, 61 trades)"). Links to `css/` and `js/` sub-modules via `<link>` and `<script>` tags.
- `app.js` — 366 lines. Entry point: `API_BASE` is **patched in-place by `LAUNCH_SCYLLA.ps1`** and `scripts\start_dev.ps1` to swap 6900↔8080 depending on whether `scylla_core.exe` exists. Contains `setupEventListeners()` and the boot IIFE. Treat the on-disk `API_BASE` value as ephemeral; if you change defaults, update the patch logic too.

### JavaScript modules: `js/`
- `state.js` — 38 lines. `state` object + `STRATEGY_DEFAULT_PARAMS`. All modules import from here; no cross-module mutable state without going through this object.
- `utils.js` — 257 lines. `fmt()` (number/currency/percent formatting), `$()` (DOM shorthand), `showView()`, `handleRouting()`, `chartDefaults()` (Chart.js defaults, legend disabled; colors must read from data series). Brand palette: teal `#005566`, green `#2d7a4a`, rust `#8b3a3a` — saturated, never pale.
- `api.js` — 482 lines. All `fetch`/API functions: `checkHealth()`, `fetchScanner()`, `fetchUnusualOptions()`, `fetchPCR()`, `fetchVolumeConcentration()`, `fetchIVSkew()`, `fetchTechnicals()`, `loadOptimalParams()`, and ML/backtest API wrappers. Every network call lives here.
- `scanner.js` — 325 lines. Scanner view: `renderScannerTable()`, `renderPCRChart()`, `renderVolumeConcentrationChart()`, `renderIVSkewChart()`, view lifecycle.
- `ml.js` — 264 lines. ML cockpit: `renderLedgerTable()`, `renderModelRunsChart()`, feature importance, prediction display.
- `backtest.js` — 590 lines. Backtest: `runBacktestSimulation()`, `renderBacktestCharts()`, walkforward equity curve, trade ledger, drawdown analysis.
- `dashboard.js` — 178 lines. Live signals: `renderOpenTradesTable()`, `getStrategyParams()`, live trade monitoring.

### CSS modules: `css/`
- `base.css` — 153 lines. CSS variables, reset, typography, `@keyframes`.
- `layout.css` — 232 lines. App shell grid, sidebar, header, footer, view switching layout.
- `components.css` — 451 lines. Buttons, cards, forms, tables, badges, modals, tooltips.
- `pages.css` — 475 lines. View-specific styles: scanner tables, backtest charts, dashboard cards, ML cockpit panels.
- `responsive.css` — 73 lines. Media queries for tablet/mobile breakpoints.

### Legacy
- `style.css` — 1370 lines. Legacy monolithic stylesheet, no longer referenced by `index.html`. Consider deleting once all styles are confirmed migrated to `css/`.

## Scripts (`scripts\`)

Active: `deploy.ps1`, `start_dev.ps1`, `fetch_vendors.ps1`, `generate_libcurl_def.ps1`, `verify_phase_a.py`, `sweep_strategies_v2.py` (2-stage walkforward sweep, writes `backend\cache\sweep_optimal_v2.json`), `generate_synthetic_ensemble.py`, `audit_synthetic_dataset.py`, `validate_synthetic_vs_real.py`.

## No-go / fragile files

- `backend\scylla_ml.db` — runtime state, gitignored.
- `backend\cache\scylla_predictor.pkl` (1.1 MB), `backend\cache\cpp_inference\` (5 quantile boosters + preprocessor JSON) — runtime state, gitignored. **The walkforward predictions cache was deleted** — the first backtest after a fresh clone will trigger a full prediction computation (~5-10 min cold, then cached).
- `cpp_core\third_party\` — vendored, managed by `fetch_vendors.ps1`.
- `cpp_core\build\`, `backend\.venv\`, `**\__pycache__\`, `.env` — gitignored build/cache/output.
- `.env` (repo root) — gitignored runtime config; never commit equivalents.
- `backend\config\_strategy_loader.py` — single source of truth loader; do not add field-specific accessors, use `get_strategy_params(strategy_type)`.
- `backend\routers\_yf_safe.py` — yfinance timeout wrapper; every yfinance call in the 5 data routers MUST go through `safe_call`.

## Gotchas

- **No tests, no lint, no typecheck, no CI.** No `pytest.ini`, no `mypy.ini`/`ruff.toml`/`.flake8`, no `.eslintrc`/`.prettierrc`, no `tsconfig.json`, no `Makefile`, no `.github\`. Do not invent a framework mid-task — ask before adding one.
- **C++ exe is optional.** Dev mode runs the full app (Python serves frontend on :6900) without ever compiling C++. `deploy.ps1` auto-falls-back to `python -m http.server 8080` if the C++ build fails.
- **C++ bridge has no HTTP timeouts.** A hung Python backend will hang the C++ request for ~120s (OS default). Not a current bug; just expected. The reverse hop (Python → C++ inference) DOES have an 8s timeout via `_cpp_batch_predict` in `models/predict.py`.
- **yfinance calls have timeouts via `_yf_safe.py`** (12s cap, 2 retries with jittered exp backoff). Without this, the curl 30s timeout + Yahoo rate limiting cascades into 30+ second page loads. If you add a new yfinance call, wrap it in `safe_call`.
- **Whale threshold values differ by layer**: `min_vol_oi=2.0` (C++ filter default, `data_fetcher.h:60`), `8.0` (Python router default for `/unusual-options`), `>= 5.0` (scanner whale label in `get_scanner`). Be explicit when changing any of the three.
- **The `docs/` directory was removed.** Inline references in source comments to `docs/PARALLELIZATION_PLAN.md` / `docs/QUESTIONS_FOR_USER.md` / `docs/SESSION_NOTES.md` point to nothing — treat as historical markers. The README still links to `docs/DEPLOYMENT.md` (stale); ignore the link.
- **Three strategy vocabularies** — see table above. This is the #1 thing agents get wrong.
- **`profit_threshold` has two meanings** that are intentionally decoupled: the backtest take-profit cap (per-strategy in `strategy_defaults.json`, currently 0.418–0.456) and the ML labeling floor (`ml_settings.profit_threshold` DB default, 0.03). **Do not conflate.**
- **`walkforward_test_increment` controls prediction cache lifetime**: changing it invalidates the cache. The current config uses 100 (matches the v2 sweep). Changing to 250 (PHASE A pass default) would force a full retrain (~30 min cold). The cache key is in `models/predict.py` and includes all 27 strategy-discriminating params.
- **No real dataset**: 14,409 "real" rows were deleted 2026-07-29 after provenance analysis confirmed they were synthetic-mislabeled-as-real (deleted older seeders had hardcoded is_synthetic=0). The current synthetic dataset (~340k rows across 5 ensembles / 9.7yr / Greeks / bimodal whales) is the sole training/backtest source. Any backtest specificity to "real" data is illusory.
- **Ensemble cache key**: walkforward prediction cache includes `ensemble_id` count (`SELECT COUNT(DISTINCT ensemble_id) FROM options_trades WHERE is_synthetic=1`). Running a backtest after adding/removing ensembles forces a fresh prediction computation (~5-10 min cold). Per-ensemble backtest filtering at request time is out of scope — the backtest API treats all ensembles as one pool unless explicitly filtered.
- **Module boundaries refactored 2026-07-30**: DB access lives in `db/`, model logic in `models/`, backtest workers in `backtest/`. `ml_model.py` (2227 lines) imports from these modules — add new code in the right module, not in `ml_model.py`. Frontend JS is split across `js/` (7 modules), CSS across `css/` (5 modules). `app.js` is the orchestrator (366 lines); new features should add modules under `js/`, not bloat `app.js`.

## Where to look first

1. `README.md` — architecture + quick start.
2. `LAUNCH_SCYLLA.ps1` — what the user actually runs.
3. `scripts\deploy.ps1` — full build pipeline, dev/prod branching.
4. `cpp_core\src\main.cpp` + `api_handlers.cpp` + `inference_engine.cpp` — C++↔Python contract and native inference.
5. `backend\main.py` + `backend\config\constants.py` + `backend\routers\ml_model.py` — Python entry, shared constants, router mounts, CORS, ML state.
6. `backend\config\strategy_defaults.json` + `backend\config\_strategy_loader.py` — single source of truth for strategy params.
7. `backend\db\schema.py` + `backend\db\queries.py` — DB initialization and query helpers.
8. `backend\models\predict.py` + `backend\models\features.py` — C++ inference bridge and feature engineering.
9. `backend\backtest\walkforward.py` — walkforward backtest step processing.
10. `frontend\js\api.js` + `frontend\js\backtest.js` — frontend API layer and backtest UI.
11. `backend/seed_grounded_real_options.py` + `scripts/generate_synthetic_ensemble.py` — synthetic dataset generation.
12. `graphify query <topic>` — fastest path to relevant code in this repo (use the `graphify` skill).
