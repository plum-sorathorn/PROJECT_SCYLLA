# AGENTS.md — PROJECT: SCYLLA // TERMINAL

High-performance options whale scanner. Hybrid 3-tier: cyberpunk HTML/JS frontend → C++ Crow (`scylla_core.exe`, :8080, now with native LightGBM inference) → Python FastAPI / yfinance + OpenBB ODP (uvicorn, :6900) → data providers. Windows-only. No paid API keys. Snapshot: HEAD `df54999` on `revamp_strats_and_optimize_via_inference_engine` (2026-07-27).

## Quick reference

| Action | Command |
|---|---|
| One-click launch (user) | `.\LAUNCH_SCYLLA.ps1` (or `.bat`) — patches `app.js` API_BASE, auto-builds C++ if missing, opens browser |
| Dev mode (Python only, no C++ build) | `.\scripts\start_dev.ps1` |
| Full production build + launch | `.\scripts\deploy.ps1` |
| C++ build only | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` |
| Python venv activate | `backend\.venv\Scripts\Activate.ps1` |
| Run Python backend | `uvicorn main:app --host 127.0.0.1 --port 6900 --reload` (from `backend\`) |
| Health check | `curl http://127.0.0.1:8080/health` (returns `{inference_engine_loaded: bool}`) and `:6900/health` |

Ports: C++ **8080**, Python **6900**. Hardcoded in `cpp_core\src\data_fetcher.cpp:4` and in `frontend\app.js` (patched by launcher). If you ever bind to a real interface, both must move together.

## Architecture (data flow)

```
frontend/index.html ─HTTP─> C++ Crow :8080
                                │ WinHTTP (no timeouts, no retry) → 127.0.0.1:6900
                                │ Native LightGBM inference (no Python) on /api/v1/ml/predict*
                                ▼
                          Python FastAPI :6900 (CORS=*, all methods/headers)
                                │ yfinance / openbb==4.3.2 / CBOE
                                │ Live inference DELEGATES back to C++ via _cpp_batch_predict (8s timeout)
                                ▼
                          data providers
```

The C++ ↔ Python bridge is `cpp_core\src\data_fetcher.cpp`. C++ outbound HTTP is **only** here; all other C++ work is local (LightGBM, threading, JSON).

**The two services form a loop**: Python calls C++ for live ML inference (port 8080); C++ calls Python for raw market data (port 6900). The C++ engine is the canonical path for both single (`/api/v1/ml/predict`) and batch (`/api/v1/ml/predict-batch`) inference — `_cpp_batch_predict` in `backend\routers\ml_model.py:129` is the only place Python performs live inference, and it falls back to local LightGBM on failure.

## C++ engine (`cpp_core\`)

- `src\main.cpp` — Crow `SimpleApp` on :8080, calls `registerRoutes(app)`.
- `src\api_handlers.cpp` — Routes:
  - `GET /health` (line 128) — JSON status + `inference_engine_loaded`
  - `POST /api/v1/ml/predict` (line 144) — single-row native inference
  - `POST /api/v1/ml/predict-batch` (line 189) — batch inference, parallel HV fetch via `std::async`
  - `GET /api/{scanner,put-call-ratio,volume-concentration,iv-skew}` — proxy to Python
  - `GET /` and `GET /<path>` — static SPA fallback (resolves `frontend/`, `../frontend/`, `frontend/dist/`)
- `src\data_fetcher.cpp` — WinHTTP, hardcoded `127.0.0.1:6900`, no timeouts, no retry; failure throws `std::runtime_error`.
- `src\inference_engine.cpp` + `include\inference_engine.h` — Loads 5 LightGBM boosters from `backend/cache/cpp_inference/`, predicts quantiles {0.10, 0.25, 0.50, 0.75, 0.90}, derives strategy. **`PredictRowInput` (header line 27)** is the schema — must stay in lock-step with Python's pass-through.
- `src\metrics_engine.cpp` — Multi-threaded when N ≥ 1000 rows (`std::thread::hardware_concurrency()`); single-threaded below that to avoid thread overhead.
- `include\data_fetcher.h` — `fetchUnusualOptions(double minVolOI = 2.0)` at line 60 is the **filter** default. Whale signal is derived at the metrics layer via `volOiRatio >= 5.0` (no constant for it — be explicit when changing).
- `CMakeLists.txt` links: `Threads::Threads`, `ws2_32`, `wsock32`, `lightgbm`, `curl`. **`winhttp.lib` is via `#pragma comment`** in `data_fetcher.cpp:15`, not CMakeLists — works because MSVC handles pragmas.
- Vendored headers in `cpp_core\third_party\` (gitignored): **Crow, Asio standalone, nlohmann/json, LightGBM, libcurl** (plus headers needed by `generate_libcurl_def.ps1`). Populated by `scripts\fetch_vendors.ps1` — do not hand-edit.

## Python backend (`backend\`)

- `main.py` — FastAPI app, `allow_origins=["*"]` + `allow_credentials=True` (local-dev only), `GET /health`, mounts 6 routers under both `/api/v1` and `/api` (compat for direct dev mode when C++ is absent), serves `frontend\` as static at `/` (resolves `../frontend` relative to `main.py`).
- `routers\` — **6 real routers** (none are stubs):
  - `unusual_options.py`, `put_call_ratio.py`, `volume_concentration.py`, `iv_skew.py`, `technicals.py`, `ml_model.py`
  - `ml_derivations.py` is **not** a router — it's pure functions (P(success), strategy classify, Kelly) imported by `ml_model.py`. Don't add `include_router` for it.
- `routers\ml_model.py` — the only stateful piece (~3000 lines):
  - SQLite: `backend\scylla_ml.db` (gitignored, with `.bak` next to it), **`PRAGMA journal_mode=WAL`** at line 209. Schema at line 218: `options_trades(... option_type TEXT NOT NULL, side TEXT NOT NULL, is_synthetic INTEGER ...)`.
  - Model pickle: `backend\cache\scylla_predictor.pkl` (gitignored). C++ boosters at `backend\cache\cpp_inference\`.
  - LightGBM quantile regression, `LABELING_VERSION = "v2_settlement"`, defaults: `horizon_days=10`, `profit_threshold=0.03`, `prob_threshold=0.55` (live) / 0.40 (backtest).
  - Retry: `_execute_with_retry` at line 670 — `max_retries=5`, `initial_delay=0.1`, exponential backoff.
  - **Parallelization** (all use `ProcessPoolExecutor(max_workers=5)` for CPU, `ThreadPoolExecutor` for I/O):
    - Training (line 1452): 5 workers × `n_jobs=2` LightGBM = ~10 cores
    - Walk-forward backtest (line 2029): `min(4, os.cpu_count())` outer workers
    - Labeling (line 1252): `min(20, ...)` thread workers for concurrent network I/O
    - HV fetch (line 770): `max(4, ...)` thread workers
  - **`_cpp_batch_predict()` (line 129)** is the only live-inference path — hits `127.0.0.1:8080/api/v1/ml/predict-batch` with 8s timeout, falls back to local LightGBM on failure.
- `routers\ml_derivations.py` — `classify_strategy()` returns `VOL_EXPANSION` / `SIDEWAYS` / `BULLISH_BREAKOUT` / `BEARISH_BREAKDOWN`. P(success) clipped to `[0.02, 0.98]`. Kelly cap `0.25`.
- `config\strategy_defaults.json` — Walk-forward optimized portfolio strategies: `quantile_confidence`, `trend_breakout`, `iv_regime_adaptive` (each with `prob_threshold`, `kelly_cap`, `hard_stop`, `profit_target`, `sharpe`).
- `seed_grounded_real_options.py` — Black-Scholes-based seeder for "grounded" real-options data; complement to the synthetic generator.

## Three layers, three strategy vocabularies

This is the single biggest source of confusion in the codebase. **Do not assume a strategy name from one layer exists in another.**

| Layer | Location | Names | Purpose |
|---|---|---|---|
| Per-trade signal | `ml_derivations.classify_strategy()` | `VOL_EXPANSION`, `SIDEWAYS`, `BULLISH_BREAKOUT`, `BEARISH_BREAKDOWN` | What the model thinks this trade IS |
| Portfolio strategy | `config\strategy_defaults.json` | `quantile_confidence`, `trend_breakout`, `iv_regime_adaptive` | Walk-forward optimized backtest runs with P/Kelly/threshold |
| UI filter labels | `frontend\app.js STRATEGY_DEFAULT_PARAMS` (~L1310) | `whale_quality`, `contrarian_trend`, `vol_regime` | Frontend dropdown / table filter labels |

C++ `derive_strategy` in `inference_engine.cpp:278-279` adds a fourth dimension: **contrarian rejection** of `BULL_ALIGNED+Put` and `BEAR_ALIGNED+Call` rows.

## Synthetic data flow

- `scripts\generate_synthetic_dataset.py` — Black-Scholes synthetic options trades (realistic pricing, RNG-seeded).
- `backend\seed_grounded_real_options.py` — Black-Scholes-seeded "real" options for backtests that should look market-shaped.
- **UI toggle** `#bt-use-synthetic` (`frontend\index.html:714`, default checked) is read in `app.js:2044, 2176, 2238` and passed as `use_synthetic` query param to the backtest API.
- DB column `is_synthetic INTEGER` on `options_trades` distinguishes the two at query time.

## Frontend (`frontend\`)

- `index.html` — Three sidebar views (line 64-90):
  1. **Main Dashboard** (`#tactical`, line 314) — default; scanner, PCR, volcon, IV, expected move
  2. **Live Signals** (`#dashboard`, line 159) — open trades, ledger, model runs
  3. **Backtester** (`#backtest`, line 512) — strategy picker, sweep heatmap, backtest ledger
- `app.js` — `API_BASE` (line 8-10) is **patched in-place by `LAUNCH_SCYLLA.ps1`** (lines 91-99) to swap 6900↔8080 depending on whether `scylla_core.exe` exists. If you change `API_BASE` defaults, also update the patch logic in `LAUNCH_SCYLLA.ps1` and `scripts\start_dev.ps1`. Treat the on-disk value as ephemeral.
- `style.css` — Key class families: `.sidebar`, `.nav-item`, `.filter-select`, `.synthetic-toggle` / `.synthetic-toggle-pill` / `.synthetic-toggle-dot`, `.brand-text`, `.accent-*`. Brand palette: teal `#005566`, green `#2d7a4a`, rust `#8b3a3a`.
- Chart legend is disabled in `chartDefaults()` (line 735) — colors must read from the data series themselves. Saturated brand colors, never pale washes.

## Scripts (`scripts\`)

Five scripts, all in active use: `deploy.ps1`, `start_dev.ps1`, `fetch_vendors.ps1`, `generate_libcurl_def.ps1`, `verify_phase_a.py`. `generate_synthetic_dataset.py` is a manual data-seeding tool (run when you need fresh synth data; not called by any other script). The old `sweep_volume.py` / `sweep_oos.py` / `sweep_quick.py` / `sweep_strategies.py` / `backtest_strategies.py` / `profile_filters.py` / `verify_synthetic_dataset.py` / `prime_default_backtest_caches.py` / `verify_inference_equivalence.py` / `benchmark_inference.py` are gone — do not recreate them.

## No-go / fragile files (do not edit casually)

- `frontend\app.js` — rewritten by launchers. If you change `API_BASE` defaults, also update the patch logic in `LAUNCH_SCYLLA.ps1` and `scripts\start_dev.ps1`.
- `backend\scylla_ml.db`, `backend\scylla_ml.db.bak`, `backend\cache\*.pkl`, `backend\cache\cpp_inference\*` — runtime state, gitignored.
- `cpp_core\third_party\` — vendored, managed by `fetch_vendors.ps1`.
- `cpp_core\build\`, `backend\.venv\`, `**\__pycache__\`, `graphify-out\`, `.env` — all gitignored build/cache/output.
- `.env` (repo root) — gitignored runtime config; never commit equivalents. Holds port/data-provider settings.
- `scylla_odp_launch.bat`, `scylla_cpp_launch.bat` — temp files written by the launcher, gitignored.

## Gotchas

- **No tests, no lint, no typecheck, no CI.** No `pytest.ini`, no `mypy.ini`/`ruff.toml`/`.flake8`, no `.eslintrc`/`.prettierrc`, no `tsconfig.json`, no `Makefile`, no `.github\`. Do not invent a framework mid-task — ask before adding one.
- **C++ exe is optional.** Dev mode runs the full app (Python serves frontend on :6900) without ever compiling C++. `deploy.ps1` auto-falls-back to `python -m http.server 8080` if the C++ build fails.
- **C++ bridge has no HTTP timeouts.** A hung Python backend will hang the C++ request for ~120s (OS default). Not a current bug; just expected. Live inference from Python to C++ DOES have an 8s timeout in `_cpp_batch_predict`.
- **Whale threshold values differ by layer**: `min_vol_oi=2.0` (C++ filter default, `data_fetcher.h:60`), `8.0` (Python router default), `≥ 5.0` (scanner logic for the whale label). Be explicit when changing any of the three.
- **Hardcoded `127.0.0.1`** in `data_fetcher.cpp` and `app.js`. Both must move together if you ever bind to a real interface.
- **Backend `allow_origins=["*"]` + `allow_credentials=True`** — local-dev only. Browsers reject this combo in real cross-origin; do not assume it's safe in any future deploy.
- **Strategy names** are not unified across layers — see "Three layers, three strategy vocabularies" above.
- **The `docs/` directory was removed** in `df54999` — `PARALLELIZATION_PLAN.md`, `QUESTIONS_FOR_USER.md`, `SESSION_NOTES.md` no longer exist. Inline references like `PARALLELIZATION_PLAN §4.1` in source comments point to nothing; treat them as historical markers.
- **MSVC generator discrepancy**: `deploy.ps1` (line 80) hardcodes `"Visual Studio 17 2022"`; `LAUNCH_SCYLLA.ps1` (line 109-113) tries `"Visual Studio 18 2026"` first, then falls back to `"Visual Studio 17 2022"`. Use VS 17 2022 in any new build script.

## OpenCode / agent conventions (project-specific)

- The repo uses a **graphify** knowledge graph (`graphify-out\`) as the primary navigation layer. Rules in `.opencode\rules\graphify.md` and `.agents\rules\graphify.md`:
  - **Query graphify before raw `grep`/`ls`/`read`** on unfamiliar code. Use `graphify query`, `graphify path`, `graphify explain`.
  - `graphify` CLI is on PATH — call it directly, never `npx`/`npm`/`pip`.
  - Run `graphify update .` after edits to keep the graph in sync.
  - For broad overviews, read `graphify-out\GRAPH_REPORT.md`; for nav, check `graphify-out\wiki\index.md` if present.
- Subagent routing (matches global rules): read-only → `@fast`, implementation → `@medium`, architecture/debug-after-2-failures → `@heavy`. Self-cap: ≤2 direct read-only tool calls per turn; dispatch `@fast` on the 3rd need.

## Where to look first when something is unclear

1. `README.md` — architecture + quick start (short, current).
2. `LAUNCH_SCYLLA.ps1` — what the user actually runs.
3. `scripts\deploy.ps1` — full build pipeline, dev/prod branching.
4. `cpp_core\src\main.cpp` + `api_handlers.cpp` + `inference_engine.cpp` — the C++↔Python contract and the native inference engine.
5. `backend\main.py` + `routers\ml_model.py:1-260` — Python entry, router mounts, CORS, ML state.
6. `graphify query <topic>` — fastest path to relevant code in a large repo.
