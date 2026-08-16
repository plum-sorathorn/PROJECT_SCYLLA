# PROJECT SCYLLA

High-performance options whale scanner: HTML/JS frontend → C++ Crow with native LightGBM (`:8080`) → Python FastAPI/yfinance (`:6900`). Windows-only; no paid API keys.

## Quick reference

| Action | Command / source |
|---|---|
| One-click launch | `.\LAUNCH_SCYLLA.ps1` — patches `frontend/app.js`, builds if needed, opens the browser |
| Dev mode | `.\scripts\start_dev.ps1` — C++ optional |
| Full build and launch | `.\scripts\deploy.ps1` |
| Fetch C++ vendors | `.\scripts\fetch_vendors.ps1` — pulls third-party dependencies required for the C++ build |
| C++ build | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` — launch scripts try VS 18 (2026), then fall back to VS 17 (2022) |
| Python dependencies | `backend\requirements.txt` — canonical dependency source |
| Python backend | Activate `backend\.venv`, then from `backend\`: `uvicorn main:app --host 127.0.0.1 --port 6900 --reload` |
| C++ health | `curl http://127.0.0.1:8080/health` |
| Python health | `curl http://127.0.0.1:6900/health` |
| Strategy defaults | `curl http://127.0.0.1:6900/api/ml/strategy-defaults` |
| Strategy sweep | `python scripts\sweep_strategies_v2.py` |
| Backtest smoke test | `curl -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d '{"mode":"walkforward","strategy_type":"vol_regime"}'` |

Launch scripts automatically free ports `6900` and `8080`. These ports must move together: C++ uses `127.0.0.1:6900` for data, while the launcher patches the frontend API base between `:6900` and `:8080`.

## Setup

- Virtual environment: `backend\.venv`
- Activate it with: `backend\.venv\Scripts\Activate.ps1`
- Install Python dependencies from `backend\requirements.txt`.
- Run `scripts\fetch_vendors.ps1` before the first C++ build.

## Architecture

Python serves data and can call C++ for inference; C++ calls Python for data. C++ outbound HTTP is limited to `cpp_core\src\data_fetcher.cpp`. Keep `PredictRowInput` in `cpp_core\include\inference_engine.h` in lock-step with the Python schema.

## Module layout

```text
backend/
  config/constants.py          Shared paths, LABELING_VERSION, SWEEP_OPTIMAL_PATHS
  config/_strategy_loader.py   Loads strategy parameters; not the source of truth
  config/strategy_defaults.json Source of truth for whale_quality, contrarian_trend, vol_regime
  db/schema.py                 Database initialization and schema
  db/queries.py                Database queries and retry handling
  models/features.py           Feature engineering and HV/VIX retrieval
  models/predict.py            C++ prediction with local LightGBM fallback and cache
  models/train.py              Model training and C++ artifact serialization
  backtest/walkforward.py      Walk-forward workers
  routers/ml_model.py          ML endpoints and backtest simulation
  routers/ml_derivations.py   Pure probability, strategy, and Kelly functions
  routers/_yf_safe.py          yfinance timeout/retry wrapper (`safe_call`)
  routers/{unusual_options,put_call_ratio,volume_concentration,iv_skew,technicals}.py
  main.py                      FastAPI app, router mounts, and static frontend

frontend/
  js/{state,utils,api,scanner,ml,backtest,dashboard}.js
  css/{base,layout,components,pages,responsive}.css
  index.html                   Main dashboard, live signals, and backtester
  app.js                       Entry point; launcher-patched API base
  style.css                    Legacy/Unreferenced - safe to delete

cpp_core/
  src/                         C++ server, inference, metrics, and data fetching
  include/inference_engine.h  C++/Python `PredictRowInput` schema
  include/data_fetcher.h      C++ `minVolOI=2.0` source
  third_party/                 Vendored Crow, Asio, nlohmann/json, LightGBM, libcurl
```

## Data tools

- `scripts/validate_synthetic_vs_real.py` — KS test for dataset validation.
- `scripts/audit_synthetic_dataset.py` — audit of synthetic data integrity.
- `scripts/generate_synthetic_ensemble.py` — generates Black-Scholes ensembles in `backend/cache/`.

## Critical gotchas

- **Strategy vocabularies differ by layer.** Per-trade signals are `VOL_EXPANSION`, `SIDEWAYS`, `BULLISH_BREAKOUT`, and `BEARISH_BREAKDOWN`; portfolio strategies are `whale_quality`, `contrarian_trend`, and `vol_regime`. Do not mix them.
- **`profit_threshold` has two meanings.** It is the per-strategy backtest take-profit cap in `strategy_defaults.json` (about `0.386–0.486`) and the ML labeling floor in `ml_settings` (default `0.03`). They are intentionally decoupled.
- **Whale thresholds differ by layer.** C++ `minVolOI=2.0` (`cpp_core/include/data_fetcher.h`), Python `/unusual-options` default `8.0`, and scanner whale labeling `>=5.0`.
- **Every yfinance call must use `_yf_safe.safe_call()`** for timeout, retry, and backoff behavior.
