# PROJECT SCYLLA — Agent & Developer Blueprint

High-performance options whale scanner and quantitative analytics engine combining an asynchronous Python FastAPI data pipeline (`:6900`), a native C++ Crow microservice with embedded LightGBM inference (`:8080`), and a modular real-time frontend.

---

## 1. System Architecture & Topology

```
┌─────────────────────────────────────────────────────────────┐
│                 Frontend (Vanilla JS / ES6+)                │
│             http://127.0.0.1:8080 or :6900 (Dev)            │
└──────────────┬───────────────────────────────▲──────────────┘
               │ HTTP / JSON API               │
               ▼                               │
┌──────────────────────────────┐               │
│      C++ Crow Core Engine    │               │
│     (cpp_core @ Port 8080)   │               │
├──────────────────────────────┤               │
│ • Native LightGBM Inference  │               │
│ • Real-time Metrics Engine   │               │
│ • Outbound Data Fetcher      │               │
└──────────────┬───────────────┘               │
               │ HTTP WinHTTP / libcurl        │
               ▼                               │
┌──────────────────────────────────────────────┴──────────────┐
│                  Python FastAPI Data Engine                 │
│                     (backend @ Port 6900)                   │
├─────────────────────────────────────────────────────────────┤
│ • Resilient Options Chain Scraping (_yf_safe retry/backoff) │
│ • Feature Engineering (HV, VIX, Skew, Vol/OI, Technicals)   │
│ • Walk-Forward Strategy Simulation & Sweep Optimizations    │
│ • SQLite Signal & Regime Persistence                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            ┌────────────────────────────────────┐
            │ Public Market Feeds (Zero API Key) │
            │      yfinance / CBOE Indices       │
            └────────────────────────────────────┘
```

---

## 2. Quick Command Reference

| Task | Command | Notes |
|---|---|---|
| **One-Click Launch** | `.\LAUNCH_SCYLLA.ps1` or `LAUNCH_SCYLLA.bat` | Auto-installs venv/deps, frees ports, compiles C++ if ready, patches frontend API base, opens browser. |
| **Dev Mode (Fast)** | `.\scripts\start_dev.ps1` | Python-only fast boot; bypasses C++ compilation. |
| **Production Build** | `.\scripts\deploy.ps1` | Fetches vendors, compiles C++ Release binary, launches full dual-engine stack. |
| **Fetch C++ Vendors** | `.\scripts\fetch_vendors.ps1` | Downloads Crow, Asio, nlohmann/json, LightGBM, and libcurl headers/binaries into `cpp_core/third_party/`. |
| **C++ Manual Build** | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` | MSVC x64 build (VS 2022 or VS 2026). |
| **Python Backend** | `backend\.venv\Scripts\Activate.ps1; cd backend; uvicorn main:app --host 127.0.0.1 --port 6900 --reload` | FastAPI reload mode. |
| **Health Checks** | `curl http://127.0.0.1:8080/health`<br>`curl http://127.0.0.1:6900/health` | Verifies C++ and Python services. |
| **Strategy Sweeps** | `python scripts\sweep_strategies_v2.py` | Multi-parameter grid search across strategy regimes. |
| **Backtest Smoke Test**| `curl -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d "{\"mode\":\"walkforward\",\"strategy_type\":\"vol_regime\"}"` | Validates walk-forward simulator. |

> **Port Management Note:** Ports `6900` (Python) and `8080` (C++) must move together. C++ fetches options data from `127.0.0.1:6900`, while the launch script patches `frontend/app.js` (`API_BASE`) to route frontend queries directly to the active coordinator.

---

## 3. Directory Blueprint

```text
PROJECT_SCYLLA/
├── backend/                        # Python FastAPI service & ML pipeline
│   ├── config/
│   │   ├── constants.py            # Global paths, labeling constants, sweep paths
│   │   ├── _strategy_loader.py     # Strategy config loader
│   │   └── strategy_defaults.json  # Source of truth for portfolio strategy regimes
│   ├── db/
│   │   ├── schema.py               # SQLite schema & table initialization
│   │   └── queries.py              # Queries, transactional logging, retry wrapper
│   ├── models/
│   │   ├── features.py             # Feature vector engineering (HV, VIX, Skew, etc.)
│   │   ├── predict.py              # C++ inference caller with local LightGBM fallback
│   │   └── train.py                # Model training & C++ artifact serialization
│   ├── backtest/
│   │   └── walkforward.py          # Walk-forward out-of-sample backtesting worker
│   ├── routers/
│   │   ├── _yf_safe.py             # Resilient yfinance wrapper (timeout, backoff, jitter)
│   │   ├── unusual_options.py      # Institutional flow / whale options endpoint
│   │   ├── put_call_ratio.py       # 30-day PCR tracker (SPY, QQQ, IWM)
│   │   ├── volume_concentration.py # Expiration cycle volume stacking
│   │   ├── iv_skew.py              # IV rank, percentile & volatility smile curves
│   │   ├── technicals.py           # Trend SMA (50d/200d) & momentum signals
│   │   ├── ml_derivations.py       # Kelly criterion, EV, and statistical metrics
│   │   └── ml_model.py             # Backtest orchestration & prediction APIs
│   ├── requirements.txt            # Locked Python dependencies
│   └── main.py                     # FastAPI entry point & router aggregation
│
├── cpp_core/                       # C++ Crow high-performance native core
│   ├── CMakeLists.txt              # CMake x64 build configuration
│   ├── include/
│   │   ├── api_handlers.h          # REST endpoint route declarations
│   │   ├── data_fetcher.h          # WinHTTP / libcurl asynchronous data client
│   │   ├── inference_engine.h      # LightGBM C-API inference bindings & structs
│   │   └── metrics_engine.h        # Real-time quantitative calculations
│   ├── src/
│   │   ├── api_handlers.cpp        # Endpoint implementation
│   │   ├── data_fetcher.cpp        # Data fetching bridge to Python backend
│   │   ├── inference_engine.cpp    # Native LightGBM prediction engine
│   │   ├── metrics_engine.cpp      # High-speed metrics calculations
│   │   └── main.cpp                # Crow server bootstrap on :8080
│   └── third_party/                # Vendored headers/libs (Crow, Asio, LightGBM, JSON)
│
├── frontend/                       # Vanilla ES6+ Web Interface
│   ├── css/                        # Modular CSS (base, layout, components, pages)
│   ├── js/                         # Modular JS (state, api, scanner, ml, backtest)
│   ├── index.html                  # Main terminal interface
│   └── app.js                      # Application orchestrator & router
│
└── scripts/                        # Automation & validation tooling
    ├── deploy.ps1                  # Full build, vendor fetch & deploy script
    ├── start_dev.ps1               # Lightweight developer launcher
    ├── fetch_vendors.ps1           # Third-party dependency installer
    ├── sweep_strategies_v2.py      # Quantitative strategy optimizer
    ├── validate_synthetic_vs_real.py # KS distribution validator
    └── audit_synthetic_dataset.py  # Dataset audit & statistical validation
```

---

## 4. Invariants & Critical Gotchas

### 1. Schema Parity (`PredictRowInput`)
`PredictRowInput` in [`cpp_core/include/inference_engine.h`](file:///C:/Users/plum/Documents/FIN%20Works/PROJECT_SCYLLA/cpp_core/include/inference_engine.h) must remain in **exact lock-step** with the feature ordering generated in [`backend/models/features.py`](file:///C:/Users/plum/Documents/FIN%20Works/PROJECT_SCYLLA/backend/models/features.py) and consumed in [`backend/models/predict.py`](file:///C:/Users/plum/Documents/FIN%20Works/PROJECT_SCYLLA/backend/models/predict.py). Any modification to input dimensions or feature keys requires updating both layers simultaneously.

### 2. Strategy Vocabularies Differ by Layer
Do not mix per-trade ML signals with portfolio-level strategies:
- **Per-Trade Signals** (Classifier labels):
  - `VOL_EXPANSION`
  - `SIDEWAYS`
  - `BULLISH_BREAKOUT`
  - `BEARISH_BREAKDOWN`
- **Portfolio Strategy Regimes** (Backtester & Strategy configs):
  - `whale_quality`: Filters high-conviction institutional flow with volume/OI spikes and technical trend alignment.
  - `contrarian_trend`: Mean-reversion signals exploiting extreme IV skew and overextended technical conditions.
  - `vol_regime`: Adaptive volatility strategy scaling position sizes based on historical vs implied volatility spreads and VIX tiers.

### 3. Dual Semantics of `profit_threshold`
- **Backtest Take-Profit Threshold (`profit_threshold` in `strategy_defaults.json`)**: Range $\approx 0.386 - 0.486$. Governs the simulated option take-profit execution barrier.
- **ML Labeling Floor (`profit_threshold` in `ml_settings`)**: Default $0.03$ ($3\%$). Governs the positive class label boundary during training sample generation.
*These two thresholds are intentionally decoupled.*

### 4. Whale Filter Thresholds Across Layers
- **C++ Data Fetcher Layer**: `minVolOI = 2.0` ([`cpp_core/include/data_fetcher.h`](file:///C:/Users/plum/Documents/FIN%20Works/PROJECT_SCYLLA/cpp_core/include/data_fetcher.h)) ensures high-throughput ingestion without dropping potential candidates.
- **Python Options Endpoint**: Default `min_vol_oi = 8.0` on `/unusual-options`.
- **Frontend Whale Scanner Display**: Highlights entries with `vol_oi >= 5.0` as institutional whale trades.

### 5. Resilient Market Data Ingestion
All third-party calls to `yfinance` **must** route through `_yf_safe.safe_call()`. Direct calls bypass the built-in rate limiter, exponential backoff, jitter, and circuit breaker, risking IP rate-limits from public quote providers.

---

## 5. Machine Learning & Walk-Forward Rules

1. **Strict Temporal Separation**: The walk-forward engine splits historical data chronologically into rolling in-sample (train) and out-of-sample (test) windows. No test fold data may be included in feature scaling or threshold tuning.
2. **Deterministic Fallback**: If the C++ inference engine (`:8080`) is offline or unreachable, Python falls back automatically to local LightGBM CPU inference without dropping incoming requests.
3. **Model Artifact Serialization**: Training runs export both the LightGBM `.txt` model file and the corresponding binary representation consumed by `InferenceEngine::load_model()`.

---

## 6. Verification & Test Checklist

Before committing changes:
```powershell
# 1. Test Python backend startup
cd backend
python -m py_compile main.py

# 2. Run statistical KS tests
python ..\scripts\validate_synthetic_vs_real.py

# 3. Test strategy defaults endpoint
curl http://127.0.0.1:6900/api/ml/strategy-defaults

# 4. Verify walk-forward backtest simulator
curl -X POST http://127.0.0.1:6900/api/ml/backtest -H "Content-Type: application/json" -d "{\"mode\":\"walkforward\",\"strategy_type\":\"vol_regime\"}"
```
