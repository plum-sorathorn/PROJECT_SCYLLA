# PROJECT: SCYLLA — Deployment Guide & Architecture Reference

## Overview

**PROJECT: SCYLLA // TERMINAL** is a hybrid options swing trading dashboard with:

- **Python FastAPI** data ingestion layer (OpenBB ODP + yfinance) on port **6900**
- **C++ Crow** processing engine + REST API server on port **8080**
- **Cyberpunk HTML5/CSS/JS** frontend served at `http://127.0.0.1:8080/`

---

## Directory Structure

```
PROJECT_SCYLLA/
├── backend/                        ← Python ODP data layer
│   ├── main.py                     ← FastAPI entry point (port 6900)
│   ├── requirements.txt
│   └── routers/
│       ├── unusual_options.py      ← Widget A: EOD scanner + SMA flags + Exp Move
│       ├── put_call_ratio.py       ← Widget B: PCR trend tracker
│       ├── volume_concentration.py ← Widget C: Vol by expiry
│       ├── iv_skew.py              ← Ext 1: IV Rank, Percentile, Smile
│       └── technicals.py          ← Ext 2: SMA alignment
│
├── cpp_core/                       ← C++ processing engine
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── data_fetcher.h          ← WinHTTP JSON puller types
│   │   ├── metrics_engine.h        ← Multi-thread compute types
│   │   └── api_handlers.h          ← Crow route declarations
│   ├── src/
│   │   ├── main.cpp                ← Entry point, Crow port 8080
│   │   ├── data_fetcher.cpp        ← WinHTTP → Python ODP bridge
│   │   ├── metrics_engine.cpp      ← Thread pool, vol/OI, trend flags
│   │   └── api_handlers.cpp        ← /api/scanner, /api/iv-skew, etc.
│   └── third_party/                ← Auto-populated by fetch_vendors.ps1
│       ├── crow/include/           ← Crow HTTP framework
│       ├── asio/include/           ← Standalone Asio
│       └── nlohmann/               ← nlohmann/json
│
├── frontend/                       ← Cyberpunk dashboard
│   ├── index.html                  ← Full layout + widget shells
│   ├── style.css                   ← Cyberpunk theme, neon palette
│   └── app.js                      ← All fetch logic + Chart.js renders
│
└── scripts/
    ├── deploy.ps1                  ← Full production deploy
    ├── start_dev.ps1               ← Dev mode (no C++ build)
    └── fetch_vendors.ps1          ← Downloads C++ header deps
```

---

## Quick Start: Dev Mode (No C++ Build Required)

If you want to run immediately without building C++:

```powershell
# From project root:
.\scripts\start_dev.ps1
```

Then open `frontend\index.html` in your browser.

> The dev script patches `app.js` to point to port 6900 (Python directly).

---

## Full Production Deployment

### Prerequisites

| Tool | Required Version | Install |
|------|-----------------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| CMake | 3.20+ | `winget install Kitware.CMake` |
| Visual Studio | 2022 Community | With "Desktop development with C++" workload |
| Git | Any | `winget install Git.Git` |

### Steps

```powershell
# 1. Set execution policy if needed
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 2. Navigate to project
Set-Location "C:\Users\plum\Documents\FIN Works\PROJECT_SCYLLA"

# 3. Full deploy (downloads vendors, builds C++, starts both services)
.\scripts\deploy.ps1
```

This will:
1. Create Python venv + install all pip packages
2. Download Crow, Asio, nlohmann/json headers into `cpp_core/third_party/`
3. Run CMake configure + build (`scylla_core.exe`)
4. Launch Python ODP on `127.0.0.1:6900`
5. Launch C++ Core on `127.0.0.1:8080`

---

## API Endpoints

### Python ODP Layer (port 6900)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service status |
| `GET /api/v1/unusual-options` | EOD scanner with SMA flags + expected move |
| `GET /api/v1/put-call-ratio` | PCR trend for SPY/QQQ/IWM |
| `GET /api/v1/volume-concentration?ticker=SPY` | Vol by expiry |
| `GET /api/v1/iv-skew?ticker=SPY` | IV Rank, Percentile, Smile |
| `GET /api/v1/technicals?tickers=SPY,AAPL` | SMA flags |
| `GET /docs` | Swagger UI (auto-generated) |

### C++ Core (port 8080)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Core status |
| `GET /api/scanner` | Processed + enriched scanner data |
| `GET /api/put-call-ratio` | PCR trend |
| `GET /api/volume-concentration?ticker=SPY` | Vol concentration |
| `GET /api/iv-skew?ticker=SPY` | IV Skew / Sandbox |
| `GET /` | Frontend dashboard |

---

## Dashboard Widgets

| Widget | Description |
|--------|-------------|
| **A — Unusual Options Scanner** | Dense table: Ticker, Exp, Strike, Type, Vol/OI Ratio. Whale rows (≥5x) glow neon blue. Columns sortable. |
| **B — Put/Call Ratio Trend** | Multi-line chart: SPY (blue), QQQ (magenta), IWM (green) across expiry cycles |
| **C — Volume Concentration** | Stacked bar: Call/Put volume per expiration cycle |
| **Ext 1 — IV Sandbox** | 30-day IV Rank + IV Percentile gauges + Volatility Smile chart (strike vs IV) |
| **Ext 2 — Swing Alignment** | Per-ticker 50d/200d SMA flags from scanner whale tickers |
| **Ext 3 — Expected Move** | ATM straddle price extracted from front-month chain; displayed as ±$ range |

---

## Data Sources (All Free, No API Keys Required)

- **yfinance** — Option chains, historical prices, SMA computation
- **CBOE/OpenBB ODP** — Free derivatives data via `openbb` package
- **No paid subscriptions required**

---

## C++ Architecture Notes

- **WinHTTP** (native Windows) used for HTTP requests to avoid libcurl dependency
- **Crow** (header-only) for the HTTP server on port 8080
- **Multi-threaded processing** in `metrics_engine.cpp` uses `std::thread` pool chunked across `hardware_concurrency()` cores
- **nlohmann/json** for JSON serialization/deserialization

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CMake not found` | Install via `winget install Kitware.CMake` |
| `VS 2022 not found` | Use `-G "Visual Studio 16 2019"` for VS 2019 |
| Port 6900 busy | `netstat -ano \| findstr :6900` then `Stop-Process -Id <PID>` |
| `yfinance` rate limits | Add `time.sleep(0.5)` between ticker fetches in unusual_options.py |
| CORS errors in browser | Already handled; verify C++ Core is running on 8080 |

---

> **DISCLAIMER**: PROJECT: SCYLLA is for **educational and research purposes only**. Not financial advice. Options trading involves substantial risk.
