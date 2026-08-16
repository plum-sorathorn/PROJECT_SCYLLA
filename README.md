# PROJECT: SCYLLA // TERMINAL
> *High-performance options whale scanner — Hybrid C++ Core + OpenBB Python ODP backend*

```
  ____  ____   ___     _ _____ ____ _____
 |  _ \|  _ \ / _ \   | | ____/ ___|_   _|
 | |_) | |_) | | | |  | |  _|| |     | |
 |  __/|  _ <| |_| |  | | |__| |___  | |
 |_|   |_| \_\___/  _/ |_____\____| |_|
                    |__/
```

## Architecture

```
[CYBERPUNK FRONTEND] ←──── HTTP/REST ────→ [C++ CROW CORE :8080]
                                                    │
                                              WinHTTP JSON
                                                    │
                                          [PYTHON ODP :6900]
                                                    │
                                           yfinance / CBOE
```

## Quick Start

```powershell
# Dev Mode (no C++ build needed):
.\scripts\start_dev.ps1

# Production (builds C++ + starts all services):
.\scripts\deploy.ps1
```

## Features
- **Whale Scanner** — EOD unusual options vol/OI ≥5x glow neon blue
- **Put/Call Ratio Tracker** — SPY/QQQ/IWM 30-day trend
- **Volume Concentration** — Stacked bar by expiration cycle
- **IV Sandbox & Skew** — IV Rank, Percentile, Volatility Smile
- **Swing Alignment** — 50d/200d SMA trend flags on whale tickers
- **Expected Move** — ATM straddle ±$ range calculator

**No paid API keys required.** All data via free yfinance + OpenBB ODP.
