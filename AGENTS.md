# AGENTS.md — PROJECT: SCYLLA // TERMINAL

High-performance options whale scanner. Hybrid 3-tier: cyberpunk HTML/JS frontend → C++ Crow (`scylla_core.exe`, :8080) → Python FastAPI/OpenBB ODP (uvicorn, :6900) → yfinance + CBOE. Windows-only. No paid API keys.

## Quick reference

| Action | Command |
|---|---|
| One-click launch (user) | `.\LAUNCH_SCYLLA.ps1` (or `.bat`) |
| Dev mode (Python only, no C++ build) | `.\scripts\start_dev.ps1` |
| Full production build + launch | `.\scripts\deploy.ps1` |
| C++ build only | `cd cpp_core\build && cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --parallel` |
| Python venv activate | `backend\.venv\Scripts\Activate.ps1` |
| Run Python backend | `uvicorn main:app --host 127.0.0.1 --port 6900 --reload` (from `backend\`) |
| Health check | `curl http://127.0.0.1:8080/health` and `:6900/health` |

Ports: C++ **8080**, Python **6900**. Do not change without updating `data_fetcher.cpp` and `frontend\app.js`.

## Architecture (data flow)

```
frontend/index.html ─HTTP─> C++ Crow (8080, multithreaded)
                                │ WinHTTP, hardcoded 127.0.0.1:6900
                                ▼
                          Python FastAPI (6900, CORS=*)
                                │ yfinance / openbb==4.3.2 / CBOE
                                ▼
                          data providers
```

The C++ ↔ Python bridge is `cpp_core\src\data_fetcher.cpp`. It uses **WinHTTP** (`#pragma comment(lib, "winhttp.lib")`), sets **no timeouts** (OS defaults ~120s), and has **no retry** — any failure throws `std::runtime_error` and the request fails. C++ routes call into `data_fetcher` and `metrics_engine` only; no other outbound HTTP in C++.

## Backend layout

- `backend\main.py` — FastAPI app; mounts 6 routers under both `/api/v1` and `/api` (compat for direct dev mode), serves `frontend\` as static at `/`, CORS wide open, `GET /health`.
- `backend\routers\` — `unusual_options.py`, `put_call_ratio.py`, `volume_concentration.py`, `iv_skew.py`, `technicals.py`, `ml_model.py`, `ml_derivations.py`. All 7 are real; none are stubs.
- `cpp_core\src\main.cpp` — Crow `SimpleApp` on :8080, calls `registerRoutes(app)`.
- `cpp_core\src\api_handlers.cpp` — HTTP routes: `/health`, `/api/scanner`, `/api/put-call-ratio`, `/api/volume-concentration`, `/api/iv-skew`, `/` (static), `/<path>` (SPA catch-all from `frontend\dist\`).
- `frontend\app.js` — `API_BASE` defaults to `http://127.0.0.1:6900` (line ~9). **This file is patched in-place by the launcher scripts** to swap 6900↔8080 depending on whether `scylla_core.exe` exists. Treat the on-disk value as ephemeral.

## C++ build (Windows, MSVC required)

- Generator: **Visual Studio 17 2022**, x64, C++17.
- Vendored headers (Crow, Asio standalone, nlohmann/json) in `cpp_core\third_party\`. Populated by `scripts\fetch_vendors.ps1` — do not hand-edit.
- Output: `cpp_core\build\Release\scylla_core.exe`.
- Links `ws2_32` + `wsock32` (Windows sockets).
- `deploy.ps1` calls `fetch_vendors.ps1` before `cmake`; if C++ build fails, deploy auto-falls-back to dev mode (Python serves frontend directly via `python -m http.server 8080`).

## ML pipeline (worth knowing)

`backend\routers\ml_model.py` is the only stateful piece:
- SQLite DB: `backend\scylla_ml.db` (gitignored). Has a `.bak` next to it.
- Model pickle: `backend\cache\scylla_predictor.pkl` (gitignored).
- LightGBM quantile regression, `LABELING_VERSION = "v2_settlement"`, default `horizon_days=10`, `profit_threshold=0.03`, `prob_threshold=0.55`, `_execute_with_retry` = 5 retries + exp backoff, DB in WAL mode.
- `backend\routers\ml_derivations.py` is pure functions (P(success), strategy, Kelly). Clip bounds `0.02`/`0.98`, `kelly_cap=0.25`.

## No-go / fragile files (do not edit casually)

- `frontend\app.js` — rewritten by launchers. If you change `API_BASE`, also update the patch logic in `LAUNCH_SCYLLA.ps1` and `scripts\start_dev.ps1`.
- `backend\scylla_ml.db`, `backend\scylla_ml.db.bak`, `backend\cache\*.pkl` — runtime state, gitignored.
- `cpp_core\third_party\` — vendored, managed by `fetch_vendors.ps1`.
- `cpp_core\build\`, `backend\.venv\`, `**\__pycache__\`, `graphify-out\`, `.env` — all gitignored build/cache/output.
- `.env` (repo root) — gitignored runtime config; never commit equivalents. Holds port/data-provider settings.

## Gotchas

- **No tests, no lint, no typecheck, no CI** are configured. There is no `pytest.ini`, no `mypy.ini`/`ruff.toml`/`.flake8`, no `.eslintrc`/`.prettierrc`, no `tsconfig.json`, no `Makefile`, no `.github\`. Do not invent a framework mid-task — ask before adding one.
- **C++ exe is optional.** Full app runs in dev mode (Python serves frontend) without ever compiling C++.
- **Whale threshold**: vol/OI ≥ 5x. Default `min_vol_oi=2.0` in C++ (`data_fetcher.h:60`), `8.0` in Python router default, ≥ 5.0 in scanner logic. These are three different defaults — be explicit when changing.
- **No HTTP timeouts in C++ bridge.** A hung Python backend will hang the C++ request for ~120s. Not a current bug, just expected.
- **Backend `allow_origins=["*"]`** — local-dev only, do not assume this is safe in any future deploy.
- **Hardcoded `127.0.0.1`** in `data_fetcher.cpp` and `app.js`. Both must move together if you ever bind to a real interface.

## OpenCode / agent conventions (project-specific)

The repo uses a **graphify** knowledge graph (`graphify-out\`) as the primary navigation layer. Rules in `.opencode\rules\graphify.md` and `.agents\rules\graphify.md`:

- **Query graphify before raw `grep`/`ls`/`read`** on unfamiliar code. Use `graphify query`, `graphify path`, `graphify explain`.
- `graphify` CLI is on PATH — call it directly, never `npx`/`npm`/`pip`.
- Run `graphify update .` after edits to keep the graph in sync.
- For broad overviews, read `graphify-out\GRAPH_REPORT.md`; for nav, check `graphify-out\wiki\index.md` if present.

## Subagent routing (matches repo `AGENTS.md`/global rules)

- Read-only exploration → `Task(subagent_type="fast", ...)` (default).
- Implementation / edits → `Task(subagent_type="medium", ...)` (default for this orchestrator).
- Architecture/debug after ≥2 failures → `Task(subagent_type="heavy", ...)`. Before dispatching @heavy, gather concrete context via @fast.
- Hard self-cap: ≤2 direct read-only tool calls per turn; dispatch @fast on the 3rd need.

## Where to look first when something is unclear

1. `README.md` (architecture + quick start).
2. `LAUNCH_SCYLLA.ps1` — what the user actually runs.
3. `scripts\deploy.ps1` — full build pipeline, dev/prod branching.
4. `cpp_core\src\main.cpp` + `data_fetcher.cpp` — the C++↔Python contract.
5. `backend\main.py` — Python entry, router mounts, CORS, static serving.
6. `graphify query <topic>` — fastest path to relevant code in a large repo.
