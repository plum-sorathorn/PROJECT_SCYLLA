"""
PROJECT: SCYLLA // OpenBB Data Ingestion API
Python FastAPI server running on port 6900.
Provides free market data via OpenBB ODP (yfinance + CBOE providers).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Param
import asyncio
import inspect
import logging

from routers import unusual_options, put_call_ratio, volume_concentration, iv_skew, technicals, ml_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scylla.backend")

app = FastAPI(
    title="PROJECT: SCYLLA // OpenBB Data Gateway",
    description="Free options market data ingestion layer via OpenBB ODP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(unusual_options.router, prefix="/api/v1", tags=["Unusual Options"])
app.include_router(put_call_ratio.router, prefix="/api/v1", tags=["Put/Call Ratio"])
app.include_router(volume_concentration.router, prefix="/api/v1", tags=["Volume Concentration"])
app.include_router(iv_skew.router, prefix="/api/v1", tags=["IV Skew"])
app.include_router(technicals.router, prefix="/api/v1", tags=["Technicals"])
app.include_router(ml_model.router, prefix="/api/v1", tags=["ML Model"])

# Compatibility routes for frontend calling directly in DEV MODE (without C++ Core running)
app.include_router(unusual_options.router, prefix="/api", tags=["Scanner API"])
app.include_router(put_call_ratio.router, prefix="/api", tags=["Put/Call Ratio API"])
app.include_router(volume_concentration.router, prefix="/api", tags=["Volume Concentration API"])
app.include_router(iv_skew.router, prefix="/api", tags=["IV Skew API"])
app.include_router(technicals.router, prefix="/api", tags=["Technicals API"])
app.include_router(ml_model.router, prefix="/api", tags=["ML Model API"])


def _call_with_resolved_defaults(func, **overrides):
    """Call a router handler as a plain function, resolving any FastAPI Param
    (Query/Path/Body/etc.) defaults to their underlying value. The handler's
    declared signature is the single source of truth for defaults; explicit
    kwargs in `overrides` take precedence.
    """
    sig = inspect.signature(func)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in overrides:
            kwargs[name] = overrides[name]
        else:
            default = param.default
            if isinstance(default, Param):
                kwargs[name] = default.default
            else:
                kwargs[name] = default
    return func(**kwargs)


async def tactical_bundle(
    min_vol_oi: float = 2.0,
    volcon_ticker: str = "SPY",
    iv_ticker: str = "SPY",
):
    scanner, pcr, volcon, iv = await asyncio.gather(
        asyncio.to_thread(_call_with_resolved_defaults, unusual_options.get_scanner, min_vol_oi=min_vol_oi),
        asyncio.to_thread(_call_with_resolved_defaults, put_call_ratio.get_put_call_ratio),
        asyncio.to_thread(_call_with_resolved_defaults, volume_concentration.get_volume_concentration, ticker=volcon_ticker),
        asyncio.to_thread(_call_with_resolved_defaults, iv_skew.get_iv_skew, ticker=iv_ticker),
    )
    return {
        "scanner": scanner,
        "put_call_ratio": pcr,
        "volume_concentration": volcon,
        "iv_skew": iv,
    }


app.add_api_route("/api/v1/tactical-bundle", tactical_bundle, methods=["GET"])
app.add_api_route("/api/tactical-bundle", tactical_bundle, methods=["GET"])


@app.get("/health")
def health():
    return {"status": "online", "service": "SCYLLA OpenBB Gateway", "port": 6900}


import os
from fastapi.staticfiles import StaticFiles

# Resolve frontend directory path relative to this script
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6900, reload=False, log_level="info")
