"""
PROJECT: SCYLLA // OpenBB Data Ingestion API
Python FastAPI server running on port 6900.
Provides free market data via OpenBB ODP (yfinance + CBOE providers).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from routers import unusual_options, put_call_ratio, volume_concentration, iv_skew, technicals

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


@app.get("/health")
def health():
    return {"status": "online", "service": "SCYLLA OpenBB Gateway", "port": 6900}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=6900, reload=False, log_level="info")
