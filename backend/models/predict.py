"""Prediction / inference utilities.

Extracted from ml_model.py: C++ batch predict, prediction cache helpers,
quantile monotonicity enforcement, and the async HTTP client.
"""
import hashlib
import json
import datetime
import sqlite3
import logging
from typing import Optional

import httpx

try:
    from ..config.constants import DB_PATH, _CPP_CORE_URL
except ImportError:
    from config.constants import DB_PATH, _CPP_CORE_URL

logger = logging.getLogger("scylla.ml_model")

_async_http_client: Optional[httpx.AsyncClient] = None


def _get_async_http_client() -> httpx.AsyncClient:
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = httpx.AsyncClient()
    return _async_http_client


async def _cpp_batch_predict(rows: list[dict], timeout: float = 8.0) -> list[dict] | None:
    """
    PARALLELIZATION_PLAN §3.3 / Steps 15-16: delegate batch ML inference to the
    C++ InferenceEngine via /api/v1/ml/predict-batch on port 8080.

    Sends a list of row dicts (each with option fields) and returns a parallel
    list of prediction dicts containing quantiles, p_success, strategy,
    kelly_fraction.  Returns None if C++ is unreachable or returns an error,
    so callers can fall back to the Python sklearn model path.

    Args:
        rows:    list of dicts with keys: ticker, underlier_price, strike,
                 volume, open_interest, implied_vol, premium, option_type,
                 side, dte, is_weekly, trend_alignment.
        timeout: per-request timeout in seconds (default 8s).
    Returns:
        list of prediction dicts on success, or None on failure.
    """
    if not rows:
        return []
    client = _get_async_http_client()
    try:
        resp = await client.post(
            f"{_CPP_CORE_URL}/api/v1/ml/predict-batch",
            json={"rows": rows},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(
                f"[cpp_batch] HTTP {resp.status_code}: {resp.text[:200]}"
            )
            return None
        data = resp.json()
        predictions = data.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(rows):
            logger.warning(
                f"[cpp_batch] Unexpected response shape: {str(data)[:200]}"
            )
            return None
        return predictions
    except httpx.ConnectError:
        # scylla_core.exe not running — fall back to Python path silently
        return None
    except Exception as exc:
        logger.warning(f"[cpp_batch] Error calling C++ predict-batch: {exc}")
        return None


def enforce_monotonic_quantiles(preds: dict) -> dict:
    """Enforces monotonicity across predictions: P10 <= P25 <= P50 <= P75 <= P90."""
    keys = ["p10", "p25", "p50", "p75", "p90"]
    vals = sorted([preds[k] for k in keys])
    return {k: v for k, v in zip(keys, vals)}


def _get_cache_hash(features_dict: dict) -> str:
    s = json.dumps(features_dict, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _check_prediction_cache(input_hash: str, model_version: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT prediction_json FROM prediction_cache WHERE input_hash = ? AND model_version = ?", (input_hash, model_version))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def _save_prediction_cache(input_hash: str, model_version: str, prediction_dict: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO prediction_cache (input_hash, model_version, prediction_json, created_at)
        VALUES (?, ?, ?, ?)
    """, (input_hash, model_version, json.dumps(prediction_dict), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
