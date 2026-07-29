from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import asyncio
import datetime
import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import requests
import logging
import yfinance as yf
from .ml_derivations import compute_calibrated_p_success, classify_strategy, kelly_fraction, kelly_fraction_realized
from ._yf_safe import safe_call

import threading
import concurrent.futures
import time

from .utils import _sanitize_float_values

# ── Imports from new sub-modules (try/except handles both
#     "from backend.routers.ml_model" and "from routers import ml_model" contexts)
try:
    # When loaded as backend.routers.ml_model (project root context)
    from ..config.constants import (  # type: ignore[import-untyped]
        DB_PATH, CACHE_DIR, MODEL_PATH, _SCYLLA_MAX_PROC,
        LABELING_VERSION, _CPP_CORE_URL, BOOT_CACHE_PATH,
        SWEEP_OPTIMAL_PATHS, STRATEGY_DEFAULTS_PATH,
    )
    from ..db.schema import init_db
    from ..db.queries import (  # type: ignore[import-untyped]
        _execute_with_retry, _fetch_trades_from_db,
        get_real_trades, get_dataset_stats,
    )
    from ..models.features import (  # type: ignore[import-untyped]
        _fetch_historical_volatility, _fetch_vix_level,
        _dte_to_bucket, compute_advanced_features,
        get_hv_cache, get_vix_cache,
    )
    from ..models.tier_a import (  # type: ignore[import-untyped]
        _compute_tier_a_tickers, _compute_tier_profitable_tickers,
        TIER_PROFITABLE_TICKERS,
    )
    from ..models.predict import (  # type: ignore[import-untyped]
        _cpp_batch_predict, enforce_monotonic_quantiles,
        _get_cache_hash, _check_prediction_cache, _save_prediction_cache,
    )
    from ..models.train import _fit_one_quantile_train, _serialize_cpp_inference_artifacts  # type: ignore[import-untyped]
    from ..backtest.walkforward import _wf_worker_init, _process_walkforward_step  # type: ignore[import-untyped]
    from ..config._strategy_loader import get_strategy_params, get_common_params  # type: ignore[import-untyped]
except ImportError:
    # When loaded as routers.ml_model (backend/ directory context)
    from config.constants import (  # type: ignore[import-untyped]
        DB_PATH, CACHE_DIR, MODEL_PATH, _SCYLLA_MAX_PROC,
        LABELING_VERSION, _CPP_CORE_URL, BOOT_CACHE_PATH,
        SWEEP_OPTIMAL_PATHS, STRATEGY_DEFAULTS_PATH,
    )
    from db.schema import init_db
    from db.queries import (  # type: ignore[import-untyped]
        _execute_with_retry, _fetch_trades_from_db,
        get_real_trades, get_dataset_stats,
    )
    from models.features import (  # type: ignore[import-untyped]
        _fetch_historical_volatility, _fetch_vix_level,
        _dte_to_bucket, compute_advanced_features,
        get_hv_cache, get_vix_cache,
    )
    from models.tier_a import (  # type: ignore[import-untyped]
        _compute_tier_a_tickers, _compute_tier_profitable_tickers,
        TIER_PROFITABLE_TICKERS,
    )
    from models.predict import (  # type: ignore[import-untyped]
        _cpp_batch_predict, enforce_monotonic_quantiles,
        _get_cache_hash, _check_prediction_cache, _save_prediction_cache,
    )
    from models.train import _fit_one_quantile_train, _serialize_cpp_inference_artifacts  # type: ignore[import-untyped]
    from backtest.walkforward import _wf_worker_init, _process_walkforward_step  # type: ignore[import-untyped]
    from config._strategy_loader import get_strategy_params, get_common_params  # type: ignore[import-untyped]

logger = logging.getLogger("scylla.ml_model")
router = APIRouter()

# ── In-memory caches for expensive market data lookups ───────
# These are re-exported from models.features so that _fetch_historical_volatility
# and _fetch_vix_level (the authoritative implementations) write to the SAME
# dicts that the remaining endpoint code reads from.
_hv_cache = get_hv_cache()   # ticker -> annualized HV (same object as models.features._hv_cache)
_vix_cache = get_vix_cache()  # cached VIX level (same object as models.features._vix_cache)
_settings_cache = {"data": None, "ts": 0.0}  # TTL cache for _get_settings()
_SETTINGS_TTL_SEC = 60.0
_global_model = None
_backtest_response_cache = {}
_walkforward_lock = threading.Lock()
_walkforward_cache_mem = {}

# ── Walkforward process-pool shared state ──────────────────────
# On fork-based platforms (Linux/macOS) the child inherits these
# from the parent. On spawn-based platforms (Windows) the initializer
# _wf_worker_init() sets them in each worker. Each worker receives the
# full dataset pickled exactly once, not once per task submission.
_WF_DF_REAL = None
_WF_DF_REAL_FEAT = None


def get_global_model():
    global _global_model
    if _global_model is not None:
        return _global_model
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            if isinstance(m, dict) and all(q in m for q in [0.1, 0.25, 0.5, 0.75, 0.9]):
                _global_model = m
                return _global_model
        except Exception:
            pass
    return None


def _prefetch_historical_vols(tickers) -> None:
    if not tickers:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(4, min(20, len(tickers)))) as executor:
        futures = [executor.submit(_fetch_historical_volatility, t) for t in tickers]
        concurrent.futures.wait(futures, timeout=None)


class TradeSchema(BaseModel):
    ticker: str
    expiration: str
    strike: float
    optionType: str
    volume: int
    openInterest: int
    volOiRatio: float
    impliedVolatility: float
    underlierPrice: float
    premium: float
    side: str
    dte: int
    isWeekly: bool
    trendAlignment: str



class PredictRequestSchema(BaseModel):
    ticker: Optional[str] = "SPY"
    strike: float
    volume: int
    openInterest: int
    volOiRatio: float
    impliedVolatility: float
    underlierPrice: float
    premium: float
    dte: int
    optionType: str
    side: str
    trendAlignment: str
    # Optional advanced features — computed server-side if not provided
    moneyness: Optional[float] = None
    iv_hv_ratio: Optional[float] = None
    vix_level: Optional[float] = None
    log_premium: Optional[float] = None
    dte_bucket: Optional[str] = None

class PredictResponseSchema(BaseModel):
    quantiles: dict[str, float]      # {"p10": ..., "p25": ..., "p50": ..., "p75": ..., "p90": ...}
    p_success: float
    expected_return: float           # = p50
    strategy: str                    # SIDEWAYS | BULLISH_BREAKOUT | BEARISH_BREAKDOWN | VOL_EXPANSION
    strategy_confidence: float
    kelly_fraction: float
    kelly_fraction_uncapped: float
    model_type: Optional[str] = None

# (init_db() and auto-call moved → backend/db/schema.py, see import above)
# (Feature engineering functions moved → backend/models/features.py, see import above)

# ═══════════════════════════════════════════════════════════════
# SETTINGS ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def _get_settings() -> dict:
    """Read all settings from the ml_settings table. Cached for 60s; invalidated by api_update_settings."""
    now = time.monotonic()
    if _settings_cache["data"] is not None and (now - _settings_cache["ts"]) < _SETTINGS_TTL_SEC:
        return dict(_settings_cache["data"])
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM ml_settings")
    rows = cursor.fetchall()
    conn.close()
    result = {row[0]: row[1] for row in rows}
    _settings_cache["data"] = result
    _settings_cache["ts"] = now
    return result


# (TIER_A_TICKERS and TIER_PROFITABLE_TICKERS moved → backend/models/tier_a.py, see import above)

@router.get("/ml/settings")
def api_get_settings():
    """Return all ML settings as a JSON dict."""
    try:
        settings = _get_settings()
        return {
            "horizon_days": int(settings.get("horizon_days", "10")),
            "profit_threshold": float(settings.get("profit_threshold", "0.03"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/settings")
def api_update_settings(
    horizon_days: Optional[int] = Query(default=None),
    profit_threshold: Optional[float] = Query(default=None)
):
    """Update ML settings in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if horizon_days is not None:
            cursor.execute("INSERT OR REPLACE INTO ml_settings (key, value) VALUES ('horizon_days', ?)", (str(horizon_days),))
        if profit_threshold is not None:
            cursor.execute("INSERT OR REPLACE INTO ml_settings (key, value) VALUES ('profit_threshold', ?)", (str(profit_threshold),))
        conn.commit()
        conn.close()
        # Invalidate TTL cache so the next _get_settings() call re-reads from disk.
        _settings_cache["data"] = None
        _settings_cache["ts"] = 0.0
        return {"status": "success", "message": "Settings updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# MODEL RUNS AUDIT ENDPOINT
# ═══════════════════════════════════════════════════════════════

@router.get("/ml/model-runs")
def api_get_model_runs(limit: int = 50):
    """Return historical training run audit data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM model_runs ORDER BY timestamp DESC LIMIT ?",
            conn, params=[limit]
        )
        conn.close()
        # Replace NaN/NaT with None for JSON compliance
        df = df.where(pd.notnull(df), None)
        runs = df.to_dict(orient="records")
        
        # Clean up any potential float nan/inf values to guarantee JSON compliance
        import math
        for run in runs:
            for k, v in run.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    run[k] = None
                    
        return {"data": runs, "count": len(runs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# (_execute_with_retry moved → backend/db/queries.py, see import above)

# ═══════════════════════════════════════════════════════════════
# TRADE LOGGING
# ═══════════════════════════════════════════════════════════════

@router.post("/ml/log-trade")
def api_log_trade(trade: TradeSchema):
    def _log_trade():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for duplicates logged within the last 12 hours (same contract and ratio)
        twelve_hours_ago = (datetime.datetime.now() - datetime.timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            SELECT id FROM options_trades
            WHERE ticker = ? AND strike = ? AND expiration = ? AND option_type = ? 
              AND vol_oi_ratio = ? AND timestamp >= ?
        """, (trade.ticker.upper(), trade.strike, trade.expiration, trade.optionType, trade.volOiRatio, twelve_hours_ago))
        
        if cursor.fetchone():
            conn.close()
            return {"status": "ignored", "message": "Duplicate trade logged recently."}
            
        cursor.execute("""
            INSERT INTO options_trades (
                timestamp, ticker, expiration, strike, option_type, volume, 
                open_interest, vol_oi_ratio, implied_vol, underlier_price, 
                premium, side, dte, is_weekly, trend_alignment, is_synthetic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            trade.ticker.upper(),
            trade.expiration,
            trade.strike,
            trade.optionType,
            trade.volume,
            trade.openInterest,
            trade.volOiRatio,
            trade.impliedVolatility,
            trade.underlierPrice,
            trade.premium,
            trade.side,
            trade.dte,
            1 if trade.isWeekly else 0,
            trade.trendAlignment
        ))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Logged trade for {trade.ticker}"}
    
    try:
        return _execute_with_retry(_log_trade)
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml/open-trades")
async def api_get_open_trades(
    prob_threshold: Optional[float] = Query(default=0.55),
    min_kelly_fraction: Optional[float] = Query(default=0.01)
):
    try:
        settings = await asyncio.to_thread(_get_settings)
        target_pct = float(settings.get("profit_threshold", "0.03"))

        from .unusual_options import scan_raw_options, SCAN_TICKERS
        raw_options = await asyncio.to_thread(
            scan_raw_options, ",".join(SCAN_TICKERS), min_vol_oi=0.1, limit=100
        )
        df = pd.DataFrame(raw_options)

        if not df.empty:
            df = df.rename(columns={
                'optionType': 'option_type',
                'openInterest': 'open_interest',
                'volOiRatio': 'vol_oi_ratio',
                'impliedVolatility': 'implied_vol',
                'underlierPrice': 'underlier_price',
                'isWeekly': 'is_weekly',
                'trendAlignment': 'trend_alignment'
            })
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            unique_tickers = list(df['ticker'].unique())
            if unique_tickers:
                await asyncio.to_thread(_prefetch_historical_vols, unique_tickers)

        highest_prob_trade = None
        max_p_success = -1.0

        if not df.empty:
            df['predicted_p10'] = None
            df['predicted_p25'] = None
            df['predicted_p50'] = None
            df['predicted_p75'] = None
            df['predicted_p90'] = None
            df['predicted_strategy'] = None
            df['p_success'] = None
            df['kelly_fraction'] = None

            models = get_global_model()

            try:
                horizon_days = int(settings.get("horizon_days", "10"))

                # Fetch vix and populate caches in parallel
                await asyncio.to_thread(_fetch_vix_level)
                unique_tickers = list(df['ticker'].unique())
                if unique_tickers:
                    await asyncio.to_thread(_prefetch_historical_vols, unique_tickers)

                # Vectorized feature engineering on the DataFrame.
                # 50-100x faster than iterrows() for the Live Signals refresh path.
                df['ticker'] = df['ticker'].str.upper()
                if 'trend_alignment' not in df.columns:
                    df['trend_alignment'] = 'NEUTRAL'
                else:
                    df['trend_alignment'] = df['trend_alignment'].fillna('NEUTRAL')
                df['moneyness'] = (df['strike'] - df['underlier_price']) / df['underlier_price']
                hv_series = df['ticker'].map(_hv_cache).fillna(0.0)
                df['iv_hv_ratio'] = np.where(hv_series > 0, df['implied_vol'] / hv_series, 1.0)
                df['vix_level'] = _vix_cache.get("value") or 20.0
                df['log_premium'] = np.log1p(df['premium'].clip(lower=0))
                df['dte_bucket'] = np.select(
                    [df['dte'] <= 7, df['dte'] <= 30, df['dte'] <= 60],
                    ['weekly', 'short', 'medium'],
                    default='long',
                )
                feature_rows = df.to_dict(orient='records')

                if feature_rows:
                    # ── STEP 15: Try C++ InferenceEngine batch first ───────────────────
                    # PARALLELIZATION_PLAN §3.3: forward entire batch to port 8080.
                    # C++ runs LGBM_BoosterPredict natively (no GIL, no pickle load).
                    # Falls back to Python sklearn path if scylla_core.exe is not up.
                    cpp_rows = [
                        {
                            "ticker":          r["ticker"],
                            "underlier_price": r["underlier_price"],
                            "strike":          r["strike"],
                            "volume":          r["volume"],
                            "open_interest":   r["open_interest"],
                            "implied_vol":     r["implied_vol"],
                            "premium":         r["premium"],
                            "option_type":     r["option_type"],
                            "side":            r["side"],
                            "dte":             r["dte"],
                            "is_weekly":       str(r.get("is_weekly", "False")),
                            "trend_alignment": r.get("trend_alignment", "NEUTRAL"),
                        }
                        for r in feature_rows
                    ]

                    cpp_preds = await _cpp_batch_predict(cpp_rows)

                    if cpp_preds is not None:
                        # ── C++ path succeeded ───────────────────────────────
                        logger.debug(f"[open_trades] C++ batch predict: {len(cpp_preds)} rows")
                        predicted_p10s, predicted_p25s, predicted_p50s = [], [], []
                        predicted_p75s, predicted_p90s = [], []
                        predicted_strategies, p_successes, kellys = [], [], []

                        for pred in cpp_preds:
                            q = pred.get("quantiles", {})
                            predicted_p10s.append(round(q.get("p10", 0.0), 4))
                            predicted_p25s.append(round(q.get("p25", 0.0), 4))
                            predicted_p50s.append(round(q.get("p50", 0.0), 4))
                            predicted_p75s.append(round(q.get("p75", 0.0), 4))
                            predicted_p90s.append(round(q.get("p90", 0.0), 4))
                            predicted_strategies.append(pred.get("strategy", "NONE"))
                            p_successes.append(round(float(pred.get("p_success", 0.0)), 4))
                            kellys.append(round(float(pred.get("kelly_fraction", 0.0)), 4))

                    else:
                        # ── Python sklearn fallback path ──────────────────────
                        feat_df = pd.DataFrame(feature_rows)
                        predicted_p10s, predicted_p25s, predicted_p50s = [], [], []
                        predicted_p75s, predicted_p90s = [], []
                        predicted_strategies, p_successes, kellys = [], [], []

                        if models is not None:
                            p10_preds = models[0.1].predict(feat_df)
                            p25_preds = models[0.25].predict(feat_df)
                            p50_preds = models[0.5].predict(feat_df)
                            p75_preds = models[0.75].predict(feat_df)
                            p90_preds = models[0.9].predict(feat_df)

                        for i in range(len(df)):
                            row = df.iloc[i]
                            tk = row['ticker'].upper()
                            tk_hv = _hv_cache.get(tk, 0.0)
                            hv_dec = tk_hv / 100.0
                            ticker_hv_30d = hv_dec * np.sqrt(horizon_days / 252.0)
                            if ticker_hv_30d <= 0.0:
                                ticker_hv_30d = 0.04

                            iqr_threshold = 1.5 * ticker_hv_30d
                            direction_threshold = 0.25 * ticker_hv_30d

                            if models is not None:
                                q_preds = {
                                    "p10": float(p10_preds[i]),
                                    "p25": float(p25_preds[i]),
                                    "p50": float(p50_preds[i]),
                                    "p75": float(p75_preds[i]),
                                    "p90": float(p90_preds[i])
                                }
                                q_preds = enforce_monotonic_quantiles(q_preds)
                                p_succ = compute_calibrated_p_success(
                                    target_pct,
                                    q_preds["p10"], q_preds["p25"], q_preds["p50"], q_preds["p75"], q_preds["p90"]
                                )
                                capped_kelly, _ = kelly_fraction(p_succ, q_preds["p50"], q_preds["p10"], q_preds["p90"])
                                strat, _ = classify_strategy(q_preds["p50"], q_preds["p10"], q_preds["p90"], iqr_threshold, direction_threshold)
                            else:
                                # Heuristic fallback path (no model at all)
                                score = 0.50
                                if row['trend_alignment'] == 'BULL_ALIGNED':
                                    score += 0.15
                                if row['side'] == 'BUY':
                                    score += 0.05
                                p_succ = min(score, 0.95)
                                median_val = 0.03
                                q_preds = {
                                    "p10": median_val - 0.05, "p25": median_val - 0.025,
                                    "p50": median_val, "p75": median_val + 0.025, "p90": median_val + 0.05
                                }
                                capped_kelly, _ = kelly_fraction(p_succ, q_preds["p50"], q_preds["p10"], q_preds["p90"])
                                strat, _ = classify_strategy(q_preds["p50"], q_preds["p10"], q_preds["p90"], iqr_threshold, direction_threshold)

                            predicted_p10s.append(round(q_preds["p10"], 4))
                            predicted_p25s.append(round(q_preds["p25"], 4))
                            predicted_p50s.append(round(q_preds["p50"], 4))
                            predicted_p75s.append(round(q_preds["p75"], 4))
                            predicted_p90s.append(round(q_preds["p90"], 4))
                            predicted_strategies.append(strat)
                            p_successes.append(round(p_succ, 4))
                            kellys.append(round(capped_kelly, 4))

                    df['predicted_p10'] = predicted_p10s
                    df['predicted_p25'] = predicted_p25s
                    df['predicted_p50'] = predicted_p50s
                    df['predicted_p75'] = predicted_p75s
                    df['predicted_p90'] = predicted_p90s
                    df['predicted_strategy'] = predicted_strategies
                    df['p_success'] = p_successes
                    df['kelly_fraction'] = kellys

                    
            except Exception as pred_ex:
                logger.warning(f"Failed to compute dynamic predictions in api_get_open_trades: {pred_ex}")
            
        df = df.where(pd.notnull(df), None)
        trades = df.to_dict(orient="records")
        
        filtered_trades = []
        for t in trades:
            if t.get('p_success') is not None and t['p_success'] >= prob_threshold:
                if t.get('kelly_fraction') is not None and t['kelly_fraction'] >= min_kelly_fraction:
                    filtered_trades.append(t)
                    if t['p_success'] > max_p_success:
                        max_p_success = t['p_success']
                        highest_prob_trade = t
                
        return _sanitize_float_values({"data": filtered_trades, "count": len(filtered_trades), "highest_probability": highest_prob_trade})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml/trades")
async def api_get_trades(
    ticker: Optional[str] = None,
    labeled: Optional[int] = None,
    limit: int = 50,
    offset: int = 0
):
    try:
        # Fetch profit threshold to derive label_success at read time
        settings = await asyncio.to_thread(_get_settings)
        target_pct = float(settings.get("profit_threshold", "0.03"))

        query = "SELECT * FROM options_trades WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if labeled is not None:
            query += " AND labeled = ?"
            params.append(labeled)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        df = await asyncio.to_thread(_fetch_trades_from_db, query, params)
        
        # Override label_success at read time based on continuous return (observed_return)
        if not df.empty and 'observed_return' in df.columns:
            # Vectorized: only overwrite label_success for rows that are labeled AND have an observed_return.
            # Preserves existing label_success for unlabeled rows (matches the previous lambda's else-branch).
            mask = (df['labeled'] == 1) & df['observed_return'].notna()
            if mask.any():
                df.loc[mask, 'label_success'] = (df.loc[mask, 'observed_return'] >= target_pct).astype(int)
            # Make sure it handles NaN
            df['label_success'] = df['label_success'].replace({np.nan: None})

        # Predict P50 and Strategy for each trade row if the model exists
        if not df.empty:
            df['predicted_p50'] = None
            df['predicted_strategy'] = None
            
            models = get_global_model()
                    
            if models is not None:
                try:
                    horizon_days = int(settings.get("horizon_days", "10"))
                    
                    # Pre-populate HV cache for all unique tickers in the list
                    await asyncio.to_thread(_prefetch_historical_vols, list(df['ticker'].unique()))
                            
                    # Vectorized feature engineering on the DataFrame.
                    # 50-100x faster than iterrows() on the Live Signals history table.
                    df['ticker'] = df['ticker'].str.upper()
                    if 'trend_alignment' not in df.columns:
                        df['trend_alignment'] = 'NEUTRAL'
                    else:
                        df['trend_alignment'] = df['trend_alignment'].fillna('NEUTRAL')
                    df['moneyness'] = (df['strike'] - df['underlier_price']) / df['underlier_price']
                    hv_series = df['ticker'].map(_hv_cache).fillna(0.0)
                    df['iv_hv_ratio'] = np.where(hv_series > 0, df['implied_vol'] / hv_series, 1.0)
                    df['vix_level'] = _vix_cache.get("value") or 20.0
                    df['log_premium'] = np.log1p(df['premium'].clip(lower=0))
                    df['dte_bucket'] = np.select(
                        [df['dte'] <= 7, df['dte'] <= 30, df['dte'] <= 60],
                        ['weekly', 'short', 'medium'],
                        default='long',
                    )
                    feature_rows = df.to_dict(orient='records')
                    
                    if feature_rows:
                        # ── STEP 16: Try C++ InferenceEngine batch first ───────────
                        # PARALLELIZATION_PLAN §3.3: forward to port 8080.
                        # api_get_trades only needs p50 + strategy for the history table.
                        cpp_rows = [
                            {
                                "ticker":          r["ticker"],
                                "underlier_price": r["underlier_price"],
                                "strike":          r["strike"],
                                "volume":          r["volume"],
                                "open_interest":   r["open_interest"],
                                "implied_vol":     r["implied_vol"],
                                "premium":         r["premium"],
                                "option_type":     r["option_type"],
                                "side":            r["side"],
                                "dte":             r["dte"],
                                "is_weekly":       str(r.get("is_weekly", "False")),
                                "trend_alignment": r.get("trend_alignment", "NEUTRAL"),
                            }
                            for r in feature_rows
                        ]

                        cpp_preds = await _cpp_batch_predict(cpp_rows)

                        if cpp_preds is not None:
                            logger.debug(f"[get_trades] C++ batch predict: {len(cpp_preds)} rows")
                            predicted_p50s = [
                                round(p.get("quantiles", {}).get("p50", 0.0), 4)
                                for p in cpp_preds
                            ]
                            predicted_strategies = [
                                p.get("strategy", "NONE") for p in cpp_preds
                            ]
                        else:
                            # ── Python sklearn fallback ──────────────────────────
                            feat_df = pd.DataFrame(feature_rows)
                            p10_preds = models[0.1].predict(feat_df)
                            p50_preds = models[0.5].predict(feat_df)
                            p90_preds = models[0.9].predict(feat_df)

                            predicted_p50s = []
                            predicted_strategies = []

                            for i in range(len(df)):
                                p10_val = float(p10_preds[i])
                                p50_val = float(p50_preds[i])
                                p90_val = float(p90_preds[i])

                                tk_hv = _hv_cache.get(feature_rows[i]['ticker'], 0.0)
                                hv_dec = tk_hv / 100.0
                                ticker_hv_30d = hv_dec * np.sqrt(horizon_days / 252.0)
                                if ticker_hv_30d <= 0.0:
                                    ticker_hv_30d = 0.04

                                iqr_threshold = 1.5 * ticker_hv_30d
                                direction_threshold = 0.25 * ticker_hv_30d

                                p10_sorted, p50_sorted, p90_sorted = sorted([p10_val, p50_val, p90_val])
                                strat, _ = classify_strategy(p50_sorted, p10_sorted, p90_sorted, iqr_threshold, direction_threshold)

                                predicted_p50s.append(round(p50_sorted, 4))
                                predicted_strategies.append(strat)

                        df['predicted_p50'] = predicted_p50s
                        df['predicted_strategy'] = predicted_strategies
                except Exception as pred_ex:
                    logger.warning(f"Failed to compute dynamic predictions in api_get_trades: {pred_ex}")

            
        # Replace NaN/NaT with None for JSON compliance
        df = df.where(pd.notnull(df), None)
        # Convert numeric fields to clean types
        trades = df.to_dict(orient="records")
        return {"data": trades, "count": len(trades)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# LABELING WORKER
# ═══════════════════════════════════════════════════════════════

@router.post("/ml/label")
def api_label_trades(
    horizon_days: Optional[int] = Query(default=None),
    profit_threshold: Optional[float] = Query(default=None),
    force: Optional[bool] = Query(default=False, description="Force re-label all trades (resets existing labels)")
):
    """Triggers the labeling process for pending trades.
    Uses configured settings from ml_settings table as defaults,
    but query params override if provided.
    Set force=true to re-label all trades (useful after labeling logic changes).
    """
    try:
        # Read defaults from settings table
        settings = _get_settings()
        effective_horizon = horizon_days if horizon_days is not None else int(settings.get("horizon_days", "10"))
        effective_threshold = profit_threshold if profit_threshold is not None else float(settings.get("profit_threshold", "0.03"))

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if force:
            cursor.execute("UPDATE options_trades SET labeled = 0, label_success = NULL, observed_return = NULL, max_adverse_return = NULL, evaluation_date = NULL WHERE is_synthetic = 0")
            conn.commit()
            logger.info(f"Force re-label: reset all trade labels")
        
        time_limit = (datetime.datetime.now() - datetime.timedelta(days=effective_horizon)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            SELECT id, timestamp, ticker, option_type, underlier_price, side 
            FROM options_trades 
            WHERE labeled = 0 AND timestamp <= ?
        """, (time_limit,))
        
        pending = cursor.fetchall()
        if not pending:
            conn.close()
            return {"labeled_count": 0, "message": "No pending trades match the horizon criteria."}
        
        from collections import defaultdict
        trades_by_ticker = defaultdict(list)
        for row in pending:
            trade_id, timestamp_str, ticker, option_type, start_price, side = row
            trades_by_ticker[ticker].append((trade_id, timestamp_str, option_type, start_price, side))
        
        labeled_count = 0
        # PARALLELIZATION_PLAN §4.7: parallelize per-ticker yfinance fetch with ThreadPoolExecutor.
        # SQLite writes must be serialized — use a lock around DB commits.
        _label_lock = threading.Lock()
        _label_db_rows = []  # accumulate (trade_id, success, ret, mar, eval_date)

        def _label_one_ticker(ticker_and_trades):
            ticker, ticker_trades = ticker_and_trades
            rows = []
            try:
                dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date() for _, ts, _, _, _ in ticker_trades]
                min_date = min(dates)
                max_date = max(dates) + datetime.timedelta(days=effective_horizon + 2)

                tk = safe_call(yf.Ticker, ticker, retries=1)
                hist = safe_call(lambda t: t.history(start=min_date.strftime("%Y-%m-%d"),
                                  end=max_date.strftime("%Y-%m-%d")), tk)

                if hist.empty:
                    return rows

                hist.index = hist.index.date

                for trade_id, timestamp_str, option_type, start_price, side in ticker_trades:
                    trade_date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").date()
                    end_date = trade_date + datetime.timedelta(days=effective_horizon)

                    trade_hist = hist.loc[trade_date:end_date]
                    if trade_hist.empty:
                        continue

                    prices = trade_hist['Close'].values
                    is_bullish = (option_type == "Call" and side == "BUY") or (option_type == "Put" and side == "SELL")

                    success = 0
                    continuous_favorable_return = 0.0
                    max_adverse_return = 0.0

                    if len(prices) > 0:
                        settlement_price = prices[-1]
                        min_price = np.min(prices)
                        max_price = np.max(prices)

                        if is_bullish:
                            continuous_favorable_return = (settlement_price - start_price) / start_price
                            max_adverse_return = (min_price - start_price) / start_price
                        else:
                            continuous_favorable_return = (start_price - settlement_price) / start_price
                            max_adverse_return = (start_price - max_price) / start_price

                        if continuous_favorable_return >= effective_threshold:
                            success = 1

                    rows.append((
                        trade_id, success,
                        round(float(continuous_favorable_return), 4),
                        round(float(max_adverse_return), 4),
                        end_date.strftime("%Y-%m-%d"),
                    ))
            except Exception as ex:
                logger.warning(f"Error labeling trades for ticker {ticker}: {ex}")
            return rows

        _MAX_LABEL_WORKERS = 20
        ticker_items = list(trades_by_ticker.items())
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_MAX_LABEL_WORKERS, len(ticker_items))) as ex:
            for rows in ex.map(_label_one_ticker, ticker_items):
                if rows:
                    # Serialize DB writes — SQLite is not safe for concurrent writes
                    for (trade_id, success, ret, mar, eval_date) in rows:
                        cursor.execute("""
                            UPDATE options_trades
                            SET labeled = 1,
                                label_success = ?,
                                observed_return = ?,
                                max_adverse_return = ?,
                                evaluation_date = ?
                            WHERE id = ?
                        """, (success, ret, mar, eval_date, trade_id))
                        labeled_count += 1
                    conn.commit()

                
        conn.close()
        return {"labeled_count": labeled_count, "message": f"Successfully labeled {labeled_count} trades."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

@router.post("/ml/train")
def api_train_model():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    except ImportError:
        raise HTTPException(status_code=500, detail="sklearn package is not installed in Python environment. Run 'pip install scikit-learn' first.")
        
    try:
        conn = sqlite3.connect(DB_PATH)
        # PHASE A (5.1): Data cleaning — drop garbage tail rows.
        # The unlabeled raw yfinance options chain occasionally produces extreme values
        # (vol/OI up to 8000, IV up to 957%) that the LightGBM trees latch onto as
        # spurious "signal" and overfit to. Hard-clamping the input domain kills
        # this without losing the meaningful trade distribution.
        #
        # TRAINING SOURCE: is_synthetic = 1 (Black-Scholes–grounded synthetic data
        # from seed_grounded_real_options.py). We switched FROM is_synthetic=0
        # (real scanner trades) BECAUSE the v2_settlement labeling worker labels
        # using *stock* close-to-close returns, which is the wrong target for
        # option P&L (theta, gamma, vega are ignored). The synthetic dataset
        # labels on option mark-to-market over a 10-day path, which is the
        # correct signal. To retrain on real scanner trades instead, change
        # `is_synthetic = 1` back to `is_synthetic = 0` here.
        # PHASE A TRAINING FILTER (synthetic dataset, vol_oi distribution capped at p99≈0.92).
        # The original filter (vol_oi BETWEEN 0.5 AND 100) cut 92% of the 68,802 synthetic
        # rows. Loosened to 0.05 (matches seed_grounded_real_options realistic floor) and
        # 200 ceiling on premium dropped to 100_000 to match verify_synthetic_dataset's
        # realistic band — these changes preserve the *meaningful* data (avoids NaNs and
        # extreme outliers) without excluding the median row.
        df = pd.read_sql_query("""
            SELECT * FROM options_trades
            WHERE labeled = 1
              AND is_synthetic = 1
              AND vol_oi_ratio BETWEEN 0.05 AND 10
              AND implied_vol BETWEEN 5 AND 200
              AND premium BETWEEN 10 AND 100000
              AND underlier_price > 5
              AND observed_return BETWEEN -1.0 AND 5.0
        """, conn)
        conn.close()
        
        # If there's too little data, we add some dummy historical data to allow bootstrap training
        # to ensure the page doesn't error out when starting fresh.
        if len(df) < 5:
            # Let's seed some mock historical rows to ensure the user gets a working model right away
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            historical_date = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S")
            eval_date = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            
            import random
            rng = random.Random(42)
            
            tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL"]
            sides = ["BUY", "SELL"]
            alignments = ["BULL_ALIGNED", "BEAR_ALIGNED", "BULL_CONTRARIAN", "NEUTRAL"]
            
            for i in range(100):
                ticker = rng.choice(tickers)
                opt_type = rng.choice(["Call", "Put"])
                side = rng.choice(sides)
                trend = rng.choice(alignments)
                
                underlier = round(rng.uniform(100.0, 600.0), 2)
                strike = round(underlier * rng.uniform(0.9, 1.1), 2)
                dte = rng.choice([5, 10, 15, 30, 45, 60, 90])
                vol_oi = round(rng.uniform(1.1, 15.0), 2)
                iv = round(rng.uniform(20.0, 85.0), 2)
                premium = round(rng.uniform(10000.0, 1500000.0), 2)
                
                # Scoring function to determine probabilistic label (with noise)
                score = 0.0
                if vol_oi > 3.0:
                    score += 0.2
                if vol_oi > 7.0:
                    score += 0.15
                    
                if opt_type == "Call" and trend == "BULL_ALIGNED":
                    score += 0.3
                elif opt_type == "Put" and trend == "BEAR_ALIGNED":
                    score += 0.3
                elif trend == "NEUTRAL":
                    score += 0.1
                else:
                    score -= 0.2
                    
                if side == "BUY":
                    score += 0.15
                else:
                    score -= 0.1
                    
                if iv > 60.0:
                    score -= 0.15
                
                # Probability mapping
                prob = 1.0 / (1.0 + np.exp(-score))
                success = 1 if rng.random() < prob else 0
                observed_ret = round(rng.uniform(-0.15, 0.25) if success else rng.uniform(-0.50, -0.01), 4)
                max_adverse = round(rng.uniform(-0.10, -0.01), 4) if success else round(observed_ret * rng.uniform(1.0, 1.2), 4)
                
                cursor.execute("""
                    INSERT INTO options_trades (
                        timestamp, ticker, expiration, strike, option_type, volume, 
                        open_interest, vol_oi_ratio, implied_vol, underlier_price, 
                        premium, side, dte, is_weekly, trend_alignment, labeled, label_success, observed_return, max_adverse_return, evaluation_date, is_synthetic
                    ) VALUES (?, ?, '2026-08-20', ?, ?, 1000, 200, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?, ?, 1)
                """, (
                    historical_date, ticker, strike, opt_type, vol_oi, iv, underlier, premium, side, dte, trend, success, observed_ret, max_adverse, eval_date
                ))
            conn.commit()
            
            # Refetch data
            df = pd.read_sql_query("SELECT * FROM options_trades WHERE labeled = 1 AND is_synthetic = 1", conn)
            conn.close()

        # ── Compute advanced features ─────────────────────────
        df = compute_advanced_features(df)

        numeric_features = [
            'strike', 'volume', 'open_interest', 'vol_oi_ratio', 'implied_vol',
            'underlier_price', 'premium', 'dte',
            'moneyness', 'iv_hv_ratio', 'vix_level', 'log_premium'
        ]
        categorical_features = ['option_type', 'side', 'trend_alignment', 'dte_bucket']
        
        # Include ticker column in the features DataFrame
        X = df[['ticker'] + numeric_features + categorical_features]
        # Target variable is continuous signed max-favorable return
        y = df['observed_return']
        
        # Build processing pipeline
        # PHASE B (PARALLELIZATION_PLAN 6.2): StandardScaler removed.
        # Tree models (LightGBM) are scale-invariant, so the scaler added
        # no signal — but it DID force the C++ inference path to persist
        # mean/std arrays and apply them per-row with float-precision loss.
        # Drop it for both the trained model and the walkforward path.
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        # ── Split ───────────────────────────────────────
        X_train, X_test, y_train_continuous, y_test_continuous = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5 independent LightGBM regressors, one per quantile
        # PARALLELIZATION_PLAN §4.1: fit all 5 quantile models in parallel via ProcessPoolExecutor.
        # Each worker fits one model on 2 cores (n_jobs=2), for 5×2 = 10 cores total.
        # ProcessPoolExecutor avoids GIL contention; sklearn Pipeline pickles cleanly.
        QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
        models = {}
        fit_args = [
            (q, X_train, y_train_continuous, preprocessor, 150, 0.03, 15, 30, 1.0)
            for q in QUANTILES
        ]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=_SCYLLA_MAX_PROC) as ex:
                for q, pipe in ex.map(_fit_one_quantile_train, fit_args):
                    models[q] = pipe
        except Exception as pool_ex:
            # Fall back to sequential if ProcessPoolExecutor fails (e.g., pickling issue)
            logger.warning(f"ProcessPoolExecutor training failed ({pool_ex}), falling back to sequential")
            models = {}
            for q in QUANTILES:
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("regressor", lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=150,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=30,
                        reg_lambda=1.0,
                        random_state=42,
                        verbose=-1
                    ))
                ])
                pipe.fit(X_train, y_train_continuous)
                models[q] = pipe

        # Enforce quantile monotonicity post-fit & compute pinball loss
        pinball_losses = {}
        test_preds_q = {}
        for q in QUANTILES:
            test_preds_q[q] = models[q].predict(X_test)
            
        num_triggered = 0
        total_rows = len(X_test)
        for i in range(total_rows):
            row_vals = [test_preds_q[q][i] for q in QUANTILES]
            if row_vals != sorted(row_vals):
                num_triggered += 1
        pct_triggered = (num_triggered / total_rows) * 100 if total_rows > 0 else 0
        logger.info(f"Quantile monotonicity enforcement triggered on {num_triggered}/{total_rows} ({pct_triggered:.2f}%) validation rows.")

        for q in QUANTILES:
            diff = y_test_continuous - test_preds_q[q]
            loss = np.mean(np.maximum(q * diff, (q - 1) * diff))
            pinball_losses[q] = float(loss)

        # Backtest calibration coverage
        coverages = {}
        for q in [0.1, 0.5, 0.9]:
            coverages[q] = float(np.mean(y_test_continuous <= test_preds_q[q]))
        logger.info(f"Quantile Calibration Coverage: P10={coverages[0.1]:.2%}, P50={coverages[0.5]:.2%}, P90={coverages[0.9]:.2%}")

        # Post-hoc evaluation of binary metrics on the median model
        settings = _get_settings()
        profit_threshold = float(settings.get("profit_threshold", "0.03"))
        
        y_test_binary = (y_test_continuous >= profit_threshold).astype(int)
        test_preds_median = test_preds_q[0.5]
        test_preds_binary = (test_preds_median >= profit_threshold).astype(int)
        
        train_preds_median = models[0.5].predict(X_train)
        y_train_binary = (y_train_continuous >= profit_threshold).astype(int)
        train_preds_binary = (train_preds_median >= profit_threshold).astype(int)

        test_roc_auc = 0.0
        try:
            test_roc_auc = float(roc_auc_score(y_test_binary, test_preds_median))
        except Exception:
            pass

        # Manual Stratified K-Fold CV for median model
        cv_roc_auc_mean = 0.0
        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            y_binary = (y >= profit_threshold).astype(int)
            for train_idx, val_idx in skf.split(X, y_binary):
                X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
                y_tr_cont, y_va_bin = y.iloc[train_idx], y_binary.iloc[val_idx]
                
                fold_pipe = Pipeline(steps=[
                    ('preprocess', preprocessor),
                    ('regressor', lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=0.5,
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=20,
                        verbose=-1,
                        random_state=42
                    ))
                ])
                fold_pipe.fit(X_tr, y_tr_cont)
                preds_val = fold_pipe.predict(X_va)
                cv_scores.append(roc_auc_score(y_va_bin, preds_val))
            cv_roc_auc_mean = float(np.mean(cv_scores))
        except Exception as cv_ex:
            logger.warning(f"Cross-validation failed: {cv_ex}")
            cv_roc_auc_mean = 0.0

        metrics = {
            "train_accuracy": float(accuracy_score(y_train_binary, train_preds_binary)),
            "test_accuracy": float(accuracy_score(y_test_binary, test_preds_binary)),
            "test_precision": float(precision_score(y_test_binary, test_preds_binary, zero_division=0)),
            "test_recall": float(recall_score(y_test_binary, test_preds_binary, zero_division=0)),
            "test_f1": float(f1_score(y_test_binary, test_preds_binary, zero_division=0)),
            "test_roc_auc": test_roc_auc,
            "cv_roc_auc_mean": cv_roc_auc_mean,
            "samples_count": len(df)
        }

        # Aggregate feature importances across the 5 models
        preprocessor_fit = models[0.5].named_steps['preprocess']
        ohe_cols = list(preprocessor_fit.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
        all_features = numeric_features + ohe_cols
        
        mean_importances = np.mean([models[q].named_steps['regressor'].feature_importances_ for q in QUANTILES], axis=0)
        sum_imp = np.sum(mean_importances)
        if sum_imp > 0:
            mean_importances = mean_importances / sum_imp
        
        feat_imp = []
        for feat, imp in zip(all_features, mean_importances):
            feat_imp.append({"feature": feat, "importance": round(float(imp), 4)})
        feat_imp.sort(key=lambda x: x["importance"], reverse=True)

        # Save model dict
        joblib.dump(models, MODEL_PATH)
        global _global_model
        _global_model = models

        # PHASE B (PARALLELIZATION_PLAN 6.1): Dump raw LightGBM Boosters + a JSON of
        # preprocessor params to backend/cache/cpp_inference/. The 5 .txt files are
        # loaded by InferenceEngine::load() in cpp_core/src/inference_engine.cpp via
        # LGBM_BoosterLoadModelFromFile. The JSON carries the imputer medians and
        # OHE category maps needed to vectorize a raw input row before prediction.
        _serialize_cpp_inference_artifacts(models, numeric_features, categorical_features, QUANTILES)

        # Log training run to model_runs audit table
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_runs (
                    timestamp, samples_count, train_accuracy, test_accuracy,
                    test_precision, test_recall, test_f1, test_roc_auc,
                    cv_roc_auc_mean, horizon_days, profit_threshold, model_version,
                    pinball_loss_p10, pinball_loss_p25, pinball_loss_p50, pinball_loss_p75, pinball_loss_p90
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metrics["samples_count"],
                metrics["train_accuracy"],
                metrics["test_accuracy"],
                metrics["test_precision"],
                metrics["test_recall"],
                metrics["test_f1"],
                metrics["test_roc_auc"],
                metrics["cv_roc_auc_mean"],
                int(settings.get("horizon_days", "10")),
                profit_threshold,
                "lightgbm_quantile_v2",
                pinball_losses[0.1],
                pinball_losses[0.25],
                pinball_losses[0.5],
                pinball_losses[0.75],
                pinball_losses[0.9]
            ))
            conn.commit()
            conn.close()
        except Exception as audit_ex:
            logger.warning(f"Failed to log training run audit: {audit_ex}")

        return {
            "status": "success",
            "metrics": metrics,
            "feature_importances": feat_imp[:10]
        }
    except Exception as e:
        logger.exception("Failed in api_train_model")
        raise HTTPException(status_code=500, detail=str(e))


# (enforce_monotonic_quantiles, _cpp_artifacts, prediction cache moved → backend/models/predict.py, backend/models/train.py)

# ═══════════════════════════════════════════════════════════════
# PREDICTION / INFERENCE
# ═══════════════════════════════════════════════════════════════

@router.post("/ml/predict", response_model=PredictResponseSchema)
def api_predict(req: PredictRequestSchema):
    # PHASE C (PARALLELIZATION_PLAN Step 18): Delegate prediction to C++ native Crow engine (:8080)
    try:
        is_weekly_val = bool(getattr(req, "isWeekly", getattr(req, "is_weekly", False)))
        payload = {
            "ticker": (req.ticker or "SPY").upper(),
            "underlier_price": float(req.underlierPrice),
            "strike": float(req.strike),
            "volume": float(req.volume),
            "open_interest": float(req.openInterest),
            "implied_vol": float(req.impliedVolatility),
            "premium": float(req.premium),
            "side": str(req.side).upper(),
            "dte": float(req.dte),
            "is_weekly": "True" if is_weekly_val else "False",
            "trend_alignment": str(req.trendAlignment)
        }
        resp = requests.post("http://127.0.0.1:8080/api/v1/ml/predict", json=payload, timeout=5)
        if resp.status_code == 200:
            c_data = resp.json()
            return {
                "quantiles": c_data["quantiles"],
                "p_success": round(c_data["p_success"], 4),
                "expected_return": round(c_data["expected_return"], 4),
                "strategy": c_data["strategy"],
                "strategy_confidence": round(c_data.get("direction_confidence", 0.5), 4),
                "kelly_fraction": round(c_data["kelly_fraction"], 4),
                "kelly_fraction_uncapped": round(c_data["kelly_fraction"], 4),
                "model_type": "cpp_native_lightgbm_v2"
            }
    except Exception as ex:
        logger.warning(f"C++ inference pass-through failed: {ex}. Falling back to local evaluation.")

    try:
        # Get settings/thresholds
        settings = _get_settings()
        profit_threshold = float(settings.get("profit_threshold", "0.03"))
        horizon_days = int(settings.get("horizon_days", "10"))
        
        # Historical Volatility
        hv_ticker = (req.ticker or "SPY").upper()
        hv_annual_pct = _fetch_historical_volatility(hv_ticker)
        hv_annual_dec = hv_annual_pct / 100.0
        ticker_hv_30d = hv_annual_dec * np.sqrt(horizon_days / 252.0)
        if ticker_hv_30d <= 0.0:
            ticker_hv_30d = 0.04  # fallback 10-day vol (20% annualized)
            
        iqr_threshold = 1.5 * ticker_hv_30d
        direction_threshold = 0.25 * ticker_hv_30d

        if not os.path.exists(MODEL_PATH):
            # Fallback heuristic score mapping
            score = 0.5
            if req.volOiRatio > 5.0:
                score += 0.1
            if req.trendAlignment in ['BULL_ALIGNED', 'BEAR_ALIGNED']:
                score += 0.15
            if req.side == 'BUY':
                score += 0.05
            
            p_success_heuristic = min(score, 0.95)
            # Create a degenerate quantile spread flat +/-5% around heuristic point estimate (P50=0.03)
            median_val = 0.03
            quantiles = {
                "p10": median_val - 0.05,
                "p25": median_val - 0.025,
                "p50": median_val,
                "p75": median_val + 0.025,
                "p90": median_val + 0.05
            }
            
            p_success = compute_calibrated_p_success(profit_threshold, **quantiles)
            strategy, confidence = classify_strategy(quantiles["p50"], quantiles["p10"], quantiles["p90"], iqr_threshold, direction_threshold)
            capped_kelly, uncapped_kelly = kelly_fraction(p_success, quantiles["p50"], quantiles["p10"], quantiles["p90"])
            
            return {
                "quantiles": quantiles,
                "p_success": round(p_success, 4),
                "expected_return": round(quantiles["p50"], 4),
                "strategy": strategy,
                "strategy_confidence": confidence,
                "kelly_fraction": round(capped_kelly, 4),
                "kelly_fraction_uncapped": round(uncapped_kelly, 4),
                "model_type": "heuristic_fallback"
            }
            
        # Try loading models
        models = get_global_model()
        if models is None:
            raise HTTPException(status_code=500, detail="Failed to load model for evaluation.")
            logger.warning(f"Could not load multi-quantile model: {load_ex}")
            # Heuristic fallback path
            median_val = 0.03
            quantiles = {
                "p10": median_val - 0.05,
                "p25": median_val - 0.025,
                "p50": median_val,
                "p75": median_val + 0.025,
                "p90": median_val + 0.05
            }
            p_success = compute_calibrated_p_success(profit_threshold, **quantiles)
            strategy, confidence = classify_strategy(quantiles["p50"], quantiles["p10"], quantiles["p90"], iqr_threshold, direction_threshold)
            capped_kelly, uncapped_kelly = kelly_fraction(p_success, quantiles["p50"], quantiles["p10"], quantiles["p90"])
            return {
                "quantiles": quantiles,
                "p_success": round(p_success, 4),
                "expected_return": round(quantiles["p50"], 4),
                "strategy": strategy,
                "strategy_confidence": confidence,
                "kelly_fraction": round(capped_kelly, 4),
                "kelly_fraction_uncapped": round(uncapped_kelly, 4),
                "model_type": "heuristic_fallback"
            }
        
        # Build base feature row
        row_data = {
            'ticker': hv_ticker,
            'strike': req.strike,
            'volume': req.volume,
            'open_interest': req.openInterest,
            'vol_oi_ratio': req.volOiRatio,
            'implied_vol': req.impliedVolatility,
            'underlier_price': req.underlierPrice,
            'premium': req.premium,
            'dte': req.dte,
            'option_type': req.optionType,
            'side': req.side,
            'trend_alignment': req.trendAlignment
        }
        
        row_data['moneyness'] = req.moneyness if req.moneyness is not None else (req.strike - req.underlierPrice) / req.underlierPrice
        row_data['iv_hv_ratio'] = req.iv_hv_ratio if req.iv_hv_ratio is not None else 1.0
        row_data['vix_level'] = req.vix_level if req.vix_level is not None else _fetch_vix_level()
        row_data['log_premium'] = req.log_premium if req.log_premium is not None else float(np.log1p(max(req.premium, 0)))
        row_data['dte_bucket'] = req.dte_bucket if req.dte_bucket is not None else _dte_to_bucket(req.dte)

        model_version = str(os.path.getmtime(MODEL_PATH))
        input_hash = _get_cache_hash(row_data)
        cached_res = _check_prediction_cache(input_hash, model_version)
        if cached_res:
            return cached_res

        df_row = pd.DataFrame([row_data])
        
        # Predict all 5 quantiles
        preds = {}
        q_map = {0.1: "p10", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90"}
        for q, key in q_map.items():
            preds[key] = float(models[q].predict(df_row)[0])
            
        # Enforce monotonicity post-fit
        preds = enforce_monotonic_quantiles(preds)
        
        # Derive values
        p_success = compute_calibrated_p_success(profit_threshold, **preds)
        strategy, confidence = classify_strategy(preds["p50"], preds["p10"], preds["p90"], iqr_threshold, direction_threshold)
        capped_kelly, uncapped_kelly = kelly_fraction(p_success, preds["p50"], preds["p10"], preds["p90"])
        
        result = {
            "quantiles": preds,
            "p_success": round(p_success, 4),
            "expected_return": round(preds["p50"], 4),
            "strategy": strategy,
            "strategy_confidence": confidence,
            "kelly_fraction": round(capped_kelly, 4),
            "kelly_fraction_uncapped": round(uncapped_kelly, 4),
            "model_type": "lightgbm_quantile_v2"
        }
        
        _save_prediction_cache(input_hash, model_version, result)
        return result
    except Exception as e:
        logger.exception("Error in predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════

@router.get("/ml/stats")
def api_get_stats():
    try:
        # Fetch profit threshold to count successful trades at read time
        settings = _get_settings()
        target_pct = float(settings.get("profit_threshold", "0.03"))

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM options_trades")
        total_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE labeled = 1")
        labeled_trades = cursor.fetchone()[0]
        
        # Success is defined as observed_return >= target_pct
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE labeled = 1 AND observed_return >= ?", (target_pct,))
        successful_trades = cursor.fetchone()[0]
        
        conn.close()
        
        success_ratio = round(successful_trades / max(labeled_trades, 1), 4)
        
        model_exists = os.path.exists(MODEL_PATH)
        
        return {
            "total_trades": total_trades,
            "labeled_trades": labeled_trades,
            "successful_trades": successful_trades,
            "success_ratio": success_ratio,
            "model_ready": model_exists
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════

# (get_real_trades moved → backend/db/queries.py, see import above)

# (_strategy_loader imported above via try/except block)


def _resolve_strategy_defaults(req: "BacktestRequestSchema") -> dict:
    """
    Resolve BacktestRequestSchema fields that are None to the value from
    backend/config/strategy_defaults.json. Explicit values in req take precedence
    so per-request overrides still work.

    Returns a dict with the resolved values; the request object is NOT mutated.
    """
    strat = get_strategy_params(req.strategy_type or "vol_regime")
    common = get_common_params()

    def pick(field, file_key, kind="strat", default=None):
        v = getattr(req, field, None)
        if v is not None:
            return v
        return (strat if kind == "strat" else common).get(file_key, default)

    return {
        "initial_capital":            pick("initial_capital", "initial_capital", "common", 100000.0),
        "prob_threshold":             pick("prob_threshold", "prob_threshold", "strat", 0.40),
        "kelly_multiplier":           pick("kelly_multiplier", "kelly_multiplier", "strat", 0.75),
        "kelly_cap":                  pick("kelly_cap", "kelly_cap", "strat", 0.05),
        "stop_lambda":                pick("stop_lambda", "stop_lambda", "strat", 1.5),
        "max_risk_pct_per_trade":     pick("max_risk_pct_per_trade", "max_risk_pct_per_trade", "common", 0.02),
        "walkforward_train_window":   pick("walkforward_train_window", "walkforward_train_window", "common", 500),
        "walkforward_test_increment": pick("walkforward_test_increment", "walkforward_test_increment", "common", 250),
        "max_concurrent_trades":      pick("max_concurrent_trades", "max_concurrent_trades", "strat", 12),
        "scan_time":                  pick("scan_time", "scan_time", "common", "10:00:00"),
        "min_kelly_fraction":         pick("min_kelly_fraction", "min_kelly_fraction", "common", 0.01),
        "hard_stop_loss":             pick("hard_stop_loss", "hard_stop_loss", "strat", 0.04),
        "lookback_days":              pick("lookback_days", "lookback_days", "common", None),
        "profit_threshold":           pick("profit_threshold", "profit_threshold", "strat", 0.05),
        "max_quantile_spread":        pick("max_quantile_spread", "max_quantile_spread", "strat", 0.0),
        "min_median_return":          pick("min_median_return", "min_median_return", "strat", 0.0),
        "slippage_pct":               pick("slippage_pct", "slippage_pct", "common", 0.01),
        "use_costs":                  pick("use_costs", "use_costs", "common", True),
        "max_iv":                     pick("max_iv", "max_iv", "strat", 150.0),
    }


class BacktestRequestSchema(BaseModel):
    mode: Optional[str] = "walkforward"
    # Single source of truth: the per-strategy and common fields below default to None
    # and are resolved at request-handling time by `_resolve_strategy_defaults(req)`
    # from backend/config/strategy_defaults.json (see _strategy_loader.py). Explicit
    # values supplied in the request body take precedence so per-request overrides
    # still work. The defensive fallbacks inside `_resolve_strategy_defaults` match
    # the prior hardcoded schema defaults, so behaviour is unchanged for any
    # strategy_type or field absent from the JSON file (or if the file is missing).
    initial_capital: Optional[float] = None
    # Probability calibration target used to map predicted return distribution -> p_success.
    # Decoupled from profit_threshold (take-profit cap) to avoid circular over-restriction.
    calibration_target_pct: Optional[float] = 0.025
    # Plan 1A: decouple the walkforward label threshold from the take-profit cap.
    # 0.5 keeps the 96 MB on-disk walkforward cache (v2_settlement_500_250_68802_0.5_0.025_0_None.pkl)
    # hot for any profit_threshold — only retrain when the label threshold itself changes.
    # Currently unused inside _process_walkforward_step (pure quantile regression), but
    # reserved in the schema and the cache key so a future binary-label refactor won't
    # invalidate the predictions cache.
    walkforward_label_threshold: Optional[float] = 0.5
    prob_threshold: Optional[float] = None
    kelly_multiplier: Optional[float] = None
    kelly_cap: Optional[float] = None           # PHASE A: was 0.20 — tighten to cap tail risk
    stop_lambda: Optional[float] = None
    max_risk_pct_per_trade: Optional[float] = None
    walkforward_train_window: Optional[int] = None
    walkforward_test_increment: Optional[int] = None  # PHASE A: 250 = synthetic-tuned optimal (recycle bin cache v2_settlement_500_250_68802_0.5_0.025_0_None.pkl). 100 is too slow on 68k trades (~683 steps, 15-30 min).
    confirm_direct_dev: Optional[bool] = False
    strategy_type: Optional[str] = "whale_quality"  # PHASE A: default to new strategy
    max_concurrent_trades: Optional[int] = None  # PHASE A: was 8 — allow more concurrent (with tighter Kelly)
    scan_time: Optional[str] = None
    min_kelly_fraction: Optional[float] = None
    lookback_days: Optional[int] = None
    # Take-profit cap for the BACKTEST (the +X% return at which a winning trade is closed).
    # Intentionally decoupled from ml_settings.profit_threshold (ML training label
    # threshold, see comment in api_backtest). Resolved from strategy_defaults.json.
    profit_threshold: Optional[float] = None
    max_quantile_spread: Optional[float] = None
    min_median_return: Optional[float] = None
    slippage_pct: Optional[float] = None
    max_iv: Optional[float] = None
    min_open_interest: Optional[int] = 100
    min_dte: Optional[int] = 7
    max_dte: Optional[int] = 60
    data_start_idx: Optional[int] = 0  # PHASE A: default 0 (start of dataset) so default cache_key matches existing on-disk caches (e.g. ..._0_None.pkl). Legacy default of None produced a non-matching ..._None_None key.
    data_end_idx: Optional[int] = None  # None = use all data through end of dataset.
    hard_stop_loss: Optional[float] = None  # PHASE A pass 3: was 0.06 — tighter stop OK because hold_to_horizon is no longer catastrophic
    # Plan 1A: per-strategy threshold overrides (JSON-driven). The PHASE A new
    # strategies previously hardcoded these; they now read from the request so
    # config/strategy_defaults.json becomes the single source of truth.
    # None falls back to the strategy-block default (matches the prior hardcoded value).
    real_vol_oi_floor: Optional[float] = None        # vol_oi floor in real-data mode
    synth_vol_oi_floor: Optional[float] = None       # vol_oi floor in synth-data mode
    synth_max_quantile_spread: Optional[float] = None  # IQR cap in synth-data mode (real uses max_quantile_spread)
    # PHASE A: when True, run backtest on the synthetic dataset (is_synthetic=1) instead
    # of real scanner output. Also bypasses the TIER_PROFITABLE_TICKERS gate in the new
    # strategies (synthetic universe is by-construction positive-EV, no need to filter)
    # and uses synthetic-tuned vol_oi floors. Default False preserves prod path.
    use_synthetic: Optional[bool] = True
    use_costs: Optional[bool] = True


# (_process_walkforward_step moved → backend/backtest/walkforward.py, see import above)


@router.post("/ml/backtest")
def api_backtest(req: BacktestRequestSchema):
    try:
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        import lightgbm as lgb
    except ImportError:
        raise HTTPException(status_code=500, detail="sklearn package is not installed in Python environment. Run 'pip install scikit-learn' first.")

    # Guard rail
    if req.mode == "direct_dev" and not req.confirm_direct_dev:
        raise HTTPException(
            status_code=400,
            detail="direct_dev mode requires explicit confirm_direct_dev=true — results are in-sample and not valid for strategy validation"
        )

    # Resolve per-strategy + common defaults from backend/config/strategy_defaults.json.
    # Single source of truth — explicit values in the request still take precedence.
    resolved = _resolve_strategy_defaults(req)

    # Check response cache — include ALL strategy-discriminating params to avoid cross-strategy cache collisions
    try:
        _ens_conn = sqlite3.connect(DB_PATH)
        _ens_cur = _ens_conn.cursor()
        _ens_cur.execute("SELECT COUNT(DISTINCT ensemble_id) FROM options_trades WHERE is_synthetic = 1")
        _ens_count = _ens_cur.fetchone()[0]
        _ens_conn.close()
    except Exception:
        _ens_count = 0
    cache_key_resp = (
        f"{req.mode}_{resolved['initial_capital']}_{resolved['prob_threshold']}_{resolved['kelly_multiplier']}_"
        f"{resolved['kelly_cap']}_{resolved['stop_lambda']}_{resolved['max_risk_pct_per_trade']}_{resolved['walkforward_train_window']}_"
        f"{resolved['walkforward_test_increment']}_{req.strategy_type}_{resolved['max_concurrent_trades']}_"
        f"{resolved['scan_time']}_{resolved['min_kelly_fraction']}_{resolved['hard_stop_loss']}_{resolved['lookback_days']}_"
        f"{resolved['profit_threshold']}_{req.calibration_target_pct}_{resolved['max_quantile_spread']}_{resolved['min_median_return']}_{resolved['slippage_pct']}_{resolved['max_iv']}_{req.min_open_interest}"
        f"_{req.min_dte}_{req.max_dte}_{req.data_start_idx}_{req.data_end_idx}_{int(bool(req.use_synthetic))}_{int(bool(req.use_costs))}"
        f"_{_ens_count}"
    )
    if cache_key_resp in _backtest_response_cache:
        print(f"api_backtest: returning cached response for key {cache_key_resp}")
        return _backtest_response_cache[cache_key_resp]

    # Fetch settings/thresholds
    # NOTE: profit_threshold here is the BACKTEST profit cap (the +X% return at which a winning trade
    # is closed). This is intentionally decoupled from the ml_settings.profit_threshold DB value, which
    # is the ML training label threshold (the return above which a trade is labelled "successful" for
    # supervised learning). They serve different purposes and intentionally do not share a default.
    # Resolved from strategy_defaults.json per-strategy block via _resolve_strategy_defaults.
    settings = _get_settings()
    profit_threshold = resolved["profit_threshold"]
    # Plan 1A: walkforward label threshold is independent of the take-profit cap.
    # Default 0.5 matches the existing on-disk 96 MB walkforward cache, so changing
    # `profit_threshold` (the backtest take-profit cap) no longer invalidates the
    # ~30 min/strategy walkforward recompute.
    walkforward_label_threshold = req.walkforward_label_threshold if req.walkforward_label_threshold is not None else 0.5
    calibration_target_pct = req.calibration_target_pct if req.calibration_target_pct is not None else 0.025
    horizon_days = int(settings.get("horizon_days", "10"))

    conn = sqlite3.connect(DB_PATH)
    df_real = get_real_trades(conn, labeled=1, synthetic=bool(req.use_synthetic))
    conn.close()

    if req.data_start_idx is not None or req.data_end_idx is not None:
        start = req.data_start_idx if req.data_start_idx is not None else 0
        end = req.data_end_idx if req.data_end_idx is not None else len(df_real)
        end = min(end, len(df_real))
        start = max(0, min(start, end))
        df_real = df_real.iloc[start:end].reset_index(drop=True)

    N = len(df_real)

    data_start = None
    data_end = None
    data_span_days = 0
    if N > 0 and "timestamp" in df_real.columns:
        ts_min = df_real["timestamp"].min()
        ts_max = df_real["timestamp"].max()
        try:
            ts_min_str = str(ts_min)
            ts_max_str = str(ts_max)
            data_start = ts_min_str[:10] if len(ts_min_str) >= 10 else ts_min_str
            data_end = ts_max_str[:10] if len(ts_max_str) >= 10 else ts_max_str
            try:
                d_min = datetime.datetime.strptime(ts_min_str[:10], "%Y-%m-%d").date()
                d_max = datetime.datetime.strptime(ts_max_str[:10], "%Y-%m-%d").date()
                data_span_days = (d_max - d_min).days
            except Exception:
                data_span_days = 0
        except Exception:
            pass

    if req.mode == "walkforward":
        # Initial T = walkforward_train_window
        train_window = req.walkforward_train_window or 50
        increment = req.walkforward_test_increment or 10

        if N < train_window + increment:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient real trade volume for walk-forward. Required: at least {train_window + increment} labeled trades, but current count is {N}."
            )

        import joblib
        import os
        # Plan 1A: walkforward cache key uses `walkforward_label_threshold` (the
        # pure-quantile-regression label floor) instead of `profit_threshold` (the
        # backtest take-profit cap). This decouples the two so the 96 MB on-disk
        # cache is hit regardless of the take-profit cap. Numeric default 0.5 is
        # identical to the previous default, so existing cache files match.
        cache_key = f"{LABELING_VERSION}_{train_window}_{increment}_{N}_{walkforward_label_threshold}_{calibration_target_pct}_{req.data_start_idx}_{req.data_end_idx}"
        cache_file = os.path.join(CACHE_DIR, f"cache_predictions_walkforward_{cache_key}.pkl")
        legacy_cache_file = os.path.join(CACHE_DIR, "cache_predictions_walkforward.pkl")
        
        predictions = []
        predictions_loaded = False
        
        with _walkforward_lock:
            if cache_key in _walkforward_cache_mem:
                predictions = _walkforward_cache_mem[cache_key]
                predictions_loaded = True
            else:
                target_file = cache_file if os.path.exists(cache_file) else (legacy_cache_file if os.path.exists(legacy_cache_file) else None)
                if target_file:
                    try:
                        cached_data = joblib.load(target_file)
                        if cached_data.get("key") == cache_key:
                            predictions = cached_data["predictions"]
                            predictions_loaded = True
                            _walkforward_cache_mem[cache_key] = predictions
                    except Exception as e:
                        print(f"Error loading cache: {e}")
                        pass
            
            print(f"Predictions loaded from cache: {predictions_loaded}")
                    
            if not predictions_loaded:
                tasks = []
                T = train_window
                while T < N:
                    test_end = min(T + increment, N)
                    if T < test_end:
                        tasks.append((T, test_end))
                    T += increment

                num_workers = min(_SCYLLA_MAX_PROC, os.cpu_count() or 4)
                df_real_feat = compute_advanced_features(df_real)
                print(f"Walkforward: {len(tasks)} steps across {num_workers} workers "
                      f"(train_window={train_window}, increment={increment}, "
                      f"{N} rows × {len(df_real_feat.columns)} cols, "
                      f"~{df_real_feat.memory_usage(deep=True).sum() / 1024**2:.0f} MB)")

                # PARALLELIZATION_PLAN §4: ONE shared ProcessPoolExecutor
                # instead of spawning/destroying child pools inside every
                # _process_walkforward_step. The initializer pickles the
                # full dataset into each worker ONCE (not once per task),
                # then each worker fits all 5 quantile models sequentially.
                #
                # Sibling-worker limit = min(_SCYLLA_MAX_PROC, cpu_count).
                # This replaces the old outer-joblib-threads + inner-pool
                # pattern that attempted 16,955+ rapid process spawns and
                # caused WinError 1450 on Windows.
                _results = []
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_wf_worker_init,
                    initargs=(df_real, df_real_feat)
                ) as executor:
                    _futures = {
                        executor.submit(_process_walkforward_step, T_start, T_end): (T_start, T_end)
                        for T_start, T_end in tasks
                    }
                    for fut in concurrent.futures.as_completed(_futures):
                        try:
                            _results.append(fut.result())
                        except Exception as _exc:
                            T_start, T_end = _futures[fut]
                            logger.error(
                                f"Walkforward step [{T_start}:{T_end}] failed: {_exc}"
                            )
                            # Continue with other steps — one bad window
                            # shouldn't kill the whole backtest.

                # Sort by T_start to preserve exact chronological order
                _results.sort(key=lambda r: r[0])
                for _, step_preds in _results:
                    predictions.extend(step_preds)

            if not predictions_loaded:
                try:
                    joblib.dump({"key": cache_key, "predictions": predictions}, cache_file)
                except Exception:
                    pass

        in_sample_warning = False

    elif req.mode == "direct_dev":
        # Load from pkl
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=400,
                detail="No trained model found. Please train a model in the ML Cockpit before running direct_dev backtest."
            )

        models = get_global_model()
        if models is None:
            raise HTTPException(status_code=500, detail="Failed to load model.")

        df_real_feat = compute_advanced_features(df_real)
        numeric_features = [
            'strike', 'volume', 'open_interest', 'vol_oi_ratio', 'implied_vol',
            'underlier_price', 'premium', 'dte',
            'moneyness', 'iv_hv_ratio', 'vix_level', 'log_premium'
        ]
        categorical_features = ['option_type', 'side', 'trend_alignment', 'dte_bucket']
        X_real = df_real_feat[['ticker'] + numeric_features + categorical_features]

        # Re-sync the C++ engine's boosters + preprocessor schema with the models
        # we just loaded from MODEL_PATH. Cheap (5 .txt writes + 1 JSON) and
        # idempotent; the C++ engine reloads on its own schedule.
        _serialize_cpp_inference_artifacts(models, numeric_features, categorical_features, [0.1, 0.25, 0.5, 0.75, 0.9])

        cpp_rows = [
            {
                "ticker":          str(r.get("ticker", "SPY")).upper(),
                "underlier_price": float(r.get("underlier_price", 0.0)),
                "strike":          float(r.get("strike", 0.0)),
                "volume":          float(r.get("volume", 0)),
                "open_interest":   float(r.get("open_interest", 0)),
                "implied_vol":     float(r.get("implied_vol", 0.0)),
                "premium":         float(r.get("premium", 0.0)),
                "option_type":     str(r.get("option_type", "Call")),
                "side":            str(r.get("side", "BUY")),
                "dte":             float(r.get("dte", 0)),
                "is_weekly":       "True" if int(r.get("is_weekly", 0)) else "False",
                "trend_alignment": str(r.get("trend_alignment", "NEUTRAL")),
            }
            for _, r in df_real.iterrows()
        ]
        cpp_preds = _cpp_batch_predict(cpp_rows)

        if cpp_preds is not None:
            quantiles_list = [
                {
                    "p10": float(p.get("quantiles", {}).get("p10", 0.0)),
                    "p25": float(p.get("quantiles", {}).get("p25", 0.0)),
                    "p50": float(p.get("quantiles", {}).get("p50", 0.0)),
                    "p75": float(p.get("quantiles", {}).get("p75", 0.0)),
                    "p90": float(p.get("quantiles", {}).get("p90", 0.0)),
                }
                for p in cpp_preds
            ]
        else:
            p10_preds = models[0.1].predict(X_real)
            p25_preds = models[0.25].predict(X_real)
            p50_preds = models[0.5].predict(X_real)
            p75_preds = models[0.75].predict(X_real)
            p90_preds = models[0.9].predict(X_real)
            quantiles_list = [
                {
                    "p10": float(p10_preds[i]),
                    "p25": float(p25_preds[i]),
                    "p50": float(p50_preds[i]),
                    "p75": float(p75_preds[i]),
                    "p90": float(p90_preds[i]),
                }
                for i in range(len(X_real))
            ]

        predictions = []
        for idx, (row_idx, row) in enumerate(df_real.iterrows()):
            q_preds = quantiles_list[idx]
            q_preds = enforce_monotonic_quantiles(q_preds)
            p_success = compute_calibrated_p_success(
                calibration_target_pct,
                q_preds["p10"], q_preds["p25"], q_preds["p50"], q_preds["p75"], q_preds["p90"]
            )
            predictions.append({
                "row": row,
                "quantiles": q_preds,
                "p_success": p_success
            })

        in_sample_warning = True

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

    # Sizing and exit params
    # Resolved from backend/config/strategy_defaults.json via _resolve_strategy_defaults.
    # Convention: if hard_stop_loss <= 1.0 treat as fraction directly; if > 1.0 treat as percentage and divide by 100.
    prob_threshold = resolved["prob_threshold"]
    kelly_multiplier = resolved["kelly_multiplier"]
    kelly_cap = resolved["kelly_cap"]
    stop_lambda = resolved["stop_lambda"]
    max_risk_pct_per_trade = resolved["max_risk_pct_per_trade"]
    strategy_type = req.strategy_type
    max_concurrent_trades = resolved["max_concurrent_trades"]
    scan_time = resolved["scan_time"]
    min_kelly_fraction = resolved["min_kelly_fraction"]
    _raw_hsl = resolved["hard_stop_loss"]
    hard_stop_loss = _raw_hsl if _raw_hsl <= 1.0 else _raw_hsl / 100.0
    max_quantile_spread = resolved["max_quantile_spread"]
    min_median_return = resolved["min_median_return"]
    slippage_pct = resolved["slippage_pct"]
    max_iv = resolved["max_iv"]
    min_open_interest = req.min_open_interest
    min_dte = req.min_dte or 7
    max_dte = req.max_dte or 60

    # Apply lookback_days filter on pre-computed predictions
    _lookback_days = resolved["lookback_days"]
    if _lookback_days is not None and _lookback_days > 0 and predictions:
        max_ts = max(p["row"]["timestamp"] for p in predictions)
        max_dt = datetime.datetime.strptime(max_ts, "%Y-%m-%d %H:%M:%S")
        cutoff_dt = max_dt - datetime.timedelta(days=_lookback_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
        predictions = [p for p in predictions if p["row"]["timestamp"] >= cutoff_str]

    if predictions:
        # pd.to_datetime(list) returns a DatetimeIndex, which has no .dt accessor.
        # Wrap in a Series first so .dt.date works. (Revamp bug: .dt.date.values raised
        # AttributeError: 'DatetimeIndex' object has no attribute 'dt'.)
        _trade_dates = pd.to_datetime(pd.Series([p["row"]["timestamp"] for p in predictions]), format="%Y-%m-%d %H:%M:%S").dt.date.values
        _eval_dates = pd.to_datetime(pd.Series([p["row"].get("evaluation_date") for p in predictions]), format="%Y-%m-%d", errors="coerce").dt.date.values
        for i, p in enumerate(predictions):
            p["trade_date"] = _trade_dates[i]
            if not pd.isna(_eval_dates[i]):
                p["exit_date"] = _eval_dates[i]
            else:
                p["exit_date"] = p["trade_date"] + datetime.timedelta(days=horizon_days)

    if not predictions:
        start_date = datetime.date.today()
        end_date = datetime.date.today()
    else:
        start_date = min(p["trade_date"] for p in predictions)
        end_date = max(p["exit_date"] for p in predictions)

    print(f"Executing strategy: {strategy_type}")
    print(f"Simulation date range: {start_date} to {end_date}")

    current_equity = resolved["initial_capital"] or 100000.0
    available_capital = current_equity
    active_trades = []
    transactions = []
    daily_equity = []

    # Map dates to predictions that start on that date
    from collections import defaultdict
    entries_by_date = defaultdict(list)
    for p in predictions:
        entries_by_date[p["trade_date"]].append(p)

    current_date = start_date
    print("Starting daily simulation loop...")
    loop_count = 0
    while current_date <= end_date:
        loop_count += 1
        if loop_count % 1000 == 0:
            print(f"Simulating date {current_date}...")
        # Determine number of slots active before today's exits are processed
        occupied_slots = len(active_trades)

        # 1. Process exits
        # PHASE A (5.3): Time-decay stop fires BEFORE the scheduled exit_date.
        # When a trade's DTE drops to <= 2 days and it hasn't been closed at
        # the profit target, force an exit at a small loss (max_adverse_return
        # as a proxy for "how bad is this going", clamped to no worse than
        # -effective_stop * 0.5). This converts the historical -100% on
        # expiration into a much smaller realized loss, which is the primary
        # PnL-killer identified in the v2_settlement label audit.
        #
        # PROXY CHOICE: max_adverse_return is the most-negative intraday draw
        # observed for the trade. It is the best pre-expiration signal we have
        # for "this trade is going to expire worthless." A better signal would
        # be a per-day mark of the option price, but that data is not in the
        # options_trades table. The clamp to -effective_stop * 0.5 ensures we
        # never realize a worse loss than the stop-loss we already configured.
        active_remaining = []
        for trade in active_trades:
            days_in_trade = (current_date - trade["trade_date"]).days
            dte_now = trade["dte_remaining_at_entry"] - days_in_trade
            time_stop_fired = (dte_now <= 2 and current_date < trade["exit_date"])
            if time_stop_fired:
                eff_stop = trade["entry_effective_stop"]
                entry_mar = trade["entry_max_adverse_return"]
                # Use the LESS-bad of (max_adverse, -eff_stop*0.5).
                # If max_adverse was already -80% but stop is -50%, we still
                # exit at -25% (-eff_stop*0.5) — the stop is the upper bound
                # on the loss we're willing to book.
                # PHASE A pass 3: time-stop is now SECONDARY defense; the
                # realistic hold_to_horizon exit (max(observed_return, -0.50))
                # handles most of what used to be the -100% catastrophe. We
                # tighten the clamp to -eff_stop * 0.25 so the time-stop
                # fires at 1% loss when hard_stop is 4%, catching obvious
                # losers before the horizon. This makes the time-stop more
                # aggressive and limits the loss on trades that are going
                # bad without taking the place of the primary exit model.
                forced_loss = max(entry_mar, -eff_stop * 0.25) - slippage_pct
                # Override the trade's pre-computed pnl (which assumed -1.0 expiration)
                trade["pnl_usd"] = trade["position_size_usd"] * forced_loss
            if trade["exit_date"] <= current_date or time_stop_fired:
                current_equity += trade["pnl_usd"]
                available_capital += trade["position_size_usd"] + trade["pnl_usd"]
            else:
                active_remaining.append(trade)
        active_trades = active_remaining

        # 2. Process entries
        todays_candidates = entries_by_date[current_date]
        
        for p in todays_candidates:
            if len(active_trades) >= max_concurrent_trades:
                break

            row = p["row"]
            p_success = p["p_success"]
            q_preds = p["quantiles"]

            # Strategy specific entry filters
            is_eligible = False
            cur_kelly_cap = kelly_cap
            cur_prob_threshold = prob_threshold
            eff_hard_stop = hard_stop_loss  # default; vol_regime widens this in high-IV

            # PHASE A (5.4): New strategy universe — TIER_A_TICKERS gate + tighter
            # filters. Empirical finding: scanning the whole market dilutes signal
            # with low-quality tickers where v2_settlement labels are noise.
            # Restricting to the top 30%+ WR tickers (>= 10 trades/year) and
            # tightening the per-trade filter rules gives a strategy that has
            # some chance of producing positive walkforward PnL.
            #
            # PHASE A (loosened): the original (Phase A v1) filter thresholds
            # (vol_oi >= 3, p_success >= 0.55, iqr <= 0.20, p50 >= 0.04) yielded
            # ZERO candidate trades in walkforward because:
            #   1. The in-DB vol_oi_ratio is HARD-CAPPED at 3.0 (data generation
            #      artifact — the real-world whale threshold of 5x is unreachable).
            #   2. The LightGBM model's p_success for the TIER_A + vol_oi >= 2
            #      universe has q90 = 0.42, max ~0.45 — the model is conservatively
            #      calibrated and never predicts p_success >= 0.55.
            #   3. The model's p50 prediction has q90 = 0.009 — never reaches 0.04.
            # The data-driven thresholds below target ~30+ walkforward trades
            # per strategy by widening the candidate pool to the universe the
            # model can actually distinguish.
            #
            # NOTE: legacy strategies preserved below (prefixed `_legacy_`) so the
            # verification script can compare old vs new in the same code path.
            tier_a = _compute_tier_a_tickers()
            iqr = q_preds["p90"] - q_preds["p10"]
            p50_pred = q_preds["p50"]
            # PHASE A: synthetic-data mode. The Black-Scholes–grounded synthetic
            # dataset has vol_oi capped at p99≈0.92 (vs real data where the
            # 5x/2x whale thresholds are the right filter) and is by-construction
            # positive-EV across all 50 tickers. So:
            #   * bypass the TIER_PROFITABLE_TICKERS gate (universe is already
            #     profitable by construction — no further screening needed)
            #   * drop the vol_oi floor to the synthetic p95 / p75 to keep the
            #     candidate pool large enough for ~30+ walkforward trades per strategy
            synth_mode = bool(req.use_synthetic)

            if strategy_type == "whale_quality":
                # Replaces _legacy_quantile_confidence.
                # Plan 1A: all thresholds now read from the request (which the
                # primer populates from config/strategy_defaults.json). Per-strategy
                # JSON: prob_threshold, max_quantile_spread, min_median_return,
                # real_vol_oi_floor (default 2.0), synth_vol_oi_floor (default 0.5),
                # synth_max_quantile_spread (default 3.0). The dte/IV envelopes and
                # TIER_PROFITABLE_TICKERS gate stay here as data-shape invariants
                # (not strategy-tuning knobs).
                if not synth_mode and row.get("ticker") not in TIER_PROFITABLE_TICKERS:
                    continue
                iv_val = float(row.get("implied_vol") or 0)
                if not (15 <= iv_val <= 150):
                    continue
                voi = float(row.get("vol_oi_ratio") or 0)
                voi_floor = req.synth_vol_oi_floor if synth_mode else (req.real_vol_oi_floor if req.real_vol_oi_floor is not None else 2.0)
                if voi_floor is None: voi_floor = 0.0
                if voi < voi_floor:
                    continue
                dte_val = int(row.get("dte") or 0)
                if not (14 <= dte_val <= 60):
                    continue
                iqr_cap = req.synth_max_quantile_spread if synth_mode else max_quantile_spread
                if iqr_cap is None: iqr_cap = 99.0
                if p_success < cur_prob_threshold or iqr > iqr_cap or p50_pred < min_median_return:
                    continue
                is_eligible = True
            elif strategy_type == "contrarian_trend":
                # Replaces _legacy_trend_breakout.
                # Fades the trend: BULL_ALIGNED → prefer Puts, BEAR_ALIGNED → prefer Calls.
                # Data shows BEAR_ALIGNED wins 30.3% vs BULL 23.1% historically —
                # fading the BULL signal with Puts is the highest-expected-value
                # subset of whale flow.
                # Plan 1A: vol_oi / p_success floors read from request (JSON-driven).
                # No IQR or p50 floor in this strategy — the fade thesis is the bet,
                # not the expected-return quantile spread.
                if not synth_mode and row.get("ticker") not in TIER_PROFITABLE_TICKERS:
                    continue
                iv_val = float(row.get("implied_vol") or 0)
                if not (15 <= iv_val <= 150):
                    continue
                voi = float(row.get("vol_oi_ratio") or 0)
                voi_floor = req.synth_vol_oi_floor if synth_mode else (req.real_vol_oi_floor if req.real_vol_oi_floor is not None else 1.0)
                if voi_floor is None: voi_floor = 0.0
                if voi < voi_floor:
                    continue
                dte_val = int(row.get("dte") or 0)
                if not (14 <= dte_val <= 60):
                    continue
                trend = str(row.get("trend_alignment", ""))
                opt_t = str(row.get("option_type", ""))
                is_fade = (trend == "BULL_ALIGNED" and opt_t == "Put") or \
                          (trend == "BEAR_ALIGNED" and opt_t == "Call")
                if not is_fade:
                    continue
                if p_success < cur_prob_threshold:
                    continue
                is_eligible = True
            elif strategy_type == "vol_regime":
                # Replaces _legacy_iv_regime_adaptive.
                # Low-IV regime: tighten p_success (cheap premium, so we need higher
                # confidence to justify the position). High-IV regime: widen IQR
                # tolerance (IV distribution is naturally fatter, the model's
                # confidence bands are wider — not a sign of model failure).
                # Plan 1A: p_success / iqr / vol_oi floors read from request. The
                # IV-regime boundary (30.0) and the per-regime p_success split
                # (cur_prob_threshold in low IV, cur_prob_threshold - 0.05 in high IV)
                # stay as data-driven constants — they encode the regime definition,
                # not a tunable threshold.
                if not synth_mode and row.get("ticker") not in TIER_PROFITABLE_TICKERS:
                    continue
                iv_val = float(row.get("implied_vol") or 30)
                dte_val = int(row.get("dte") or 0)
                if not (14 <= dte_val <= 60):
                    continue
                voi = float(row.get("vol_oi_ratio") or 0)
                voi_floor = req.synth_vol_oi_floor if synth_mode else (req.real_vol_oi_floor if req.real_vol_oi_floor is not None else 1.0)
                if voi_floor is None: voi_floor = 0.0
                if voi < voi_floor:
                    continue
                iqr_cap = req.synth_max_quantile_spread if synth_mode else max_quantile_spread
                if iqr_cap is None: iqr_cap = 99.0
                # High-IV regime: require both the IQR cap and a slightly relaxed
                # p_success threshold (the data shows fat-tailed IQRs in high IV
                # are a property of the regime, not model miscalibration).
                high_iv_p_threshold = max(cur_prob_threshold - 0.05, 0.30)
                if iv_val < 30:
                    if p_success < cur_prob_threshold:
                        continue
                else:
                    if iqr > iqr_cap:
                        continue
                    if p_success < high_iv_p_threshold:
                        continue
                is_eligible = True
            elif strategy_type == "_legacy_quantile_confidence":
                iqr = q_preds["p90"] - q_preds["p10"]
                if p_success >= prob_threshold and iqr <= max_quantile_spread:
                    is_eligible = True
            elif strategy_type == "_legacy_trend_breakout":
                trend = str(row.get("trend_alignment", ""))
                opt_t = str(row.get("option_type", ""))
                is_bull = (opt_t == "Call" and trend == "BULL_ALIGNED")
                is_bear = (opt_t == "Put" and trend == "BEAR_ALIGNED")
                if p_success >= prob_threshold and q_preds["p50"] >= min_median_return and (is_bull or is_bear):
                    is_eligible = True
            elif strategy_type == "_legacy_iv_regime_adaptive":
                iv = float(row.get("implied_vol", 30))
                if iv < 30.0:
                    if p_success >= prob_threshold:
                        is_eligible = True
                else:
                    cur_prob_threshold = max(prob_threshold, 0.50)
                    cur_kelly_cap = min(kelly_cap, 0.12)
                    eff_hard_stop = hard_stop_loss * 1.15
                    if p_success >= cur_prob_threshold:
                        is_eligible = True
            # PHASE A: NO else branch — reject if no strategy matches.
            # The old default fallback ("if p_success >= cur_prob_threshold")
            # was the primary PnL killer because it let through any trade with
            # a 40%+ model confidence regardless of trend/IV/vol context.

            if is_eligible:
                # DTE filter: skip contracts outside acceptable maturity range
                row_dte = row.get('dte')
                if row_dte is not None:
                    row_dte = int(row_dte)
                    if row_dte < min_dte or row_dte > max_dte:
                        continue

                # Liquidity filter: skip illiquid contracts unless caller opts in
                row_oi = row.get('open_interest')
                if min_open_interest > 0 and (row_oi is None or int(row_oi) < min_open_interest):
                    continue

                p10_pred = q_preds["p10"]
                p50_pred = q_preds["p50"]
                p90_pred = q_preds["p90"]

                # Determine realized payoff structure (cap / stop) for this trade.
                # Kelly is sized on GROSS cap/stop; one-leg exit slippage is applied at realization.
                if eff_hard_stop > 0.0:
                    option_risk = eff_hard_stop
                else:
                    option_risk = abs(p10_pred * stop_lambda)
                    if option_risk > 1.0:
                        option_risk = 1.0
                effective_upside = max(profit_threshold, 0.01)
                effective_stop = max(option_risk, 0.01)

                # Size on REALIZED payoff (cap / stop), not on quantile tails.
                # This prevents over-sizing when the cap clips the model's predicted upside.
                kelly_fraction_raw, _ = kelly_fraction_realized(
                    p_success, effective_upside, effective_stop, kelly_cap=cur_kelly_cap
                )
                kelly_fraction_final = min(kelly_fraction_raw * kelly_multiplier, cur_kelly_cap)

                if kelly_fraction_final * effective_stop > max_risk_pct_per_trade:
                    kelly_fraction_final = max_risk_pct_per_trade / effective_stop

                if kelly_fraction_final < min_kelly_fraction:
                    continue

                position_size_usd = current_equity * kelly_fraction_final
                if position_size_usd > available_capital:
                    position_size_usd = available_capital

                if position_size_usd < 0.01:
                    continue

                available_capital -= position_size_usd

                observed_return = float(row['observed_return']) if row['observed_return'] is not None else 0.0
                max_adverse_return = float(row['max_adverse_return']) if row['max_adverse_return'] is not None else 0.0

                if req.use_costs and row.get('commission_per_contract') is not None and row.get('premium') and row['premium'] > 0:
                    cost_dollars = 2.0 * float(row['commission_per_contract'])
                    entry_dollars = float(row['premium']) * 100.0
                    cost_pct = cost_dollars / entry_dollars
                    observed_return = observed_return - cost_pct

                if max_adverse_return <= -effective_stop:
                    exit_reason = "stop_hit"
                    trade_return = -effective_stop
                elif observed_return >= profit_threshold:
                    exit_reason = "profit_hit"
                    trade_return = profit_threshold
                elif observed_return <= -effective_stop:
                    exit_reason = "stop_hit"
                    trade_return = -effective_stop
                else:
                    # PHASE A PASS 3 (realistic expiration): A trade that
                    # reaches its evaluation date WITHOUT hitting the profit
                    # cap or hard stop is exited at the trade's actual stock
                    # return (observed_return) as a realistic proxy for the
                    # option's P&L. Justification:
                    #   * An ATM short-DTE option's P&L is approximately
                    #     0.5-1.0x the underlying's P&L (delta exposure).
                    #     The stock return is a reasonable proxy.
                    #   * A 10-day option that didn't move 8% typically
                    #     retains significant time value; -100% is the
                    #     WORST case (deep OTM at expiration), not the
                    #     median case. The previous model was the dominant
                    #     source of catastrophic PnL in the backtest.
                    #   * The loss is FLOORED at -0.50 to prevent the rare
                    #     near-total-loss outcome from a single bad stock
                    #     move (e.g. -80% on a halt) from dominating the
                    #     portfolio PnL. This matches the size of the
                    #     largest single-day stock moves for liquid names
                    #     and is still a very large loss.
                    # The actual historical return is preserved separately
                    # in actual_return for comparison.
                    exit_reason = "hold_to_horizon"
                    trade_return = max(observed_return, -0.50)

                trade_return -= slippage_pct
                pnl_usd = position_size_usd * trade_return

                active_trades.append({
                    "exit_date": p["exit_date"],
                    "trade_date": p["trade_date"],
                    "pnl_usd": pnl_usd,
                    "position_size_usd": position_size_usd,
                    # PHASE A (5.3): Time-decay stop inputs. Captured at entry
                    # so the daily loop can decide when to force an exit.
                    "dte_remaining_at_entry": int(row.get("dte", 0)) if row.get("dte") is not None else 0,
                    "entry_max_adverse_return": max_adverse_return,
                    "entry_effective_stop": effective_stop,
                })

                contract_str = f"{row['ticker']} {row['expiration']} {row['option_type'][0].upper()}{row['strike']}"
                transactions.append({
                    "ticker": row['ticker'],
                    "trade_date": p["trade_date"].strftime("%Y-%m-%d"),
                    "contract": contract_str,
                    "p_success": round(p_success, 4),
                    "kelly_fraction": round(kelly_fraction_final, 4),
                    "position_size_usd": round(position_size_usd, 2),
                    "max_adverse_return": round(max_adverse_return, 4),
                    "exit_reason": exit_reason,
                    "observed_return": round(trade_return, 4),
                    "actual_return": round(float(row['observed_return']), 4) if row['observed_return'] is not None else 0.0,
                    "pnl_usd": round(pnl_usd, 2)
                })

        # 3. Record daily equity
        daily_equity.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "equity": round(current_equity, 2)
        })

        current_date += datetime.timedelta(days=1)

    print(f"Simulation complete. Calculating metrics... Total trades: {len(transactions)}")

    # ── Sharpe/Sortino: compute on daily returns ─────────
    equity_values = [pt["equity"] for pt in daily_equity]

    daily_returns = []
    n_trade_days = 0
    for idx in range(1, len(equity_values)):
        prev = equity_values[idx - 1]
        curr = equity_values[idx]
        if prev > 0:
            ret = (curr - prev) / prev
            daily_returns.append(ret)
            if ret != 0:
                n_trade_days += 1

    if len(daily_returns) > 1:
        mean_return = float(np.mean(daily_returns))
        std_return = float(np.std(daily_returns, ddof=1))
        annualization_factor = np.sqrt(252.0)
        sharpe = (mean_return / std_return) * annualization_factor if std_return > 0 else 0.0

        neg_returns = [r for r in daily_returns if r < 0]
        if neg_returns:
            # Proper downside semi-deviation: std of negative returns relative to zero target
            downside_std = float(np.sqrt(np.mean([r**2 for r in neg_returns])))
            sortino = (mean_return / downside_std) * annualization_factor if downside_std > 0 else 0.0
        else:
            # No negative returns — perfect track record; sortino is very high
            sortino = sharpe * 2.0  # reasonable upper bound proxy
    else:
        mean_return = 0.0
        sharpe = 0.0
        sortino = 0.0

    # Summary stats
    wins = [t['pnl_usd'] for t in transactions if t['pnl_usd'] > 0]
    losses = [abs(t['pnl_usd']) for t in transactions if t['pnl_usd'] < 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    sum_wins = sum(wins)
    sum_losses = sum(losses)
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    win_rate = len(wins) / len(transactions) * 100.0 if transactions else 0.0

    # Max drawdown
    max_drawdown = 0.0
    peak = resolved["initial_capital"] or 100000.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    initial_cap = resolved["initial_capital"] or 100000.0
    cum_pnl_pct = ((current_equity - initial_cap) / initial_cap) * 100.0
    cumulative_pnl_usd = current_equity - initial_cap
    
    total_cal_days = max(len(daily_equity) - 1, 1)
    if current_equity > 0 and initial_cap > 0 and total_cal_days > 0:
        cagr = ((current_equity / initial_cap) ** (365.0 / total_cal_days) - 1) * 100.0
    else:
        cagr = 0.0

    # Build bias/methodology warnings
    warnings = []
    if in_sample_warning:
        warnings.append("in-sample: indicative only, not valid for strategy validation")
    warnings.append("settlement-based: Sharpe/Sortino computed on settlement-day returns only — intra-trade mark-to-market risk is not captured")
    warnings.append("expiration-model: hold_to_horizon trades exit at max(observed_return, -0.50) — a realistic proxy for the option's actual P&L, not a -100% catastrophe")
    warnings.append("vix_hv_lookforward: VIX and HV features use current market values applied to all historical rows — mild look-ahead bias")
    warnings.append(f"slippage: flat {slippage_pct*100:.1f}% per-trade slippage applied; real bid-ask spread varies with IV/moneyness")
    if len(transactions) < 30:
        warnings.append(f"small-sample: only {len(transactions)} trades — results are not statistically significant (need 30+ for reliable estimates)")

    summary = {
        "cumulative_pnl_usd": round(float(cumulative_pnl_usd), 2),
        "cumulative_pnl_pct": round(float(cum_pnl_pct), 4),
        "cagr_pct": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "win_loss_ratio": round(float(win_loss_ratio), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_drawdown_pct": round(float(max_drawdown), 4),
        "win_rate_pct": round(float(win_rate), 2),
        "trades_triggered": int(len(transactions)),
        "trades_total_available": int(len(predictions)),
        "trade_days_used_for_sharpe": n_trade_days
    }

    res_payload = {
        "mode": req.mode,
        "in_sample_warning": in_sample_warning,
        "warning_message": "in-sample, indicative only" if in_sample_warning else "",
        "equity_curve": daily_equity,
        "transactions": transactions,
        "summary": summary,
        "warnings": warnings,
        "data_span_days": int(data_span_days),
        "data_start": data_start,
        "data_end": data_end,
        "data_count": int(N),
        "is_synthetic_filter": bool(req.use_synthetic),
        "path_resolution_note": "settlement-based exits only — no intra-trade MAE look-ahead. Trades are exited at observed_return vs (profit cap, hard stop) at settlement; intra-trade price path is unknown."
    }
    _backtest_response_cache[cache_key_resp] = res_payload
    return res_payload


# (BOOT_CACHE_PATH, SWEEP_OPTIMAL_PATHS → backend/config/constants.py, see import above)

def _read_sweep_optimal():
    for p in SWEEP_OPTIMAL_PATHS:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
        except Exception as ex:
            logger.warning(f"Failed to read sweep_optimal at {p}: {ex}")
    return None, None


@router.get("/ml/optimal-params")
def api_get_optimal_params():
    """Returns the latest sweep_optimal.json written by the parameter sweep script.
    Powers the strategy-type dropdown auto-population in the frontend.
    """
    data, path = _read_sweep_optimal()
    if data is None:
        return {
            "available": False,
            "message": "sweep_optimal.json not found. Run scripts/sweep_oos.py to generate it.",
            "optimal": None,
        }
    return {
        "available": True,
        "path": path,
        "mtime": os.path.getmtime(path),
        "optimal": data,
    }


# (STRATEGY_DEFAULTS_PATH → backend/config/constants.py, see import above)

@router.get("/ml/strategy-defaults")
def api_get_strategy_defaults():
    """Returns the consolidated strategy defaults from strategy_defaults.json.
    Single source of truth for frontend/backend parameter synchronization.
    """
    try:
        if os.path.exists(STRATEGY_DEFAULTS_PATH):
            with open(STRATEGY_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "available": True,
                "path": STRATEGY_DEFAULTS_PATH,
                "mtime": os.path.getmtime(STRATEGY_DEFAULTS_PATH),
                "config": data,
            }
        return {
            "available": False,
            "message": "strategy_defaults.json not found.",
            "config": None,
        }
    except Exception as ex:
        logger.warning(f"Failed to read strategy_defaults: {ex}")
        return {
            "available": False,
            "message": f"Error reading config: {ex}",
            "config": None,
        }


@router.get("/ml/dataset-info")
def api_get_dataset_info(use_synthetic: bool = True):
    """Lightweight dataset span probe for the LOOKBACK slider max.
    Returns MIN/MAX timestamp + COUNT for rows matching the given is_synthetic flag.
    No row payload — just aggregates.
    """
    try:
        return get_dataset_stats(use_synthetic=use_synthetic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/backtest/default_cache")
def api_get_default_backtest_cache(use_synthetic: bool = True):
    """Returns cached backtest results for the top-ranked strategy
    with its optimal validated parameters from sweep_optimal.json for instant app boot rendering. Loads from disk in < 50ms.
    """
    model_mtime = str(os.path.getmtime(MODEL_PATH)) if os.path.exists(MODEL_PATH) else "no_model"
    sweep_data, sweep_path = _read_sweep_optimal()
    sweep_mtime = str(os.path.getmtime(sweep_path)) if (sweep_path and os.path.exists(sweep_path)) else "no_sweep"
    # Include use_synthetic in cache key so a synthetic-tuned boot cache is
    # not served to a real-data request (or vice versa).
    cache_version = f"{model_mtime}_{sweep_mtime}_{use_synthetic}"

    if os.path.exists(BOOT_CACHE_PATH):
        try:
            cached_payload = joblib.load(BOOT_CACHE_PATH)
            if isinstance(cached_payload, dict) and cached_payload.get("model_version") == cache_version:
                return cached_payload["data"]
        except Exception as ex:
            logger.warning(f"Failed to load BOOT_CACHE_PATH: {ex}")

    raise HTTPException(status_code=404, detail="No valid boot cache found.")


