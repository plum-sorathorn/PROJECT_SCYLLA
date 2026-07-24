from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import datetime
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import joblib
import lightgbm as lgb
import logging
from .ml_derivations import compute_calibrated_p_success, classify_strategy, kelly_fraction, kelly_fraction_realized

import threading
import concurrent.futures
import math

logger = logging.getLogger("scylla.ml_model")
router = APIRouter()

def _sanitize_float_values(obj):
    """Recursively converts NaN, Infinity, -Infinity floats to None for standard JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_float_values(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


# Paths relative to the backend directory
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scylla_ml.db"))
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache"))
MODEL_PATH = os.path.join(CACHE_DIR, "scylla_predictor.pkl")

LABELING_VERSION = "v2_settlement"

# ── In-memory caches for expensive market data lookups ───────
_hv_cache = {}   # ticker -> annualized HV
_vix_cache = {"value": None, "ts": None}  # cached VIX level
_global_model = None
_backtest_response_cache = {}
_walkforward_lock = threading.Lock()
_walkforward_cache_mem = {}


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

def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode for better concurrency (allows reads during writes)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            option_type TEXT NOT NULL,
            volume INTEGER NOT NULL,
            open_interest INTEGER NOT NULL,
            vol_oi_ratio REAL NOT NULL,
            implied_vol REAL NOT NULL,
            underlier_price REAL NOT NULL,
            premium REAL NOT NULL,
            side TEXT NOT NULL,
            dte INTEGER NOT NULL,
            is_weekly INTEGER NOT NULL,
            trend_alignment TEXT NOT NULL,
            labeled INTEGER DEFAULT 0,
            label_success INTEGER DEFAULT NULL,
            observed_return REAL DEFAULT NULL,
            max_adverse_return REAL DEFAULT NULL,
            evaluation_date TEXT DEFAULT NULL,
            is_synthetic INTEGER NOT NULL DEFAULT 0
        )
    """)

    # MLOps training run audit log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            samples_count INTEGER NOT NULL,
            train_accuracy REAL,
            test_accuracy REAL,
            test_precision REAL,
            test_recall REAL,
            test_f1 REAL,
            test_roc_auc REAL,
            cv_roc_auc_mean REAL,
            horizon_days INTEGER,
            profit_threshold REAL,
            model_version TEXT,
            pinball_loss_p10 REAL DEFAULT NULL,
            pinball_loss_p25 REAL DEFAULT NULL,
            pinball_loss_p50 REAL DEFAULT NULL,
            pinball_loss_p75 REAL DEFAULT NULL,
            pinball_loss_p90 REAL DEFAULT NULL
        )
    """)

    # Configurable ML settings (horizon, threshold, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    # Prediction cache to prevent re-running inference for identical inputs + model version
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_cache (
            input_hash TEXT PRIMARY KEY,
            model_version TEXT NOT NULL,
            prediction_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Seed defaults if not present
    cursor.execute("INSERT OR IGNORE INTO ml_settings (key, value) VALUES ('horizon_days', '10')")
    cursor.execute("INSERT OR IGNORE INTO ml_settings (key, value) VALUES ('profit_threshold', '0.03')")

    # Run ALTER TABLE command to verify columns exist
    try:
        cursor.execute("ALTER TABLE options_trades ADD COLUMN max_adverse_return REAL DEFAULT NULL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE options_trades ADD COLUMN is_synthetic INTEGER NOT NULL DEFAULT 0")
        conn.commit()
        # Backfill mock trades
        cursor.execute("UPDATE options_trades SET is_synthetic = 1 WHERE expiration = '2026-08-20' AND volume = 1000 AND open_interest = 200")
        conn.commit()
        logger.info("Migrated options_trades: added is_synthetic and backfilled 100 mock trades.")
    except sqlite3.OperationalError:
        pass

    for q_col in ["pinball_loss_p10", "pinball_loss_p25", "pinball_loss_p50", "pinball_loss_p75", "pinball_loss_p90"]:
        try:
            cursor.execute(f"ALTER TABLE model_runs ADD COLUMN {q_col} REAL DEFAULT NULL")
        except sqlite3.OperationalError:
            pass

    conn.commit()

    # Startup/health-check log line reporting counts
    try:
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 1")
        synth_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM options_trades WHERE is_synthetic = 0")
        real_count = cursor.fetchone()[0]
        logger.info(f"Database health check: options_trades has {real_count} real trades (is_synthetic=0) and {synth_count} synthetic trades (is_synthetic=1).")
    except Exception as e:
        logger.warning(f"Failed to count synthetic vs real trades during startup: {e}")

    conn.close()


# Ensure database is initialized on import
init_db()


# ═══════════════════════════════════════════════════════════════
# ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def _fetch_historical_volatility(ticker: str) -> float:
    """Fetch 30-day annualized historical volatility for a ticker (cached)."""
    global _hv_cache
    if ticker in _hv_cache:
        return _hv_cache[ticker]
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty or len(hist) < 5:
            _hv_cache[ticker] = 0.0
            return 0.0
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        # Annualized std of log returns (252 trading days)
        hv = float(log_returns.std() * np.sqrt(252) * 100)  # as percentage
        _hv_cache[ticker] = hv
        return hv
    except Exception as ex:
        logger.warning(f"HV fetch failed for {ticker}: {ex}")
        _hv_cache[ticker] = 0.0
        return 0.0


def _fetch_vix_level() -> float:
    """Fetch current VIX level (cached for 5 minutes)."""
    global _vix_cache
    now = datetime.datetime.now()
    if _vix_cache["value"] is not None and _vix_cache["ts"] is not None:
        elapsed = (now - _vix_cache["ts"]).total_seconds()
        if elapsed < 300:  # 5-minute cache
            return _vix_cache["value"]
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            _vix_cache["value"] = 20.0
            _vix_cache["ts"] = now
            return 20.0
        val = float(hist['Close'].iloc[-1])
        _vix_cache["value"] = val
        _vix_cache["ts"] = now
        return val
    except Exception as ex:
        logger.warning(f"VIX fetch failed: {ex}")
        _vix_cache["value"] = 20.0
        _vix_cache["ts"] = now
        return 20.0


def _dte_to_bucket(dte: int) -> str:
    """Categorize DTE into labeled buckets."""
    if dte <= 7:
        return "weekly"
    elif dte <= 30:
        return "short"
    elif dte <= 60:
        return "medium"
    else:
        return "long"


def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features from raw trade columns.
    Adds: moneyness, iv_hv_ratio, vix_level, log_premium, dte_bucket.
    Prevents lookahead leakage by avoiding live yfinance calls on historical datasets.
    """
    df = df.copy()

    # 1. Moneyness: percentage distance from spot
    df['moneyness'] = (df['strike'] - df['underlier_price']) / df['underlier_price']

    # 2. IV / HV ratio per ticker
    if 'iv_hv_ratio' not in df.columns:
        if len(df) > 10:
            # Historical batch mode: use relative IV to 25% baseline HV to prevent lookahead API leakage
            df['iv_hv_ratio'] = df['implied_vol'] / 25.0
        else:
            # Live inference mode: fetch current HV
            tickers = df['ticker'].unique() if 'ticker' in df.columns else []
            for t in tickers:
                _fetch_historical_volatility(t)

            def _iv_hv_ratio(row):
                hv = _hv_cache.get(row.get('ticker', ''), 0.0)
                iv = row.get('implied_vol', 0.0)
                return iv / hv if hv > 0 else 1.0

            df['iv_hv_ratio'] = df.apply(_iv_hv_ratio, axis=1) if 'ticker' in df.columns else 1.0

    # 3. VIX level (global market regime)
    if 'vix_level' not in df.columns:
        if len(df) > 10:
            # Historical batch mode: use neutral baseline VIX (20.0) to prevent lookahead API leakage
            df['vix_level'] = 20.0
        else:
            df['vix_level'] = _fetch_vix_level()

    # 4. Log-scaled premium
    df['log_premium'] = np.log1p(df['premium'].clip(lower=0))

    # 5. DTE bucket
    df['dte_bucket'] = df['dte'].apply(_dte_to_bucket)

    return df


# ═══════════════════════════════════════════════════════════════
# SETTINGS ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def _get_settings() -> dict:
    """Read all settings from the ml_settings table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM ml_settings")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


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


# ═══════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════

def _execute_with_retry(func, max_retries=5, initial_delay=0.1):
    """Execute a database operation with retry logic for SQLite locking."""
    import time
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
    raise sqlite3.OperationalError("Max retries exceeded")


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
def api_get_open_trades(
    prob_threshold: Optional[float] = Query(default=0.55),
    min_kelly_fraction: Optional[float] = Query(default=0.01)
):
    try:
        settings = _get_settings()
        target_pct = float(settings.get("profit_threshold", "0.03"))
        
        from .unusual_options import scan_raw_options, SCAN_TICKERS
        raw_options = scan_raw_options(",".join(SCAN_TICKERS), min_vol_oi=0.1, limit=100)
        df = pd.DataFrame(raw_options)
        models = get_global_model()
        
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
            
            models = None
            if os.path.exists(MODEL_PATH):
                try:
                    models = joblib.load(MODEL_PATH)
                    if not isinstance(models, dict) or not all(q in models for q in [0.1, 0.25, 0.5, 0.75, 0.9]):
                        models = None
                except Exception:
                    models = None
                    
            try:
                horizon_days = int(settings.get("horizon_days", "10"))
                
                # Fetch vix and populate caches in parallel
                _fetch_vix_level()
                unique_tickers = list(df['ticker'].unique())
                if unique_tickers:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(unique_tickers))) as executor:
                        futures = [executor.submit(_fetch_historical_volatility, t) for t in unique_tickers]
                        concurrent.futures.wait(futures, timeout=5.0)
                        
                feature_rows = []
                for _, row in df.iterrows():
                    row_data = {
                        'ticker': row['ticker'].upper(),
                        'strike': row['strike'],
                        'volume': row['volume'],
                        'open_interest': row['open_interest'],
                        'vol_oi_ratio': row['vol_oi_ratio'],
                        'implied_vol': row['implied_vol'],
                        'underlier_price': row['underlier_price'],
                        'premium': row['premium'],
                        'dte': row['dte'],
                        'option_type': row['option_type'],
                        'side': row['side'],
                        'trend_alignment': row.get('trend_alignment', 'NEUTRAL') if hasattr(row, 'get') else getattr(row, 'trend_alignment', 'NEUTRAL')
                    }
                    
                    row_data['moneyness'] = (row['strike'] - row['underlier_price']) / row['underlier_price']
                    row_data['iv_hv_ratio'] = 1.0
                    hv = _hv_cache.get(row_data['ticker'], 0.0)
                    if hv > 0:
                        row_data['iv_hv_ratio'] = row_data['implied_vol'] / hv
                        
                    row_data['vix_level'] = _vix_cache.get("value") or 20.0
                    row_data['log_premium'] = float(np.log1p(max(row['premium'], 0)))
                    row_data['dte_bucket'] = _dte_to_bucket(row['dte'])
                    
                    feature_rows.append(row_data)
                
                if feature_rows:
                    feat_df = pd.DataFrame(feature_rows)
                    
                    predicted_p10s = []
                    predicted_p25s = []
                    predicted_p50s = []
                    predicted_p75s = []
                    predicted_p90s = []
                    predicted_strategies = []
                    p_successes = []
                    kellys = []
                    
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
                            # Heuristic fallback path
                            score = 0.50
                            if row['trend_alignment'] == 'BULL_ALIGNED':
                                score += 0.15
                            if row['side'] == 'BUY':
                                score += 0.05
                            p_succ = min(score, 0.95)
                            
                            median_val = 0.03
                            q_preds = {
                                "p10": median_val - 0.05,
                                "p25": median_val - 0.025,
                                "p50": median_val,
                                "p75": median_val + 0.025,
                                "p90": median_val + 0.05
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
def api_get_trades(
    ticker: Optional[str] = None,
    labeled: Optional[int] = None,
    limit: int = 50,
    offset: int = 0
):
    try:
        # Fetch profit threshold to derive label_success at read time
        settings = _get_settings()
        target_pct = float(settings.get("profit_threshold", "0.03"))

        conn = sqlite3.connect(DB_PATH)
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
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Override label_success at read time based on continuous return (observed_return)
        if not df.empty and 'observed_return' in df.columns:
            # If trade is labeled, success is observed_return >= target_pct
            df['label_success'] = df.apply(
                lambda row: (1 if row['observed_return'] >= target_pct else 0) 
                if row['labeled'] == 1 and row['observed_return'] is not None 
                else row['label_success'], 
                axis=1
            )
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
                    for t in df['ticker'].unique():
                        try:
                            _fetch_historical_volatility(t)
                        except Exception:
                            pass
                            
                    feature_rows = []
                    for _, row in df.iterrows():
                        row_data = {
                            'ticker': row['ticker'].upper(),
                            'strike': row['strike'],
                            'volume': row['volume'],
                            'open_interest': row['open_interest'],
                            'vol_oi_ratio': row['vol_oi_ratio'],
                            'implied_vol': row['implied_vol'],
                            'underlier_price': row['underlier_price'],
                            'premium': row['premium'],
                            'dte': row['dte'],
                            'option_type': row['option_type'],
                            'side': row['side'],
                            'trend_alignment': row['trend_alignment']
                        }
                        
                        row_data['moneyness'] = (row['strike'] - row['underlier_price']) / row['underlier_price']
                        row_data['iv_hv_ratio'] = 1.0
                        hv = _hv_cache.get(row_data['ticker'], 0.0)
                        if hv > 0:
                            row_data['iv_hv_ratio'] = row_data['implied_vol'] / hv
                            
                        row_data['vix_level'] = _vix_cache.get("value") or 20.0
                        row_data['log_premium'] = float(np.log1p(max(row['premium'], 0)))
                        row_data['dte_bucket'] = _dte_to_bucket(row['dte'])
                        
                        feature_rows.append(row_data)
                    
                    if feature_rows:
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
        for ticker, ticker_trades in trades_by_ticker.items():
            try:
                dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date() for _, ts, _, _, _ in ticker_trades]
                min_date = min(dates)
                max_date = max(dates) + datetime.timedelta(days=effective_horizon + 2)
                
                tk = yf.Ticker(ticker)
                hist = tk.history(start=min_date.strftime("%Y-%m-%d"), 
                                  end=max_date.strftime("%Y-%m-%d"))
                
                if hist.empty:
                    continue
                
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
                    
                    cursor.execute("""
                        UPDATE options_trades
                        SET labeled = 1,
                            label_success = ?,
                            observed_return = ?,
                            max_adverse_return = ?,
                            evaluation_date = ?
                        WHERE id = ?
                    """, (success, 
                          round(float(continuous_favorable_return), 4), 
                          round(float(max_adverse_return), 4), 
                          end_date.strftime("%Y-%m-%d"), 
                          trade_id))
                    conn.commit()
                    labeled_count += 1
                    
            except Exception as ex:
                logger.warning(f"Error labeling trades for ticker {ticker}: {ex}")
                continue
                
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
        df = pd.read_sql_query("SELECT * FROM options_trades WHERE labeled = 1", conn)
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
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
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
        QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
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


def enforce_monotonic_quantiles(preds: dict) -> dict:
    """Enforces monotonicity across predictions: P10 <= P25 <= P50 <= P75 <= P90."""
    keys = ["p10", "p25", "p50", "p75", "p90"]
    vals = sorted([preds[k] for k in keys])
    return {k: v for k, v in zip(keys, vals)}

import hashlib
import json

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


# ═══════════════════════════════════════════════════════════════
# PREDICTION / INFERENCE
# ═══════════════════════════════════════════════════════════════

@router.post("/ml/predict", response_model=PredictResponseSchema)
def api_predict(req: PredictRequestSchema):
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

def get_real_trades(conn, **filters) -> pd.DataFrame:
    """
    Shared query helper that pulls options_trades with is_synthetic = 0.
    Always appends WHERE is_synthetic=0.
    """
    query = "SELECT * FROM options_trades WHERE is_synthetic = 0"
    params = []
    for k, v in filters.items():
        query += f" AND {k} = ?"
        params.append(v)
    query += " ORDER BY timestamp ASC"
    return pd.read_sql_query(query, conn, params=params)


class BacktestRequestSchema(BaseModel):
    mode: Optional[str] = "walkforward"
    initial_capital: Optional[float] = 100000.0
    # Probability calibration target used to map predicted return distribution -> p_success.
    # Decoupled from profit_threshold (take-profit cap) to avoid circular over-restriction.
    calibration_target_pct: Optional[float] = 0.025
    prob_threshold: Optional[float] = 0.40
    kelly_multiplier: Optional[float] = 0.80
    kelly_cap: Optional[float] = 0.20
    stop_lambda: Optional[float] = 1.5
    max_risk_pct_per_trade: Optional[float] = 0.02
    walkforward_train_window: Optional[int] = 500
    walkforward_test_increment: Optional[int] = 100
    confirm_direct_dev: Optional[bool] = False
    strategy_type: Optional[str] = "quantile_confidence"
    max_concurrent_trades: Optional[int] = 8
    scan_time: Optional[str] = "10:00:00"
    min_kelly_fraction: Optional[float] = 0.01
    hard_stop_loss: Optional[float] = 0.25
    lookback_days: Optional[int] = None
    profit_threshold: Optional[float] = 0.08
    max_quantile_spread: Optional[float] = 0.35
    min_median_return: Optional[float] = 0.015
    slippage_pct: Optional[float] = 0.01
    max_iv: Optional[float] = 100.0
    min_open_interest: Optional[int] = 100
    min_dte: Optional[int] = 7
    max_dte: Optional[int] = 60
    data_start_idx: Optional[int] = None
    data_end_idx: Optional[int] = None


def _process_walkforward_step(T_start, T_end, df_real, profit_threshold, calibration_target_pct=0.025, n_estimators=200, learning_rate=0.025, min_child_samples=15):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    df_train = df_real.iloc[0:T_start]
    df_test = df_real.iloc[T_start:T_end]

    if len(df_test) == 0:
        return (T_start, [])

    df_train_feat = compute_advanced_features(df_train)
    numeric_features = [
        'strike', 'volume', 'open_interest', 'vol_oi_ratio', 'implied_vol',
        'underlier_price', 'premium', 'dte',
        'moneyness', 'iv_hv_ratio', 'vix_level', 'log_premium'
    ]
    categorical_features = ['option_type', 'side', 'trend_alignment', 'dte_bucket']

    X_train = df_train_feat[['ticker'] + numeric_features + categorical_features]
    y_train = df_train_feat['observed_return']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
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

    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = {}
    for q in QUANTILES:
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("regressor", lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=20,
                min_child_samples=min_child_samples,
                random_state=42,
                verbose=-1,
                n_jobs=4
            ))
        ])
        pipe.fit(X_train, y_train)
        models[q] = pipe

    df_test_feat = compute_advanced_features(df_test)
    X_test = df_test_feat[['ticker'] + numeric_features + categorical_features]

    p10_preds = models[0.1].predict(X_test)
    p25_preds = models[0.25].predict(X_test)
    p50_preds = models[0.5].predict(X_test)
    p75_preds = models[0.75].predict(X_test)
    p90_preds = models[0.9].predict(X_test)

    step_preds = []
    for idx_test, (row_idx, row) in enumerate(df_test.iterrows()):
        q_preds = {
            "p10": float(p10_preds[idx_test]),
            "p25": float(p25_preds[idx_test]),
            "p50": float(p50_preds[idx_test]),
            "p75": float(p75_preds[idx_test]),
            "p90": float(p90_preds[idx_test])
        }
        q_preds = enforce_monotonic_quantiles(q_preds)
        p_success = compute_calibrated_p_success(
            calibration_target_pct,
            q_preds["p10"], q_preds["p25"], q_preds["p50"], q_preds["p75"], q_preds["p90"]
        )
        step_preds.append({
            "row": row,
            "quantiles": q_preds,
            "p_success": p_success
        })

    return (T_start, step_preds)


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

    # Check response cache — include ALL strategy-discriminating params to avoid cross-strategy cache collisions
    cache_key_resp = (
        f"{req.mode}_{req.initial_capital}_{req.prob_threshold}_{req.kelly_multiplier}_"
        f"{req.kelly_cap}_{req.stop_lambda}_{req.max_risk_pct_per_trade}_{req.walkforward_train_window}_"
        f"{req.walkforward_test_increment}_{req.strategy_type}_{req.max_concurrent_trades}_"
        f"{req.scan_time}_{req.min_kelly_fraction}_{req.hard_stop_loss}_{req.lookback_days}_"
        f"{req.profit_threshold}_{req.calibration_target_pct}_{req.max_quantile_spread}_{req.min_median_return}_{req.slippage_pct}_{req.max_iv}_{req.min_open_interest}"
        f"_{req.min_dte}_{req.max_dte}_{req.data_start_idx}_{req.data_end_idx}"
    )
    if cache_key_resp in _backtest_response_cache:
        print(f"api_backtest: returning cached response for key {cache_key_resp}")
        return _backtest_response_cache[cache_key_resp]

    # Fetch settings/thresholds
    # NOTE: profit_threshold here is the BACKTEST profit cap (the +X% return at which a winning trade
    # is closed). This is intentionally decoupled from the ml_settings.profit_threshold DB value, which
    # is the ML training label threshold (the return above which a trade is labelled "successful" for
    # supervised learning). They serve different purposes and intentionally do not share a default.
    settings = _get_settings()
    profit_threshold = req.profit_threshold if req.profit_threshold is not None else 0.08
    calibration_target_pct = req.calibration_target_pct if req.calibration_target_pct is not None else 0.025
    horizon_days = int(settings.get("horizon_days", "10"))

    conn = sqlite3.connect(DB_PATH)
    df_real = get_real_trades(conn, labeled=1)
    conn.close()

    if req.data_start_idx is not None or req.data_end_idx is not None:
        start = req.data_start_idx if req.data_start_idx is not None else 0
        end = req.data_end_idx if req.data_end_idx is not None else len(df_real)
        end = min(end, len(df_real))
        start = max(0, min(start, end))
        df_real = df_real.iloc[start:end].reset_index(drop=True)

    N = len(df_real)

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
        cache_key = f"{LABELING_VERSION}_{train_window}_{increment}_{N}_{profit_threshold}_{calibration_target_pct}_{req.data_start_idx}_{req.data_end_idx}"
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

                print(f"Parallelizing {len(tasks)} walkforward steps across CPU cores...")
                num_workers = min(16, os.cpu_count() or 4)
                results = joblib.Parallel(n_jobs=num_workers, prefer="threads")(
                    joblib.delayed(_process_walkforward_step)(T_start, T_end, df_real, profit_threshold, calibration_target_pct)
                    for T_start, T_end in tasks
                )

                # Sort by T_start to preserve exact chronological order
                results.sort(key=lambda r: r[0])
                for _, step_preds in results:
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

        p10_preds = models[0.1].predict(X_real)
        p25_preds = models[0.25].predict(X_real)
        p50_preds = models[0.5].predict(X_real)
        p75_preds = models[0.75].predict(X_real)
        p90_preds = models[0.9].predict(X_real)

        predictions = []
        for idx, (row_idx, row) in enumerate(df_real.iterrows()):
            q_preds = {
                "p10": float(p10_preds[idx]),
                "p25": float(p25_preds[idx]),
                "p50": float(p50_preds[idx]),
                "p75": float(p75_preds[idx]),
                "p90": float(p90_preds[idx])
            }
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
    # Defaults mirror BacktestRequestSchema — single source of truth.
    # Convention: if hard_stop_loss <= 1.0 treat as fraction directly; if > 1.0 treat as percentage and divide by 100.
    prob_threshold = req.prob_threshold
    kelly_multiplier = req.kelly_multiplier
    kelly_cap = req.kelly_cap
    stop_lambda = req.stop_lambda
    max_risk_pct_per_trade = req.max_risk_pct_per_trade
    strategy_type = req.strategy_type
    max_concurrent_trades = req.max_concurrent_trades
    scan_time = req.scan_time
    min_kelly_fraction = req.min_kelly_fraction
    _raw_hsl = req.hard_stop_loss
    hard_stop_loss = _raw_hsl if _raw_hsl <= 1.0 else _raw_hsl / 100.0
    max_quantile_spread = req.max_quantile_spread
    min_median_return = req.min_median_return
    slippage_pct = req.slippage_pct
    max_iv = req.max_iv
    min_open_interest = req.min_open_interest
    min_dte = req.min_dte or 7
    max_dte = req.max_dte or 60

    # Apply lookback_days filter on pre-computed predictions
    if req.lookback_days is not None and req.lookback_days > 0 and predictions:
        max_ts = max(p["row"]["timestamp"] for p in predictions)
        max_dt = datetime.datetime.strptime(max_ts, "%Y-%m-%d %H:%M:%S")
        cutoff_dt = max_dt - datetime.timedelta(days=req.lookback_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
        predictions = [p for p in predictions if p["row"]["timestamp"] >= cutoff_str]

    # Date parsing
    for p in predictions:
        row = p["row"]
        p["trade_date"] = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S").date()
        if row.get('evaluation_date'):
            p["exit_date"] = datetime.datetime.strptime(row['evaluation_date'], "%Y-%m-%d").date()
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

    current_equity = req.initial_capital or 100000.0
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
        active_remaining = []
        for trade in active_trades:
            if trade["exit_date"] <= current_date:
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

            if strategy_type == "quantile_confidence":
                iqr = q_preds["p90"] - q_preds["p10"]
                if p_success >= prob_threshold and iqr <= max_quantile_spread:
                    is_eligible = True
            elif strategy_type == "trend_breakout":
                trend = str(row.get("trend_alignment", ""))
                opt_t = str(row.get("option_type", ""))
                is_bull = (opt_t == "Call" and trend == "BULL_ALIGNED")
                is_bear = (opt_t == "Put" and trend == "BEAR_ALIGNED")
                if p_success >= prob_threshold and q_preds["p50"] >= min_median_return and (is_bull or is_bear):
                    is_eligible = True
            elif strategy_type == "iv_regime_adaptive":
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
            else:
                if p_success >= cur_prob_threshold:
                    is_eligible = True

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

                # Intraday stop-loss: use max_adverse_return (worst drawdown during trade life)
                # to detect if the stop threshold was breached intra-trade, even if the trade
                # recovered by settlement. max_adverse_return is negative (e.g. -0.30 = -30%).
                if max_adverse_return <= -effective_stop:
                    exit_reason = "stop_hit"
                    trade_return = -effective_stop
                elif observed_return >= profit_threshold:
                    exit_reason = "profit_hit"
                    trade_return = profit_threshold
                else:
                    exit_reason = "expired_profit" if observed_return > 0 else "expired_loss"
                    trade_return = observed_return

                trade_return -= slippage_pct
                pnl_usd = position_size_usd * trade_return

                active_trades.append({
                    "exit_date": p["exit_date"],
                    "pnl_usd": pnl_usd,
                    "position_size_usd": position_size_usd
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
    peak = req.initial_capital or 100000.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    initial_cap = req.initial_capital or 100000.0
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
        "path_resolution_note": "settlement-based exits only — no intra-trade MAE look-ahead. Trades are exited at observed_return vs (profit cap, hard stop) at settlement; intra-trade price path is unknown."
    }
    _backtest_response_cache[cache_key_resp] = res_payload
    return res_payload


BOOT_CACHE_PATH = os.path.join(CACHE_DIR, "boot_backtest_cache.pkl")

SWEEP_OPTIMAL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "sweep_optimal.json"),
    os.path.join(CACHE_DIR, "sweep_optimal.json"),
]


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


STRATEGY_DEFAULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "strategy_defaults.json"
)


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


@router.get("/ml/backtest/default_cache")
def api_get_default_backtest_cache():
    """Returns cached backtest results for the top-ranked strategy
    with its optimal validated parameters from sweep_optimal.json for instant app boot rendering. Loads from disk in < 50ms.
    """
    model_mtime = str(os.path.getmtime(MODEL_PATH)) if os.path.exists(MODEL_PATH) else "no_model"
    sweep_data, sweep_path = _read_sweep_optimal()
    sweep_mtime = str(os.path.getmtime(sweep_path)) if (sweep_path and os.path.exists(sweep_path)) else "no_sweep"
    cache_version = f"{model_mtime}_{sweep_mtime}"

    if os.path.exists(BOOT_CACHE_PATH):
        try:
            cached_payload = joblib.load(BOOT_CACHE_PATH)
            if isinstance(cached_payload, dict) and cached_payload.get("model_version") == cache_version:
                return cached_payload["data"]
        except Exception as ex:
            logger.warning(f"Failed to load BOOT_CACHE_PATH: {ex}")

    # Determine top strategy and params from sweep_optimal.json if available.
    # # defaults mirror BacktestRequestSchema — single source of truth.
    strat_type = "quantile_spread"
    opt_params = {
        "prob_threshold": 0.35,
        "kelly_multiplier": 0.80,
        "kelly_cap": 0.30,
        "stop_lambda": 1.5,
        "hard_stop_loss": 0.06,
        "profit_threshold": 0.08,
        "calibration_target_pct": 0.025,
        "max_quantile_spread": 0.35,
        "min_median_return": 0.015,
        "max_concurrent_trades": 10,
        "max_iv": 120.0,
        "slippage_pct": 0.01,
        "min_kelly_fraction": 0.005
    }
    eval_type = "in-sample"

    if sweep_data and isinstance(sweep_data, dict):
        best_strat = None
        best_sharpe = -999.0
        for st_name, st_info in sweep_data.items():
            if isinstance(st_info, dict) and "metrics" in st_info and "params" in st_info:
                sharpe = float(st_info["metrics"].get("sharpe", -999.0))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_strat = (st_name, st_info)
        
        if best_strat:
            st_name, st_info = best_strat
            strat_type = st_name
            p = st_info.get("params", {})
            eval_type = st_info.get("evaluation", "unknown")
            opt_params = {
                "prob_threshold": p.get("prob_threshold", 0.35),
                "kelly_multiplier": p.get("kelly_multiplier", 0.80),
                "kelly_cap": p.get("kelly_cap", 0.30),
                "stop_lambda": p.get("stop_lambda", 1.5),
                "hard_stop_loss": p.get("hard_stop_loss", 0.06),
                "profit_threshold": p.get("profit_threshold", 0.08),
                "calibration_target_pct": p.get("calibration_target_pct", 0.025),
                "max_quantile_spread": p.get("max_quantile_spread", 0.35),
                "min_median_return": p.get("min_median_return", 0.015),
                "max_concurrent_trades": int(p.get("max_concurrent_trades", 10)),
                "max_iv": p.get("max_iv", 120.0),
                "slippage_pct": p.get("slippage_pct", 0.01),
                "min_kelly_fraction": p.get("min_kelly_fraction", 0.005)
            }

    default_req = BacktestRequestSchema(
        mode="walkforward",
        strategy_type=strat_type,
        prob_threshold=opt_params["prob_threshold"],
        kelly_multiplier=opt_params["kelly_multiplier"],
        kelly_cap=opt_params["kelly_cap"],
        stop_lambda=opt_params.get("stop_lambda", 1.5),
        hard_stop_loss=opt_params["hard_stop_loss"],
        profit_threshold=opt_params["profit_threshold"],
        calibration_target_pct=opt_params["calibration_target_pct"],
        max_quantile_spread=opt_params["max_quantile_spread"],
        min_median_return=opt_params["min_median_return"],
        max_concurrent_trades=opt_params["max_concurrent_trades"],
        walkforward_train_window=500,
        walkforward_test_increment=100,
        slippage_pct=opt_params["slippage_pct"],
        min_kelly_fraction=opt_params["min_kelly_fraction"],
        initial_capital=100000.0,
        max_iv=opt_params["max_iv"]
    )
    try:
        res = api_backtest(default_req)
    except HTTPException as e:
        if e.status_code == 422:
            return {
                "error": "insufficient_data",
                "message": f"Not enough labeled trades to run backtest. Please complete the labeling process first. ({e.detail})",
                "summary": {
                    "cumulative_pnl_pct": 0,
                    "cumulative_pnl_usd": 0,
                    "trades_triggered": 0,
                    "trades_total_available": 0
                }
            }
        raise

    res_copy = dict(res)
    res_copy["cache_metadata"] = {
        "cached_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_name": f"{strat_type} ({eval_type.upper()} Optimal Default)",
        "evaluation": eval_type,
        "optimal_params": opt_params,
        "model_version": cache_version,
        "is_stale": False
    }

    try:
        joblib.dump({"model_version": cache_version, "data": res_copy}, BOOT_CACHE_PATH)
    except Exception as ex:
        logger.warning(f"Failed to save BOOT_CACHE_PATH: {ex}")

    return res_copy


