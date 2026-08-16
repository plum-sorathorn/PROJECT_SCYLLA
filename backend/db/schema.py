"""Database schema initialization for the Scylla ML backend.

Extracted from ml_model.py: the `init_db()` function containing all
CREATE TABLE / CREATE INDEX / ALTER TABLE migration statements.
"""
import sqlite3
import logging

try:
    from ..config.constants import DB_PATH
except ImportError:
    from config.constants import DB_PATH

logger = logging.getLogger("scylla.ml_model")


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
            is_synthetic INTEGER NOT NULL DEFAULT 0,
            delta_entry REAL DEFAULT NULL,
            gamma_entry REAL DEFAULT NULL,
            vega_entry REAL DEFAULT NULL,
            theta_entry REAL DEFAULT NULL,
            rho_entry REAL DEFAULT NULL,
            delta_exit REAL DEFAULT NULL,
            gamma_exit REAL DEFAULT NULL,
            vega_exit REAL DEFAULT NULL,
            theta_exit REAL DEFAULT NULL,
            rho_exit REAL DEFAULT NULL,
            ensemble_id INTEGER NOT NULL DEFAULT 0,
            commission_per_contract REAL DEFAULT NULL,
            bid_ask_spread_pct REAL DEFAULT NULL,
            early_exercise_risk INTEGER DEFAULT 0,
            delta_hedged_return REAL DEFAULT NULL,
            days_to_earnings INTEGER DEFAULT NULL,
            is_earnings_window INTEGER DEFAULT 0,
            days_to_fomc INTEGER DEFAULT NULL,
            is_fomc_day INTEGER DEFAULT 0,
            vix_regime TEXT DEFAULT NULL,
            market_regime TEXT DEFAULT NULL
        )
    """)
    # Indexes (idempotent via IF NOT EXISTS)
    # 1. ticker+timestamp DESC: serves api_get_trades (filter by ticker, sort by timestamp)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON options_trades(ticker, timestamp DESC)")
    # 2. labeled+timestamp: serves labeling worker scan of unlabeled rows
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_labeled_ts ON options_trades(labeled, timestamp)")
    # 3. dedup tuple: serves dedup equality lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_dedup ON options_trades(ticker, strike, expiration, option_type, vol_oi_ratio, timestamp)")

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

    # Run ALTER TABLE commands to verify columns exist
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

    for greek_col in [
        "delta_entry", "gamma_entry", "vega_entry", "theta_entry", "rho_entry",
        "delta_exit", "gamma_exit", "vega_exit", "theta_exit", "rho_exit",
    ]:
        try:
            cursor.execute(f"ALTER TABLE options_trades ADD COLUMN {greek_col} REAL DEFAULT NULL")
        except sqlite3.OperationalError:
            pass

    try:
        cursor.execute("ALTER TABLE options_trades ADD COLUMN ensemble_id INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    for col_3 in [
        "commission_per_contract REAL DEFAULT NULL",
        "bid_ask_spread_pct REAL DEFAULT NULL",
        "early_exercise_risk INTEGER DEFAULT 0",
        "delta_hedged_return REAL DEFAULT NULL",
    ]:
        try:
            cursor.execute(f"ALTER TABLE options_trades ADD COLUMN {col_3}")
        except sqlite3.OperationalError:
            pass

    for col_4 in [
        "days_to_earnings INTEGER DEFAULT NULL",
        "is_earnings_window INTEGER DEFAULT 0",
        "days_to_fomc INTEGER DEFAULT NULL",
        "is_fomc_day INTEGER DEFAULT 0",
        "vix_regime TEXT DEFAULT NULL",
        "market_regime TEXT DEFAULT NULL",
    ]:
        try:
            cursor.execute(f"ALTER TABLE options_trades ADD COLUMN {col_4}")
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
