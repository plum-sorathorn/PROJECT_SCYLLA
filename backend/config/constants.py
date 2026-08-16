"""Shared constants for the Scylla ML backend.

Extracted from ml_model.py so all sub-modules can import from a single
source without circular dependencies.
"""
import os

# Paths relative to the backend directory
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scylla_ml.db"))
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache"))
MODEL_PATH = os.path.join(CACHE_DIR, "scylla_predictor.pkl")
_SCYLLA_MAX_PROC = max(1, int(os.environ.get("SCYLLA_MAX_WORKERS", "5")))  # cap all child pools

LABELING_VERSION = "v2_settlement"

# C++ native inference engine URL (port 8080, hardcoded per AGENTS.md convention)
_CPP_CORE_URL = "http://127.0.0.1:8080"

# Boot cache path for pre-computed backtest results
BOOT_CACHE_PATH = os.path.join(CACHE_DIR, "boot_backtest_cache.pkl")

# Paths to sweep_optimal output files (multiple locations for fallback)
SWEEP_OPTIMAL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "sweep_optimal.json"),
    os.path.join(CACHE_DIR, "sweep_optimal_v2.json"),
    os.path.join(CACHE_DIR, "sweep_optimal.json"),
]

# Consolidated strategy defaults JSON (single source of truth)
STRATEGY_DEFAULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "strategy_defaults.json"
)
