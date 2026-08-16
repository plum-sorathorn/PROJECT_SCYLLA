"""
Single source of truth for strategy defaults. Reads backend/config/strategy_defaults.json
and exposes typed accessors. Used by both the API endpoint and the backtest request
default path in ml_model.py so they can't drift apart.

Import patterns supported (both resolved from this file's location, not sys.path):
    from config._strategy_loader import get_strategy_params, get_common_params   # backend/
    from backend.config._strategy_loader import get_strategy_params, get_common_params  # repo root

The first form works because uvicorn is launched from backend/ (see AGENTS.md), so
backend/ is on sys.path and `config` is a package (this file's directory). The second
form works when the repo root is on sys.path instead.

ml_model.py imports this module via `from config._strategy_loader import ...` (running
from backend/), mirroring how it already resolves STRATEGY_DEFAULTS_PATH via
`os.path.dirname(os.path.dirname(...))` -> backend/config/.
"""
import json
import os
from typing import Optional

_DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_defaults.json")


def get_defaults_path() -> str:
    """Return the canonical strategy defaults path used by all consumers."""
    return _DEFAULTS_PATH


def load_defaults() -> dict:
    """Load and return the full strategy_defaults.json document."""
    with open(_DEFAULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_strategy_params(strategy_type: str) -> dict:
    """
    Return the per-strategy parameter block for the given strategy_type.

    Raises ValueError if the requested strategy_type is unknown.
    """
    data = load_defaults()
    strategies = data.get("strategies", {})
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy_type: {strategy_type}. Valid: {sorted(strategies)}")
    return strategies[strategy_type]


def get_common_params() -> dict:
    """Return the common (non-strategy-specific) parameter block."""
    return load_defaults().get("common", {})
