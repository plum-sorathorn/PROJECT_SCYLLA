import math
import numpy as np


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
