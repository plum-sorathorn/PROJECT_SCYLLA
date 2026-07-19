import numpy as np

def compute_calibrated_p_success(target_pct: float, p10: float, p25: float, p50: float, p75: float, p90: float) -> float:
    """
    Calibrate P(success) by piecewise-linear interpolation of the quantile grid.
    evaluated at the user's target return threshold.
    Clips to [0.02, 0.98] outside the P10-P90 range to handle fat tails.
    """
    # Monotonicity is assumed enforced already, but sort to be safe
    x_pts = sorted([p10, p25, p50, p75, p90])
    
    # If target_pct is outside the range P10-P90, clip to [0.02, 0.98]
    # target_pct < p10 means high probability of success (since target is below P10, and P(ret >= P10) = 0.90)
    # target_pct > p90 means low probability of success (since target is above P90, and P(ret >= P90) = 0.10)
    if target_pct < x_pts[0]:
        return 0.98
    if target_pct > x_pts[-1]:
        return 0.02

    y_pts = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    # np.interp gets CDF value F(target_pct) = P(return <= target_pct)
    cdf_val = float(np.interp(target_pct, x_pts, y_pts))
    p_success = 1.0 - cdf_val
    return float(np.clip(p_success, 0.02, 0.98))

def classify_strategy(median: float, p10: float, p90: float, iqr_threshold: float, direction_threshold: float) -> tuple[str, float]:
    """
    Rule-based strategy classification on median and IQR.
    Returns: (strategy_name, confidence_score_percentage)
    """
    iqr = p90 - p10
    
    if iqr >= iqr_threshold:
        strategy = "VOL_EXPANSION"
        dist = iqr - iqr_threshold
        confidence = min(100.0, max(0.0, (dist / max(iqr_threshold, 1e-6)) * 100.0))
        return strategy, round(confidence, 2)
    
    dist_to_iqr_boundary = iqr_threshold - iqr
    
    if abs(median) < direction_threshold:
        strategy = "SIDEWAYS"
        dist_to_direction_boundary = direction_threshold - abs(median)
        pct_iqr = dist_to_iqr_boundary / max(iqr_threshold, 1e-6)
        pct_dir = dist_to_direction_boundary / max(direction_threshold, 1e-6)
        confidence = min(pct_iqr, pct_dir) * 100.0
        return strategy, round(min(100.0, max(0.0, confidence)), 2)
    else:
        strategy = "BULLISH_BREAKOUT" if median > 0 else "BEARISH_BREAKDOWN"
        dist_to_direction_boundary = abs(median) - direction_threshold
        pct_iqr = dist_to_iqr_boundary / max(iqr_threshold, 1e-6)
        pct_dir = dist_to_direction_boundary / max(direction_threshold, 1e-6)
        confidence = min(pct_iqr, pct_dir) * 100.0
        return strategy, round(min(100.0, max(0.0, confidence)), 2)

def kelly_fraction(p_success: float, median: float, p10: float, p90: float) -> tuple[float, float]:
    """
    Compute Kelly criterion position sizing fraction.
    Returns: (capped_fraction, uncapped_fraction)
    """
    b = (p90 - median) / max(median - p10, 1e-6)  # asymmetric payoff ratio proxy
    
    # Kelly Formula: f* = (p * b - q) / b = (p * b - (1 - p)) / b
    f_star = (p_success * b - (1.0 - p_success)) / b
    
    capped = max(0.0, min(f_star, 0.25))
    return float(capped), float(f_star)
