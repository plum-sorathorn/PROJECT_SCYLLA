"""Walkforward backtest helpers.

Extracted from ml_model.py: the per-worker initializer and per-step
walkforward processing function.
"""
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    from ..routers.ml_derivations import compute_calibrated_p_success
    from ..models.features import compute_advanced_features
    from ..models.predict import enforce_monotonic_quantiles
except ImportError:
    from routers.ml_derivations import compute_calibrated_p_success
    from models.features import compute_advanced_features
    from models.predict import enforce_monotonic_quantiles

# ── Walkforward process-pool shared state ──────────────────────
# On fork-based platforms (Linux/macOS) the child inherits these
# from the parent. On spawn-based platforms (Windows) the initializer
# _wf_worker_init() sets them in each worker. Each worker receives the
# full dataset pickled exactly once, not once per task submission.
_WF_DF_REAL = None
_WF_DF_REAL_FEAT = None


def _wf_worker_init(df_real_data, df_real_feat_data):
    """Per-worker initializer for the walkforward ProcessPoolExecutor.

    Called exactly once per worker process. Receives the full (large)
    DataFrame so that _process_walkforward_step can read it from
    module-level globals rather than re-pickling it on every task.
    """
    global _WF_DF_REAL, _WF_DF_REAL_FEAT
    _WF_DF_REAL = df_real_data
    _WF_DF_REAL_FEAT = df_real_feat_data


def _process_walkforward_step(T_start, T_end, df_real=None, calibration_target_pct=0.025, n_estimators=200, learning_rate=0.025, min_child_samples=15, df_real_feat=None):
    # Plan 1A: removed the unused `profit_threshold` parameter. The walkforward step
    # is pure quantile regression — no binary label is computed here. The
    # `walkforward_label_threshold` field in BacktestRequestSchema is reserved for
    # a future binary-label refactor and is consumed only by the walkforward
    # cache key.
    #
    # PARALLELIZATION_PLAN §4: inner ProcessPoolExecutor removed. The calling
    # context (api_backtest) now runs ONE ProcessPoolExecutor at the outer level.
    # Each worker fits 5 quantile models SEQUENTIALLY — no more WinError 1450
    # from 16,955+ rapid process spawns.

    # Resolve DataFrames: function args take priority; fall back to module-level
    # globals set by _wf_worker_init (the ProcessPoolExecutor per-worker
    # initializer). On spawn-based platforms (Windows) this avoids re-pickling
    # the full 100MB+ dataset on every task submission.
    if df_real is None:
        df_real = _WF_DF_REAL
    if df_real_feat is None:
        df_real_feat = _WF_DF_REAL_FEAT

    if df_real_feat is not None:
        df_train_feat = df_real_feat.iloc[0:T_start]
        df_test_feat = df_real_feat.iloc[T_start:T_end]
    else:
        df_train = df_real.iloc[0:T_start]
        df_test = df_real.iloc[T_start:T_end]
        if len(df_test) == 0:
            return (T_start, [])
        df_train_feat = compute_advanced_features(df_train)
        df_test_feat = compute_advanced_features(df_test)

    if len(df_test_feat) == 0:
        return (T_start, [])
    numeric_features = [
        'strike', 'volume', 'open_interest', 'vol_oi_ratio', 'implied_vol',
        'underlier_price', 'premium', 'dte',
        'moneyness', 'iv_hv_ratio', 'vix_level', 'log_premium'
    ]
    categorical_features = ['option_type', 'side', 'trend_alignment', 'dte_bucket']

    X_train = df_train_feat[['ticker'] + numeric_features + categorical_features]
    y_train = df_train_feat['observed_return']

    # PHASE B (PARALLELIZATION_PLAN 6.2): StandardScaler removed for consistency
    # with the training pipeline in api_train_model. Both paths must agree on
    # preprocessing, otherwise the walkforward inner models and the outer
    # production model see differently-scaled features.
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

    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = {}
    # Fit 5 quantile models SEQUENTIALLY inside this worker process.
    # Each LGBMRegressor uses n_jobs=4 for internal multi-threading.
    # The outer ProcessPoolExecutor isolates workers into separate
    # processes so there is no GIL contention between the outer and
    # inner parallelism levels.
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

    X_test = df_test_feat[['ticker'] + numeric_features + categorical_features]

    p10_preds = models[0.1].predict(X_test)
    p25_preds = models[0.25].predict(X_test)
    p50_preds = models[0.5].predict(X_test)
    p75_preds = models[0.75].predict(X_test)
    p90_preds = models[0.9].predict(X_test)

    step_preds = []
    # Use df_test_feat (always defined in both if/else paths above). df_test is only
    # assigned in the else branch — referencing it here raised UnboundLocalError after
    # the df_real_feat pre-compute optimization was added. df_test_feat has the same
    # original columns as df_test plus the engineered features, so the `row` payload is
    # a superset of what callers expect.
    for idx_test, (row_idx, row) in enumerate(df_test_feat.iterrows()):
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
