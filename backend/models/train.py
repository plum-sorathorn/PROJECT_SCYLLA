"""Training pipeline helpers.

Extracted from ml_model.py: the parallelized quantile model fitting worker
and the C++ inference artifact serializer.
"""
import os
import json

from sklearn.pipeline import Pipeline
import lightgbm as lgb

try:
    from ..config.constants import CACHE_DIR
except ImportError:
    from config.constants import CACHE_DIR


def _fit_one_quantile_train(args):
    """
    Module-level helper for ProcessPoolExecutor (PARALLELIZATION_PLAN §4.1 / §4.2).
    Must be at module scope so multiprocessing can pickle it by name on Windows.

    args: (q, X_train, y_train, preprocessor, n_estimators, learning_rate,
            num_leaves, min_child_samples, reg_lambda)
    Returns: (q, fitted_Pipeline)
    """
    (q, X_train, y_train, preprocessor, n_estimators,
     learning_rate, num_leaves, min_child_samples, reg_lambda) = args

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            n_jobs=2,          # 2 cores per model × 5 workers = 10 cores
            random_state=42,
            verbose=-1,
        ))
    ])
    pipe.fit(X_train, y_train)
    return q, pipe


def _serialize_cpp_inference_artifacts(models, numeric_features, categorical_features, quantiles):
    """
    Persist trained LightGBM quantile Pipelines to backend/cache/cpp_inference/ for
    the C++ InferenceEngine. Writes:
      - scylla_q{10,25,50,75,90}.txt  (raw LightGBM text format, loaded by
        InferenceEngine::load via LGBM_BoosterLoadModelFromFile)
      - scylla_preprocessor.json      (numeric/categorical feature schema, imputer
        medians, OHE category maps)
    Idempotent — overwrites existing files. Safe to call from any path that has a
    freshly fitted {0.1, 0.25, 0.5, 0.75, 0.9} -> Pipeline dict (training,
    direct_dev, or walkforward).
    """
    ARTIFACT_DIR = os.path.join(CACHE_DIR, "cpp_inference")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    for q, pipe in models.items():
        booster = pipe.named_steps['regressor'].booster_
        booster.save_model(os.path.join(ARTIFACT_DIR, f"scylla_q{int(q*100)}.txt"))
    preprocessor_fit = models[0.5].named_steps['preprocess']
    num_medians = list(preprocessor_fit.named_transformers_['num']
                       .named_steps['imputer'].statistics_)
    ohe_categories = [list(c) for c in preprocessor_fit.named_transformers_['cat']
                      .named_steps['onehot'].categories_]
    with open(os.path.join(ARTIFACT_DIR, "scylla_preprocessor.json"), "w") as f:
        json.dump({
            "version": 1,
            "quantiles": quantiles,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_medians": num_medians,
            "ohe_categories": ohe_categories,
            "model_version_tag": "lightgbm_quantile_v2_no_scaler",
            "is_synthetic": True,
        }, f, indent=2)
