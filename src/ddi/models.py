from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBOOST_AVAILABLE = False

try:
    XGBOOST_PACKAGE_VERSION = version("xgboost")
except PackageNotFoundError:
    XGBOOST_PACKAGE_VERSION = "not installed"

BOOSTING_BACKEND = "XGBoost" if XGBOOST_AVAILABLE else "sklearn HistGradientBoosting fallback"


def train_logistic_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, seed: int
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3_000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_boosted_tree(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
    n_jobs: int = 1,
) -> Any:
    positives = max(1, int(np.sum(y_train == 1)))
    negatives = max(1, int(np.sum(y_train == 0)))
    if XGBOOST_AVAILABLE:
        assert XGBClassifier is not None
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=2.0,
            subsample=0.85,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=2.0,
            scale_pos_weight=negatives / positives,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
        )
        # XGBoost rejects otherwise valid pandas column names containing square
        # brackets (the AI4I sensor units use them extensively).  Train and
        # infer positionally while retaining the DataFrame schema externally.
        model.fit(X_train.to_numpy(), y_train)
        return model

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=n_estimators,
        max_leaf_nodes=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    sample_weight = np.where(y_train == 1, negatives / positives, 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def positive_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    model_input = X.to_numpy() if XGBOOST_AVAILABLE and isinstance(model, XGBClassifier) else X
    probabilities = model.predict_proba(model_input)  # type: ignore[attr-defined]
    return np.asarray(probabilities[:, 1], dtype=float)


def model_feature_importance(
    model: object,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)  # type: ignore[attr-defined]
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return np.asarray(result.importances_mean, dtype=float)
