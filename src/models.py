"""XGBoost classification and regression model wrappers."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor


# ---------------------------------------------------------------------------
# Shared hyper-parameters tuned for small-to-medium logistics datasets
# ---------------------------------------------------------------------------
_COMMON = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)


def build_classifier() -> XGBClassifier:
    return XGBClassifier(
        **_COMMON,
        eval_metric="auc",
        scale_pos_weight=1,
    )


def build_regressor() -> XGBRegressor:
    return XGBRegressor(
        **_COMMON,
        eval_metric="rmse",
        objective="reg:squarederror",
    )


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def cv_classify(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """Return mean AUC-ROC and accuracy over stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc").mean()
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1").mean()
    return {"cv_auc": round(auc, 4), "cv_accuracy": round(acc, 4), "cv_f1": round(f1, 4)}


def cv_regress(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """Return mean neg-RMSE and neg-MAE over k-fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error").mean()
    )
    mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    return {"cv_rmse": round(rmse, 4), "cv_mae": round(mae, 4)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
