"""
Return-Load Probability Model (p_return) -- FreightIQ
======================================================
Predicts the probability that a truck secures a backhaul (return) load
on a given lane and day, quantifying the empty-return sunk-cost risk.

Target variable proxy: is_market  (Market=1 / Regular=0)
  - A "Market" booking means the shipper found the truck through the spot
    market, indicating active freight demand in both directions on that
    corridor and day => higher return-load availability.
  - "Regular" = dedicated contract route => typically lower p_return.

Models trained (as specified in the dynamic-pricing specification):
  1. Logistic Regression  (baseline, interpretable, class_weight=balanced)
  2. XGBoost Classifier   (primary, handles non-linear interactions,
                           scale_pos_weight = neg/pos ratio)

Saved artefacts (all under ml/):
  model_p_return_xgb.pkl    -- XGBoost model
  model_p_return_lr.pkl     -- Logistic Regression pipeline
  model_p_return_best.pkl   -- Winner by Test AUC-ROC
  p_return_meta.pkl         -- Feature column list + winner name
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import build_model_dataset
from src.preprocessor import clean


# ---------------------------------------------------------------------------
# Weather severity map (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------

_WEATHER_SEVERITY = {
    "Sunny": 0, "Clear": 0,
    "Partly cloudy": 1, "Cloudy": 1,
    "Overcast": 2,
    "Mist": 3, "Patchy rain possible": 3,
    "Light rain shower": 4, "Patchy light rain with thunder": 4,
    "Rainy": 5,
    "Moderate or heavy rain shower": 6,
}

_ADVERSE = {
    "Rainy", "Light rain shower", "Moderate or heavy rain shower",
    "Patchy rain possible", "Mist",
}

_REGION_MAP = {
    "Tamil Nadu": 0, "Karnataka": 1, "Pondichery": 2, "Maharashtra": 3
}


# ---------------------------------------------------------------------------
# Feature engineering for p_return
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # Datetime features from Planned_ETA (day-of-week, season, hour)
    eta = pd.to_datetime(X.get("Planned_ETA"), errors="coerce")
    X["day_of_week"] = eta.dt.dayofweek.fillna(2).astype(int)   # 0=Mon
    X["month"] = eta.dt.month.fillna(6).astype(int)             # season proxy
    X["hour"] = eta.dt.hour.fillna(12).astype(int)
    X["is_weekend"] = (eta.dt.dayofweek >= 5).fillna(False).astype(int)

    # Route frequency (nearby industrial hub proxy via lane volume)
    if "Origin_Location" in X.columns and "Destination_Location" in X.columns:
        X["route_id"] = (
            X["Origin_Location"].fillna("UNK") + ">>"
            + X["Destination_Location"].fillna("UNK")
        )
        freq = X["route_id"].value_counts()
        X["route_freq"] = X["route_id"].map(freq).fillna(1).astype(float)
        X = X.drop(columns=["route_id"])
    else:
        X["route_freq"] = 1.0

    # Distance (median imputation 160 km)
    dist = pd.to_numeric(
        X.get("TRANSPORTATION_DISTANCE_IN_KM", X.get("distance_km")),
        errors="coerce",
    ).fillna(160.0)
    X["distance_km"] = dist
    X["log_distance_km"] = np.log1p(dist)

    # Weather
    cond = X.get("condition_text", pd.Series(["Sunny"] * len(X)))
    X["weather_severity"] = cond.map(_WEATHER_SEVERITY).fillna(2).astype(int)
    X["adverse_weather"] = cond.isin(_ADVERSE).astype(int)

    # Region ordinal
    X["region_code"] = X["region"].map(_REGION_MAP).fillna(1).astype(int) if "region" in X.columns else 1

    # Cost features (route tier signal)
    for col in ("Fixed Costs", "Maintenance", "Difference"):
        if col in X.columns:
            med = pd.to_numeric(X[col], errors="coerce").median()
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(med)

    # Customer rating
    if "Customer_rating" in X.columns:
        med = pd.to_numeric(X["Customer_rating"], errors="coerce").median()
        X["Customer_rating"] = pd.to_numeric(
            X["Customer_rating"], errors="coerce"
        ).fillna(med)

    # Route reliability proxies from preprocessor outputs
    if "on_time" in X.columns:
        X["on_time"] = pd.to_numeric(X["on_time"], errors="coerce").fillna(0.73)
    else:
        X["on_time"] = 0.73

    if "time_delta_hours" in X.columns:
        X["time_delta_hours"] = (
            pd.to_numeric(X["time_delta_hours"], errors="coerce").fillna(0.0)
        )
    else:
        X["time_delta_hours"] = 0.0

    return X


# Feature set for the model
FEATURES = [
    "log_distance_km",
    "distance_km",
    "day_of_week",
    "month",
    "hour",
    "is_weekend",
    "route_freq",
    "weather_severity",
    "adverse_weather",
    "region_code",
    "Fixed Costs",
    "Maintenance",
    "Difference",
    "Customer_rating",
    "on_time",
    "time_delta_hours",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_return_load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean, and engineer all features for p_return model."""
    df = build_model_dataset()
    df = clean(df)
    df = _engineer_features(df)

    # Target: Market=1 (spot lane => return load likely), Regular=0
    df["is_market"] = (
        df["market_or_regular"]
        .str.strip()
        .str.lower()
        .eq("market")
        .astype(int)
    )
    df = df.dropna(subset=["is_market"])

    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].fillna(0)
    y = df["is_market"]

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_return_load_model(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: str = "ml",
) -> dict:
    """
    Train Logistic Regression + XGBoost for p_return prediction.

    Both models address the lane-level class imbalance (Regular >> Market).
    Saves artefacts to save_dir/.  Returns full evaluation dict.
    """
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    ratio = n_neg / n_pos

    print(f"\n  Dataset  : {len(y)} trips  |  {n_pos} Market (pos)  "
          f"| {n_neg} Regular (neg)  |  imbalance {ratio:.1f}x")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    sep = "-" * 52

    # ── 1. Logistic Regression ────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  MODEL 1 -- Logistic Regression  (class_weight=balanced)")
    print(f"  {sep}")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=0.5,
            max_iter=2000,
            random_state=42,
        )),
    ])
    lr_cv_auc = cross_val_score(
        lr_pipe, X_tr, y_tr, cv=skf, scoring="roc_auc"
    ).mean()
    lr_cv_ap = cross_val_score(
        lr_pipe, X_tr, y_tr, cv=skf, scoring="average_precision"
    ).mean()
    lr_pipe.fit(X_tr, y_tr)
    lr_pred = lr_pipe.predict(X_te)
    lr_prob = lr_pipe.predict_proba(X_te)[:, 1]
    lr_auc  = roc_auc_score(y_te, lr_prob)
    lr_ap   = average_precision_score(y_te, lr_prob)

    print(f"    CV AUC-ROC          : {lr_cv_auc:.4f}")
    print(f"    CV Avg Precision    : {lr_cv_ap:.4f}")
    print(f"    Test AUC-ROC        : {lr_auc:.4f}")
    print(f"    Test Avg Precision  : {lr_ap:.4f}")
    print(f"\n  {sep}")
    print("  Classification Report (Logistic Regression on test set):")
    print(f"  {sep}")
    print(classification_report(y_te, lr_pred, target_names=["Regular", "Market"]))
    print("  Confusion Matrix:")
    cm_lr = confusion_matrix(y_te, lr_pred)
    print(f"    TN={cm_lr[0,0]}  FP={cm_lr[0,1]}")
    print(f"    FN={cm_lr[1,0]}  TP={cm_lr[1,1]}")

    # ── 2. XGBoost ────────────────────────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  MODEL 2 -- XGBoost  (scale_pos_weight={ratio:.1f})")
    print(f"  {sep}")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb_cv_auc = cross_val_score(
        xgb, X_tr, y_tr, cv=skf, scoring="roc_auc"
    ).mean()
    xgb_cv_ap = cross_val_score(
        xgb, X_tr, y_tr, cv=skf, scoring="average_precision"
    ).mean()
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)
    xgb_prob = xgb.predict_proba(X_te)[:, 1]
    xgb_auc  = roc_auc_score(y_te, xgb_prob)
    xgb_ap   = average_precision_score(y_te, xgb_prob)

    print(f"    CV AUC-ROC          : {xgb_cv_auc:.4f}")
    print(f"    CV Avg Precision    : {xgb_cv_ap:.4f}")
    print(f"    Test AUC-ROC        : {xgb_auc:.4f}")
    print(f"    Test Avg Precision  : {xgb_ap:.4f}")
    print(f"\n  {sep}")
    print("  Classification Report (XGBoost on test set):")
    print(f"  {sep}")
    print(classification_report(y_te, xgb_pred, target_names=["Regular", "Market"]))
    print("  Confusion Matrix:")
    cm_xgb = confusion_matrix(y_te, xgb_pred)
    print(f"    TN={cm_xgb[0,0]}  FP={cm_xgb[0,1]}")
    print(f"    FN={cm_xgb[1,0]}  TP={cm_xgb[1,1]}")

    # ── Feature importance (XGBoost) ──────────────────────────────────────
    fi = (
        pd.Series(xgb.feature_importances_, index=list(X.columns))
        .sort_values(ascending=False)
    )
    print(f"\n  {sep}")
    print("  XGBoost Feature Importances (top 10):")
    print(f"  {sep}")
    for fname, score in fi.head(10).items():
        bar = "#" * int(score * 200)
        print(f"    {fname:<25s}: {score:.4f}  {bar}")

    # ── p_return distribution on full dataset ──────────────────────────────
    all_probs_xgb = xgb.predict_proba(X)[:, 1]
    all_probs_lr  = lr_pipe.predict_proba(X)[:, 1]

    print(f"\n  {sep}")
    print("  p_return Distribution on Full Dataset:")
    print(f"  {sep}")
    for name, probs in [("XGBoost", all_probs_xgb), ("LogReg", all_probs_lr)]:
        print(f"  {name}:")
        print(f"    Min    : {probs.min():.4f}")
        print(f"    Max    : {probs.max():.4f}")
        print(f"    Mean   : {probs.mean():.4f}")
        print(f"    Median : {np.median(probs):.4f}")
        print(f"    Std    : {probs.std():.4f}")
        q = np.percentile(probs, [10, 25, 50, 75, 90])
        print(f"    P10/25/50/75/90: "
              f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f}")

    # ── Comparison summary ─────────────────────────────────────────────────
    print(f"\n  {sep}")
    print("  MODEL COMPARISON SUMMARY:")
    print(f"  {sep}")
    print(f"  {'Model':<26} {'CV AUC':>8} {'Test AUC':>10} {'Avg Prec':>10}")
    print(f"  {'-'*26} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'Logistic Regression':<26} {lr_cv_auc:>8.4f} {lr_auc:>10.4f} {lr_ap:>10.4f}")
    print(f"  {'XGBoost':<26} {xgb_cv_auc:>8.4f} {xgb_auc:>10.4f} {xgb_ap:>10.4f}")

    # ── Select best ────────────────────────────────────────────────────────
    if xgb_auc >= lr_auc:
        best_model = xgb
        best_name  = "XGBoost"
        best_auc   = xgb_auc
    else:
        best_model = lr_pipe
        best_name  = "LogisticRegression"
        best_auc   = lr_auc

    print(f"\n  >> Winner: {best_name}  (Test AUC-ROC = {best_auc:.4f})")

    # ── Save artefacts ─────────────────────────────────────────────────────
    out = Path(save_dir)
    out.mkdir(exist_ok=True)

    joblib.dump(xgb,        out / "model_p_return_xgb.pkl")
    joblib.dump(lr_pipe,    out / "model_p_return_lr.pkl")
    joblib.dump(best_model, out / "model_p_return_best.pkl")
    joblib.dump(
        {"features": list(X.columns), "best_model_name": best_name},
        out / "p_return_meta.pkl",
    )

    print(f"\n  Saved: ml/model_p_return_xgb.pkl")
    print(f"  Saved: ml/model_p_return_lr.pkl")
    print(f"  Saved: ml/model_p_return_best.pkl")
    print(f"  Saved: ml/p_return_meta.pkl")

    return {
        "best_model":  best_model,
        "best_name":   best_name,
        "xgb":         xgb,
        "lr_pipe":     lr_pipe,
        "xgb_auc":     xgb_auc,
        "lr_auc":      lr_auc,
        "xgb_cv_auc":  xgb_cv_auc,
        "lr_cv_auc":   lr_cv_auc,
        "xgb_ap":      xgb_ap,
        "lr_ap":       lr_ap,
        "fi":          fi,
        "X_te":        X_te,
        "y_te":        y_te,
        "xgb_prob":    xgb_prob,
        "lr_prob":     lr_prob,
        "cm_xgb":      cm_xgb,
        "cm_lr":       cm_lr,
        "all_probs_xgb": all_probs_xgb,
        "all_probs_lr":  all_probs_lr,
        "y_te":        y_te,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_p_return_model(model_dir: str = "ml"):
    """Load the saved best p_return model."""
    return joblib.load(Path(model_dir) / "model_p_return_best.pkl")


def predict_p_return(model, trip_features: dict, model_dir: str = "ml") -> float:
    """
    Predict p_return for a single trip dict.

    Parameters
    ----------
    model : fitted XGBoost or Pipeline
    trip_features : dict with raw numeric values for each feature in FEATURES

    Returns
    -------
    float : p_return in [0, 1]
    """
    meta = joblib.load(Path(model_dir) / "p_return_meta.pkl")
    row = pd.DataFrame([{f: trip_features.get(f, 0) for f in meta["features"]}])
    return float(model.predict_proba(row)[:, 1][0])
