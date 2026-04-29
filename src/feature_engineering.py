"""Feature engineering and encoding for the logistics prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


WEATHER_SEVERITY = {
    "Sunny": 0,
    "Clear": 0,
    "Partly cloudy": 1,
    "Cloudy": 1,
    "Overcast": 2,
    "Mist": 3,
    "Patchy rain possible": 3,
    "Light rain shower": 4,
    "Patchy light rain with thunder": 4,
    "Rainy": 5,
    "Moderate or heavy rain shower": 6,
}

REGION_ORDER = ["Tamil Nadu", "Karnataka", "Pondichery", "Maharashtra"]


class WeatherEncoder(BaseEstimator, TransformerMixin):
    """Maps free-text weather conditions to an ordinal severity score."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "condition_text" in X.columns:
            X["weather_severity"] = (
                X["condition_text"]
                .map(WEATHER_SEVERITY)
                .fillna(2)  # unknown → moderate
                .astype(int)
            )
            X = X.drop(columns=["condition_text"])
        return X


class RegionEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encodes region; unseen labels get the median rank."""

    def __init__(self):
        self._enc = OrdinalEncoder(
            categories=[REGION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    def fit(self, X, y=None):
        if "region" in X.columns:
            self._enc.fit(X[["region"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "region" in X.columns:
            X["region_code"] = self._enc.transform(X[["region"]]).astype(int)
            X = X.drop(columns=["region"])
        return X


class VehicleTypeEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode vehicleType; collapse rare categories into 'Other'."""

    TOP_N = 4

    def fit(self, X, y=None):
        if "vehicleType" in X.columns:
            counts = X["vehicleType"].value_counts()
            self._top = set(counts.head(self.TOP_N).index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "vehicleType" in X.columns:
            X["vehicleType"] = X["vehicleType"].where(
                X["vehicleType"].isin(self._top), other="Other"
            ).fillna("Other")
            dummies = pd.get_dummies(
                X["vehicleType"], prefix="vtype", drop_first=True, dtype=int
            )
            X = pd.concat([X.drop(columns=["vehicleType"]), dummies], axis=1)
        return X


class MarketTypeEncoder(BaseEstimator, TransformerMixin):
    """Binary encode Market / Regular."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        col = "market_or_regular"
        if col in X.columns:
            X["is_market"] = (X[col].str.strip() == "Market").astype(int)
            X = X.drop(columns=[col])
        return X


class DistanceImputer(BaseEstimator, TransformerMixin):
    """Median-impute missing distance values."""

    def fit(self, X, y=None):
        if "distance_km" in X.columns:
            self._median = X["distance_km"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "distance_km" in X.columns:
            X["distance_km"] = X["distance_km"].fillna(self._median)
            X["log_distance_km"] = np.log1p(X["distance_km"])
        return X


def build_feature_pipeline() -> list:
    """Return ordered list of sklearn-compatible transformers."""
    return [
        WeatherEncoder(),
        RegionEncoder(),
        VehicleTypeEncoder(),
        MarketTypeEncoder(),
        DistanceImputer(),
    ]


def apply_transformers(
    df: pd.DataFrame,
    transformers: list,
    fit: bool = True,
) -> pd.DataFrame:
    """Fit-transform or transform-only through a list of transformers."""
    for t in transformers:
        if fit:
            t.fit(df)
        df = t.transform(df)
    return df


FEATURE_COLUMNS = [
    "distance_km",
    "log_distance_km",
    "weather_severity",
    "adverse_weather",
    "region_code",
    "Fixed Costs",
    "Maintenance",
    "Difference",
    "Customer_rating",
    "time_delta_hours",
    "is_market",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only columns used as model inputs (fills any gaps with 0)."""
    vtype_cols = [c for c in df.columns if c.startswith("vtype_")]
    cols = FEATURE_COLUMNS + vtype_cols
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = 0
    return df[[c for c in cols if c in df.columns]].fillna(0)
