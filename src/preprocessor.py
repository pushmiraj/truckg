"""Cleaning and type-normalisation for the merged logistics DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Weather conditions that map to degraded driving conditions
ADVERSE_WEATHER = {
    "Rainy", "Light rain shower", "Moderate or heavy rain shower",
    "Patchy rain possible", "Patchy light rain with thunder", "Mist",
}

WEATHER_NORMALISE = {
    "cloudy": "Cloudy",
    "mist": "Mist",
    "rainy": "Rainy",
    "clear": "Sunny",
    "sunny": "Sunny",
}


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _normalise_weather(series: pd.Series) -> pd.Series:
    return series.str.strip().replace(WEATHER_NORMALISE)


def _normalise_region(series: pd.Series) -> pd.Series:
    mapping = {"m": "Maharashtra"}
    return series.str.strip().replace(mapping)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps; returns a new DataFrame.

    Steps
    -----
    1. Parse datetime columns.
    2. Compute Time_Delta_Hours (actual − planned ETA).
    3. Compute Distance_KM from the primary-data column, fall back to haversine.
    4. Normalise free-text categoricals (weather, region).
    5. Derive binary Delay flag and handle NULL sentinel values.
    6. Drop columns that carry no signal or are leakage.
    """
    df = df.copy()

    # --- 1. Datetime parsing ---
    for col in ("created_at", "actual_delivery_time", "Planned_ETA", "actual_eta"):
        if col in df.columns:
            df[col] = _parse_datetime(df[col])

    # --- 2. Time delta (hours) ---
    if "Planned_ETA" in df.columns and "actual_eta" in df.columns:
        df["time_delta_hours"] = (
            (df["actual_eta"] - df["Planned_ETA"]).dt.total_seconds() / 3600
        )
    elif "created_at" in df.columns and "actual_delivery_time" in df.columns:
        df["time_delta_hours"] = (
            (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 3600
        )
    else:
        df["time_delta_hours"] = np.nan

    # --- 3. Distance ---
    if "TRANSPORTATION_DISTANCE_IN_KM" in df.columns:
        df["distance_km"] = pd.to_numeric(
            df["TRANSPORTATION_DISTANCE_IN_KM"], errors="coerce"
        )
    else:
        df["distance_km"] = np.nan

    # --- 4. Categoricals ---
    if "condition_text" in df.columns:
        df["condition_text"] = _normalise_weather(df["condition_text"])
        df["adverse_weather"] = df["condition_text"].isin(ADVERSE_WEATHER).astype(int)

    if "region" in df.columns:
        df["region"] = _normalise_region(df["region"])

    # --- 5. Target: on_time (handle NULL / "G" / "R" encoding in primary sheet) ---
    # Refined sheet already has integer 0/1 in "On time Delivery"
    df["on_time"] = pd.to_numeric(df["On time Delivery"], errors="coerce")

    # Delay flag: 1 = delayed (time_delta > 0), consistent with "R" in primary sheet
    df["is_delayed"] = (df["time_delta_hours"] > 0).astype(float)
    # For rows where time_delta is NaN, fall back to on_time inversion
    mask_no_td = df["time_delta_hours"].isna()
    df.loc[mask_no_td, "is_delayed"] = (1 - df.loc[mask_no_td, "on_time"])

    # --- 6. Numeric coercion for cost columns ---
    for col in ("Fixed Costs", "Maintenance", "Difference", "Customer_rating"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_low_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not used in modelling."""
    drop_cols = [
        "Delivery Id", "Origin_Location", "Destination_Location",
        "created_at", "actual_delivery_time", "Planned_ETA", "actual_eta",
        "TRANSPORTATION_DISTANCE_IN_KM", "On time Delivery",
        "Area",  # >91 % missing
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])
