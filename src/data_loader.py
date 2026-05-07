"""Data loading and sheet merging for the Logistics Prediction System."""

from __future__ import annotations

import pandas as pd


EXCEL_FILE = "Transportation and Logistics Tracking Dataset..xlsx"


def load_primary() -> pd.DataFrame:
    """Load the raw GPS / booking sheet."""
    return pd.read_excel(EXCEL_FILE, sheet_name="Primary data")


def load_refined() -> pd.DataFrame:
    """Load the pre-joined analytical sheet (main modelling source)."""
    return pd.read_excel(EXCEL_FILE, sheet_name="Refined")


def load_q10_maintenance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ev_df, petrol_df) yearly maintenance cost tables."""
    raw = pd.read_excel(EXCEL_FILE, sheet_name="Q10")

    ev = (
        raw[["S.no", "Vehicle Type", "Yearly Maintenance Cost (INR)(LKS)"]]
        .dropna(subset=["Vehicle Type"])
        .rename(columns={
            "S.no": "id",
            "Vehicle Type": "vehicle_type",
            "Yearly Maintenance Cost (INR)(LKS)": "yearly_maintenance_lks",
        })
        .query("vehicle_type == 'EV'")
        .reset_index(drop=True)
    )

    petrol = (
        raw[["S.no.1", "Vehicle Type.1", "Yearly Maintenance Cost (INR)(THS)"]]
        .dropna(subset=["Vehicle Type.1"])
        .rename(columns={
            "S.no.1": "id",
            "Vehicle Type.1": "vehicle_type",
            "Yearly Maintenance Cost (INR)(THS)": "yearly_maintenance_ths",
        })
        .query("vehicle_type == 'Petrol'")
        .reset_index(drop=True)
    )

    return ev, petrol


def load_regional_costs() -> pd.DataFrame:
    """Load Q8 regional fixed-cost / maintenance breakdown."""
    return pd.read_excel(EXCEL_FILE, sheet_name="Q8")[
        ["region", "Fixed Costs", "Maintenance", "Difference"]
    ].dropna(subset=["region"])


def build_model_dataset() -> pd.DataFrame:
    """
    Merge Refined sheet with key columns from Primary data.
    Returns a single DataFrame ready for preprocessing.
    """
    refined = load_refined()
    primary = load_primary()[
        [
            "BookingID",
            "TRANSPORTATION_DISTANCE_IN_KM",
            "vehicleType",
            "Market/Regular ",
            "Planned_ETA",
            "actual_eta",
        ]
    ].rename(columns={"Market/Regular ": "market_or_regular"})

    merged = pd.merge(
        refined,
        primary,
        left_on="Delivery Id",
        right_on="BookingID",
        how="left",
    ).drop(columns=["BookingID"])

    return merged
