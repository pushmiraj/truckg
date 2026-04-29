"""Evaluation utilities: classification report, regression metrics, feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def print_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label: str = "On-Time Delivery",
) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Classification Report — {label}")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Delayed", "On-Time"]))

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC : {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")


def print_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label: str = "Delivery Time",
) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    print(f"\n{'=' * 55}")
    print(f"  Regression Report — {label}")
    print("=" * 55)
    print(f"  RMSE : {rmse:.4f} hours")
    print(f"  MAE  : {mae:.4f} hours")
    print(f"  MAPE : {mape:.2f} %")


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: str | None = None,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Feature importance chart saved to {save_path}")
    else:
        plt.show()


def plot_maintenance_comparison(
    ev_df: pd.DataFrame,
    petrol_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Bar chart: EV vs Petrol yearly maintenance costs."""
    ev_mean = ev_df["yearly_maintenance_lks"].mean()
    petrol_mean = petrol_df["yearly_maintenance_ths"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution
    axes[0].hist(ev_df["yearly_maintenance_lks"], bins=8, color="seagreen", alpha=0.7, label="EV (LKS)")
    axes[0].hist(petrol_df["yearly_maintenance_ths"], bins=8, color="tomato", alpha=0.7, label="Petrol (THS)")
    axes[0].set_title("Maintenance Cost Distribution")
    axes[0].set_xlabel("Yearly Maintenance Cost")
    axes[0].legend()

    # Means
    axes[1].bar(["EV (LKS avg)", "Petrol (THS avg)"], [ev_mean, petrol_mean],
                color=["seagreen", "tomato"])
    axes[1].set_title("Average Yearly Maintenance Cost")
    axes[1].set_ylabel("Cost (respective units)")
    for i, v in enumerate([ev_mean, petrol_mean]):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

    plt.suptitle("EV vs Petrol Vehicle Maintenance — FreightIQ Analysis", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Maintenance chart saved to {save_path}")
    else:
        plt.show()


def plot_weather_delay_impact(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Grouped bar: on-time rate per weather condition."""
    summary = (
        df.groupby("condition_text")["on_time"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "on_time_rate", "count": "n"})
        .sort_values("on_time_rate")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(summary.index, summary["on_time_rate"] * 100, color="steelblue")
    ax.set_xlabel("On-Time Delivery Rate (%)")
    ax.set_title("Weather Condition vs On-Time Delivery Rate")
    ax.axvline(x=summary["on_time_rate"].mean() * 100, color="red", linestyle="--",
               label="Mean rate")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Weather impact chart saved to {save_path}")
    else:
        plt.show()
