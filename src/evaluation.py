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
    roc_curve,
    precision_recall_curve,
    average_precision_score,
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


def plot_pricing_scenarios(
    results: list,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing key pricing outputs across scenarios."""
    labels = [r.label for r in results]
    base_costs = [r.base_cost_modified for r in results]
    risk_costs = [r.risk_cost for r in results]
    quoted_prices = [r.quoted_price for r in results]
    profits = [r.expected_profit for r in results]

    x = np.arange(len(labels))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cost stack + quoted price
    axes[0].bar(x - width, base_costs, width, label="Base Cost (modified)", color="steelblue")
    axes[0].bar(x, risk_costs, width, label="Risk Cost", color="tomato")
    axes[0].bar(x + width, quoted_prices, width, label="Quoted Price", color="seagreen")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("INR")
    axes[0].set_title("Cost Components vs Quoted Price by Scenario")
    axes[0].legend()

    # Expected profit + booking probability
    ax2b = axes[1].twinx()
    axes[1].bar(x, profits, width * 2, label="Expected Profit (INR)", color="gold", alpha=0.8)
    p_book_vals = [r.p_book * 100 for r in results]
    ax2b.plot(x, p_book_vals, "o--", color="purple", label="Booking Prob (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Expected Profit (INR)", color="goldenrod")
    ax2b.set_ylabel("Booking Probability (%)", color="purple")
    axes[1].set_title("Expected Profit & Booking Probability by Scenario")
    axes[1].legend(loc="upper left")
    ax2b.legend(loc="upper right")

    plt.suptitle("FreightIQ — Risk-Quantified Dynamic Pricing: Scenario Comparison", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Pricing scenario chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pricing_sensitivity(
    sweep_df: pd.DataFrame,
    title: str = "p_return Sensitivity Analysis",
    save_path: str | None = None,
) -> None:
    """Line charts: how risk cost, quoted price, and expected profit vary with p_return."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(sweep_df["p_return"], sweep_df["risk_cost"], color="tomato", linewidth=2)
    axes[0].set_xlabel("Return-Load Probability (p_return)")
    axes[0].set_ylabel("Risk Cost (INR)")
    axes[0].set_title("Risk Cost vs p_return")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_df["p_return"], sweep_df["model_price"], "--", color="steelblue",
                 label="Model Price", linewidth=2)
    axes[1].plot(sweep_df["p_return"], sweep_df["quoted_price"], color="seagreen",
                 label="Quoted Price", linewidth=2)
    axes[1].set_xlabel("Return-Load Probability (p_return)")
    axes[1].set_ylabel("Price (INR)")
    axes[1].set_title("Model Price vs Quoted Price")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(sweep_df["p_return"], sweep_df["expected_profit"], color="gold", linewidth=2)
    axes[2].set_xlabel("Return-Load Probability (p_return)")
    axes[2].set_ylabel("Expected Profit (INR)")
    axes[2].set_title("Expected Profit vs p_return")
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"FreightIQ Pricing Sensitivity — {title}", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Pricing sensitivity chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_distance_sensitivity(
    sweep_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Line chart: how costs and quoted price scale with distance."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(sweep_df["distance_km"], sweep_df["base_cost"], "o-",
            label="Base Cost", color="steelblue", linewidth=2)
    ax.plot(sweep_df["distance_km"], sweep_df["risk_cost"], "s--",
            label="Risk Cost", color="tomato", linewidth=2)
    ax.plot(sweep_df["distance_km"], sweep_df["quoted_price"], "^-",
            label="Quoted Price", color="seagreen", linewidth=2)
    ax.plot(sweep_df["distance_km"], sweep_df["expected_profit"], "D--",
            label="Expected Profit", color="gold", linewidth=2)

    ax.set_xlabel("One-Way Distance (km)")
    ax.set_ylabel("INR")
    ax.set_title("FreightIQ Pricing Components vs Trip Distance")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Distance sensitivity chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


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


def plot_p_return_roc_pr(
    eval_dict: dict,
    save_path: str | None = None,
) -> None:
    """
    2x2 grid: ROC curves, PR curves, probability distributions,
    and feature importance for the p_return model.
    """
    y_te      = eval_dict["y_te"]
    xgb_prob  = eval_dict["xgb_prob"]
    lr_prob   = eval_dict["lr_prob"]
    fi        = eval_dict["fi"]
    all_xgb   = eval_dict["all_probs_xgb"]
    all_lr    = eval_dict["all_probs_lr"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── ROC curves ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    for probs, name, color in [
        (xgb_prob, "XGBoost", "steelblue"),
        (lr_prob,  "LogReg",  "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y_te, probs)
        auc = roc_auc_score(y_te, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- p_return Model")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── PR curves ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    for probs, name, color in [
        (xgb_prob, "XGBoost", "steelblue"),
        (lr_prob,  "LogReg",  "tomato"),
    ]:
        prec, rec, _ = precision_recall_curve(y_te, probs)
        ap = average_precision_score(y_te, probs)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name}  (AP = {ap:.4f})")
    baseline = y_te.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- p_return Model")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── p_return distribution on full dataset ────────────────────────────
    ax = axes[1, 0]
    ax.hist(all_xgb, bins=30, color="steelblue", alpha=0.65, label="XGBoost p_return")
    ax.hist(all_lr,  bins=30, color="tomato",    alpha=0.55, label="LogReg p_return")
    ax.axvline(all_xgb.mean(), color="navy",    linestyle="--",
               label=f"XGB mean {all_xgb.mean():.3f}")
    ax.axvline(all_lr.mean(),  color="darkred", linestyle=":",
               label=f"LR mean {all_lr.mean():.3f}")
    ax.set_xlabel("Predicted p_return")
    ax.set_ylabel("Count")
    ax.set_title("p_return Distribution (Full Dataset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── XGBoost feature importance ───────────────────────────────────────
    ax = axes[1, 1]
    top_fi = fi.head(12)
    ax.barh(top_fi.index[::-1], top_fi.values[::-1], color="seagreen")
    ax.set_xlabel("Importance Score")
    ax.set_title("XGBoost Feature Importances -- p_return Model")
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "FreightIQ -- Return-Load Probability Model Evaluation", fontsize=13
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"p_return evaluation chart saved to {save_path}")
    else:
        plt.show()
    plt.close()
