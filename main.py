"""
Logistics Intelligence & Delivery Prediction System
====================================================
Entry point for training and evaluating the two-model pipeline:

    Task 1 — Binary Classification : On-Time Delivery (0 = Delayed, 1 = On-Time)
    Task 2 — Regression             : Actual Delivery Time (hours)

Usage
-----
    python main.py                    # train, evaluate, save models
    python main.py --no-plots         # skip matplotlib output
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import build_model_dataset, load_q10_maintenance
from src.preprocessor import clean, drop_low_signal_columns
from src.feature_engineering import (
    build_feature_pipeline,
    apply_transformers,
    select_features,
)
from src.models import (
    build_classifier,
    build_regressor,
    cv_classify,
    cv_regress,
    save_model,
)
from src.evaluation import (
    print_classification_metrics,
    print_regression_metrics,
    plot_feature_importance,
    plot_maintenance_comparison,
    plot_weather_delay_impact,
    plot_pricing_scenarios,
    plot_pricing_sensitivity,
    plot_distance_sensitivity,
    plot_p_return_roc_pr,
)
from src.pricing import FreightPricingEngine, DEMO_SCENARIOS, PricingInput
from src.return_load_model import build_return_load_dataset, train_return_load_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = Path("ml")
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def prepare_data(
    show_plots: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and engineer features.  Returns (X, y_clf, y_reg)."""
    print("Loading dataset …")
    df = build_model_dataset()
    print(f"  Raw shape : {df.shape}")

    print("Cleaning data …")
    df = clean(df)

    # Exploratory chart: weather vs on-time rate
    if show_plots:
        plot_weather_delay_impact(
            df[df["condition_text"].notna()],
            save_path="ml/weather_delay_impact.png",
        )

    # Separate targets before dropping columns that would cause leakage
    y_clf = df["on_time"].dropna()
    y_reg = df["Delivery_Time"].dropna()

    df = drop_low_signal_columns(df)

    print("Engineering features …")
    transformers = build_feature_pipeline()
    df = apply_transformers(df, transformers, fit=True)
    X = select_features(df)

    # Align indices after dropna
    common_clf = X.index.intersection(y_clf.index)
    common_reg = X.index.intersection(y_reg.index)

    X_clf = X.loc[common_clf]
    y_clf = y_clf.loc[common_clf]

    X_reg = X.loc[common_reg]
    y_reg = y_reg.loc[common_reg]

    print(f"  Classification samples : {len(X_clf)}")
    print(f"  Regression    samples  : {len(X_reg)}")

    return X_clf, y_clf, X_reg, y_reg


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_classification(
    X: pd.DataFrame,
    y: pd.Series,
    show_plots: bool = True,
) -> None:
    print("\n--- Task 1: On-Time Delivery Classification ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = build_classifier()

    print("Running 5-fold cross-validation …")
    cv_scores = cv_classify(clf, X_train, y_train)
    print(f"  CV AUC-ROC  : {cv_scores['cv_auc']}")
    print(f"  CV Accuracy : {cv_scores['cv_accuracy']}")
    print(f"  CV F1       : {cv_scores['cv_f1']}")

    print("Fitting final classifier …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print_classification_metrics(y_test, y_pred, y_prob)

    if show_plots:
        plot_feature_importance(
            clf,
            feature_names=list(X.columns),
            title="Feature Importance — On-Time Classification",
            save_path="ml/fi_classification.png",
        )

    save_model(clf, "ml/model_classification.pkl")


def train_regression(
    X: pd.DataFrame,
    y: pd.Series,
    show_plots: bool = True,
) -> None:
    print("\n--- Task 2: Delivery Time Regression ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = build_regressor()

    print("Running 5-fold cross-validation …")
    cv_scores = cv_regress(reg, X_train, y_train)
    print(f"  CV RMSE : {cv_scores['cv_rmse']:.4f} hours")
    print(f"  CV MAE  : {cv_scores['cv_mae']:.4f} hours")

    print("Fitting final regressor …")
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print_regression_metrics(y_test, y_pred)

    if show_plots:
        plot_feature_importance(
            reg,
            feature_names=list(X.columns),
            title="Feature Importance — Delivery Time Regression",
            save_path="ml/fi_regression.png",
        )

    save_model(reg, "ml/model_regression.pkl")


def train_p_return(show_plots: bool = True) -> None:
    """Train the return-load probability model and save all artefacts."""
    print("\n" + "=" * 62)
    print("  Task 3: Return-Load Probability Model  (p_return)")
    print("=" * 62)
    print(
        "\n  Target proxy: is_market  (Market=1 = spot lane active"
        " => return load available)"
    )
    print("  Models: Logistic Regression  +  XGBoost  (PDF spec)")

    print("\nBuilding return-load dataset ...")
    X_rl, y_rl = build_return_load_dataset()
    print(f"  Feature matrix : {X_rl.shape}")
    print(f"  Features       : {list(X_rl.columns)}")

    eval_dict = train_return_load_model(X_rl, y_rl, save_dir="ml")

    if show_plots:
        plot_p_return_roc_pr(eval_dict, save_path="ml/p_return_evaluation.png")


def run_pricing_demo(show_plots: bool = True) -> None:
    """Run the 5-layer Risk-Quantified Dynamic Pricing Engine on all demo scenarios."""
    print("\n" + "=" * 62)
    print("  FreightIQ — Risk-Quantified Dynamic Pricing Engine")
    print("=" * 62)

    engine = FreightPricingEngine()
    results = []

    for scenario in DEMO_SCENARIOS:
        result = engine.compute(scenario)
        results.append(result)
        print(result.summary())

    # ── Sensitivity: p_return sweep (using Scenario 1 as baseline) ────────
    print("\n--- p_return Sensitivity Table (Scenario: TN Metro baseline) ---")
    sweep_df = engine.sweep_p_return(DEMO_SCENARIOS[0], n=11)
    print(sweep_df.to_string(index=False))

    # ── Sensitivity: distance sweep ────────────────────────────────────────
    print("\n--- Distance Sensitivity Table ---")
    dist_df = engine.sweep_distance(DEMO_SCENARIOS[0])
    print(dist_df.to_string(index=False))

    # ── Summary table across all scenarios ────────────────────────────────
    print("\n--- All-Scenario Summary ---")
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Scenario": r.label,
            "Base Cost": f"INR {r.base_cost_modified:,.0f}",
            "Risk Cost": f"INR {r.risk_cost:,.0f}",
            "Risk Factor": f"{r.risk_factor:.4f}",
            "Market Adj": f"{r.market_adj:.4f}",
            "Model Price": f"INR {r.model_price:,.0f}",
            "Quoted Price": f"INR {r.quoted_price:,.0f}",
            "Booking Prob": f"{r.p_book * 100:.1f}%",
            "Exp Profit": f"INR {r.expected_profit:,.0f}",
            "Driver Earn": f"INR {r.driver_earnings:,.0f}",
            "Feasible": "YES" if r.feasible else "NO",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    if show_plots:
        plot_pricing_scenarios(results, save_path="ml/pricing_scenarios.png")
        plot_pricing_sensitivity(
            sweep_df,
            title="TN Metro Baseline",
            save_path="ml/pricing_sensitivity_p_return.png",
        )
        plot_distance_sensitivity(dist_df, save_path="ml/pricing_sensitivity_distance.png")


def analyse_maintenance(show_plots: bool = True) -> None:
    print("\n--- EV vs Petrol Maintenance Cost Analysis ---")
    ev, petrol = load_q10_maintenance()

    ev_mean = ev["yearly_maintenance_lks"].mean()
    petrol_mean = petrol["yearly_maintenance_ths"].mean()
    print(f"  EV    avg yearly maintenance : {ev_mean:.2f} LKS")
    print(f"  Petrol avg yearly maintenance: {petrol_mean:.2f} THS")
    print(
        "  Note: EV units are LKS (Lakhs INR), Petrol units are THS (Thousands INR).\n"
        "  EV vehicles carry significantly higher absolute maintenance overhead —\n"
        "  a key input when computing per-trip fixed cost allocation."
    )

    if show_plots:
        plot_maintenance_comparison(ev, petrol, save_path="ml/ev_vs_petrol.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FreightIQ Logistics Prediction")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show = not args.no_plots

    X_clf, y_clf, X_reg, y_reg = prepare_data(show_plots=show)
    train_classification(X_clf, y_clf, show_plots=show)
    train_regression(X_reg, y_reg, show_plots=show)
    analyse_maintenance(show_plots=show)
    train_p_return(show_plots=show)
    run_pricing_demo(show_plots=show)

    print("\nDone. Artefacts written to ml/")


if __name__ == "__main__":
    main()
