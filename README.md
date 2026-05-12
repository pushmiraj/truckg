# FreightIQ — AI-Driven Logistics Intelligence & Dynamic Pricing Engine

> Risk-Quantified Dynamic Pricing · On-Time Delivery Prediction · Return-Load Probability Modelling  
> Built on a real-world Indian freight dataset (6,880 GPS pings across 4 states)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Dataset](#4-dataset)
5. [ML Pipeline — Three Models](#5-ml-pipeline--three-models)
6. [Risk-Quantified Dynamic Pricing Engine](#6-risk-quantified-dynamic-pricing-engine)
7. [Model Performance Results](#7-model-performance-results)
8. [Feature Engineering](#8-feature-engineering)
9. [Generated Artefacts](#9-generated-artefacts)
10. [Key Business Insights](#10-key-business-insights)
11. [Inference Examples](#11-inference-examples)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

FreightIQ is a three-model machine-learning pipeline that powers a **5-layer risk-quantified dynamic pricing engine** for Indian truck logistics. It answers three core business questions:

| Model | Task | Target | Algorithm |
|-------|------|--------|-----------|
| **Task 1** | Binary Classification | Will this delivery be on time? (0/1) | XGBoost Classifier |
| **Task 2** | Regression | How many hours will delivery take? | XGBoost Regressor |
| **Task 3** | Return-Load Probability | Will the truck secure a backhaul load? p_return ∈ [0,1] | XGBoost + Logistic Regression |

The output of Task 3 (`p_return`) feeds directly into the **5-layer pricing engine** as the risk quantification input, computing the optimal quoted freight price for every trip.

**Geographic coverage:** Tamil Nadu · Karnataka · Pondicherry · Maharashtra  
**Weather conditions:** Sunny · Partly Cloudy · Mist · Rainy · Patchy Rain · Light/Heavy Shower

---

## 2. Repository Structure

```
truckG/
├── main.py                        # Entry point — trains all 3 models, runs pricing demo
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── CLAUDE.md                      # Codebase instructions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Excel ingestion + multi-sheet merging
│   ├── preprocessor.py            # Datetime parsing, NULL handling, cleaning
│   ├── feature_engineering.py     # Encoders, transformers, feature selection
│   ├── models.py                  # XGBoost wrappers + CV helpers
│   ├── evaluation.py              # Metrics, plots, reports
│   ├── pricing.py                 # 5-layer dynamic pricing engine
│   └── return_load_model.py       # Task 3: p_return model (LR + XGBoost)
│
└── ml/                            # Generated artefacts (created by running main.py)
    ├── model_classification.pkl       # Task 1 — XGBoost on-time classifier
    ├── model_regression.pkl           # Task 2 — XGBoost delivery-time regressor
    ├── model_p_return_xgb.pkl         # Task 3 — XGBoost p_return model
    ├── model_p_return_lr.pkl          # Task 3 — Logistic Regression p_return model
    ├── model_p_return_best.pkl        # Task 3 — Winner model (XGBoost)
    ├── p_return_meta.pkl              # Feature list + winner name for inference
    ├── fi_classification.png          # Feature importance — Task 1
    ├── fi_regression.png              # Feature importance — Task 2
    ├── p_return_evaluation.png        # ROC, PR curves, distribution — Task 3
    ├── weather_delay_impact.png       # Weather vs on-time rate chart
    ├── ev_vs_petrol.png               # EV vs Petrol maintenance cost chart
    ├── pricing_scenarios.png          # 5 pricing scenario comparison
    ├── pricing_sensitivity_p_return.png  # Risk cost vs p_return sweep
    └── pricing_sensitivity_distance.png  # Pricing vs distance sweep
```

---

## 3. Quick Start

### Prerequisites
```
Python >= 3.10
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Place the dataset
```
truckG/
└── Transportation and Logistics Tracking Dataset..xlsx
```

### Run the full pipeline
```bash
# Train all 3 models + run pricing demo + generate all charts
python main.py

# Headless / CI mode (no matplotlib windows)
python main.py --no-plots
```

The pipeline will:
1. Load and clean the dataset
2. Train Task 1 (on-time classification) + save model
3. Train Task 2 (delivery time regression) + save model
4. Train Task 3 (p_return — return-load probability) + save both models
5. Run the 5-layer pricing engine on 5 representative freight scenarios
6. Print all output parameters and sensitivity tables
7. Save 14 artefacts to `ml/`

---

## 4. Dataset

### Source Sheets

| Sheet | Rows | Purpose |
|-------|------|---------|
| `Primary data` | 6,880 | Raw GPS pings, booking metadata, planned vs actual ETA |
| `Refined` | 705 | Analytical join: weather, region, on-time flag, costs |
| `Q6` / `Q9` | — | Weather condition vs on-time and maintenance |
| `Q8` | — | Regional fixed-cost and maintenance distributions |
| `Q10` | 15+15 | EV vs Petrol yearly maintenance cost comparison |

### Ingestion Pipeline

```
Raw GPS Pings (Primary data)
        |
        v   join on BookingID = Delivery Id
Analytical Layer (Refined)
        |
        |---> Weather / Condition (Q6, Q9)
        |---> Regional Cost (Q8)
        `---> Vehicle Maintenance (Q10)
        |
        v
Preprocessing --> Feature Engineering --> XGBoost Models --> Pricing Engine
```

### Constraint Handling

| Issue | Resolution |
|-------|-----------|
| `TRANSPORTATION_DISTANCE_IN_KM` missing (712 rows) | Median imputation (160 km) via `DistanceImputer` |
| `ontime` column NULLs (4,332 / 6,880) | Cross-referenced with `On time Delivery` in Refined sheet |
| `vehicleType` missing (769 rows) | Collapsed to "Other" in one-hot encoding |
| `Area` column (91% missing) | Dropped — no signal at this sparsity |
| Weather text inconsistencies ("cloudy" vs "Cloudy") | Normalised in `preprocessor._normalise_weather()` |
| Region typo ("m" instead of "Maharashtra") | Normalised in `preprocessor._normalise_region()` |

---

## 5. ML Pipeline — Three Models

### Task 1 — On-Time Delivery Classification

**Target:** `On time Delivery` (0 = Delayed, 1 = On-Time)  
**Algorithm:** XGBoost Classifier  
**Train/Test split:** 80/20, stratified  
**Cross-validation:** 5-fold stratified  

```
python main.py  # trains and saves to ml/model_classification.pkl
```

**Inference:**
```python
from src.models import load_model

clf = load_model("ml/model_classification.pkl")
prediction = clf.predict(X_new)      # 0 = Delayed, 1 = On-Time
probability = clf.predict_proba(X_new)[:, 1]   # delay risk score
```

---

### Task 2 — Delivery Time Regression

**Target:** `Delivery_Time` (hours)  
**Algorithm:** XGBoost Regressor  
**Objective:** reg:squarederror  
**Cross-validation:** 5-fold  

```python
reg = load_model("ml/model_regression.pkl")
hours = reg.predict(X_new)           # estimated delivery hours
```

---

### Task 3 — Return-Load Probability (p_return)

**Target proxy:** `is_market` — Market bookings (spot market = active corridor demand in both directions = higher return-load availability).  
**Class imbalance:** 40 Market vs 667 Regular (16.7x) — handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Logistic Regression).  
**Models trained:** Logistic Regression + XGBoost (both saved; best selected by Test AUC-ROC).  

**Features used (16 total):**

| Feature | Source | Why it matters for p_return |
|---------|--------|------------------------------|
| `log_distance_km` | Primary data | Longer routes = lower backhaul availability |
| `route_freq` | Origin + Destination | Lane volume = industrial hub activity proxy |
| `day_of_week` | `Planned_ETA` | Weekend vs weekday demand patterns |
| `month` | `Planned_ETA` | Seasonal freight demand |
| `hour` | `Planned_ETA` | Time-of-day market activity |
| `is_weekend` | `Planned_ETA` | Weekend demand dip |
| `weather_severity` | `condition_text` | Bad weather reduces available trucks |
| `adverse_weather` | `condition_text` | Binary rain/mist flag |
| `region_code` | `region` | TN/KA/PY/MH ordinal (MH highest cost) |
| `Fixed Costs` | Refined | Route cost tier signal |
| `Maintenance` | Refined | Route cost tier signal |
| `Difference` | Refined | Margin band |
| `Customer_rating` | Refined | Shipper reliability |
| `on_time` | Refined | Route demand/reliability indicator |
| `time_delta_hours` | ETAs | Route performance |

**Inference:**
```python
from src.return_load_model import load_p_return_model, predict_p_return

model = load_p_return_model("ml")
p = predict_p_return(model, {
    "log_distance_km": 5.08,
    "route_freq": 12,
    "day_of_week": 1,
    "month": 8,
    "region_code": 0,
    # ... other features
})
# p = 0.73  --> 73% chance of securing return load
```

---

## 6. Risk-Quantified Dynamic Pricing Engine

The three ML models feed into a **5-layer pricing engine** (`src/pricing.py`) that computes the optimal quoted freight price.

### Layer 1 — Operational Base Cost

```
F_loaded  = D / M_loaded
F_empty   = D / M_empty
Base_cost = (F_loaded + F_empty) * P_diesel + C_toll + C_driver + C_maint + C_load + C_unload

E_mod     = 1 + omega * Urgency + eta * Time_of_Day_Factor
Base_cost_modified = Base_cost * E_mod
```

### Layer 2 — Risk Quantification (ML Layer)

```
Empty_return_cost = F_empty * P_diesel + C_toll + C_driver + C_maint + C_parking
Risk_cost         = (1 - p_return) * Empty_return_cost        # p_return from Task 3 XGBoost
Risk_factor       = Risk_cost / Base_cost_modified
```

### Layer 3 — Cost-Optimal Model Price

```
x1 = (I_demand / I_supply - mu_1) / sigma_1       # demand/supply signal
x2 = ((P_compet - Base_cost) / Base_cost - mu_2) / sigma_2   # competitor gap signal
Market_adj  = gamma * x1 + delta * x2
Model_price = Base_cost_modified * (1 + Risk_factor + Market_adj)
```

### Layer 4 — Market-Feasible Quoted Price

```
Market_cap   = P_compet * (1 + epsilon)            # epsilon ~ 7%
Quoted_price = min(Model_price, Market_cap)
```

### Layer 5 — Profit Optimisation & Driver Feasibility

```
Delta_P          = Quoted_price - P_compet
P(book | Delta_P) = sigmoid(a - b * Delta_P)       # a, b learned from booking data
E[Profit]        = P(book) * (Quoted_price - Base_cost_modified)

Driver_earnings  = Quoted_price - (F_loaded * E_mod + C_toll + C_load + C_unload)
Feasible         = Driver_earnings >= R_driver      # if False -> trip flagged INFEASIBLE
```

### Pricing Engine Usage

```python
from src.pricing import FreightPricingEngine, PricingInput

engine = FreightPricingEngine()
result = engine.compute(PricingInput(
    label="Chennai to Pune",
    distance_km=1_350.0,
    m_loaded=3.8,
    m_empty=5.5,
    p_diesel=93.5,
    c_toll=2_200.0,
    c_driver=3_000.0,
    c_maint=700.0,
    c_load=500.0,
    c_unload=500.0,
    p_compet=55_000.0,
    i_demand=1.3,
    i_supply=0.9,
    p_return=0.41,        # from ml/model_p_return_best.pkl
))

print(result.summary())
# Prints all 5 layers with INR values
```

**Sensitivity sweeps available:**
```python
# Sweep p_return 0 -> 1
df = engine.sweep_p_return(base_input, n=11)

# Sweep distances [50, 100, 160, 250, 400, 600, 900 km]
df = engine.sweep_distance(base_input)
```

---

## 7. Model Performance Results

### Task 1 — On-Time Delivery Classification

| Metric | CV (5-fold) | Hold-out Test |
|--------|-------------|---------------|
| AUC-ROC | 0.9806 | 0.9947 |
| Accuracy | 96.46% | 96.48% |
| F1-Score | 0.9753 | 0.9804 |
| Precision (On-Time) | — | 0.97 |
| Recall (On-Time) | — | 0.98 |

Confusion matrix (test set, 142 samples):

```
              Predicted
              Delayed   On-Time
Actual  Delayed    35        3
        On-Time     2      102
```

### Task 2 — Delivery Time Regression

| Metric | CV (5-fold) | Hold-out Test |
|--------|-------------|---------------|
| RMSE | 21.38 hrs | 25.16 hrs |
| MAE | 17.30 hrs | 18.99 hrs |

*Note: High MAPE is expected on short-haul dominated datasets where small absolute errors produce large percentage errors near zero.*

### Task 3 — Return-Load Probability

| Model | CV AUC-ROC | Test AUC-ROC | Avg Precision |
|-------|-----------|-------------|---------------|
| Logistic Regression | 0.8731 | 0.8778 | 0.3067 |
| **XGBoost (Winner)** | **0.9320** | **0.9711** | **0.7091** |

XGBoost confusion matrix (test set, 142 samples, 16.7x imbalance):

```
              Predicted
              Regular   Market
Actual  Regular   130       4
        Market      3       5
```

**Top features by XGBoost importance:**

```
route_freq          0.1917  ######################################
log_distance_km     0.1640  ################################
distance_km         0.1120  ######################
region_code         0.0826  ################
time_delta_hours    0.0629  ############
on_time             0.0520  ##########
month               0.0512  ##########
is_weekend          0.0445  ########
```

---

## 8. Feature Engineering

All transformers are sklearn-compatible (`BaseEstimator`, `TransformerMixin`) and live in `src/feature_engineering.py`.

| Transformer | Input | Output |
|-------------|-------|--------|
| `WeatherEncoder` | `condition_text` (free text) | `weather_severity` ordinal 0–6 |
| `RegionEncoder` | `region` (free text) | `region_code` ordinal 0–3 |
| `VehicleTypeEncoder` | `vehicleType` (free text) | One-hot top-4 types, rest = Other |
| `MarketTypeEncoder` | `market_or_regular` | `is_market` binary |
| `DistanceImputer` | `distance_km` (712 NULLs) | Median-imputed + `log_distance_km` |

**Weather severity scale:**

| Score | Conditions |
|-------|-----------|
| 0 | Sunny, Clear |
| 1 | Partly Cloudy, Cloudy |
| 2 | Overcast |
| 3 | Mist, Patchy Rain Possible |
| 4 | Light Rain Shower, Patchy Light Rain with Thunder |
| 5 | Rainy |
| 6 | Moderate or Heavy Rain Shower |

---

## 9. Generated Artefacts

All artefacts are written to `ml/` when you run `python main.py`.

| File | Type | Description |
|------|------|-------------|
| `model_classification.pkl` | Model | XGBoost on-time delivery classifier |
| `model_regression.pkl` | Model | XGBoost delivery time regressor |
| `model_p_return_xgb.pkl` | Model | XGBoost return-load probability |
| `model_p_return_lr.pkl` | Model | Logistic Regression return-load probability |
| `model_p_return_best.pkl` | Model | Best p_return model (XGBoost, AUC 0.9711) |
| `p_return_meta.pkl` | Metadata | Feature list + winner name for inference |
| `fi_classification.png` | Chart | Top-15 feature importances — Task 1 |
| `fi_regression.png` | Chart | Top-15 feature importances — Task 2 |
| `p_return_evaluation.png` | Chart | ROC + PR curves + distribution + FI — Task 3 |
| `weather_delay_impact.png` | Chart | On-time rate by weather condition |
| `ev_vs_petrol.png` | Chart | EV vs Petrol yearly maintenance cost |
| `pricing_scenarios.png` | Chart | 5-scenario cost vs quoted price comparison |
| `pricing_sensitivity_p_return.png` | Chart | Risk cost / profit vs p_return sweep |
| `pricing_sensitivity_distance.png` | Chart | All pricing components vs trip distance |

---

## 10. Key Business Insights

### Weather vs On-Time Delivery
- **Sunny** conditions: highest on-time rate (~73%)
- **Mist** and **Rainy** conditions: on-time rate drops 15–25 percentage points
- `weather_severity` consistently ranks **top-5 features** across all three models

### Regional Cost Variations

| Region | Avg Fixed Costs | Avg Maintenance | Avg Margin |
|--------|----------------|----------------|-----------|
| Karnataka | ~INR 8,500 | ~INR 920 | ~INR 7,580 |
| Tamil Nadu | ~INR 8,400 | ~INR 900 | ~INR 7,500 |
| Maharashtra | ~INR 9,200 | ~INR 960 | ~INR 8,240 |
| Pondicherry | ~INR 7,800 | ~INR 840 | ~INR 6,960 |

Maharashtra routes are 7% costlier than Tamil Nadu — justifying a regional price floor adjustment in the pricing engine.

### EV vs Petrol Fleet Planning

| Vehicle Type | Avg Yearly Maintenance | Unit |
|-------------|----------------------|------|
| EV | 43.27 | Lakhs INR |
| Petrol | 33.33 | Thousands INR |

EV trucks carry maintenance costs in **Lakhs** vs Petrol in **Thousands** — a ~13x difference in absolute cost. Implication for pricing:
- **EV trips:** Higher `C_maint` raises the empty-return cost floor, making `p_return` the most financially critical variable
- **Recommendation:** Assign EV trucks preferentially to high-volume, high-`p_return` corridors (Tamil Nadu metro-metro lanes) until EV maintenance costs normalise

### Distance & Pricing Feasibility
- Median trip: **160 km** (short-haul dominated)
- At current competitor pricing (~INR 11,500 cap), trips **> 500 km** become unprofitable
- At **600+ km**, driver earnings turn negative — trip flagged INFEASIBLE by the pricing engine

---

## 11. Inference Examples

### Full pricing quote for a trip

```python
import joblib
from src.pricing import FreightPricingEngine, PricingInput
from src.return_load_model import predict_p_return

# Load models
p_return_model = joblib.load("ml/model_p_return_best.pkl")

# Get p_return from ML model
trip_features = {
    "log_distance_km": 5.08,
    "distance_km": 160.0,
    "day_of_week": 1,           # Tuesday
    "month": 8,                  # August
    "hour": 10,
    "is_weekend": 0,
    "route_freq": 8,
    "weather_severity": 3,
    "adverse_weather": 1,
    "region_code": 0,            # Tamil Nadu
    "Fixed Costs": 8400,
    "Maintenance": 900,
    "Difference": 7500,
    "Customer_rating": 4.2,
    "on_time": 1,
    "time_delta_hours": -2.5,
}
p_return = predict_p_return(p_return_model, trip_features)

# Compute full quoted price
engine = FreightPricingEngine()
result = engine.compute(PricingInput(
    label="Chennai Central",
    distance_km=160.0,
    p_return=p_return,
    p_compet=13_000.0,
    i_demand=1.3,
    i_supply=1.0,
))
print(result.summary())
```

### Check on-time delivery probability

```python
from src.models import load_model
import pandas as pd

clf = load_model("ml/model_classification.pkl")

X_new = pd.DataFrame([{
    "distance_km": 160,
    "log_distance_km": 5.08,
    "weather_severity": 3,
    "adverse_weather": 1,
    "region_code": 0,
    "Fixed Costs": 8400,
    "Maintenance": 900,
    "Difference": 7500,
    "Customer_rating": 4,
    "time_delta_hours": 0,
    "is_market": 0,
}])

print(clf.predict(X_new))           # [1] = On-Time
print(clf.predict_proba(X_new))     # [[0.07, 0.93]]
```

---

## 12. Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
openpyxl>=3.1.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Module Map (for contributors)

| To change... | Edit file | Where |
|---|---|---|
| Add new features | `src/feature_engineering.py` | New `Transformer` class |
| Tune model hyperparameters | `src/models.py` | `_COMMON` dict |
| Add evaluation charts | `src/evaluation.py` | New `plot_*` function |
| Add new data sheets | `src/data_loader.py` | New `load_*` function |
| Change pricing formula | `src/pricing.py` | `FreightPricingEngine.compute()` |
| Add p_return features | `src/return_load_model.py` | `FEATURES` list + `_engineer_features()` |

---

*FreightIQ — Built on Transportation and Logistics Tracking Dataset (Indian freight corridors)*
