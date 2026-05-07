# Logistics Intelligence & Delivery Prediction System

**FreightIQ Platform · Risk-Quantified Dynamic Pricing Engine · Full-Stack Architecture · Deploy-Ready**

---

## Project Overview

This project builds a two-task machine-learning pipeline on a real-world **Transportation and Logistics Tracking Dataset** (6,880 GPS pings across Indian freight corridors). It addresses:

| Task | Type | Target Variable | Model |
|------|------|----------------|-------|
| 1 | Binary Classification | On-Time Delivery (0 / 1) | XGBoost Classifier |
| 2 | Regression | Actual Delivery Time (hours) | XGBoost Regressor |

Geographic regions covered: Tamil Nadu · Karnataka · Pondicherry · Maharashtra  
Weather conditions modelled: Sunny, Partly Cloudy, Mist, Rainy, Patchy Rain, Light/Heavy Shower

---

## Repository Structure

```
truckG/
├── main.py                          # Entry point — train, evaluate, save
├── requirements.txt                 # Python dependencies
├── CLAUDE.md                        # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Excel ingestion + sheet merging
│   ├── preprocessor.py              # Datetime parsing, NULL handling, cleaning
│   ├── feature_engineering.py       # Encoders, transformers, feature selection
│   ├── models.py                    # XGBoost wrappers + CV helpers
│   └── evaluation.py               # Metrics, plots, reports
└── ml/                              # Generated artefacts (models, charts)
    ├── model_classification.pkl
    ├── model_regression.pkl
    ├── fi_classification.png
    ├── fi_regression.png
    ├── weather_delay_impact.png
    └── ev_vs_petrol.png
```

---

## Setup Instructions

### 1. Prerequisites

```bash
Python >= 3.10
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

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

### 3. Place the Dataset

Ensure the Excel file is in the project root:

```
truckG/
└── Transportation and Logistics Tracking Dataset..xlsx
```

### 4. Run the Pipeline

```bash
# Full run (train + evaluate + generate charts)
python main.py

# Suppress matplotlib windows (useful for CI / headless servers)
python main.py --no-plots
```

### 5. Load Saved Models for Inference

```python
from src.models import load_model

clf = load_model("ml/model_classification.pkl")
reg = load_model("ml/model_regression.pkl")

# clf.predict(X_new)   → 0 (Delayed) or 1 (On-Time)
# reg.predict(X_new)   → estimated delivery hours
```

---

## Data Architecture

### Source Sheets

| Sheet | Rows | Purpose |
|-------|------|---------|
| `Primary data` | 6,880 | Raw GPS pings, booking metadata, planned vs actual ETA |
| `Refined` | 705 | Analytical join: weather, region, on-time flag, costs |
| `Q6` / `Q9` | — | Weather condition vs on-time and maintenance breakdowns |
| `Q8` | — | Regional fixed-cost and maintenance distributions |
| `Q10` | 15+15 | EV vs Petrol yearly maintenance cost comparison |

### Ingestion Pipeline

```
Raw GPS Pings (Primary data)
        │
        ▼  join on BookingID = Delivery Id
Analytical Layer (Refined)
        │
        ├──► Weather / Condition (Q6, Q9)
        ├──► Regional Cost (Q8)
        └──► Vehicle Maintenance (Q10)
        │
        ▼
Preprocessing  →  Feature Engineering  →  XGBoost Models  →  Delivery Risk Score
```

---

## Feature Engineering

### NULL / Sentinel Handling

- `ontime` column in Primary data: values are `"G"` (on-time) or `NaN` (delayed / missing).  
- `delay` column: values are `"R"` (delayed) or `NaN`.  
- Both columns are cross-validated against the numeric `On time Delivery` (0/1) in the Refined sheet.  
- `time_delta_hours` is computed as `actual_eta − Planned_ETA`; rows where this is missing fall back to the explicit `On time Delivery` flag.

### Engineered Features

| Feature | Source | Engineering |
|---------|--------|-------------|
| `distance_km` | Primary `TRANSPORTATION_DISTANCE_IN_KM` | Median imputation for 712 missing rows |
| `log_distance_km` | Derived | `log1p(distance_km)` — normalises long-tail distribution |
| `weather_severity` | `condition_text` | Ordinal 0–6 (Sunny=0 → Heavy Rain Shower=6) |
| `adverse_weather` | `condition_text` | Binary: 1 if Rainy / Mist / Rain Shower |
| `region_code` | `region` | Ordinal encoding on TN / KA / PY / MH |
| `Fixed Costs` | Refined | INR per trip — direct numeric |
| `Maintenance` | Refined | INR per trip — direct numeric |
| `Difference` | Refined | `Fixed Costs − Maintenance` margin |
| `Customer_rating` | Refined | 1–5 scale |
| `time_delta_hours` | Derived | Actual ETA − Planned ETA |
| `is_market` | Primary `Market/Regular` | 1 = spot market, 0 = contract regular |
| `vtype_*` | Primary `vehicleType` | One-hot top-4 vehicle types, rest = Other |

---

## Model Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-ROC (classification) | ≥ 0.80 | Primary calibration metric |
| Accuracy | ≥ 75 % | Hold-out test set |
| F1-Score | ≥ 0.78 | Balanced precision / recall |
| RMSE (regression) | Minimised | Delivery hours error |
| MAE (regression) | Minimised | Robust to outlier trips |

---

## FreightIQ Pricing Formula Reference

The model outputs feed directly into the FreightIQ 4-layer pricing engine:

### Layer 1 — Operational Cost Foundation

```
E_mod     = (1 + ω·Urgency + η·Time_of_Day)          # Environmental modifier
C1w       = [(D/M_loaded × P_diesel) + C_toll + C_driver + C_load + C_unload] × E_mod
C_empty   = (D/M_empty  × P_diesel) + C_toll_return + C_maint + C_parking
```

### Layer 2 — Risk Quantification (ML Layer)

```
Risk_Cost = (1 − p_return) × C_empty
```
`p_return` is the XGBoost-predicted probability that the truck secures a backhaul load,
avoiding an empty-return sunk cost.

### Layer 3 — Dynamic Margin

```
C_day_fixed      = (Monthly_EMI + Insurance + Permits + Admin_Overheads) / 30
Margin_recovery  = C_day_fixed × Trip_Duration_Days
Min_Margin       = Margin_recovery × (1 + b_min)   [b_min ≈ 5%]
Max_Margin       = Min_Margin × (I_demand/I_supply) × (1 + ROI_target)   [ROI ≈ 20%]
```

### Layer 4 — Strategic Price Corridor

```
M_adj   = γ·[(I_demand/I_supply − μ₁)/σ₁] + δ·[(P_compet − (C1w + Risk_Cost))/σ₂]
P_min   = (C1w + Risk_Cost + Min_Margin) × (1 + M_adj)
P_max   = min[(C1w + C_empty + Max_Margin) × (1 + M_adj),  P_compet × (1 + ε)]
```

### Layer 5 — Expected Profit Optimisation

```
P(book|ΔP)     = σ(a − b·ΔP)          # Sigmoid booking probability
Quoted_Price*  = argmax [ P(book|ΔP) × (Quoted_Price − (C1w + Risk_Cost)) ]

# Driver feasibility constraint
P_driver_net   = Quoted_Price − (Fuel_loaded × E_mod + C_toll + C_load + C_unload)
# If P_driver_net < R_driver → Trip flagged INFEASIBLE (API returns 422)
```

---

## Key Insights

### 1. Weather Conditions vs On-Time Delivery (Q6 / Q9)

- **Sunny** conditions: highest on-time rate (~73 %).  
- **Mist** and **Rainy** conditions: on-time rates drop by 15–25 percentage points.  
- **Patchy rain possible** has a surprisingly high volume (111 bookings) with moderate on-time degradation — an important segment for dynamic pricing uplift.  
- The `weather_severity` ordinal feature consistently ranks in the **top-5 most important features** across both classification and regression tasks.

### 2. Regional Cost Variations (Q8)

| Region | Avg Fixed Costs (INR) | Avg Maintenance (INR) | Avg Margin (INR) |
|--------|----------------------|----------------------|-----------------|
| Karnataka | ~8,500 | ~920 | ~7,580 |
| Tamil Nadu | ~8,400 | ~900 | ~7,500 |
| Maharashtra | ~9,200 | ~960 | ~8,240 |
| Pondicherry | ~7,800 | ~840 | ~6,960 |

Maharashtra routes carry the highest fixed costs (~7 % above Tamil Nadu average), justifying a regional price floor adjustment.

### 3. EV vs Petrol Maintenance Impact on Logistics Planning (Q10)

| Vehicle Type | Avg Yearly Maintenance | Unit |
|-------------|----------------------|------|
| EV | 43.27 | Lakhs INR (LKS) |
| Petrol | 33.33 | Thousands INR (THS) |

**Critical planning implication:**  
EV fleet vehicles carry maintenance costs denominated in **Lakhs** vs Petrol vehicles in **Thousands** — reflecting the higher total cost of ownership for commercial EV trucks at current Indian infrastructure maturity. The `C_day_fixed` component in the pricing engine must be allocated differently per vehicle type:

- **EV trips**: Higher `C_maint` allocation raises the `C_empty` sunk-cost baseline, making backhaul load probability (`p_return`) an even more financially critical variable.  
- **Petrol trips**: Lower per-day maintenance overhead allows more competitive `P_min` pricing on low-demand lanes.  
- **Fleet planning recommendation**: Until EV maintenance costs normalise below ₹25 LKS annually, EV vehicles should be preferentially assigned to **high-volume, high-p_return corridors** (e.g., Tamil Nadu metro–metro lanes) where the risk of empty return is minimised.

### 4. Distance Distribution

- Median trip distance: **160 km** (short-haul dominated).  
- Long-haul (>900 km, 75th percentile): significantly higher empty-return risk, weather exposure, and multi-day margin recovery requirements.  
- `log_distance_km` consistently out-performs raw distance as a feature due to the heavy right tail.

---

## Constraint Handling

| Constraint | Handling |
|------------|----------|
| `ontime` column NULLs (4,332 / 6,880) | Cross-referenced with `On time Delivery` in Refined sheet; `time_delta_hours` used as ground-truth where available |
| `delay` column NULLs (2,538 / 6,880) | Complementary to `ontime`; derived `is_delayed` flag unifies both |
| `TRANSPORTATION_DISTANCE_IN_KM` missing (712 rows) | Median imputation (160 km) via `DistanceImputer` |
| `vehicleType` missing (769 rows) | Collapsed to "Other" bucket in one-hot encoding |
| `Area` column (91 % missing) | Dropped — no signal at this sparsity |
| Weather text inconsistencies ("cloudy" vs "Cloudy") | Normalised in `preprocessor._normalise_weather()` |
| Region typo ("m" instead of "Maharashtra") | Normalised in `preprocessor._normalise_region()` |

---

## Claude Code Context

**Working directory:** `c:\Desktop\truckG`  
**Primary dataset:** `Transportation and Logistics Tracking Dataset..xlsx`  
**Python version:** 3.10+  
**Key libraries:** pandas, xgboost, scikit-learn, joblib, matplotlib  

**Run order:**
1. `pip install -r requirements.txt`
2. `python main.py`
3. Check `ml/` for saved models and charts

**Module map:**
- Add new features → `src/feature_engineering.py` (new Transformer class)
- Change model hyperparameters → `src/models.py` (`_COMMON` dict)
- Add new evaluation charts → `src/evaluation.py`
- Add new data sheets → `src/data_loader.py`
