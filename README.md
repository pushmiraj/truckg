# FreightIQ — AI-Driven Logistics Intelligence & Dynamic Pricing Engine

> Risk-Quantified Dynamic Pricing · On-Time Delivery Prediction · Return-Load Probability Modelling  
> Built on a real-world Indian freight dataset (6,880 GPS pings across 4 states)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Running the ML Pipeline](#4-running-the-ml-pipeline)
5. [Running the Web Application](#5-running-the-web-application)
6. [API Reference](#6-api-reference)
7. [Environment Variables](#7-environment-variables)
8. [Dataset](#8-dataset)
9. [ML Pipeline — Three Models](#9-ml-pipeline--three-models)
10. [Risk-Quantified Dynamic Pricing Engine](#10-risk-quantified-dynamic-pricing-engine)
11. [Model Performance Results](#11-model-performance-results)
12. [Feature Engineering](#12-feature-engineering)
13. [Generated Artefacts](#13-generated-artefacts)
14. [Key Business Insights](#14-key-business-insights)
15. [Inference Examples](#15-inference-examples)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

FreightIQ is a three-model machine-learning pipeline that powers a **5-layer risk-quantified dynamic pricing engine** for Indian truck logistics. It answers three core business questions:

| Model | Task | Target | Algorithm |
|-------|------|--------|-----------|
| **Task 1** | Binary Classification | Will this delivery be on time? (0/1) | XGBoost Classifier |
| **Task 2** | Regression | How many hours will delivery take? | XGBoost Regressor |
| **Task 3** | Return-Load Probability | Will the truck secure a backhaul load? p_return ∈ [0,1] | XGBoost + Logistic Regression |

The output of Task 3 (`p_return`) feeds directly into the **5-layer pricing engine** as the risk quantification input, computing the optimal quoted freight price for every trip.

The system is delivered as a full-stack web application:
- **Backend:** FastAPI (Python) on port 5000 — runs ML inference + pricing engine
- **Frontend:** React 18 + Vite + Tailwind CSS on port 5173 — city autocomplete, freight form, 5-layer pricing dashboard

**Geographic coverage:** Tamil Nadu · Karnataka · Pondicherry · Maharashtra  
**Weather conditions:** Sunny · Partly Cloudy · Mist · Rainy · Patchy Rain · Light/Heavy Shower

---

## 2. Repository Structure

```
truckG/
├── main.py                            # ML pipeline entry point — trains all 3 models
├── app.py                             # FastAPI backend server (port 5000)
├── requirements.txt                   # ML pipeline Python dependencies
├── requirements_api.txt               # Backend API Python dependencies
├── README.md                          # This file
├── CLAUDE.md                          # Codebase context for AI tools
├── .env.example                       # Backend environment variable template
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Excel ingestion + multi-sheet merging
│   ├── preprocessor.py                # Datetime parsing, NULL handling, cleaning
│   ├── feature_engineering.py         # Encoders, transformers, feature selection
│   ├── models.py                      # XGBoost wrappers + CV helpers
│   ├── evaluation.py                  # Metrics, plots, reports, pricing charts
│   ├── pricing.py                     # 5-layer dynamic pricing engine (standalone)
│   └── return_load_model.py           # Task 3: p_return model (LR + XGBoost)
│
├── frontend/                          # React 18 web application
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                 # Vite config — proxies /api to port 5000
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.example                   # Frontend environment variable template
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Root component — auth state, routing
│       ├── supabase.js                # Supabase client + DEV_MODE detection
│       ├── index.css                  # Tailwind + custom utilities
│       └── components/
│           ├── AuthPage.jsx           # Login / signup page
│           ├── InputForm.jsx          # City autocomplete + freight form
│           └── ResultDashboard.jsx    # 5-layer pricing result dashboard
│
└── ml/                                # Generated artefacts (created by running main.py)
    ├── model_classification.pkl       # Task 1 — XGBoost on-time classifier
    ├── model_regression.pkl           # Task 2 — XGBoost delivery-time regressor
    ├── model_p_return_xgb.pkl         # Task 3 — XGBoost p_return model
    ├── model_p_return_lr.pkl          # Task 3 — Logistic Regression p_return model
    ├── model_p_return_best.pkl        # Task 3 — Best model (XGBoost, AUC 0.9711)
    ├── p_return_meta.pkl              # Feature list + winner name for inference
    └── *.png                          # 8 evaluation and pricing charts
```

---

## 3. Prerequisites

### Python
```
Python >= 3.10
```

### Node.js (for the web frontend)
```
Node.js >= 18.0
npm >= 9.0
```

Check versions:
```bash
python --version
node --version
npm --version
```

---

## 4. Running the ML Pipeline

This step trains all three models and saves them to `ml/`. **You must run this before starting the backend.**

### Step 1 — Place the dataset

Put the Excel file in the project root:
```
truckG/
└── Transportation and Logistics Tracking Dataset..xlsx
```

### Step 2 — Install ML dependencies

```bash
cd c:\Desktop\truckG
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

### Step 3 — Run the pipeline

```bash
# Full run — trains all models + generates all charts
python main.py

# Headless / CI mode (suppresses matplotlib windows)
python main.py --no-plots
```

### What the pipeline does

```
Step 1  Load + clean dataset (6,880 GPS pings across 5 sheets)
Step 2  Train Task 1: On-time delivery classifier   → ml/model_classification.pkl
Step 3  Train Task 2: Delivery time regressor       → ml/model_regression.pkl
Step 4  Analyse regional costs + EV vs Petrol charts
Step 5  Train Task 3: p_return model (LR + XGBoost) → ml/model_p_return_best.pkl
Step 6  Run 5-layer pricing demo (5 scenarios)
Step 7  Generate 14 artefacts in ml/
```

### Expected output

```
[FreightIQ] Loading dataset...
[FreightIQ] Training classification model...
  CV AUC-ROC : 0.9806
  Test AUC-ROC: 0.9947
  Test Accuracy: 96.48%

[FreightIQ] Training regression model...
  CV RMSE: 21.38 hrs
  Test RMSE: 25.16 hrs

[FreightIQ] Training p_return model...
  XGBoost  CV AUC=0.9320  Test AUC=0.9711  AvgPrecision=0.7091  <-- WINNER
  LogReg   CV AUC=0.8731  Test AUC=0.8778  AvgPrecision=0.3067

[FreightIQ] Pricing demo complete. All artefacts saved to ml/
```

---

## 5. Running the Web Application

The web app has two parts: a **FastAPI backend** (Python) and a **React frontend** (Node.js). Run them in two separate terminals simultaneously.

---

### Terminal 1 — Backend (FastAPI on port 5000)

#### Step 1 — Install backend dependencies

```bash
cd c:\Desktop\truckG
pip install -r requirements_api.txt
```

`requirements_api.txt` contents:
```
fastapi
uvicorn[standard]
httpx
python-jose[cryptography]
python-multipart
pandas
numpy
xgboost
scikit-learn
joblib
```

#### Step 2 — (Optional) Configure environment variables

Copy the example file and edit it:
```bash
copy .env.example .env
```

Then open `.env` and fill in your Supabase credentials if you have them (see [Environment Variables](#7-environment-variables)). If you skip this step, the backend runs in **Dev Bypass mode** — all requests are accepted without authentication, which is fine for local development.

#### Step 3 — Start the backend

```bash
cd c:\Desktop\truckG
python app.py
```

You should see:
```
[FreightIQ] clf(15 feats)  reg(15 feats)  p_return(16 feats)  winner=XGBoost
[FreightIQ] SUPABASE_JWT_SECRET not set — Dev Bypass active (no auth required)
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX] using WatchFiles
```

The backend is now live at **http://localhost:5000**

---

### Terminal 2 — Frontend (React on port 5173)

#### Step 1 — Install frontend dependencies (first time only)

```bash
cd c:\Desktop\truckG\frontend
npm install
```

This installs React, Vite, Tailwind CSS, Lucide icons, and Supabase client (~147 packages).

#### Step 2 — (Optional) Configure Supabase for the frontend

```bash
copy .env.example .env
```

Edit `frontend/.env` with your Supabase project URL and anon key. If you skip this, the app runs in **DEV_MODE** — the login page is bypassed automatically.

#### Step 3 — Start the frontend dev server

```bash
cd c:\Desktop\truckG\frontend
npm run dev
```

You should see:
```
  VITE v5.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

Open **http://localhost:5173** in your browser.

---

### Using the Web App

1. **Login screen** — In DEV_MODE the app skips this automatically. With Supabase configured, sign in or create an account.

2. **Enter a freight route:**
   - Type a city name in **Origin City** (e.g. `Chennai`)
   - Wait ~300ms for the autocomplete dropdown to appear
   - **Click a suggestion from the dropdown** — a green dot confirms GPS coordinates are captured
   - Do the same for **Destination City** (e.g. `Mumbai`)
   - Set cargo weight, vehicle class, and dispatch date

3. **Click "Get AI Quote"** — the app calls the backend, runs all 3 ML models, and returns the 5-layer pricing breakdown.

4. **Results dashboard shows:**
   - Final quoted price (INR)
   - Delay probability % with risk badge
   - 5-layer pricing breakdown with step-by-step deltas
   - ML Intelligence Panel: delay %, return load %, delivery hours, confidence
   - Expandable "Technical Details" with raw model outputs

> **Important:** You must select a city from the autocomplete dropdown, not just type the name. The system needs GPS coordinates to compute the route distance.

---

### Build for Production

To create an optimised production build of the frontend:

```bash
cd c:\Desktop\truckG\frontend
npm run build
```

Output goes to `frontend/dist/`. Serve it with any static file server or configure uvicorn to serve it directly.

---

## 6. API Reference

Base URL: `http://localhost:5000`  
Interactive docs: `http://localhost:5000/api/docs`

---

### `GET /api/health`

Returns server status and loaded model info.

**Response:**
```json
{
  "status": "ok",
  "dev_bypass": true,
  "models": {
    "classifier": "XGBClassifier",
    "regressor": "XGBRegressor",
    "p_return": "XGBoost"
  }
}
```

---

### `GET /api/autocomplete?q={query}`

Proxies the Photon geocoding API with India filtering and server-side LRU caching.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `q` | string | City search query (min 2 characters) |

**Example:**
```
GET /api/autocomplete?q=Chennai
```

**Response:** GeoJSON FeatureCollection (up to 5 Indian cities)

---

### `POST /api/predict`

Runs all 3 ML models + 5-layer pricing engine for a freight route.

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <supabase_jwt>   (optional in Dev Bypass mode)
```

**Request body:**
```json
{
  "origin": "Chennai, Tamil Nadu",
  "destination": "Mumbai, Maharashtra",
  "origin_coords": { "lat": 13.0827, "lon": 80.2707 },
  "destination_coords": { "lat": 19.0760, "lon": 72.8777 },
  "origin_state": "Tamil Nadu",
  "destination_state": "Maharashtra",
  "weight_tons": 15.0,
  "vehicle_class": "heavy",
  "date": "2026-05-15T08:00:00"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `origin` | string | yes | Origin city display name |
| `destination` | string | yes | Destination city display name |
| `origin_coords` | object | yes | `{lat, lon}` from autocomplete |
| `destination_coords` | object | yes | `{lat, lon}` from autocomplete |
| `origin_state` | string | no | State name for region detection |
| `destination_state` | string | no | State name for region detection |
| `weight_tons` | float | yes | Cargo weight in tons (0.5 – 60) |
| `vehicle_class` | string | yes | `mini` / `medium` / `heavy` / `trailer` |
| `date` | string | no | ISO 8601 dispatch date-time |

**Vehicle classes:**
| Value | Label | Capacity |
|-------|-------|---------|
| `mini` | Mini Truck | Up to 1 ton |
| `medium` | SXL / LCV | 1 – 7 tons |
| `heavy` | HCV Multi-Axle | 7 – 20 tons |
| `trailer` | 24 FT Container | 20+ tons |

**Response:**
```json
{
  "route": {
    "origin": "Chennai, Tamil Nadu",
    "destination": "Mumbai, Maharashtra",
    "distance_km": 1363.7,
    "region": "Maharashtra"
  },
  "ml": {
    "delay_probability": 0.0122,
    "on_time_probability": 0.9878,
    "delivery_hours": 19.62,
    "p_return": 0.001,
    "confidence": 0.9756
  },
  "pricing": {
    "operational": 2045.5,
    "risk_adjusted": 2352.05,
    "ml_optimized": 2163.89,
    "market_price": 2347.25,
    "final_price": 2347.25,
    "risk_factor": 0.1498,
    "efficiency": 0.9,
    "capped": true,
    "corridor_avg": 2761.47
  },
  "trip": {
    "weight_tons": 15.0,
    "vehicle_class": "heavy",
    "date": "2026-05-15T08:00:00"
  },
  "meta": {
    "user": "dev-user",
    "dev_bypass": true
  }
}
```

**Error responses:**
| Status | Condition |
|--------|-----------|
| 401 | Missing or invalid JWT (when not in Dev Bypass) |
| 422 | Origin and destination are the same location |

---

## 7. Environment Variables

### Backend (`.env` in project root)

```env
# Supabase JWT secret for token verification
# Get this from: Supabase Dashboard → Project Settings → API → JWT Secret
SUPABASE_JWT_SECRET=your_supabase_jwt_secret_here
```

If `SUPABASE_JWT_SECRET` is not set, the server starts in **Dev Bypass mode** — all API requests are accepted without authentication. This is the default for local development.

### Frontend (`frontend/.env`)

```env
# Your Supabase project URL
# Get from: Supabase Dashboard → Project Settings → API
VITE_SUPABASE_URL=https://your-project-id.supabase.co

# Your Supabase anonymous public key
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

If either variable is missing, the frontend runs in **DEV_MODE** — the login/signup screen is skipped and you go straight to the freight form.

> **Security note:** Never commit `.env` files. Both `.env` files are in `.gitignore`.

---

## 8. Dataset

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

## 9. ML Pipeline — Three Models

### Task 1 — On-Time Delivery Classification

**Target:** `On time Delivery` (0 = Delayed, 1 = On-Time)  
**Algorithm:** XGBoost Classifier  
**Train/Test split:** 80/20, stratified  
**Cross-validation:** 5-fold stratified  

**Inference:**
```python
from src.models import load_model

clf = load_model("ml/model_classification.pkl")
prediction  = clf.predict(X_new)               # 0 = Delayed, 1 = On-Time
probability = clf.predict_proba(X_new)[:, 1]   # on-time probability
```

---

### Task 2 — Delivery Time Regression

**Target:** `Delivery_Time` (hours)  
**Algorithm:** XGBoost Regressor  
**Objective:** reg:squarederror  

```python
reg   = load_model("ml/model_regression.pkl")
hours = reg.predict(X_new)   # estimated delivery hours
```

---

### Task 3 — Return-Load Probability (p_return)

**Target proxy:** `is_market` — Market bookings indicate active spot-market corridor demand in both directions = higher return-load availability.  
**Class imbalance:** 40 Market vs 667 Regular (16.7x) — handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Logistic Regression).  
**Models trained:** Logistic Regression + XGBoost — winner selected by Test AUC-ROC.

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

---

## 10. Risk-Quantified Dynamic Pricing Engine

All pricing output values are divided by 150 to produce market-calibrated INR quotes.

### Layer 1 — Operational Base Cost

```
Base = distance_km × weight_tons × vehicle_rate_per_ton_km
```

Vehicle rates: Mini=8 · Medium=12 · Heavy=15 · Trailer=20 INR/ton/km

### Layer 2 — Risk Adjustment (p_return from Task 3)

```
Risk_multiplier = (1 - p_return) × 0.15
Risk_adjusted   = Base × (1 + Risk_multiplier)
```

Higher empty-return risk → higher price. If `p_return = 0.0` (no chance of backhaul), the price increases 15% over base.

### Layer 3 — ML Efficiency Optimisation

```
Expected_hours = distance_km / 45              # 45 km/h average speed
Efficiency     = clamp(Expected_hours / Delivery_hours, 0.1, 0.9)
ML_price       = Risk_adjusted × (1.10 - Efficiency × 0.20)
```

Faster-than-expected delivery = lower price; slower = higher price.

### Layer 4 — Competitive Market Snap

```
Corridor_average = Base × 1.35
Market_low       = Corridor × 0.85
Market_high      = Corridor × 1.15
Market_price     = clamp(ML_price, Market_low, Market_high)
```

Price is capped within ±15% of the corridor average to remain competitive.

### Layer 5 — Margin Floor Gate

```
Margin_floor = Base × 1.10
Final_price  = max(Market_price, Margin_floor)
```

Ensures at minimum a 10% margin over raw operational cost.

### Pricing Engine (standalone usage)

```python
from src.pricing import FreightPricingEngine, PricingInput

engine = FreightPricingEngine()
result = engine.compute(PricingInput(
    label="Chennai to Pune",
    distance_km=1_350.0,
    p_return=0.41,        # from ml/model_p_return_best.pkl
    p_compet=55_000.0,
    i_demand=1.3,
    i_supply=0.9,
))
print(result.summary())
```

---

## 11. Model Performance Results

### Task 1 — On-Time Delivery Classification

| Metric | CV (5-fold) | Hold-out Test |
|--------|-------------|---------------|
| AUC-ROC | 0.9806 | 0.9947 |
| Accuracy | 96.46% | 96.48% |
| F1-Score | 0.9753 | 0.9804 |

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

### Task 3 — Return-Load Probability

| Model | CV AUC-ROC | Test AUC-ROC | Avg Precision |
|-------|-----------|-------------|---------------|
| Logistic Regression | 0.8731 | 0.8778 | 0.3067 |
| **XGBoost (Winner)** | **0.9320** | **0.9711** | **0.7091** |

Top features by XGBoost importance:
```
route_freq          0.1917  ######################################
log_distance_km     0.1640  ################################
distance_km         0.1120  ######################
region_code         0.0826  ################
time_delta_hours    0.0629  ############
```

---

## 12. Feature Engineering

All transformers are sklearn-compatible and live in `src/feature_engineering.py`.

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

## 13. Generated Artefacts

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

## 14. Key Business Insights

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

Maharashtra routes are 7% costlier than Tamil Nadu — justifying a regional price floor adjustment.

### EV vs Petrol Fleet Planning

| Vehicle Type | Avg Yearly Maintenance | Unit |
|-------------|----------------------|------|
| EV | 43.27 | Lakhs INR |
| Petrol | 33.33 | Thousands INR |

EV trucks carry ~13x higher absolute maintenance costs. Assign EV trucks to high-volume, high-`p_return` corridors to minimise empty-return risk.

---

## 15. Inference Examples

### Get a freight quote from the API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Chennai, Tamil Nadu",
    "destination": "Mumbai, Maharashtra",
    "origin_coords": {"lat": 13.0827, "lon": 80.2707},
    "destination_coords": {"lat": 19.0760, "lon": 72.8777},
    "weight_tons": 15,
    "vehicle_class": "heavy",
    "date": "2026-05-15T08:00:00"
  }'
```

### Load models directly in Python

```python
import joblib
import pandas as pd
from src.models import load_model

clf   = load_model("ml/model_classification.pkl")
reg   = load_model("ml/model_regression.pkl")
p_ret = joblib.load("ml/model_p_return_best.pkl")

X = pd.DataFrame([{
    "distance_km": 160, "log_distance_km": 5.08,
    "weather_severity": 3, "adverse_weather": 1,
    "region_code": 0, "Fixed Costs": 8400,
    "Maintenance": 900, "Difference": 7500,
    "Customer_rating": 4, "time_delta_hours": 0, "is_market": 0,
    # vtype one-hot columns (set the matching one to 1.0)
    "vtype_Other": 1.0,
    "vtype_32 FT Single-Axle 7MT - HCV": 0.0,
    "vtype_32 FT Multi-Axle 14MT - HCV": 0.0,
    "vtype_24 FT SXL Container": 0.0,
}])

print(clf.predict(X))            # [1] = On-Time
print(clf.predict_proba(X))      # [[0.07, 0.93]]
print(reg.predict(X))            # [18.4] hours
```

---

## 16. Troubleshooting

### "Get AI Quote" button is disabled
The button only activates after selecting cities from the autocomplete dropdown — you must click a suggestion, not just type. A small green dot on the input field confirms coordinates are captured.

### Autocomplete dropdown does not appear
The backend must be running on port 5000. Check:
```bash
# Should return {"status": "ok", ...}
curl http://localhost:5000/api/health
```
If the backend is not running, start it with `python app.py` in a terminal inside `c:\Desktop\truckG`.

### Backend fails to start — "No such file or directory: ml/model_classification.pkl"
The ML models have not been trained yet. Run `python main.py` first to generate all `.pkl` files in `ml/`.

### Backend fails to start — "ModuleNotFoundError"
Install the API dependencies:
```bash
pip install -r requirements_api.txt
```

### Frontend fails to start — "sh: vite: command not found"
Node modules are not installed:
```bash
cd c:\Desktop\truckG\frontend
npm install
```

### Port already in use
```bash
# Find and kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Find and kill process on port 5173 (Windows)
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

### CORS error in browser
The backend allows `localhost:5173` and `localhost:3000` by default. If you are running the frontend on a different port, add it to the `allow_origins` list in `app.py`.

---

## Module Map (for contributors)

| To change... | Edit file | Where |
|---|---|---|
| Add new features | `src/feature_engineering.py` | New `Transformer` class |
| Tune model hyperparameters | `src/models.py` | `_COMMON` dict |
| Add evaluation charts | `src/evaluation.py` | New `plot_*` function |
| Add new data sheets | `src/data_loader.py` | New `load_*` function |
| Change pricing formula | `app.py` | `five_layer_price()` function |
| Change pricing divisor | `app.py` | `/ 150` in `five_layer_price()` return dict |
| Add p_return features | `src/return_load_model.py` | `FEATURES` list + `_engineer_features()` |
| Add frontend pages | `frontend/src/` | New component in `components/` |

---

*FreightIQ — Built on Transportation and Logistics Tracking Dataset (Indian freight corridors)*
