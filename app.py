"""
FreightIQ FastAPI Backend
=========================
Port  : 5000
Models: ml/model_classification.pkl  (Task 1 — delay probability)
        ml/model_regression.pkl       (Task 2 — delivery hours)
        ml/model_p_return_best.pkl    (Task 3 — return-load probability)

Endpoints:
  GET  /api/health        — liveness + model info
  GET  /api/autocomplete  — Photon geocoding proxy (server-side LRU cache)
  POST /api/predict       — 5-layer pricing computation

Auth:
  JWT HS256 via SUPABASE_JWT_SECRET env var.
  If the variable is absent the server runs in "Dev Bypass" mode and all
  requests are accepted without a token.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from jose import JWTError
    from jose import jwt as jose_jwt
    _JOSE_OK = True
except ImportError:
    _JOSE_OK = False

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FreightIQ API", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

_ML = "ml"

clf        = joblib.load(f"{_ML}/model_classification.pkl")
reg        = joblib.load(f"{_ML}/model_regression.pkl")
p_ret_mdl  = joblib.load(f"{_ML}/model_p_return_best.pkl")
p_ret_meta = joblib.load(f"{_ML}/p_return_meta.pkl")

CLF_FEATS  = list(clf.feature_names_in_)
REG_FEATS  = list(reg.feature_names_in_)
PRET_FEATS = list(p_ret_meta["features"])

print(f"[FreightIQ] clf({len(CLF_FEATS)} feats)  reg({len(REG_FEATS)} feats)  "
      f"p_return({len(PRET_FEATS)} feats)  winner={p_ret_meta['best_model_name']}")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
DEV_BYPASS = not SUPABASE_JWT_SECRET

if DEV_BYPASS:
    print("[FreightIQ] SUPABASE_JWT_SECRET not set — Dev Bypass active (no auth required)")


async def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    if DEV_BYPASS:
        return {"sub": "dev-user", "mode": "bypass"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing Authorization header")
    token = authorization.split(" ", 1)[1]
    if not _JOSE_OK:
        return {"sub": "unknown", "mode": "jose-unavailable"}
    try:
        return jose_jwt.decode(token, SUPABASE_JWT_SECRET,
                               algorithms=["HS256"],
                               options={"verify_aud": False})
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Invalid token: {exc}")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Regional average cost values derived from Q8 analysis
_REGION_COSTS: dict[str, dict] = {
    "Tamil Nadu":  {"Fixed Costs": 8400.0, "Maintenance": 900.0,  "Difference": 7500.0},
    "Karnataka":   {"Fixed Costs": 8500.0, "Maintenance": 920.0,  "Difference": 7580.0},
    "Maharashtra": {"Fixed Costs": 9200.0, "Maintenance": 960.0,  "Difference": 8240.0},
    "Pondicherry": {"Fixed Costs": 7800.0, "Maintenance": 840.0,  "Difference": 6960.0},
    "default":     {"Fixed Costs": 8400.0, "Maintenance": 900.0,  "Difference": 7500.0},
}
_REGION_KEYS = list(_REGION_COSTS.keys())

# INR per km per ton
_VEHICLE_RATE = {"mini": 8.0, "medium": 12.0, "heavy": 15.0, "trailer": 20.0}

# Map frontend vehicle class → vtype one-hot column name (from training data)
_VTYPE_MAP = {
    "mini":    "vtype_Other",
    "medium":  "vtype_32 FT Single-Axle 7MT - HCV",
    "heavy":   "vtype_32 FT Multi-Axle 14MT - HCV",
    "trailer": "vtype_24 FT SXL Container",
}

# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line Haversine distance × 1.32 road-distance factor."""
    R = 6_371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(0, a))) * 1.32


def _detect_region(text: str) -> str:
    t = (text or "").lower()
    for k in _REGION_KEYS[:-1]:   # skip "default"
        if k.lower() in t:
            return k
    return "default"

# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _clf_features(distance_km: float, region: str,
                  dt: datetime, vehicle_class: str,
                  weather_sev: int = 2) -> pd.DataFrame:
    """1-row DataFrame matching CLF_FEATS / REG_FEATS."""
    costs = _REGION_COSTS.get(region, _REGION_COSTS["default"])
    region_code = _REGION_KEYS.index(region) if region in _REGION_KEYS else 1

    row: dict = {f: 0.0 for f in CLF_FEATS}
    row.update({
        "distance_km":      distance_km,
        "log_distance_km":  np.log1p(distance_km),
        "weather_severity": float(weather_sev),
        "adverse_weather":  float(weather_sev >= 4),
        "region_code":      float(region_code),
        "Fixed Costs":      costs["Fixed Costs"],
        "Maintenance":      costs["Maintenance"],
        "Difference":       costs["Difference"],
        "Customer_rating":  4.0,
        "time_delta_hours": 0.0,
        "is_market":        float(vehicle_class in ("mini", "medium")),
    })
    # Set the matching vtype column
    vtype_col = _VTYPE_MAP.get(vehicle_class, "vtype_Other")
    if vtype_col in row:
        row[vtype_col] = 1.0

    return pd.DataFrame([row])[CLF_FEATS]


def _pret_features(distance_km: float, region: str,
                   dt: datetime, weather_sev: int = 2,
                   route_freq: float = 8.0) -> pd.DataFrame:
    """1-row DataFrame matching PRET_FEATS."""
    costs = _REGION_COSTS.get(region, _REGION_COSTS["default"])
    region_code = _REGION_KEYS.index(region) if region in _REGION_KEYS else 1

    row: dict = {f: 0.0 for f in PRET_FEATS}
    row.update({
        "log_distance_km":  np.log1p(distance_km),
        "distance_km":      distance_km,
        "day_of_week":      float(dt.weekday()),
        "month":            float(dt.month),
        "hour":             float(dt.hour),
        "is_weekend":       float(dt.weekday() >= 5),
        "route_freq":       route_freq,
        "weather_severity": float(weather_sev),
        "adverse_weather":  float(weather_sev >= 4),
        "region_code":      float(region_code),
        "Fixed Costs":      costs["Fixed Costs"],
        "Maintenance":      costs["Maintenance"],
        "Difference":       costs["Difference"],
        "Customer_rating":  4.0,
        "on_time":          0.73,   # dataset average prior
        "time_delta_hours": 0.0,
    })
    return pd.DataFrame([row])[PRET_FEATS]

# ---------------------------------------------------------------------------
# 5-Layer Pricing Engine
# ---------------------------------------------------------------------------

def five_layer_price(
    distance_km:   float,
    weight_tons:   float,
    vehicle_class: str,
    delay_prob:    float,
    p_return:      float,
    delivery_hours: float,
) -> dict:
    """
    Layer 1  Operational Base   : distance × weight × vehicle_rate
    Layer 2  Risk Adjustment    : Base × (1 + (1−p_return) × 0.15)
    Layer 3  ML Optimisation    : Risk × efficiency factor from delivery model
    Layer 4  Competitive Snap   : Clamp to corridor average ± 15 %
    Layer 5  Feasibility Gate   : max(Market, Base × 1.10 margin floor)
    """
    rate       = _VEHICLE_RATE.get(vehicle_class.lower(), 12.0)
    base       = distance_km * weight_tons * rate

    # Layer 2
    risk_mult  = (1.0 - p_return) * 0.15
    risk_adj   = base * (1.0 + risk_mult)

    # Layer 3 — efficiency score: ratio of expected speed to predicted speed
    expected_h  = distance_km / 45.0            # 45 km/h effective average
    efficiency  = float(np.clip(expected_h / max(delivery_hours, 0.5), 0.1, 0.9))
    ml_price    = risk_adj * (1.10 - efficiency * 0.20)

    # Layer 4
    corridor    = base * 1.35
    mkt_low     = corridor * 0.85
    mkt_high    = corridor * 1.15
    market      = float(np.clip(ml_price, mkt_low, mkt_high))

    # Layer 5
    margin_floor = base * 1.10
    final        = max(market, margin_floor)

    return {
        "operational":   round(base,     2),
        "risk_adjusted": round(risk_adj, 2),
        "ml_optimized":  round(ml_price, 2),
        "market_price":  round(market,   2),
        "final_price":   round(final,    2),
        "risk_factor":   round(risk_mult, 4),
        "efficiency":    round(efficiency, 4),
        "capped":        not (mkt_low <= ml_price <= mkt_high),
        "corridor_avg":  round(corridor, 2),
    }

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Coords(BaseModel):
    lat: float
    lon: float

class PredictRequest(BaseModel):
    origin:            str
    destination:       str
    origin_coords:     Coords
    destination_coords: Coords
    origin_state:      Optional[str] = None
    destination_state: Optional[str] = None
    weight_tons:       float = Field(default=10.0, gt=0, le=60)
    vehicle_class:     str   = "medium"
    date:              str   = ""

# ---------------------------------------------------------------------------
# Autocomplete cache (server-side LRU)
# ---------------------------------------------------------------------------

_ac_cache: dict[str, dict] = {}
_AC_MAX = 2_048

@app.get("/api/autocomplete")
async def autocomplete(q: str = ""):
    if len(q) < 2:
        return {"features": []}
    key = q.strip().lower()
    if key in _ac_cache:
        return _ac_cache[key]
    try:
        async with httpx.AsyncClient(timeout=5.0, headers={"User-Agent": "Mozilla/5.0 FreightIQ/1.0"}) as cli:
            r = await cli.get("https://photon.komoot.io/api/",
                              params={"q": q, "limit": 6, "lang": "en"})
        data = r.json()
    except Exception:
        return {"features": []}
    if data.get("features"):
        if len(_ac_cache) >= _AC_MAX:
            _ac_cache.pop(next(iter(_ac_cache)))
        _ac_cache[key] = data
    return data

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "dev_bypass": DEV_BYPASS,
        "models": {
            "classifier":  type(clf).__name__,
            "regressor":   type(reg).__name__,
            "p_return":    p_ret_meta.get("best_model_name", "XGBoost"),
        },
    }

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(req: PredictRequest, user: dict = Depends(verify_token)):
    # Distance
    dist = haversine_km(
        req.origin_coords.lat, req.origin_coords.lon,
        req.destination_coords.lat, req.destination_coords.lon,
    )
    if dist < 1.0:
        raise HTTPException(status_code=422,
                            detail="Origin and destination appear to be the same location.")

    # Date
    try:
        dt = datetime.fromisoformat(req.date) if req.date else datetime.now()
    except ValueError:
        dt = datetime.now()

    # Region
    dest_region = _detect_region(req.destination_state or req.destination)

    # Feature matrices
    X_clf  = _clf_features(dist, dest_region, dt, req.vehicle_class)
    X_reg  = pd.DataFrame([{f: X_clf.iloc[0].get(f, 0.0) for f in REG_FEATS}])[REG_FEATS]
    X_pret = _pret_features(dist, dest_region, dt)

    # Model inference
    delay_prob     = float(clf.predict_proba(X_clf)[0, 0])     # P(delayed)
    on_time_prob   = float(clf.predict_proba(X_clf)[0, 1])     # P(on-time)
    delivery_hours = float(reg.predict(X_reg)[0])
    p_return       = float(p_ret_mdl.predict_proba(X_pret)[0, 1])

    # Confidence: model certainty (0 = at decision boundary, 1 = fully certain)
    confidence = abs(on_time_prob - 0.5) * 2.0

    # 5-layer pricing
    pricing = five_layer_price(
        dist, req.weight_tons, req.vehicle_class,
        delay_prob, p_return, delivery_hours,
    )

    return {
        "route": {
            "origin":      req.origin,
            "destination": req.destination,
            "distance_km": round(dist, 1),
            "region":      dest_region,
        },
        "ml": {
            "delay_probability":  round(delay_prob,     4),
            "on_time_probability": round(on_time_prob,  4),
            "delivery_hours":      round(delivery_hours, 2),
            "p_return":            round(p_return,       4),
            "confidence":          round(confidence,     4),
        },
        "pricing": pricing,
        "trip": {
            "weight_tons":   req.weight_tons,
            "vehicle_class": req.vehicle_class,
            "date":          req.date,
        },
        "meta": {"user": user.get("sub", "anon"), "dev_bypass": DEV_BYPASS},
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
