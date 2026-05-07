"""
Risk-Quantified Dynamic Pricing Engine -- FreightIQ
====================================================
Implements the 5-layer model from the pricing specification:

  Layer 1 -- Operational Base Cost  (fuel, toll, driver, handling)
  Layer 2 -- Risk Quantification    (empty-return probability x cost)
  Layer 3 -- Model Price            (base x (1 + risk_factor + market_adj))
  Layer 4 -- Market-Feasible Price  (min(model_price, competitor_cap))
  Layer 5 -- Profit Optimisation    (booking probability, expected profit,
                                     driver-acceptance feasibility check)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Input / Output data structures
# ---------------------------------------------------------------------------

@dataclass
class PricingInput:
    """All parameters required to quote a single freight trip."""

    # Route
    distance_km: float = 160.0          # one-way distance (km)
    label: str = "Default"

    # Vehicle efficiency
    m_loaded: float = 4.0               # loaded mileage (km/L)
    m_empty: float = 6.0                # empty-return mileage (km/L)

    # Variable costs (INR)
    p_diesel: float = 92.0             # diesel price per litre
    c_toll: float = 500.0              # total toll cost
    c_driver: float = 1_000.0          # driver wages per trip
    c_maint: float = 200.0             # maintenance (per trip share)
    c_load: float = 300.0              # loading cost
    c_unload: float = 300.0            # unloading cost
    c_parking: float = 100.0           # parking (empty-return leg)

    # Market context
    p_compet: float = 13_000.0         # competitor quoted price (INR)
    i_demand: float = 1.2              # demand index (normalised)
    i_supply: float = 1.0              # supply index (normalised)

    # ML model output
    p_return: float = 0.60             # probability of securing return load

    # Market-adjustment weights
    gamma: float = 0.10                # demand/supply signal weight
    delta: float = 0.05                # competitor-gap signal weight
    epsilon: float = 0.07             # market cap premium over competitor (7%)

    # Historical normalisation constants (empirical)
    mu_1: float = 1.0                  # mean of I_demand/I_supply
    sigma_1: float = 0.30             # std  of I_demand/I_supply
    mu_2: float = 0.0                  # mean of competitor margin ratio
    sigma_2: float = 0.20             # std  of competitor margin ratio

    # Booking-probability model (sigmoid, learned from accept/reject data)
    a: float = 2.0                     # intercept
    b: float = 0.0002                  # price-sensitivity slope

    # Environmental / urgency modifiers
    urgency: float = 0.0               # normalised urgency [0, 1]
    time_of_day_factor: float = 0.0    # normalised time-of-day [0, 1]
    omega: float = 0.10                # urgency weight
    eta: float = 0.05                  # time-of-day weight

    # Driver feasibility threshold
    r_driver: float = 800.0            # minimum acceptable driver earnings (INR)


@dataclass
class PricingResult:
    """Full computation output -- every intermediate value is exposed."""

    label: str = ""

    # Layer 1
    fuel_loaded: float = 0.0
    fuel_empty: float = 0.0
    base_cost_raw: float = 0.0
    env_modifier: float = 1.0
    base_cost_modified: float = 0.0

    # Layer 2
    empty_return_cost: float = 0.0
    p_return: float = 0.0
    risk_cost: float = 0.0
    risk_factor: float = 0.0

    # Layer 3
    x1: float = 0.0
    x2: float = 0.0
    market_adj: float = 0.0
    model_price: float = 0.0

    # Layer 4
    market_cap: float = 0.0
    quoted_price: float = 0.0

    # Layer 5
    delta_p: float = 0.0
    p_book: float = 0.0
    expected_profit: float = 0.0
    driver_earnings: float = 0.0
    feasible: bool = True

    def summary(self) -> str:
        sep = "=" * 62
        feasibility = "FEASIBLE" if self.feasible else "INFEASIBLE - driver constraint violated"
        lines = [
            "",
            sep,
            "  FreightIQ Pricing Result  |  Scenario: " + self.label,
            sep,
            "  LAYER 1 -- OPERATIONAL BASE COST",
            "    Fuel cost (loaded trip)   : INR {:>10,.2f}".format(self.fuel_loaded),
            "    Fuel cost (empty return)  : INR {:>10,.2f}".format(self.fuel_empty),
            "    Base Cost (raw)           : INR {:>10,.2f}".format(self.base_cost_raw),
            "    Environmental modifier    :      {:>10.4f}".format(self.env_modifier),
            "    Base Cost (modified)      : INR {:>10,.2f}".format(self.base_cost_modified),
            "",
            "  LAYER 2 -- RISK QUANTIFICATION",
            "    Return-load probability   :      {:>10.4f}".format(self.p_return),
            "    Empty-return cost         : INR {:>10,.2f}".format(self.empty_return_cost),
            "    Risk Cost                 : INR {:>10,.2f}".format(self.risk_cost),
            "    Risk Factor               :      {:>10.4f}".format(self.risk_factor),
            "",
            "  LAYER 3 -- COST-OPTIMAL MODEL PRICE",
            "    x1  (demand/supply signal):      {:>10.4f}".format(self.x1),
            "    x2  (competitor gap signal):     {:>10.4f}".format(self.x2),
            "    Market Adjustment (M_adj) :      {:>10.4f}".format(self.market_adj),
            "    Model Price               : INR {:>10,.2f}".format(self.model_price),
            "",
            "  LAYER 4 -- MARKET-FEASIBLE QUOTED PRICE",
            "    Market Cap (comp x 1+eps) : INR {:>10,.2f}".format(self.market_cap),
            "    Quoted Price              : INR {:>10,.2f}".format(self.quoted_price),
            "",
            "  LAYER 5 -- PROFIT OPTIMISATION & FEASIBILITY",
            "    Delta P (quoted - compet) : INR {:>10,.2f}".format(self.delta_p),
            "    Booking Probability       :      {:>10.4f}  ({:.1f} %)".format(
                self.p_book, self.p_book * 100
            ),
            "    Expected Profit           : INR {:>10,.2f}".format(self.expected_profit),
            "    Driver Net Earnings       : INR {:>10,.2f}".format(self.driver_earnings),
            "    Trip Feasibility          :  >> " + feasibility,
            sep,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Pricing Engine
# ---------------------------------------------------------------------------

class FreightPricingEngine:
    """
    Computes the risk-adjusted quoted price for a truck freight trip.

    Usage::

        engine = FreightPricingEngine()
        result = engine.compute(PricingInput(distance_km=200, p_return=0.65, ...))
        print(result.summary())
    """

    def compute(self, inp: PricingInput) -> PricingResult:
        res = PricingResult(label=inp.label)

        # Layer 1: Operational base cost
        res.fuel_loaded = (inp.distance_km / inp.m_loaded) * inp.p_diesel
        res.fuel_empty = (inp.distance_km / inp.m_empty) * inp.p_diesel
        res.base_cost_raw = (
            res.fuel_loaded + res.fuel_empty
            + inp.c_toll + inp.c_driver + inp.c_maint
            + inp.c_load + inp.c_unload
        )
        res.env_modifier = 1.0 + inp.omega * inp.urgency + inp.eta * inp.time_of_day_factor
        res.base_cost_modified = res.base_cost_raw * res.env_modifier

        # Layer 2: Empty-return risk cost
        res.empty_return_cost = (
            res.fuel_empty + inp.c_toll + inp.c_driver + inp.c_maint + inp.c_parking
        )
        res.p_return = float(np.clip(inp.p_return, 0.0, 1.0))
        res.risk_cost = (1.0 - res.p_return) * res.empty_return_cost
        res.risk_factor = res.risk_cost / (res.base_cost_modified + 1e-9)

        # Layer 3: Market-adjusted model price
        res.x1 = (inp.i_demand / inp.i_supply - inp.mu_1) / (inp.sigma_1 + 1e-9)
        compet_ratio = (inp.p_compet - res.base_cost_modified) / (res.base_cost_modified + 1e-9)
        res.x2 = (compet_ratio - inp.mu_2) / (inp.sigma_2 + 1e-9)
        res.market_adj = inp.gamma * res.x1 + inp.delta * res.x2
        res.model_price = res.base_cost_modified * (1.0 + res.risk_factor + res.market_adj)

        # Layer 4: Market-feasible quoted price (capped at competitor + epsilon)
        res.market_cap = inp.p_compet * (1.0 + inp.epsilon)
        res.quoted_price = min(res.model_price, res.market_cap)

        # Layer 5: Booking probability, expected profit, driver feasibility
        res.delta_p = res.quoted_price - inp.p_compet
        res.p_book = _sigmoid(inp.a - inp.b * res.delta_p)
        res.expected_profit = res.p_book * (res.quoted_price - res.base_cost_modified)

        # Driver net earnings = quoted price minus costs the driver directly bears
        res.driver_earnings = res.quoted_price - (
            res.fuel_loaded * res.env_modifier + inp.c_toll + inp.c_load + inp.c_unload
        )
        res.feasible = res.driver_earnings >= inp.r_driver

        return res

    def sweep_p_return(
        self,
        base_inp: PricingInput,
        n: int = 11,
    ) -> pd.DataFrame:
        """Sweep p_return 0->1 and return a DataFrame of key pricing metrics."""
        rows = []
        for p in np.linspace(0.0, 1.0, n):
            d = base_inp.__dict__.copy()
            d["p_return"] = p
            d["label"] = "p={:.2f}".format(p)
            inp = PricingInput(**d)
            r = self.compute(inp)
            rows.append({
                "p_return": round(p, 2),
                "risk_cost": round(r.risk_cost, 2),
                "risk_factor": round(r.risk_factor, 4),
                "model_price": round(r.model_price, 2),
                "quoted_price": round(r.quoted_price, 2),
                "p_book": round(r.p_book, 4),
                "expected_profit": round(r.expected_profit, 2),
                "driver_earnings": round(r.driver_earnings, 2),
                "feasible": r.feasible,
            })
        return pd.DataFrame(rows)

    def sweep_distance(
        self,
        base_inp: PricingInput,
        distances: list[float] | None = None,
    ) -> pd.DataFrame:
        """Sweep distance values and return key metrics."""
        if distances is None:
            distances = [50, 100, 160, 250, 400, 600, 900]
        rows = []
        for d in distances:
            inp_d = base_inp.__dict__.copy()
            inp_d["distance_km"] = d
            inp_d["label"] = "d={}km".format(d)
            inp = PricingInput(**inp_d)
            r = self.compute(inp)
            rows.append({
                "distance_km": d,
                "base_cost": round(r.base_cost_modified, 2),
                "risk_cost": round(r.risk_cost, 2),
                "model_price": round(r.model_price, 2),
                "quoted_price": round(r.quoted_price, 2),
                "expected_profit": round(r.expected_profit, 2),
                "driver_earnings": round(r.driver_earnings, 2),
                "feasible": r.feasible,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


# ---------------------------------------------------------------------------
# Pre-built demo scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: list[PricingInput] = [
    PricingInput(
        label="TN Metro - Short Haul, High Return",
        distance_km=120.0,
        m_loaded=4.2,
        m_empty=6.2,
        p_diesel=92.0,
        c_toll=350.0,
        c_driver=900.0,
        c_maint=150.0,
        c_load=250.0,
        c_unload=250.0,
        c_parking=80.0,
        p_compet=11_500.0,
        i_demand=1.4,
        i_supply=1.0,
        p_return=0.78,
        urgency=0.0,
        time_of_day_factor=0.2,
        r_driver=800.0,
    ),
    PricingInput(
        label="MH Rural - Long Haul, Low Return",
        distance_km=550.0,
        m_loaded=3.8,
        m_empty=5.5,
        p_diesel=93.5,
        c_toll=1_800.0,
        c_driver=2_500.0,
        c_maint=600.0,
        c_load=400.0,
        c_unload=400.0,
        c_parking=200.0,
        p_compet=42_000.0,
        i_demand=0.9,
        i_supply=1.2,
        p_return=0.32,
        urgency=0.0,
        time_of_day_factor=0.5,
        r_driver=1_800.0,
    ),
    PricingInput(
        label="Rainy Weather + Urgency Premium",
        distance_km=160.0,
        m_loaded=3.9,
        m_empty=5.8,
        p_diesel=92.0,
        c_toll=520.0,
        c_driver=1_100.0,
        c_maint=250.0,
        c_load=300.0,
        c_unload=300.0,
        c_parking=120.0,
        p_compet=13_500.0,
        i_demand=1.1,
        i_supply=1.3,
        p_return=0.45,
        urgency=0.8,
        time_of_day_factor=0.7,
        omega=0.12,
        eta=0.06,
        r_driver=900.0,
    ),
    PricingInput(
        label="Demand Surge - High I_demand",
        distance_km=160.0,
        m_loaded=4.0,
        m_empty=6.0,
        p_diesel=92.0,
        c_toll=500.0,
        c_driver=1_000.0,
        c_maint=200.0,
        c_load=300.0,
        c_unload=300.0,
        c_parking=100.0,
        p_compet=13_000.0,
        i_demand=2.1,
        i_supply=0.8,
        p_return=0.72,
        urgency=0.2,
        time_of_day_factor=0.1,
        r_driver=800.0,
    ),
    PricingInput(
        label="EV Truck - High Maintenance, Metro Lane",
        distance_km=140.0,
        m_loaded=5.5,
        m_empty=7.0,
        p_diesel=6.5,       # electricity cost per unit (INR/kWh equivalent)
        c_toll=450.0,
        c_driver=1_000.0,
        c_maint=950.0,      # higher per-trip share (EV maintenance in LKS/yr)
        c_load=270.0,
        c_unload=270.0,
        c_parking=90.0,
        p_compet=12_800.0,
        i_demand=1.3,
        i_supply=1.0,
        p_return=0.70,
        urgency=0.0,
        time_of_day_factor=0.15,
        r_driver=800.0,
    ),
]
