"""Simple, explainable Customer Lifetime Value and churn-risk estimates.

CLV = average order value x orders per year x expected lifespan (years) x margin

Orders per year is measured over each customer's own tenure, floored at one
year, so short histories are never extrapolated upwards (a one-time buyer from
last week is not projected to buy 50 times a year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_clv(
    rfm: pd.DataFrame, margin: float = 0.25, lifespan_years: float = 3.0
) -> pd.DataFrame:
    out = rfm.copy()
    tenure_years = np.maximum(out["tenure"], 365) / 365
    orders_per_year = out["frequency"] / tenure_years

    # Churn risk: how overdue the customer is relative to their own typical
    # gap between orders, squashed into 0-1.
    avg_gap = np.where(out["frequency"] > 1, out["tenure"] / out["frequency"], 90)
    overdue_ratio = out["recency"] / np.maximum(avg_gap, 1)
    out["churn_risk"] = (1 - np.exp(-overdue_ratio / 2)).round(3)

    retention = 1 - out["churn_risk"]
    out["predicted_clv"] = (
        out["avg_order_value"] * orders_per_year * lifespan_years * margin * retention
    ).round(2)
    out["avg_order_value"] = out["avg_order_value"].round(2)
    return out
