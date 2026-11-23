"""Rule-based segmentation.

The original SQL CASE statement left many score combinations in an "Other"
bucket (e.g. R=4, F=1). This version uses the widely used R x FM grid, where
FM is the rounded mean of the F and M scores, so all 25 cells map to a named
segment and no customer is left unclassified.
"""

from __future__ import annotations

import pandas as pd

# Rows: R score 1..5, columns: FM score 1..5
_GRID = {
    5: ["New Customers", "Promising", "Potential Loyalists", "Loyal Customers", "Champions"],
    4: ["Promising", "Potential Loyalists", "Potential Loyalists", "Loyal Customers", "Champions"],
    3: ["About to Sleep", "Need Attention", "Need Attention", "Loyal Customers", "Loyal Customers"],
    2: ["Hibernating", "About to Sleep", "At Risk", "At Risk", "Cannot Lose Them"],
    1: ["Lost", "Hibernating", "At Risk", "At Risk", "Cannot Lose Them"],
}

SEGMENT_ACTIONS = {
    "Champions": "Reward them: early access, VIP perks, referral programme.",
    "Loyal Customers": "Upsell higher-value products; ask for reviews.",
    "Potential Loyalists": "Offer a membership or loyalty programme.",
    "New Customers": "Onboarding emails and a second-purchase incentive.",
    "Promising": "Build brand awareness; free-shipping thresholds.",
    "Need Attention": "Limited-time, personalised offers based on history.",
    "About to Sleep": "Recommend popular products; reconnect with a discount.",
    "At Risk": "Personalised win-back campaign with strong incentives.",
    "Cannot Lose Them": "High-value but lapsing: direct outreach, best offers.",
    "Hibernating": "Low-cost reactivation; recommend relevant categories.",
    "Lost": "Minimal spend; include only in broad seasonal campaigns.",
}

# Display order from best to worst, used by charts.
SEGMENT_ORDER = [
    "Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising",
    "Need Attention", "About to Sleep", "Cannot Lose Them", "At Risk", "Hibernating", "Lost",
]


def assign_segment(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    fm = (out["f_score"] + out["m_score"] + 1) // 2  # mean of F and M, rounded half up
    out["fm_score"] = fm
    out["segment"] = [_GRID[r][f - 1] for r, f in zip(out["r_score"], fm)]
    out["recommended_action"] = out["segment"].map(SEGMENT_ACTIONS)
    return out
