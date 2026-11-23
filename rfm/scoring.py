"""Recency / Frequency / Monetary calculation and 1-5 scoring."""

from __future__ import annotations

import pandas as pd


def compute_rfm(tx: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer with recency (days), frequency (orders), monetary (revenue).

    The snapshot date defaults to the day after the last transaction, so the most
    recent buyer has recency 1 rather than 0.
    """
    if snapshot_date is None:
        snapshot_date = tx["invoice_date"].max().normalize() + pd.Timedelta(days=1)

    rfm = tx.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        country=("country", lambda s: s.mode().iat[0]),
    )
    rfm["recency"] = (snapshot_date - rfm["last_purchase"].dt.normalize()).dt.days
    rfm["tenure"] = (snapshot_date - rfm["first_purchase"].dt.normalize()).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm.reset_index()


def _quintile(series: pd.Series, ascending: bool) -> pd.Series:
    """Score 1-5 by quintile. Ranking first breaks ties so qcut never fails
    on heavily repeated values (e.g. the many customers with frequency == 1)."""
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add r/f/m scores (5 = best), the combined code and a 3-15 total score."""
    out = rfm.copy()
    out["r_score"] = _quintile(out["recency"], ascending=False)  # recent = 5
    out["f_score"] = _quintile(out["frequency"], ascending=True)
    out["m_score"] = _quintile(out["monetary"], ascending=True)
    out["rfm_code"] = (
        out["r_score"].astype(str) + out["f_score"].astype(str) + out["m_score"].astype(str)
    )
    out["rfm_total"] = out[["r_score", "f_score", "m_score"]].sum(axis=1)
    return out
