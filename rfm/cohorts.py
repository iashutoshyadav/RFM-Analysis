"""Monthly acquisition-cohort retention."""

from __future__ import annotations

import pandas as pd


def cohort_retention(tx: pd.DataFrame) -> pd.DataFrame:
    """Share of each first-purchase-month cohort still buying N months later.

    Rows are cohort months (YYYY-MM), columns are months since first purchase.
    """
    df = tx[["customer_id", "invoice_date"]].copy()
    df["order_month"] = df["invoice_date"].dt.to_period("M")
    df["cohort"] = df.groupby("customer_id")["order_month"].transform("min")
    df["period"] = (df["order_month"] - df["cohort"]).apply(lambda p: p.n)

    counts = (
        df.groupby(["cohort", "period"])["customer_id"].nunique().unstack(fill_value=0)
    )
    retention = counts.div(counts[0], axis=0).round(3)
    retention.index = retention.index.astype(str)
    return retention
