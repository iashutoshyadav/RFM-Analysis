import numpy as np
import pandas as pd
import pytest

from rfm import (
    assign_segment,
    clean_transactions,
    cluster_customers,
    cohort_retention,
    compute_rfm,
    estimate_clv,
    score_rfm,
)
from rfm.segments import SEGMENT_ACTIONS


def make_tx(n_customers: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    inv = 100000
    for cid in range(1, n_customers + 1):
        for _ in range(rng.integers(1, 15)):
            inv += 1
            date = pd.Timestamp("2011-01-01") + pd.Timedelta(days=int(rng.integers(0, 340)))
            for _ in range(rng.integers(1, 4)):
                rows.append((str(inv), "A1", "ITEM", int(rng.integers(1, 20)), date,
                             float(rng.uniform(0.5, 20)), float(cid), "United Kingdom"))
    return pd.DataFrame(rows, columns=["invoice_no", "stock_code", "description", "quantity",
                                       "invoice_date", "unit_price", "customer_id", "country"])


@pytest.fixture(scope="module")
def tx():
    clean, _ = clean_transactions(make_tx())
    return clean


def test_cleaning_removes_bad_rows():
    raw = make_tx(5)
    bad = pd.DataFrame([
        ("C1", "A1", "x", 1, pd.Timestamp("2011-01-01"), 1.0, 1.0, "UK"),      # cancellation
        ("9", "A1", "x", -2, pd.Timestamp("2011-01-01"), 1.0, 1.0, "UK"),      # negative qty
        ("9", "A1", "x", 2, pd.Timestamp("2011-01-01"), 0.0, 1.0, "UK"),       # zero price
        ("9", "A1", "x", 2, pd.Timestamp("2011-01-01"), 1.0, np.nan, "UK"),    # no customer
    ], columns=raw.columns)
    dup = raw.iloc[[0]]
    clean, report = clean_transactions(pd.concat([raw, bad, dup], ignore_index=True))
    assert report["cancelled_invoices"] == 1
    assert report["non_positive_price"] == 1
    assert report["missing_customer_id"] == 1
    assert report["duplicate_rows"] >= 1
    assert (clean["quantity"] > 0).all() and (clean["unit_price"] > 0).all()
    assert len(clean) == report["clean_rows"]


def test_rfm_values(tx):
    rfm = compute_rfm(tx)
    assert rfm["customer_id"].is_unique
    assert (rfm["recency"] >= 1).all()
    c1 = tx[tx["customer_id"] == 1]
    row = rfm.set_index("customer_id").loc[1]
    assert row["frequency"] == c1["invoice_no"].nunique()
    assert row["monetary"] == pytest.approx(c1["revenue"].sum(), abs=0.01)


def test_scores_are_quintiles(tx):
    scored = score_rfm(compute_rfm(tx))
    for col in ("r_score", "f_score", "m_score"):
        counts = scored[col].value_counts()
        assert set(counts.index) == {1, 2, 3, 4, 5}
        assert counts.max() - counts.min() <= 1  # equal-sized buckets
    # The most recent buyer gets R=5, the highest spender M=5.
    assert scored.loc[scored["recency"].idxmin(), "r_score"] == 5
    assert scored.loc[scored["monetary"].idxmax(), "m_score"] == 5


def test_every_customer_gets_a_named_segment(tx):
    seg = assign_segment(score_rfm(compute_rfm(tx)))
    assert seg["segment"].isin(SEGMENT_ACTIONS.keys()).all()
    assert seg["recommended_action"].notna().all()


def test_grid_covers_all_125_score_combinations():
    combos = pd.DataFrame(
        [(r, f, m) for r in range(1, 6) for f in range(1, 6) for m in range(1, 6)],
        columns=["r_score", "f_score", "m_score"],
    )
    out = assign_segment(combos)
    assert out["segment"].notna().all()
    assert out.loc[(out.r_score == 5) & (out.f_score == 5) & (out.m_score == 5), "segment"].iat[0] == "Champions"
    assert out.loc[(out.r_score == 1) & (out.f_score == 1) & (out.m_score == 1), "segment"].iat[0] == "Lost"


def test_clv_and_churn(tx):
    clv = estimate_clv(compute_rfm(tx))
    assert clv["churn_risk"].between(0, 1).all()
    assert (clv["predicted_clv"] >= 0).all()
    # Higher margin means proportionally higher CLV.
    clv2 = estimate_clv(compute_rfm(tx), margin=0.5)
    assert clv2["predicted_clv"].sum() == pytest.approx(2 * clv["predicted_clv"].sum(), rel=0.01)


def test_clustering_orders_by_value(tx):
    seg = assign_segment(score_rfm(compute_rfm(tx)))
    out, info = cluster_customers(seg, k=3)
    assert info["k"] == 3 and out["cluster"].nunique() == 3
    means = out.groupby("cluster")["monetary"].mean()
    assert means.is_monotonic_decreasing
    assert out.loc[out["cluster"] == 0, "cluster_name"].iat[0] == "VIP"


def test_cohort_retention(tx):
    ret = cohort_retention(tx)
    assert (ret[0] == 1.0).all()
    assert ((ret >= 0) & (ret <= 1)).all().all()
