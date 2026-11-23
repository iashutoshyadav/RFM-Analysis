"""End-to-end RFM pipeline.

    python pipeline.py                       # uses Dataset/Online Retail.xlsx
    python pipeline.py --input data.csv --clusters 4 --margin 0.3

Writes to output/:
    customers.csv / customers.parquet  one row per customer, all scores + segments
    segment_summary.csv                KPIs per segment
    cluster_profile.csv                KPIs per K-Means cluster
    cohort_retention.csv               monthly retention matrix
    monthly_revenue.csv                revenue / orders / customers by month
    run_report.json                    cleaning stats, cluster selection, totals
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rfm import (
    assign_segment,
    clean_transactions,
    cluster_customers,
    cohort_retention,
    compute_rfm,
    estimate_clv,
    load_transactions,
    score_rfm,
)
from rfm.clustering import cluster_profile
from rfm.segments import SEGMENT_ORDER


def segment_summary(customers: pd.DataFrame) -> pd.DataFrame:
    total_rev = customers["monetary"].sum()
    s = (
        customers.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_clv=("predicted_clv", "mean"),
            avg_churn_risk=("churn_risk", "mean"),
        )
        .reindex(SEGMENT_ORDER)
        .dropna(how="all")
    )
    s["customer_share"] = s["customers"] / s["customers"].sum()
    s["revenue_share"] = s["revenue"] / total_rev
    return s.round(3).reset_index()


def monthly_revenue(tx: pd.DataFrame) -> pd.DataFrame:
    return (
        tx.assign(month=tx["invoice_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            revenue=("revenue", "sum"),
            orders=("invoice_no", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .round(2)
        .reset_index()
    )


def run(input_path: Path, out_dir: Path, k: int | None, margin: float, lifespan: float) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    raw = load_transactions(input_path)
    tx, clean_report = clean_transactions(raw)
    print(f"  {clean_report['raw_rows']:,} raw rows -> {clean_report['clean_rows']:,} clean rows")

    customers = assign_segment(score_rfm(compute_rfm(tx)))
    customers = estimate_clv(customers, margin=margin, lifespan_years=lifespan)
    print(f"Clustering {len(customers):,} customers ...")
    customers, cluster_info = cluster_customers(customers, k=k)
    print(f"  chose k={cluster_info['k']}")

    tx.to_parquet(out_dir / "transactions_clean.parquet", index=False)
    customers.to_parquet(out_dir / "customers.parquet", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    segment_summary(customers).to_csv(out_dir / "segment_summary.csv", index=False)
    cluster_profile(customers).to_csv(out_dir / "cluster_profile.csv", index=False)
    cohort_retention(tx).to_csv(out_dir / "cohort_retention.csv")
    monthly_revenue(tx).to_csv(out_dir / "monthly_revenue.csv", index=False)

    report = {
        "input": str(input_path),
        "cleaning": clean_report,
        "customers": len(customers),
        "total_revenue": round(float(customers["monetary"].sum()), 2),
        "date_range": [str(tx["invoice_date"].min()), str(tx["invoice_date"].max())],
        "clustering": cluster_info,
        "clv_assumptions": {"margin": margin, "lifespan_years": lifespan},
        "segments": customers["segment"].value_counts().to_dict(),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(f"Done in {report['runtime_seconds']}s -> {out_dir}/")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("Dataset/Online Retail.xlsx"))
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--clusters", type=int, default=None, help="fix K (default: best silhouette)")
    p.add_argument("--margin", type=float, default=0.25, help="gross margin for CLV")
    p.add_argument("--lifespan", type=float, default=3.0, help="expected lifespan in years for CLV")
    a = p.parse_args()
    run(a.input, a.out, a.clusters, a.margin, a.lifespan)


if __name__ == "__main__":
    main()
