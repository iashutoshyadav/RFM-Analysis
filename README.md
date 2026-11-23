# RFM Customer Intelligence: Online Retail Dataset

Customer segmentation for the UCI Online Retail dataset (541,909 transactions, 4,338 customers, Dec 2010 – Dec 2011), built with **RFM analysis, machine learning, CLV prediction and an interactive dashboard**.

> **v2** adds a reproducible Python pipeline, K-Means clustering, customer-lifetime-value and churn-risk scoring, cohort retention and a Streamlit web dashboard, all on top of the original SQL + Excel workflow (still included, see [Original workflow](#original-sql--excel-workflow)).

## Key findings

| | |
|---|---|
| **Champions** | 17% of customers, **61% of revenue** |
| **At Risk + Cannot Lose Them** | 786 lapsing customers holding **£1.1M** of lifetime revenue, the top win-back target |
| **Repeat rate** | 66% of customers have ordered more than once |
| **Retention** | about 20% of a new monthly cohort returns the next month. The Dec 2010 cohort retains about 2× better (pre-existing loyal customers) |

## What's inside

| Feature | Description |
|---|---|
| **Python pipeline** (`pipeline.py`) | One command from raw Excel to analysis outputs. The first run caches the data as Parquet, so later runs take about 8 s instead of about 90 s |
| **Better cleaning** | Also removes cancelled invoices (`C…`) and 5,192 duplicate rows, which the original SQL missed |
| **Full-coverage segmentation** | R × FM grid with 11 named segments. Every customer gets a segment and a recommended action, with no "Other" bucket |
| **ML clustering** | K-Means on log-scaled RFM values, with k chosen by silhouette score. Clusters are named (VIP / High Value / Dormant) and cross-checked against the rule-based segments |
| **CLV + churn risk** | Predicted lifetime value per customer, with adjustable margin and lifespan. Churn risk measures how overdue a customer is compared with their own usual gap between orders |
| **Cohort retention** | Monthly acquisition-cohort retention matrix |
| **Interactive dashboard** (`app.py`) | Streamlit + Plotly. Filters, KPIs, RFM heatmap, 3-D cluster explorer, cohort heatmap, churn watch-list, single-customer lookup and a campaign audience builder with CSV export |
| **Tests + CI** | `pytest` unit tests, plus a GitHub Actions workflow that also runs the full pipeline and a dashboard smoke test |
| **Docker** | `docker build` / `docker run` gives you the dashboard on port 8501 |

## Quick start

```bash
pip install -r requirements.txt
python pipeline.py          # writes results to output/
streamlit run app.py        # opens the dashboard at http://localhost:8501
```

Or with Docker:

```bash
docker build -t rfm-dashboard .
docker run -p 8501:8501 rfm-dashboard
```

Pipeline options:

```bash
python pipeline.py --input path/to/data.csv --clusters 4 --margin 0.30 --lifespan 5
```

### Outputs (`output/`)

| File | Contents |
|---|---|
| `customers.csv` / `.parquet` | One row per customer: R/F/M values and scores, segment, action, cluster, CLV, churn risk |
| `segment_summary.csv` | Customers, revenue, shares, average CLV and churn risk per segment |
| `cluster_profile.csv` | Profile of each K-Means cluster |
| `cohort_retention.csv` | Cohort × months-since-first-purchase retention |
| `monthly_revenue.csv` | Revenue, orders and active customers by month |
| `run_report.json` | Cleaning stats, silhouette scores, run metadata |

## Methodology

1. **Clean.** Drop rows with no customer ID, cancelled invoices, zero or negative quantity or price, and exact duplicates.
2. **RFM.** The snapshot date is the day after the last transaction.
   - **R** = days since last order
   - **F** = distinct invoices
   - **M** = total revenue
3. **Score.** Quintiles 1–5 (5 = best). Values are ranked before binning, so heavy ties (e.g. the many single-order customers) still split into equal groups.
4. **Segment.** FM = mean of F and M rounded half up, then looked up on the R × FM grid:

   | R \ FM | 1 | 2 | 3 | 4 | 5 |
   |---|---|---|---|---|---|
   | **5** | New Customers | Promising | Potential Loyalists | Loyal | Champions |
   | **4** | Promising | Potential Loyalists | Potential Loyalists | Loyal | Champions |
   | **3** | About to Sleep | Need Attention | Need Attention | Loyal | Loyal |
   | **2** | Hibernating | About to Sleep | At Risk | At Risk | Cannot Lose Them |
   | **1** | Lost | Hibernating | At Risk | At Risk | Cannot Lose Them |

5. **Cluster.** Apply `log1p`, then standardise, then K-Means for k = 3…8. The best silhouette score wins. Clusters are relabelled by average revenue.
6. **CLV.** `avg order value × orders per year × lifespan × margin × (1 − churn risk)`. Orders per year uses tenure floored at 1 year, so short histories are never scaled up.
7. **Churn risk.** `1 − exp(−(recency / usual gap between orders) / 2)`. A one-time buyer is assumed to have a 90-day usual gap.

The same cleaning and segmentation logic is available in pure SQL in [`sql/rfm_v2.sql`](sql/rfm_v2.sql) (PostgreSQL).

## Repository structure

```
├── app.py                  Streamlit dashboard
├── pipeline.py             End-to-end CLI pipeline
├── rfm/                    Reusable package
│   ├── data.py             loading (xlsx/csv/parquet + cache) and cleaning
│   ├── scoring.py          RFM values and quintile scores
│   ├── segments.py         R×FM segment grid and recommended actions
│   ├── clustering.py       K-Means with silhouette-based k selection
│   ├── clv.py              CLV and churn risk
│   └── cohorts.py          cohort retention
├── sql/rfm_v2.sql          SQL version of the v2 logic
├── tests/                  pytest suite
├── .github/workflows/ci.yml
├── Dockerfile
├── Dataset/Online Retail.xlsx
│
├── processing_query.sql    (v1) original SQL workflow
├── Processed Data.csv      (v1) original SQL output
├── RFM Analysis.xlsx       (v1) Excel dashboard
└── Analysis_Dashboard.jpg  (v1) dashboard screenshot
```

## Original SQL + Excel workflow

![Dashboard](./Analysis_Dashboard.jpg)

The v1 project used PostgreSQL for import, exploration, cleaning and RFM scoring ([`processing_query.sql`](processing_query.sql)), exported [`Processed Data.csv`](Processed%20Data.csv), and built the dashboard in Excel ([`RFM Analysis.xlsx`](RFM%20Analysis.xlsx)). To reproduce it, convert `Dataset/Online Retail.xlsx` to CSV, update the `COPY` path in the SQL, run the script, and load the result into the Excel workbook.

## Credits

Dataset: [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
