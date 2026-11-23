"""Interactive RFM dashboard.

    streamlit run app.py

Reads the files written by `python pipeline.py` (runs it automatically the
first time if output/ is missing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rfm.clv import estimate_clv
from rfm.segments import SEGMENT_ACTIONS, SEGMENT_ORDER

OUT = Path(__file__).parent / "output"

# Reference data-viz palette (categorical slots in fixed order; one-hue blue ramp).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE = "#2a78d6"
BLUE_RAMP = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]

st.set_page_config(page_title="RFM Customer Intelligence", page_icon="📊", layout="wide")


# ---------- data ----------
@st.cache_data(show_spinner="Loading analysis...")
def load() -> dict[str, pd.DataFrame]:
    if not (OUT / "customers.parquet").exists():
        from pipeline import run

        run(Path(__file__).parent / "Dataset" / "Online Retail.xlsx", OUT, None, 0.25, 3.0)
    return {
        "customers": pd.read_parquet(OUT / "customers.parquet"),
        "tx": pd.read_parquet(OUT / "transactions_clean.parquet"),
        "cohorts": pd.read_csv(OUT / "cohort_retention.csv", index_col=0),
    }


data = load()
customers_all, tx_all, cohorts = data["customers"], data["tx"], data["cohorts"]


def money(x: float) -> str:
    return f"£{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"£{x/1e3:,.1f}K" if abs(x) >= 1e4 else f"£{x:,.0f}"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False)
    return fig


# ---------- sidebar ----------
st.sidebar.title("📊 RFM Intelligence")
st.sidebar.caption("UCI Online Retail · Dec 2010 – Dec 2011")

countries = customers_all["country"].value_counts().index.tolist()
sel_countries = st.sidebar.multiselect("Country", countries, placeholder="All countries")
sel_segments = st.sidebar.multiselect(
    "Segment", [s for s in SEGMENT_ORDER if s in set(customers_all["segment"])], placeholder="All segments"
)

st.sidebar.divider()
st.sidebar.subheader("CLV assumptions")
margin = st.sidebar.slider("Gross margin", 0.05, 0.60, 0.25, 0.05, format="%.2f")
lifespan = st.sidebar.slider("Expected lifespan (years)", 1.0, 10.0, 3.0, 0.5)

customers = estimate_clv(customers_all, margin=margin, lifespan_years=lifespan)
if sel_countries:
    customers = customers[customers["country"].isin(sel_countries)]
if sel_segments:
    customers = customers[customers["segment"].isin(sel_segments)]
tx = tx_all[tx_all["customer_id"].isin(customers["customer_id"])]

if customers.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---------- header KPIs ----------
st.title("Customer Segmentation Dashboard")
k = st.columns(5)
k[0].metric("Customers", f"{len(customers):,}")
k[1].metric("Revenue", money(customers["monetary"].sum()))
k[2].metric("Avg order value", money(customers["monetary"].sum() / customers["frequency"].sum()))
k[3].metric("Repeat-customer rate", f"{(customers['frequency'] > 1).mean():.0%}")
k[4].metric("Predicted CLV (total)", money(customers["predicted_clv"].sum()),
            help=f"AOV × orders/yr × {lifespan:g} yrs × {margin:.0%} margin × (1 − churn risk)")

tabs = st.tabs(["Overview", "RFM grid", "ML clusters", "Cohorts", "Churn & CLV", "Customer lookup", "Campaign export"])

# ---------- Overview ----------
with tabs[0]:
    seg = (
        customers.groupby("segment")
        .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"))
        .reindex([s for s in SEGMENT_ORDER if s in set(customers["segment"])])
        .reset_index()
    )
    seg["customer_share"] = seg["customers"] / seg["customers"].sum()
    seg["revenue_share"] = seg["revenue"] / seg["revenue"].sum()

    c1, c2 = st.columns(2)
    fig = px.bar(seg, x="customers", y="segment", orientation="h", title="Customers by segment",
                 custom_data=["customer_share"], color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} customers (%{customdata[0]:.1%})<extra></extra>",
                      marker_cornerradius=4)
    fig.update_yaxes(autorange="reversed", title=None)
    c1.plotly_chart(style(fig), width="stretch")

    fig = px.bar(seg, x="revenue", y="segment", orientation="h", title="Revenue by segment",
                 custom_data=["revenue_share"], color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f} (%{customdata[0]:.1%} of revenue)<extra></extra>",
                      marker_cornerradius=4)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickprefix="£")
    c2.plotly_chart(style(fig), width="stretch")

    top = seg.sort_values("revenue_share", ascending=False).head(2)
    if len(top) == 2:
        st.info(
            f"**{top.iloc[0]['segment']}** are {top.iloc[0]['customer_share']:.0%} of customers but drive "
            f"**{top.iloc[0]['revenue_share']:.0%}** of revenue. Together with **{top.iloc[1]['segment']}** "
            f"they account for {top['revenue_share'].sum():.0%}."
        )

    monthly = (
        tx.assign(month=tx["invoice_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month").agg(revenue=("revenue", "sum"), orders=("invoice_no", "nunique"))
        .reset_index()
    )
    c1, c2 = st.columns(2)
    fig = px.line(monthly, x="month", y="revenue", title="Monthly revenue", markers=True,
                  color_discrete_sequence=[BLUE])
    fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="£", title=None, rangemode="tozero")
    fig.update_xaxes(title=None)
    c1.plotly_chart(style(fig), width="stretch")

    fig = px.line(monthly, x="month", y="orders", title="Monthly orders", markers=True,
                  color_discrete_sequence=[BLUE])
    fig.update_traces(line_width=2, marker_size=8, hovertemplate="%{x|%b %Y}<br>%{y:,} orders<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title=None, rangemode="tozero")
    fig.update_xaxes(title=None)
    c2.plotly_chart(style(fig), width="stretch")
    st.caption("December 2011 is a partial month (data ends 9 Dec 2011).")

    with st.expander("Segment playbook: recommended actions"):
        st.dataframe(
            pd.DataFrame({"Segment": list(SEGMENT_ACTIONS), "Action": list(SEGMENT_ACTIONS.values())}),
            hide_index=True, width="stretch",
        )

# ---------- RFM grid ----------
with tabs[1]:
    st.subheader("Recency × Frequency/Monetary grid")
    st.caption("Each cell is a group of customers. Top-right = recent and high-value (Champions); "
               "bottom-right = high-value but lapsing (Cannot Lose Them).")
    metric = st.radio("Cell value", ["Customers", "Revenue", "Avg CLV"], horizontal=True)
    col = {"Customers": ("customer_id", "count"), "Revenue": ("monetary", "sum"),
           "Avg CLV": ("predicted_clv", "mean")}[metric]
    grid = customers.pivot_table(index="r_score", columns="fm_score", values=col[0], aggfunc=col[1])
    grid = grid.reindex(index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5])
    seg_grid = customers.groupby(["r_score", "fm_score"])["segment"].first().unstack().reindex(
        index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5])
    fmt = "{:,.0f}" if metric == "Customers" else "£{:,.0f}"
    text = [[(f"{seg_grid.iat[i, j]}<br>{fmt.format(grid.iat[i, j])}" if pd.notna(grid.iat[i, j]) else "")
             for j in range(5)] for i in range(5)]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"FM {c}" for c in grid.columns], y=[f"R {r}" for r in grid.index],
        colorscale=BLUE_RAMP, text=text, texttemplate="%{text}", xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>", colorbar=dict(title=metric),
    ))
    st.plotly_chart(style(fig, 520), width="stretch")

# ---------- ML clusters ----------
with tabs[2]:
    st.subheader("K-Means clusters (unsupervised)")
    st.caption("Clusters are learned from log-scaled Recency, Frequency and Monetary values. "
               "They are independent of the rule-based segments and are a useful cross-check.")
    names = customers.sort_values("cluster")["cluster_name"].unique().tolist()
    colors = {n: SERIES[i % len(SERIES)] for i, n in enumerate(sorted(
        customers_all["cluster_name"].unique(), key=lambda n: customers_all.loc[customers_all.cluster_name == n, "cluster"].iat[0]))}
    sample = customers.sample(min(len(customers), 3000), random_state=1)
    fig = px.scatter_3d(
        sample, x="recency", y="frequency", z="monetary", color="cluster_name", log_y=True, log_z=True,
        color_discrete_map=colors, category_orders={"cluster_name": names}, opacity=0.75,
        hover_data={"customer_id": True, "segment": True, "monetary": ":,.0f"},
        labels={"recency": "Recency (days)", "frequency": "Orders", "monetary": "Revenue (£)",
                "cluster_name": "Cluster"},
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(style(fig, 560), width="stretch")

    prof = (
        customers.groupby(["cluster", "cluster_name"])
        .agg(Customers=("customer_id", "count"), **{"Avg recency (d)": ("recency", "mean"),
             "Avg orders": ("frequency", "mean"), "Avg revenue": ("monetary", "mean"),
             "Total revenue": ("monetary", "sum"), "Most common segment": ("segment", lambda s: s.mode().iat[0])})
        .round(1).reset_index().drop(columns="cluster").rename(columns={"cluster_name": "Cluster"})
    )
    st.dataframe(prof, hide_index=True, width="stretch",
                 column_config={"Avg revenue": st.column_config.NumberColumn(format="£%.0f"),
                                "Total revenue": st.column_config.NumberColumn(format="£%.0f")})

    st.markdown("**Agreement between segments and clusters** (customer counts)")
    xt = pd.crosstab(customers["segment"], customers["cluster_name"]).reindex(
        [s for s in SEGMENT_ORDER if s in set(customers["segment"])])[names]
    fig = go.Figure(go.Heatmap(z=xt.values, x=xt.columns, y=xt.index, colorscale=BLUE_RAMP,
                               texttemplate="%{z}", xgap=2, ygap=2,
                               hovertemplate="%{y} → %{x}: %{z} customers<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 460), width="stretch")

# ---------- Cohorts ----------
with tabs[3]:
    st.subheader("Monthly cohort retention")
    st.caption("Share of customers from each first-purchase month who bought again N months later "
               "(all customers, not filtered).")
    z = cohorts.values.astype(float)
    z[:, 0] = np.nan  # month 0 is always 100% and would flatten the colour scale
    z[z == 0] = np.nan
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"M{c}" for c in cohorts.columns], y=cohorts.index, colorscale=BLUE_RAMP,
        texttemplate="%{z:.0%}", xgap=2, ygap=2, colorbar=dict(tickformat=".0%"),
        hovertemplate="Cohort %{y}, %{x}: %{z:.1%} retained<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    st.plotly_chart(style(fig, 520), width="stretch")
    avg_m1 = cohorts["1"].replace(0, np.nan).mean()
    st.info(f"On average **{avg_m1:.0%}** of new customers return the following month. "
            "The Dec 2010 cohort retains far better than later ones; it likely holds long-standing "
            "customers who were already buying before the dataset starts.")

# ---------- Churn & CLV ----------
with tabs[4]:
    st.subheader("Churn risk and predicted lifetime value")
    c1, c2 = st.columns(2)
    fig = px.histogram(customers, x="churn_risk", nbins=20, title="Churn-risk distribution",
                       color_discrete_sequence=[BLUE])
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)",
                      hovertemplate="Risk %{x}<br>%{y} customers<extra></extra>")
    fig.update_xaxes(tickformat=".0%", title="Churn risk")
    fig.update_yaxes(title="Customers")
    c1.plotly_chart(style(fig), width="stretch")

    clv_seg = customers.groupby("segment")["predicted_clv"].mean().reindex(
        [s for s in SEGMENT_ORDER if s in set(customers["segment"])]).reset_index()
    fig = px.bar(clv_seg, x="predicted_clv", y="segment", orientation="h", title="Average predicted CLV by segment",
                 color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>", marker_cornerradius=4)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickprefix="£", title=None)
    c2.plotly_chart(style(fig), width="stretch")

    st.markdown("**🚨 High-value customers at risk of churning:** contact these first")
    at_risk = customers[(customers["churn_risk"] >= 0.5)].nlargest(25, "monetary")[
        ["customer_id", "segment", "country", "recency", "frequency", "monetary", "churn_risk", "recommended_action"]]
    st.dataframe(at_risk, hide_index=True, width="stretch", column_config={
        "monetary": st.column_config.NumberColumn("Lifetime revenue", format="£%.0f"),
        "churn_risk": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="percent"),
        "recency": "Days since last order", "frequency": "Orders"})
    st.metric("Revenue held by these at-risk customers", money(at_risk["monetary"].sum()))

# ---------- Customer lookup ----------
with tabs[5]:
    cid = st.selectbox("Customer ID", customers["customer_id"].sort_values(), index=None,
                       placeholder="Type or pick a customer ID")
    if cid is not None:
        c = customers.set_index("customer_id").loc[cid]
        st.subheader(f"Customer {cid} · {c['segment']} · {c['cluster_name']}")
        m = st.columns(5)
        m[0].metric("Last order", f"{c['recency']} days ago")
        m[1].metric("Orders", int(c["frequency"]))
        m[2].metric("Lifetime revenue", money(c["monetary"]))
        m[3].metric("Predicted CLV", money(c["predicted_clv"]))
        m[4].metric("Churn risk", f"{c['churn_risk']:.0%}")
        st.success(f"**Recommended action:** {c['recommended_action']}")
        st.caption(f"RFM code {c['rfm_code']} · Country: {c['country']} · "
                   f"Customer since {c['first_purchase']:%d %b %Y}")

        hist = tx_all[tx_all["customer_id"] == cid]
        orders = hist.groupby(["invoice_no", "invoice_date"])["revenue"].sum().reset_index()
        fig = px.bar(orders, x="invoice_date", y="revenue", title="Order history", color_discrete_sequence=[BLUE])
        fig.update_traces(hovertemplate="%{x|%d %b %Y}<br>£%{y:,.2f}<extra></extra>", marker_cornerradius=4)
        fig.update_yaxes(tickprefix="£", title=None)
        fig.update_xaxes(title=None)
        st.plotly_chart(style(fig, 300), width="stretch")

        topp = (hist.groupby("description").agg(qty=("quantity", "sum"), revenue=("revenue", "sum"))
                .nlargest(10, "revenue").reset_index())
        st.markdown("**Top products**")
        st.dataframe(topp, hide_index=True, width="stretch",
                     column_config={"revenue": st.column_config.NumberColumn(format="£%.2f")})

# ---------- Campaign export ----------
with tabs[6]:
    st.subheader("Build a campaign audience")
    c1, c2, c3 = st.columns(3)
    aud_options = [s for s in SEGMENT_ORDER if s in set(customers["segment"])]
    aud_seg = c1.multiselect("Segments", aud_options,
                             default=[s for s in ("At Risk", "Cannot Lose Them") if s in aud_options])
    min_rev = c2.number_input("Min lifetime revenue (£)", 0, value=0, step=100)
    max_risk = c3.slider("Max churn risk", 0.0, 1.0, 1.0, 0.05)
    aud = customers[(customers["monetary"] >= min_rev) & (customers["churn_risk"] <= max_risk)]
    if aud_seg:
        aud = aud[aud["segment"].isin(aud_seg)]
    st.write(f"**{len(aud):,} customers** · {money(aud['monetary'].sum())} lifetime revenue · "
             f"{money(aud['predicted_clv'].sum())} predicted CLV")
    cols = ["customer_id", "country", "segment", "cluster_name", "rfm_code", "recency", "frequency",
            "monetary", "predicted_clv", "churn_risk", "recommended_action"]
    st.dataframe(aud[cols], hide_index=True, width="stretch", height=360)
    st.download_button("⬇️ Download audience CSV", aud[cols].to_csv(index=False).encode(),
                       "campaign_audience.csv", "text/csv", type="primary")
