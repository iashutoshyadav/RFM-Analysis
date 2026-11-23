"""Machine-learning segmentation with K-Means as a data-driven complement
to the rule-based segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency", "frequency", "monetary"]


def _features(rfm: pd.DataFrame) -> np.ndarray:
    # RFM values are heavily right-skewed; log1p + standardisation gives
    # K-Means roughly spherical clusters to work with.
    return StandardScaler().fit_transform(np.log1p(rfm[FEATURES].clip(lower=0)))


CLUSTER_NAMES = ["VIP", "High Value", "Mid Value", "Low Value", "Dormant", "Cluster 5", "Cluster 6", "Cluster 7"]


def cluster_customers(
    rfm: pd.DataFrame, k: int | None = None, k_range=range(3, 9), random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Add a `cluster` label (0 = highest value), a `cluster_name`, and return
    silhouette scores.

    If `k` is None the number of clusters is chosen by best silhouette score.
    k=2 is excluded by default: it usually wins on silhouette ("active vs
    inactive") but is too coarse to act on.
    """
    X = _features(rfm)
    scores = {}
    sample = min(len(X), 5000)
    for n in k_range:
        labels = KMeans(n_clusters=n, n_init=10, random_state=random_state).fit_predict(X)
        scores[n] = float(silhouette_score(X, labels, sample_size=sample, random_state=random_state))
    best_k = k or max(scores, key=scores.get)

    model = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit(X)
    out = rfm.copy()
    out["cluster"] = model.labels_

    # Relabel clusters so 0 is the most valuable (highest mean monetary).
    order = out.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
    out["cluster"] = out["cluster"].map({old: new for new, old in enumerate(order)})
    names = CLUSTER_NAMES[:best_k]
    if best_k <= 5:  # always end on the lowest-value label
        names = CLUSTER_NAMES[: best_k - 1] + ["Dormant"]
    out["cluster_name"] = out["cluster"].map(dict(enumerate(names)))
    return out, {"k": best_k, "silhouette": scores}


def cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["cluster", "cluster_name"])
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            top_segment=("segment", lambda s: s.mode().iat[0]),
        )
        .round(1)
        .reset_index()
    )
