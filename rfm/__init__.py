"""Modern RFM customer-segmentation toolkit for the UCI Online Retail dataset."""

from .data import clean_transactions, load_transactions
from .scoring import compute_rfm, score_rfm
from .segments import SEGMENT_ACTIONS, assign_segment
from .clustering import cluster_customers
from .clv import estimate_clv
from .cohorts import cohort_retention

__all__ = [
    "load_transactions",
    "clean_transactions",
    "compute_rfm",
    "score_rfm",
    "assign_segment",
    "SEGMENT_ACTIONS",
    "cluster_customers",
    "estimate_clv",
    "cohort_retention",
]

__version__ = "2.0.0"
