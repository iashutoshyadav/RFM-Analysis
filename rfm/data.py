"""Loading and cleaning of raw transaction data.

The raw Excel file is ~23 MB and slow to parse, so the first load is cached
as Parquet next to it; later loads take well under a second.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMN_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def load_transactions(path: str | Path, use_cache: bool = True) -> pd.DataFrame:
    """Load raw transactions from .xlsx, .csv or .parquet with snake_case columns."""
    path = Path(path)
    cache = path.with_suffix(".parquet")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_parquet(cache)
    else:
        if path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
        elif path.suffix == ".csv":
            df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str}, encoding="latin-1")
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        df = df.rename(columns=COLUMN_MAP)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        for col in ("invoice_no", "stock_code", "description", "country"):
            df[col] = df[col].astype("string")
        if use_cache:
            df.to_parquet(cache, index=False)

    return df.rename(columns=COLUMN_MAP)


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the same cleaning rules as the original SQL, plus de-duplication.

    Returns the cleaned frame and a report of how many rows each rule removed.
    """
    report = {"raw_rows": len(df)}
    steps = [
        ("missing_customer_id", df["customer_id"].notna()),
        ("cancelled_invoices", ~df["invoice_no"].astype(str).str.startswith("C")),
        ("non_positive_quantity", df["quantity"] > 0),
        ("non_positive_price", df["unit_price"] > 0),
    ]
    mask = pd.Series(True, index=df.index)
    for name, rule in steps:
        report[name] = int((mask & ~rule).sum())
        mask &= rule

    out = df[mask].copy()
    before = len(out)
    out = out.drop_duplicates()
    report["duplicate_rows"] = before - len(out)

    out["customer_id"] = out["customer_id"].astype(int)
    out["invoice_no"] = out["invoice_no"].astype(str)
    out["revenue"] = out["quantity"] * out["unit_price"]
    report["clean_rows"] = len(out)
    return out.reset_index(drop=True), report
