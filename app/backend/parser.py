from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import pdfplumber  # type: ignore[import-not-found]

DATE_PATTERNS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%m-%y",
]


def _clean_amount(raw: object) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "-"}:
        return 0.0

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
    if not cleaned or cleaned == "-":
        return 0.0

    try:
        value = float(cleaned)
        return -abs(value) if negative else value
    except ValueError:
        return 0.0


def _build_amount_and_type(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    debit_col = next((c for c in ["debit", "withdrawal", "dr", "debit_amount"] if c in df.columns), None)
    credit_col = next((c for c in ["credit", "deposit", "cr", "credit_amount"] if c in df.columns), None)
    amount_col = next((c for c in ["amount", "txn_amount", "transaction_amount"] if c in df.columns), None)
    type_col = next((c for c in ["type", "txn_type", "dr_cr", "transaction_type"] if c in df.columns), None)

    if debit_col and credit_col:
        debit_series = df[debit_col].apply(_clean_amount)
        credit_series = df[credit_col].apply(_clean_amount)
        amount_series = credit_series.where(credit_series > 0, debit_series)
        type_series = pd.Series(["income" if c > 0 else "expense" for c in credit_series], index=df.index)
        fallback_expense = debit_series > 0
        type_series = type_series.where(~fallback_expense, "expense")
        amount_series = amount_series.where(amount_series > 0, debit_series.abs())
        return amount_series.fillna(0.0), type_series

    if amount_col:
        amount_raw = df[amount_col].apply(_clean_amount)
        if type_col:
            raw_types = df[type_col].astype(str).str.lower().str.strip()
            mapped = raw_types.map(
                lambda t: "income" if t in {"cr", "credit", "income", "in"} else "expense" if t in {"dr", "debit", "expense", "out"} else "unknown"
            )
            inferred = mapped.where(mapped != "unknown", "income")
            inferred = inferred.where(amount_raw >= 0, "expense")
        else:
            inferred = amount_raw.map(lambda x: "income" if x > 0 else "expense")
        return amount_raw.abs(), inferred

    raise ValueError("CSV is missing amount-like columns. Provide amount or debit/credit columns.")


def _normalize_date(value: str) -> str:
    raw = str(value).strip()
    for pattern in DATE_PATTERNS:
        try:
            return datetime.strptime(raw, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _infer_type_and_amount(amount: float, description: str) -> tuple[float, str]:
    text = description.lower()
    if amount < 0:
        return abs(float(amount)), "expense"
    if any(k in text for k in ["salary", "credited", "refund", "interest"]):
        return float(amount), "income"
    return float(amount), "income" if amount > 0 else "expense"


def parse_csv_statement(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.empty:
        return pd.DataFrame(columns=["date", "description", "amount", "type"])

    df.columns = [str(col).strip().lower() for col in df.columns]
    date_col: Optional[str] = next(
        (
            c
            for c in ["date", "transaction_date", "txn_date", "value_date", "posting_date", "txn date"]
            if c in df.columns
        ),
        None,
    )
    desc_col: Optional[str] = next(
        (
            c
            for c in ["description", "narration", "details", "merchant", "remarks", "transaction_details"]
            if c in df.columns
        ),
        None,
    )

    if not date_col or not desc_col:
        raise ValueError("CSV is missing required columns. Expected date and description-like columns.")

    clean = pd.DataFrame()
    clean["date"] = df[date_col].astype(str).apply(_normalize_date)
    clean["description"] = df[desc_col].astype(str).fillna("Unknown")
    amount_series, type_series = _build_amount_and_type(df)
    clean["amount"] = amount_series.fillna(0.0)
    clean["type"] = type_series.fillna("expense")

    # Improve type inference using description hints for ambiguous rows.
    inferred = clean.apply(lambda row: _infer_type_and_amount(row["amount"], row["description"]), axis=1)
    needs_override = clean["type"].eq("income") & clean["description"].str.lower().str.contains("debit|purchase|bill|fee", na=False)
    clean.loc[needs_override, "type"] = "expense"
    clean["type"] = clean["type"].where(clean["type"].isin(["income", "expense"]), inferred.apply(lambda x: x[1]))
    clean["amount"] = clean["amount"].abs().round(2)
    return clean


def parse_pdf_statement(file_bytes: bytes) -> pd.DataFrame:
    rows: list[tuple[str, str, float, str]] = []
    amount_pattern = re.compile(r"-?\d+[\.,]?\d{0,2}")

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", line)
                if not date_match:
                    continue
                amounts = amount_pattern.findall(line)
                if not amounts:
                    continue
                amount_raw = amounts[-1].replace(",", "")
                try:
                    amount_value = float(amount_raw)
                except ValueError:
                    continue

                date_value = _normalize_date(date_match.group(0))
                description = line.replace(date_match.group(0), "").strip(" -")
                amount, txn_type = _infer_type_and_amount(amount_value, description)
                rows.append((date_value, description or "Unknown", amount, txn_type))

    return pd.DataFrame(rows, columns=["date", "description", "amount", "type"])


def parse_statement(filename: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return parse_csv_statement(file_bytes)
    if lower_name.endswith(".pdf"):
        return parse_pdf_statement(file_bytes)
    raise ValueError("Unsupported file type. Please upload PDF or CSV.")
