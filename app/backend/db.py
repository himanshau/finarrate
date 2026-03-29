from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finance.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                type TEXT NOT NULL,
                category TEXT,
                FOREIGN KEY(upload_id) REFERENCES uploads(id)
            )
            """
        )
        conn.commit()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def save_upload(filename: str) -> int:
    with get_conn() as conn:
        cursor = conn.execute("INSERT INTO uploads(filename) VALUES(?)", (filename,))
        conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to persist upload metadata.")
        return int(cursor.lastrowid)


def save_transactions(upload_id: int, transactions: pd.DataFrame) -> None:
    if transactions.empty:
        return
    normalized = transactions.fillna("")
    records = [
        (
            upload_id,
            str(row.date),
            str(row.description),
            float(0.0 if pd.isna(pd.to_numeric(row.amount, errors="coerce")) else pd.to_numeric(row.amount, errors="coerce")),
            str(row.type),
            str(row.category) if "category" in normalized.columns else None,
        )
        for row in normalized.itertuples(index=False)
    ]
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO transactions(upload_id, date, description, amount, type, category)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def load_transactions(upload_id: int) -> pd.DataFrame:
    query = """
        SELECT date, description, amount, type, COALESCE(category, 'Other') AS category
        FROM transactions
        WHERE upload_id = ?
        ORDER BY date ASC
    """
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=(upload_id,))
    return df
