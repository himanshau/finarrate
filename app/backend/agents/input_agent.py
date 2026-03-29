from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from parser import parse_statement


@dataclass
class InputAgent:
    """Parses and standardizes raw statement files deterministically."""

    def run(self, filename: str, file_bytes: bytes) -> dict:
        df = parse_statement(filename, file_bytes)
        if df.empty:
            return {"status": "error", "message": "No transactions extracted.", "transactions": []}

        required_cols = ["date", "description", "amount", "type"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {
                "status": "error",
                "message": f"Missing required columns after parsing: {', '.join(missing)}",
                "transactions": [],
            }

        normalized = df[required_cols].copy()
        normalized["date"] = normalized["date"].astype(str)
        normalized["description"] = normalized["description"].astype(str).str.strip().replace("", "Unknown")
        normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce").fillna(0.0).abs().round(2)
        normalized["type"] = normalized["type"].astype(str).str.lower().where(
            normalized["type"].astype(str).str.lower().isin(["income", "expense"]), "expense"
        )

        return {
            "status": "ok",
            "message": "Statement parsed successfully.",
            "transactions": normalized.to_dict(orient="records"),
        }
