from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analyzer import compute_financial_metrics


@dataclass
class FinancialAnalysisAgent:
    """Computes finance metrics, health score, and deterministic risk alerts."""

    def _money_health_score(self, metrics) -> dict[str, Any]:
        # Weighted model: savings 30, EMI 30, distribution 40.
        if metrics.savings_rate >= 30:
            savings_component = 30
        elif metrics.savings_rate >= 20:
            savings_component = 25
        elif metrics.savings_rate >= 10:
            savings_component = 18
        else:
            savings_component = 10

        if metrics.emi_to_income_ratio <= 20:
            emi_component = 30
            emi_note = "EMI is within safe limits."
        elif metrics.emi_to_income_ratio <= 30:
            emi_component = 22
            emi_note = "EMI is moderate and should be monitored."
        elif metrics.emi_to_income_ratio <= 40:
            emi_component = 12
            emi_note = "EMI is high and may pressure monthly cash flow."
        else:
            emi_component = 5
            emi_note = "EMI is very high for current income."

        total_expenses = metrics.total_expenses if metrics.total_expenses > 0 else 1
        rent = metrics.category_spending.get("Rent", 0.0)
        food = metrics.category_spending.get("Food", 0.0)
        travel = metrics.category_spending.get("Travel", 0.0)
        shopping = metrics.category_spending.get("Shopping", 0.0)
        utilities = metrics.category_spending.get("Utilities", 0.0)
        other = metrics.category_spending.get("Other", 0.0)
        discretionary_ratio = ((food + travel + shopping) / total_expenses) * 100
        other_ratio = (other / total_expenses) * 100

        distribution_component = 34
        if discretionary_ratio > 50:
            distribution_component -= 14
        elif discretionary_ratio > 40:
            distribution_component -= 10
        elif discretionary_ratio > 30:
            distribution_component -= 6
        if other_ratio > 20:
            distribution_component -= 10
        elif other_ratio > 12:
            distribution_component -= 6
        elif other_ratio > 7:
            distribution_component -= 3
        if ((food + shopping) / total_expenses) * 100 > 20:
            distribution_component -= 4
        if (rent / total_expenses) * 100 > 35:
            distribution_component -= 4
        if (utilities / total_expenses) * 100 > 12:
            distribution_component -= 4
        distribution_component = max(distribution_component, 10)

        score = min(max(int(round(savings_component + emi_component + distribution_component)), 0), 100)
        explanation = (
            f"Savings contributes {savings_component}/30, EMI contributes {emi_component}/30, "
            f"and expense balance contributes {distribution_component}/40. {emi_note}"
        )
        return {
            "score": score,
            "explanation": explanation,
            "components": {
                "savings_component_out_of_30": savings_component,
                "emi_component_out_of_30": emi_component,
                "distribution_component_out_of_40": distribution_component,
            },
        }

    def _risk_alerts(self, df: pd.DataFrame, metrics) -> list[str]:
        alerts: list[str] = []
        if metrics.savings_rate < 20:
            alerts.append(
                "Your savings rate is below the recommended 20%, which may impact long-term financial stability."
            )

        if metrics.emi_to_income_ratio < 15:
            alerts.append("EMI ratio is within safe limits currently.")
        elif metrics.emi_to_income_ratio <= 25:
            alerts.append("EMI ratio is moderate and should be monitored.")
        elif metrics.emi_to_income_ratio > 30:
            alerts.append("EMI ratio is high and should be reduced when possible.")

        unknown_count = int(df["description"].str.contains("unknown", case=False, na=False).sum())
        if unknown_count > 0:
            alerts.append("Unknown transactions detected. Review and label them to avoid hidden leakage.")

        fee_count = int(df["description"].str.contains("fee|charge|penalty|gst", case=False, na=False).sum())
        if fee_count > 0:
            alerts.append("Fees and charges are present. Prevent avoidable late fees and service charges.")

        subs = int(df["description"].str.contains("subscription|netflix|spotify|google storage|autopay", case=False, na=False).sum())
        if subs > 0:
            alerts.append("Recurring subscriptions found. Audit monthly subscriptions for inactive services.")

        discretionary = (
            metrics.category_spending.get("Travel", 0.0)
            + metrics.category_spending.get("Shopping", 0.0)
            + metrics.category_spending.get("Food", 0.0)
        )
        if metrics.total_income > 0 and (discretionary / metrics.total_income) * 100 > 25:
            alerts.append("Discretionary spend is elevated. Consider a monthly cap for travel, shopping, and food.")

        return alerts[:5]

    def run(self, payload: dict) -> dict:
        transactions = payload.get("transactions", [])
        if not transactions:
            return {
                "status": "error",
                "message": "No transactions available for analysis.",
                "metrics": {},
                "health_score": {"score": 0, "explanation": "No data."},
                "risk_alerts": ["No transactions found for analysis."],
            }

        df = pd.DataFrame(transactions)
        metrics = compute_financial_metrics(df)
        health = self._money_health_score(metrics)
        risks = self._risk_alerts(df, metrics)
        annual_savings_projection = round(max(metrics.total_income - metrics.total_expenses, 0.0) * 12, 2)

        return {
            "status": "ok",
            "message": "Financial analysis completed.",
            "metrics": metrics.__dict__,
            "health_score": health,
            "risk_alerts": risks,
            "annual_savings_projection": annual_savings_projection,
        }
