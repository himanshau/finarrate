from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlanningAgent:
    """Builds practical monthly plans using deterministic portfolio-aware rules."""

    def run(self, payload: dict) -> dict:
        metrics = payload.get("metrics", {})
        category_spending = metrics.get("category_spending", {})

        income = float(metrics.get("total_income", 0.0) or 0.0)
        expenses = float(metrics.get("total_expenses", 0.0) or 0.0)
        savings_rate = float(metrics.get("savings_rate", 0.0) or 0.0)
        emi_ratio = float(metrics.get("emi_to_income_ratio", 0.0) or 0.0)

        current_monthly_savings = max(income - expenses, 0.0)
        if current_monthly_savings > 0:
            monthly_savings_target = round(min(current_monthly_savings * 1.12, income * 0.2), 2)
        else:
            monthly_savings_target = round(income * 0.05, 2)
        emergency_fund_goal = round(expenses * 6, 2)

        free_cash = max(current_monthly_savings, 0.0)
        if savings_rate < 15:
            sip_pct_of_savings = 0.25
            sip_profile = "Conservative"
        elif savings_rate < 25 or emi_ratio > 30:
            sip_pct_of_savings = 0.4
            sip_profile = "Balanced"
        elif savings_rate >= 30 and emi_ratio <= 20:
            sip_pct_of_savings = 0.6
            sip_profile = "Growth-oriented"
        else:
            sip_pct_of_savings = 0.5
            sip_profile = "Balanced"

        sip_amount = round(min(max(current_monthly_savings * sip_pct_of_savings, 0.0), free_cash), 2)

        travel = float(category_spending.get("Travel", 0.0) or 0.0)
        shopping = float(category_spending.get("Shopping", 0.0) or 0.0)
        food = float(category_spending.get("Food", 0.0) or 0.0)
        discretionary = travel + shopping + food

        suggestions = [
            "You may consider automating transfers on salary day to improve consistency.",
            "Use a disciplined saving approach before increasing discretionary spend.",
        ]
        if discretionary > 0 and income > 0 and (discretionary / income) * 100 > 25:
            suggestions.append("Set monthly caps for travel, shopping, and food to release extra savings.")

        return {
            "status": "ok",
            "message": "Planning completed.",
            "planner": {
                "monthly_savings_target": monthly_savings_target,
                "emergency_fund_goal": emergency_fund_goal,
                "emergency_fund_explanation": "This equals approximately 6 months of your current expenses.",
                "suggested_monthly_sip_like_amount": sip_amount,
                "suitable_sip_style_for_current_portfolio": sip_profile,
                "sip_guidance": (
                    f"Based on your current cash flow, you may consider allocating about {sip_amount:.2f} per month "
                    "toward long-term diversified investments."
                ),
                "suggestions": suggestions,
            },
        }
