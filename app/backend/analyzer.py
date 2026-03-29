from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    total_income: float
    total_expenses: float
    savings_rate: float
    emi_to_income_ratio: float
    category_spending: dict[str, float]


def compute_financial_metrics(df: pd.DataFrame) -> AnalysisResult:
    if df.empty:
        return AnalysisResult(0.0, 0.0, 0.0, 0.0, {})

    working = df.copy()
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce").fillna(0.0)
    working["type"] = working["type"].astype(str).str.lower().str.strip()
    if "category" not in working.columns:
        working["category"] = "Other"
    else:
        working["category"] = working["category"].astype(str).fillna("Other")

    # Normalize legacy signed records so aggregation is always positive by semantic type.
    income_df = working[working["type"] == "income"].copy()
    expense_df = working[working["type"] == "expense"].copy()
    income_df["amount"] = income_df["amount"].abs()
    expense_df["amount"] = expense_df["amount"].abs()

    # Fallback split when type tags are missing or corrupted.
    if income_df.empty and expense_df.empty:
        income_df = working[working["amount"] > 0].copy()
        expense_df = working[working["amount"] < 0].copy()
        income_df["amount"] = income_df["amount"].abs()
        expense_df["amount"] = expense_df["amount"].abs()

    total_income = float(income_df["amount"].sum())
    total_expenses = float(expense_df["amount"].sum())

    savings_rate = 0.0
    if total_income > 0:
        savings_rate = ((total_income - total_expenses) / total_income) * 100

    emi_spend = float(expense_df[expense_df["category"] == "EMI"]["amount"].sum())
    emi_to_income_ratio = (emi_spend / total_income) * 100 if total_income > 0 else 0.0

    category_spending = expense_df.groupby("category")["amount"].sum().abs().round(2).sort_values(ascending=False).to_dict()

    return AnalysisResult(
        total_income=round(total_income, 2),
        total_expenses=round(total_expenses, 2),
        savings_rate=round(savings_rate, 2),
        emi_to_income_ratio=round(emi_to_income_ratio, 2),
        category_spending={str(k): float(v) for k, v in category_spending.items()},
    )


def compute_money_health_score(metrics: AnalysisResult) -> dict[str, Any]:
    savings_component = min(max(metrics.savings_rate, 0), 40)

    if metrics.emi_to_income_ratio <= 15:
        emi_component = 30
    elif metrics.emi_to_income_ratio <= 30:
        emi_component = 20
    elif metrics.emi_to_income_ratio <= 45:
        emi_component = 10
    else:
        emi_component = 0

    essentials = metrics.category_spending.get("Rent", 0.0) + metrics.category_spending.get("Utilities", 0.0)
    wants = (
        metrics.category_spending.get("Shopping", 0.0)
        + metrics.category_spending.get("Travel", 0.0)
        + metrics.category_spending.get("Food", 0.0)
    )
    total_expenses = metrics.total_expenses if metrics.total_expenses > 0 else 1
    essentials_ratio = (essentials / total_expenses) * 100
    wants_ratio = (wants / total_expenses) * 100

    distribution_component = 30
    if wants_ratio > 45:
        distribution_component -= 10
    if essentials_ratio > 70:
        distribution_component -= 5
    distribution_component = max(0, distribution_component)

    score = int(round(savings_component + emi_component + distribution_component))
    score = min(max(score, 0), 100)

    explanation = (
        f"Savings ratio contributes {round(savings_component, 1)} points, "
        f"EMI burden contributes {emi_component} points, "
        f"and expense balance contributes {distribution_component} points."
    )
    return {"score": score, "explanation": explanation}


def generate_basic_planner(metrics: AnalysisResult) -> dict[str, float]:
    monthly_income = metrics.total_income
    monthly_expenses = metrics.total_expenses

    base_savings_target = monthly_income * 0.2
    if metrics.savings_rate < 15:
        base_savings_target = monthly_income * 0.25

    emergency_fund_goal = monthly_expenses * 6

    sip_budget = max(monthly_income - monthly_expenses - (monthly_income * 0.05), 0)
    sip_recommendation = min(sip_budget, monthly_income * 0.15)

    return {
        "monthly_savings_target": round(base_savings_target, 2),
        "emergency_fund_goal": round(emergency_fund_goal, 2),
        "suggested_monthly_sip_like_amount": round(sip_recommendation, 2),
    }
