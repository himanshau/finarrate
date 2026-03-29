from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from categorizer import CATEGORIES, KEYWORD_RULES, HybridCategorizer, build_bootstrap_training_data


@dataclass
class CategorizationAgent:
    """Categorizes transactions with rule-first, optional LLM fallback behavior."""

    llm_enabled: bool = True
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        self.engine = HybridCategorizer()
        self.engine.train_optional_ml(build_bootstrap_training_data())

    def _rule_predict(self, description: str) -> tuple[str, float]:
        text = str(description).lower().strip()

        # High-priority deterministic rules for recurring ambiguous merchant patterns.
        if any(token in text for token in ["fastag", "petrol", "fuel", "diesel", "shell"]):
            return "Utilities", 0.92
        if any(token in text for token in ["subscription", "netflix", "spotify", "google storage"]):
            return "Utilities", 0.9
        if any(token in text for token in ["refund", "reversal", "rev "]):
            return "Other", 0.85

        best_category = "Other"
        best_hits = 0
        for category, keywords in KEYWORD_RULES.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits > best_hits:
                best_hits = hits
                best_category = category

        if best_hits >= 2:
            return best_category, 0.95
        if best_hits == 1:
            return best_category, 0.82
        return "Other", 0.0

    def _ml_predict(self, description: str) -> tuple[str, float]:
        model = self.engine.ml_model
        if model is None:
            return "Other", 0.0
        if not hasattr(model.named_steps.get("clf"), "predict_proba"):
            pred = str(model.predict([description])[0])
            return pred, 0.6

        probs = model.predict_proba([description])[0]
        classes = list(model.named_steps["clf"].classes_)
        idx = int(probs.argmax())
        return str(classes[idx]), float(probs[idx])

    def _llm_predict(self, description: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify a bank transaction into one category only from: "
                    "Food, Rent, EMI, Shopping, Travel, Utilities, Other. "
                    "Return JSON only: {{\"category\": \"...\"}}.",
                ),
                ("human", "Description: {description}"),
            ]
        )
        try:
            model = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)
            chain = prompt | model
            response = chain.invoke({"description": description})
            text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        except Exception:
            return "Other"

        try:
            start = text.find("{")
            end = text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start >= 0 and end >= 0 else {}
            category = str(payload.get("category", "Other"))
            return category if category in CATEGORIES else "Other"
        except Exception:
            return "Other"

    def categorize(self, description: str) -> tuple[str, float, str]:
        rule_category, rule_conf = self._rule_predict(description)
        if rule_conf >= self.confidence_threshold:
            return rule_category, rule_conf, "rule"

        ml_category, ml_conf = self._ml_predict(description)
        chosen_category, chosen_conf, source = ml_category, ml_conf, "ml"

        if rule_conf > ml_conf:
            chosen_category, chosen_conf, source = rule_category, rule_conf, "rule"

        if self.llm_enabled and chosen_conf < self.confidence_threshold:
            llm_category = self._llm_predict(description)
            return llm_category, 0.75 if llm_category != "Other" else chosen_conf, "llm"

        return chosen_category, chosen_conf, source

    def run(self, payload: dict) -> dict:
        transactions = payload.get("transactions", [])
        if not transactions:
            return {
                "status": "error",
                "message": "No transactions available for categorization.",
                "transactions": [],
            }

        df = pd.DataFrame(transactions)
        out_rows: list[dict] = []
        for row in df.itertuples(index=False):
            category, confidence, source = self.categorize(str(row.description))
            parsed_amount = pd.to_numeric(row.amount, errors="coerce")
            amount_value = float(0.0 if pd.isna(parsed_amount) else parsed_amount)
            out_rows.append(
                {
                    "date": str(row.date),
                    "description": str(row.description),
                    "amount": amount_value,
                    "type": str(row.type),
                    "category": category,
                    "category_confidence": round(confidence, 3),
                    "category_source": source,
                }
            )

        return {
            "status": "ok",
            "message": "Categorization completed.",
            "transactions": out_rows,
            "confidence_threshold": self.confidence_threshold,
        }
