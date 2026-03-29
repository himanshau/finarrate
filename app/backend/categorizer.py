from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = ["Food", "Rent", "EMI", "Shopping", "Travel", "Utilities", "Other"]


KEYWORD_RULES = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "food", "grocery", "dine"],
    "Rent": ["rent", "landlord", "apartment", "lease"],
    "EMI": ["emi", "loan", "credit card bill", "installment"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "store", "mall"],
    "Travel": ["uber", "ola", "flight", "train", "bus", "hotel", "travel"],
    "Utilities": [
        "electricity",
        "water",
        "internet",
        "wifi",
        "gas",
        "mobile",
        "recharge",
        "fastag",
        "petrol",
        "fuel",
        "diesel",
        "subscription",
        "netflix",
        "spotify",
    ],
}


@dataclass
class HybridCategorizer:
    ml_model: Optional[Pipeline] = None

    def train_optional_ml(self, labeled_df: pd.DataFrame) -> None:
        if labeled_df.empty or "description" not in labeled_df or "category" not in labeled_df:
            return
        filtered = labeled_df.dropna(subset=["description", "category"])
        filtered = filtered[filtered["category"].isin(CATEGORIES)]
        if len(filtered) < 20:
            return
        self.ml_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        self.ml_model.fit(filtered["description"].astype(str), filtered["category"])

    def categorize_description(self, description: str) -> str:
        text = str(description).lower().strip()
        for category, keywords in KEYWORD_RULES.items():
            if any(keyword in text for keyword in keywords):
                return category
        if self.ml_model is not None:
            return str(self.ml_model.predict([description])[0])
        return "Other"

    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        result = df.copy()
        result["category"] = result["description"].apply(self.categorize_description)
        return result


def build_bootstrap_training_data() -> pd.DataFrame:
    rows = [
        ("Swiggy order", "Food"),
        ("Zomato dinner", "Food"),
        ("Grocery store", "Food"),
        ("Monthly house rent", "Rent"),
        ("Apartment lease payment", "Rent"),
        ("Home loan EMI", "EMI"),
        ("Car loan installment", "EMI"),
        ("Credit card bill", "EMI"),
        ("Amazon purchase", "Shopping"),
        ("Flipkart order", "Shopping"),
        ("Myntra shopping", "Shopping"),
        ("Uber ride", "Travel"),
        ("Ola cab", "Travel"),
        ("Flight booking", "Travel"),
        ("Electricity bill", "Utilities"),
        ("Water bill", "Utilities"),
        ("Internet recharge", "Utilities"),
        ("Gas connection payment", "Utilities"),
        ("Cafe lunch", "Food"),
        ("Restaurant dinner", "Food"),
        ("Train ticket", "Travel"),
        ("Bus ticket", "Travel"),
        ("Mall shopping", "Shopping"),
        ("Wifi bill", "Utilities"),
    ]
    return pd.DataFrame(rows, columns=["description", "category"])
