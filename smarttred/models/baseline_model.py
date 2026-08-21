from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def prepare_model_matrix(df: pd.DataFrame, target_col: str = "label_5") -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target from a labeled feature DataFrame.

    It keeps only numeric feature columns and drops rows with missing target values.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    feature_cols = [
        col
        for col in df.columns
        if col not in {"timestamp", target_col} and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found for target '{target_col}'")

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    X = X.replace([float("inf"), -float("inf")], pd.NA)
    X = X.dropna(axis=1, how="all")
    X = X.dropna().reset_index(drop=True)
    y = y.loc[X.index].reset_index(drop=True)
    return X, y


class BaselineModelTrainer:
    """A simple, dependency-light baseline trainer for directional classification.

    Prefers LogisticRegression when the dataset is small and stable, and falls back to
    HistGradientBoostingClassifier for more complex patterns. This keeps the baseline
    robust even if XGBoost/LightGBM are not installed.
    """

    def __init__(self, model_name: str = "logistic") -> None:
        self.model_name = model_name

    def build_model(self) -> Any:
        if self.model_name == "logistic":
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        return HistGradientBoostingClassifier(random_state=42, max_depth=6)

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple[Any, dict[str, float], Any, Any, Any, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )

        model = self.build_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        }
        return model, metrics, X_train, X_test, y_train, y_test

    def save(self, model: Any, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
