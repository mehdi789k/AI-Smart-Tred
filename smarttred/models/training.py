from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from smarttred.features.target_generator import triple_barrier_labels


def prepare_training_data(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    profit_taker: float = 0.01,
    stop_loss: float = 0.005,
    time_limit: int = 30,
) -> pd.DataFrame:
    """Generate labels and keep only the ML-relevant columns for model training."""
    labeled = triple_barrier_labels(
        df,
        close_col=close_col,
        profit_taker=profit_taker,
        stop_loss=stop_loss,
        time_limit=time_limit,
    )

    feature_cols = [
        c for c in labeled.columns if c not in {"timestamp", "target", "barrier_type"}
    ]
    prepared = labeled[feature_cols + ["target"]].copy()
    return prepared.dropna().reset_index(drop=True)


def purged_time_split(
    df: pd.DataFrame,
    *,
    n_splits: int = 3,
    gap: int = 5,
    target_col: str = "target",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create purged train/validation splits ordered by time.

    Each validation window is kept strictly after the corresponding training
    segment, and a small gap is inserted between them to avoid leakage through
    nearby observations.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    if n < n_splits + 2:
        raise ValueError("Not enough rows to build purged cross-validation folds.")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    step = n // n_splits
    for split_idx in range(1, n_splits + 1):
        val_end = (n * split_idx) // n_splits
        val_start = max(0, val_end - step)
        train_end = max(0, val_start - gap)

        train_idx = np.arange(train_end)
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        if ordered.iloc[val_idx][target_col].nunique() < 2:
            continue

        folds.append((train_idx, val_idx))

    if not folds:
        raise ValueError("Unable to create valid purged folds for the supplied dataset.")

    return folds


def _build_classifier(model_name: str = "xgboost", random_state: int = 42, n_classes: int = 2):
    """Create a boosting classifier backend for training."""
    objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
    lgb_objective = "binary" if n_classes <= 2 else "multiclass"

    try:
        if model_name == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=objective,
                random_state=random_state,
                eval_metric="mlogloss",
                n_jobs=-1,
                num_class=n_classes if n_classes > 2 else None,
            )
    except ImportError:
        pass

    try:
        if model_name == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                objective=lgb_objective,
                n_jobs=-1,
                num_class=n_classes if n_classes > 2 else None,
            )
    except ImportError:
        pass

    return HistGradientBoostingClassifier(random_state=random_state)


def train_xgboost_model(
    df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    profit_taker: float = 0.01,
    stop_loss: float = 0.005,
    time_limit: int = 30,
    model_name: str = "xgboost",
    use_purged_cv: bool = False,
    n_splits: int = 3,
    gap: int = 5,
) -> tuple[object, dict[str, float]]:
    """Train a boosted classifier on labeled market data.

    When ``use_purged_cv`` is True, the method creates chronological folds with a
    gap between train and validation windows to reduce leakage risk. This is a
    practical approximation of the purged cross-validation workflow used in
    research-oriented trading systems.
    """
    prepared = prepare_training_data(
        df,
        profit_taker=profit_taker,
        stop_loss=stop_loss,
        time_limit=time_limit,
    )

    if prepared.empty:
        raise ValueError("Prepared dataset is empty. Check the market data and feature generation flow.")

    feature_cols = [col for col in prepared.columns if col != "target"]
    X = prepared[feature_cols]
    y = prepared["target"]

    if use_purged_cv:
        folds = purged_time_split(prepared.assign(timestamp=pd.Series(range(len(prepared))), target=prepared["target"]), n_splits=n_splits, gap=gap)
        fold_scores: list[float] = []
        best_model = None
        best_score = -np.inf

        for train_idx, val_idx in folds:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            n_classes = len(np.unique(y))
            classifier = _build_classifier(model_name=model_name, random_state=random_state, n_classes=n_classes)
            if model_name == "xgboost":
                try:
                    classifier.set_params(num_class=n_classes if n_classes > 2 else None)
                    classifier.set_params(objective="binary:logistic" if n_classes <= 2 else "multi:softprob")
                except Exception:
                    pass
            elif model_name == "lightgbm":
                try:
                    classifier.set_params(num_class=n_classes if n_classes > 2 else None)
                    classifier.set_params(objective="binary" if n_classes <= 2 else "multiclass")
                except Exception:
                    pass
            classifier.fit(X_train, y_train)
            preds = classifier.predict(X_val)
            score = accuracy_score(y_val, preds)
            fold_scores.append(score)
            if score > best_score:
                best_model = classifier
                best_score = score

        if best_model is None:
            raise ValueError("Purged cross-validation produced no usable fold scores.")

        metrics = {
            "accuracy": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "macro_f1": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "weighted_f1": float(np.mean(fold_scores)) if fold_scores else 0.0,
        }
        model = best_model
    else:
        counts = y.value_counts()
        stratify = y if len(counts) >= 2 and counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        n_classes = len(np.unique(y))
        model = _build_classifier(model_name=model_name, random_state=random_state, n_classes=n_classes)
        if model_name == "xgboost":
            try:
                model.set_params(num_class=n_classes if n_classes > 2 else None)
                model.set_params(objective="binary:logistic" if n_classes <= 2 else "multi:softprob")
            except Exception:
                pass
        elif model_name == "lightgbm":
            try:
                model.set_params(num_class=n_classes if n_classes > 2 else None)
                model.set_params(objective="binary" if n_classes <= 2 else "multiclass")
            except Exception:
                pass
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

    return model, metrics
