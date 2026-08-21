"""Time-based train/validation splitting for labeled market data.

This module is useful after target labeling has been computed for a feature
Parquet file: it splits the rows by timestamp rather than random assignment to
avoid leakage across time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame


def time_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_ratio: float = 0.8,
    min_train_rows: int = 10,
) -> SplitResult:
    """Split a DataFrame into train/validation slices by timestamp.

    Parameters
    ----------
    df: DataFrame sorted by time. If not sorted, it will be sorted internally.
    train_ratio: fraction of rows to allocate to the training set (0 < train_ratio < 1)
    min_train_rows: minimum number of rows required in training to ensure not-too-small splits
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame")

    ordered = df.copy().sort_values(time_col).reset_index(drop=True)
    train_end = max(min_train_rows, int(len(ordered) * train_ratio))
    train_end = min(train_end, len(ordered) - 1)

    train = ordered.iloc[:train_end].copy().reset_index(drop=True)
    validation = ordered.iloc[train_end:].copy().reset_index(drop=True)

    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Output split produced an empty train or validation set.")

    return SplitResult(train=train, validation=validation)


def save_split_artifacts(
    df: pd.DataFrame,
    train_path: str | Path,
    validation_path: str | Path,
    time_col: str = "timestamp",
    train_ratio: float = 0.8,
    min_train_rows: int = 10,
) -> SplitResult:
    """Create train/validation splits and save them as Parquet files."""
    split = time_split(df, time_col=time_col, train_ratio=train_ratio, min_train_rows=min_train_rows)

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    split.train.to_parquet(train_path, index=False)
    split.validation.to_parquet(validation_path, index=False)
    return split
