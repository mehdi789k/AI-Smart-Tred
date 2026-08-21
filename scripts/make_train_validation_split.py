"""Create a time-based train/validation split from a labeled feature Parquet file.

Example:
    python scripts/make_train_validation_split.py \
        --input data/features/sample_labeled.parquet \
        --train-output data/modeling/train.parquet \
        --val-output data/modeling/validation.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smarttred.features.data_split import save_split_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Split labeled market data by timestamp into train/validation sets.")
    parser.add_argument("--input", required=True, help="Labeled feature Parquet file to split")
    parser.add_argument("--train-output", required=True, help="Path for the training parquet file")
    parser.add_argument("--val-output", required=True, help="Path for the validation parquet file")
    parser.add_argument("--time-col", default="timestamp", help="Timestamp column to sort by")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of rows for the training set")
    parser.add_argument("--min-train-rows", type=int, default=10, help="Minimum training rows required")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 2

    df = pd.read_parquet(input_path)
    split = save_split_artifacts(
        df,
        train_path=args.train_output,
        validation_path=args.val_output,
        time_col=args.time_col,
        train_ratio=args.train_ratio,
        min_train_rows=args.min_train_rows,
    )

    print(f"Train rows: {len(split.train)}")
    print(f"Validation rows: {len(split.validation)}")
    print(f"Train start: {split.train[args.time_col].min()}")
    print(f"Train end: {split.train[args.time_col].max()}")
    print(f"Validation start: {split.validation[args.time_col].min()}")
    print(f"Validation end: {split.validation[args.time_col].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
