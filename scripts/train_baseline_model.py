"""Train a baseline classification model on labeled market data.

Example:
    python scripts/train_baseline_model.py \
        --input data/features/sample_labeled.parquet \
        --target label_5 \
        --output data/models/baseline.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smarttred.models.baseline_model import BaselineModelTrainer, prepare_model_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline directional model on labeled feature data.")
    parser.add_argument("--input", required=True, help="Feature dataset parquet file with labels")
    parser.add_argument("--target", default="label_5", help="Target label column to train on")
    parser.add_argument("--output", required=True, help="Output path for model artifact")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--model", choices=["logistic", "histgbm"], default="logistic", help="Baseline model type")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 2

    df = pd.read_parquet(input_path)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found. Available columns: {list(df.columns[:10])}")
        return 3

    X, y = prepare_model_matrix(df, target_col=args.target)
    trainer = BaselineModelTrainer(model_name=args.model)
    model, metrics, _, _, _, _ = trainer.train(X, y, test_size=args.test_size)
    trainer.save(model, args.output)

    print(f"Model saved to {args.output}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
