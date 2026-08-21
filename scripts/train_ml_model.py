"""CLI script برای آموزش مدل ML (Phase 5)

نحوهٔ اجرا (فارسی):
    python scripts\train_ml_model.py --input data/labeled/labeled_data.parquet

این اسکریپت:
- دادهٔ برچسب‌خوردهٔ مرحلهٔ 4 را بارگذاری می‌کند (پارکت)
- MLTrainer را مقداردهی و داده را آماده می‌کند
- مدل را آموزش و ارزیابی می‌کند (Purged K-Fold with Embargo)
- مدل نهایی را در data/model_releases/model_v1.pkl ذخیره می‌کند
- لیست feature_names را در data/model_releases/model_v1_features.json ذخیره می‌کند

توجهات:
- اگر داده‌ها ستون 't1' و 'timestamp' را نداشته باشند، اسکریپت خطا می‌دهد.
"""
from __future__ import annotations

import argparse
import os
import json
import joblib
from typing import List

import pandas as pd

from smarttred.models.trainer import MLTrainer


def main(input_path: str) -> None:
    # بارگذاری داده
    print(f"Loading labeled data from: {input_path}")
    df = pd.read_parquet(input_path)

    # مقداردهی trainer
    trainer = MLTrainer(n_splits=5, embargo_pct=0.05)

    # آماده‌سازی داده
    print("Preparing data (separating features and target)...")
    X, y, feature_names = trainer.prepare_data(df, target_col="target")

    # آموزش و ارزیابی
    print("Training and evaluating model with Purged K-Fold + Embargo...")
    model = trainer.train_and_evaluate(X, y, feature_names)

    # ذخیره مدل و نام فیچرها
    out_dir = os.path.join(os.getcwd(), "data", "model_releases")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model_v1.pkl")
    features_path = os.path.join(out_dir, "model_v1_features.json")

    print(f"Saving trained model to: {model_path}")
    joblib.dump(model, model_path)

    print(f"Saving feature names to: {features_path}")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print("Done. Model and feature names saved.")
    print("برای استفاده در زمان اجرا (inference)، فایل مدل و لیست فیچرها را در مسیر data/model_releases پیدا کنید.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model (LightGBM) with Purged K-Fold + Embargo")
    parser.add_argument("--input", required=True, help="Path to labeled parquet file from Phase 4")
    args = parser.parse_args()
    main(args.input)
