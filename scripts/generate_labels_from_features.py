"""Script to create target labels from a feature Parquet file.

Usage:
    python scripts/generate_labels_from_features.py --input data/features/<file>.parquet --output data/features/<file>_labeled.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smarttred.features.target_labeling import generate_labels_from_features


def find_input_candidate() -> str | None:
    import glob
    candidates = glob.glob('data/features/**/*_features.parquet', recursive=True)
    if not candidates:
        candidates = glob.glob('data/features/**/*.parquet', recursive=True)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate labels from a feature Parquet file.")
    parser.add_argument("--input", required=False, help="Input feature Parquet file. If omitted the script will try to discover under data/features/")
    parser.add_argument("--output", required=False, help="Output Parquet path for labeled dataset (default: append _labeled.parquet)")
    parser.add_argument("--horizons", type=str, default="1,5,10,20", help="Comma-separated horizons (bars)")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Directional threshold for labels")
    parser.add_argument("--sl", type=float, default=0.01, help="Stop-loss as relative fraction")
    parser.add_argument("--tp", type=float, default=0.02, help="Take-profit as relative fraction")
    args = parser.parse_args()

    input_path = args.input or find_input_candidate()
    if input_path is None:
        print("No input feature Parquet found under data/features/. Provide --input to specify a file.")
        return 2

    input_path = Path(input_path)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_labeled.parquet")

    df = pd.read_parquet(input_path)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    labeled = generate_labels_from_features(
        df, horizons=horizons, threshold=args.threshold, sl=args.sl, tp=args.tp
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output_path, index=False)

    print(f"Saved labeled dataset to {output_path} with {len(labeled)} rows")
    # print a small summary of label distribution
    for h in horizons:
        col = f"label_{h}"
        if col in labeled.columns:
            print(h, labeled[col].value_counts(dropna=True).to_dict())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())