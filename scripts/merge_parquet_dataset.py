"""Merge a Parquet dataset directory into one Parquet file.

This is useful after append_to_parquet writes one fragment per batch into a
folder like <symbol>_<timeframe>_dataset/.

Example:
    python scripts/merge_parquet_dataset.py \
        --dataset-dir data/raw/forex/EURUSD_M1_dataset \
        --output data/raw/forex/EURUSD_M1_merged.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def merge_dataset_directory(dataset_dir: str | Path, output_path: str | Path, sort_by: str = "timestamp") -> pd.DataFrame:
    """Read a directory of Parquet part files and merge them into one DataFrame.

    The function uses pyarrow.dataset to read the dataset without loading the
    entire dataset into Python memory at once. It then writes a single compact
    Parquet file to `output_path`.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    dataset = ds.dataset(str(dataset_path), format="parquet")
    frame = dataset.to_table().to_pandas()

    if frame.empty:
        raise ValueError(f"No data found in dataset directory: {dataset_path}")

    if sort_by in frame.columns:
        frame = frame.sort_values(sort_by).reset_index(drop=True)

    if sort_by in frame.columns:
        frame = frame.drop_duplicates(subset=[sort_by]).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a Parquet fragment dataset into a single file.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing parquet fragments")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--sort-by", default="timestamp", help="Column used to sort and deduplicate")
    args = parser.parse_args()

    merged = merge_dataset_directory(args.dataset_dir, args.output, sort_by=args.sort_by)
    print(f"Merged {len(merged)} rows from {args.dataset_dir} into {args.output}")
    print(f"Columns: {list(merged.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
