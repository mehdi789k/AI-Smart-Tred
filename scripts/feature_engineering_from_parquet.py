"""Phase 3, Step 1: build feature engineering dataset from MT5 Parquet OHLCV data.

This script reads a local Parquet file containing raw OHLCV bars, applies a
stationarity-focused feature engineering layer (log returns + rolling statistics +
EMA/SMA/RSI + fractional differencing), and writes a clean, ML-ready dataset to
another Parquet file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smarttred.features.feature_engine import FeatureEngine


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from Parquet and normalize its column names."""
    df = pd.read_parquet(path)

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input file is missing OHLCV columns: {missing}")

    return df.sort_values("timestamp").reset_index(drop=True)


def build_feature_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Read raw OHLCV data, engineer features, and save the result."""
    raw = load_ohlcv(input_path)
    engine = FeatureEngine()

    # Step 1: technical indicators (drops rows that cannot be computed due to lookbacks)
    df_ind = engine.add_technical_indicators(raw)

    # Step 2: rolling statistical features
    df_stats = engine.compute_statistical_features(df_ind, window=20)

    # Step 3: fractional differentiation on close
    fd = engine.fractional_differentiation(df_stats, column="close", d=0.4, window=100)
    # align the df_stats to fd's index (fd is shorter because of the lookback trimming)
    df_final = df_stats.loc[fd.index].copy()
    df_final["close_fd_0_4"] = fd.values

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output, index=False)
    return df_final


def find_input_candidate() -> str | None:
    import glob
    candidates = glob.glob('data/raw/**/*.parquet', recursive=True)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ML-ready feature dataset from MT5 OHLCV Parquet data.")
    parser.add_argument("--input", type=str, required=False, help="Input Parquet file with OHLCV data. If omitted the script will try to discover under data/raw/")
    parser.add_argument("--output", type=str, required=False, help="Output Parquet file for engineered features. Default: data/features/<base>_features.parquet", default=None)
    args = parser.parse_args()

    input_path = args.input or find_input_candidate()
    if input_path is None:
        print("No input Parquet found under data/raw/. Provide --input to specify a file.")
        return 2

    if args.output:
        output = args.output
    else:
        import os
        base = os.path.splitext(os.path.basename(input_path))[0]
        output = os.path.join('data', 'features', f"{base}_features.parquet")

    df = build_feature_dataset(input_path, output)
    print(f"[INFO] Feature dataset saved to {output} with {len(df)} rows.")
    print(f"[INFO] Columns: {list(df.columns[:10])} ... total={len(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
