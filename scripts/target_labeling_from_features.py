"""Phase 4, Step 1: add triple-barrier labels to engineered feature data.

This script expects a Parquet file with OHLCV or feature-engineered data that
contains at least a timestamp and close column. It computes a target label using
an approximation of the Triple-Barrier Method and writes the labeled dataset to a
new Parquet file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smarttred.features.target_generator import triple_barrier_labels


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a parquet dataset and normalize the timestamp column when needed."""
    df = pd.read_parquet(path)
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "time"})
    
    # Ensure we have 'time' column (our standard)
    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})

    if "time" in df.columns:
        # Convert to datetime if it's numeric (unix timestamp)
        if pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit='s', errors="coerce")
        else:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Input dataset must contain a 'close' column for target generation.")
    
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("Input dataset must contain 'high' and 'low' columns for triple-barrier labeling.")

    return df.sort_values("time").reset_index(drop=True)


def generate_targets(
    input_path: str | Path,
    output_path: str | Path,
    *,
    profit_taker: float = 0.01,
    stop_loss: float = 0.005,
    time_limit: int = 30,
    use_atr: bool = False,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Run the triple-barrier labeling procedure on a feature dataset."""
    features = load_dataset(input_path)
    labels = triple_barrier_labels(
        features,
        close_col="close",
        profit_taker=profit_taker,
        stop_loss=stop_loss,
        time_limit=time_limit,
        use_atr=use_atr,
        atr_period=atr_period,
        label_col="target",
        barrier_col="barrier_type",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output, index=False)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Add Triple-Barrier labels to a Parquet dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet dataset with timestamp and close columns.")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path for labeled data.")
    parser.add_argument("--profit-taker", type=float, default=0.01, help="Take-profit threshold as fraction of price change.")
    parser.add_argument("--stop-loss", type=float, default=0.005, help="Stop-loss threshold as fraction of price change.")
    parser.add_argument("--time-limit", type=int, default=30, help="Maximum look-ahead horizon in bars.")
    parser.add_argument("--use-atr", action="store_true", help="When set, compute barriers as multiples of ATR instead of fractional returns.")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period to use when --use-atr is set.")
    args = parser.parse_args()

    df = generate_targets(
        args.input,
        args.output,
        profit_taker=args.profit_taker,
        stop_loss=args.stop_loss,
        time_limit=args.time_limit,
        use_atr=args.use_atr,
        atr_period=args.atr_period,
    )
    counts = df["target"].value_counts().to_dict()
    total = len(df)
    print(f"[INFO] Saved labeled dataset to {args.output} with {total} rows.")
    print("[INFO] Target distribution summary:")
    for label, cnt in sorted(counts.items(), reverse=True):
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  label={label}: {cnt} rows ({pct:.2f}%)")
    # also show zeros if some labels are missing
    for label in (1, -1, 0):
        if label not in counts:
            print(f"  label={label}: 0 rows (0.00%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
