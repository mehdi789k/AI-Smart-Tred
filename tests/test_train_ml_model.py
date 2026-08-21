from __future__ import annotations

import pandas as pd


def load_labeled_dataset(path: str) -> pd.DataFrame:
    """Load labeled dataset from parquet file."""
    return pd.read_parquet(path)


def test_load_labeled_dataset(tmp_path) -> None:
    input_path = tmp_path / "labeled.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="min"),
            "close": [1.0 + i * 0.0004 for i in range(40)],
            "target": [1, -1, 0] * 13 + [1],
            "feature_a": [i for i in range(40)],
            "feature_b": [i * 0.1 for i in range(40)],
        }
    )
    df.to_parquet(input_path, index=False)

    loaded = load_labeled_dataset(str(input_path))
    assert not loaded.empty
    assert {"target", "close", "feature_a", "feature_b"}.issubset(loaded.columns)
