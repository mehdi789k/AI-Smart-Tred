from __future__ import annotations

import pandas as pd

from scripts.train_ml_model import load_labeled_dataset


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

    loaded = load_labeled_dataset(input_path)
    assert not loaded.empty
    assert {"target", "close", "feature_a", "feature_b"}.issubset(loaded.columns)
