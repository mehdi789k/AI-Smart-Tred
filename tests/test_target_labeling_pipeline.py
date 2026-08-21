from __future__ import annotations

import pandas as pd

from scripts.target_labeling_from_features import generate_targets


def test_generate_targets_from_feature_dataset(tmp_path) -> None:
    input_path = tmp_path / "eurusd_features.parquet"
    output_path = tmp_path / "labeled" / "eurusd_features_labeled.parquet"

    rows = []
    base = 1.10
    for i in range(120):
        price = base + i * 0.0002
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                "open": price,
                "high": price + 0.0005,
                "low": price - 0.0005,
                "close": price + 0.0001,
                "volume": 1000 + i,
                "returns": 0.0,
                "rsi_14": 50.0,
                "sma_10": price,
            }
        )

    pd.DataFrame(rows).to_parquet(input_path, index=False)
    labeled = generate_targets(input_path, output_path, profit_taker=0.01, stop_loss=0.005, time_limit=10)

    assert not labeled.empty
    assert {"timestamp", "close", "target", "barrier_type"}.issubset(labeled.columns)
    assert output_path.exists()
    assert set(labeled["target"].unique()).issubset({-1, 0, 1})
