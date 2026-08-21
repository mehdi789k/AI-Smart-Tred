from __future__ import annotations

import pandas as pd

from scripts.feature_engineering_from_parquet import build_feature_dataset


def test_build_feature_dataset_from_parquet(tmp_path) -> None:
    input_path = tmp_path / "eurusd_m15.parquet"
    output_path = tmp_path / "features" / "eurusd_m15_features.parquet"

    rows = []
    base = 1.1000
    for i in range(80):
        price = base + i * 0.0005
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                "open": price,
                "high": price + 0.0008,
                "low": price - 0.0008,
                "close": price + 0.0003,
                "volume": 1000 + i,
            }
        )

    pd.DataFrame(rows).to_parquet(input_path, index=False)
    df = build_feature_dataset(input_path, output_path)

    assert not df.empty
    assert {"timestamp", "close", "returns", "sma_10", "rsi_14", "close_fd_0_5"}.issubset(df.columns)
    assert output_path.exists()
