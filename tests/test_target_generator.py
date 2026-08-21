import pandas as pd

from smarttred.features import triple_barrier_labels


def test_triple_barrier_labels_profit() -> None:
    price = [100.0, 101.0, 102.0, 103.0, 104.0, 103.5, 103.0]
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(price), freq="min").astype('int64') // 10**9,
        "close": price,
        "high": [p * 1.002 for p in price],  # Add high column
        "low": [p * 0.998 for p in price],   # Add low column
    })

    result = triple_barrier_labels(df, profit_taker=0.01, stop_loss=0.005, time_limit=3)
    assert result["target"].iloc[0] == 1


def test_triple_barrier_labels_loss() -> None:
    price = [100.0, 99.0, 98.0, 97.0, 96.5, 96.0, 95.0]
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(price), freq="min").astype('int64') // 10**9,
        "close": price,
        "high": [p * 1.002 for p in price],  # Add high column
        "low": [p * 0.998 for p in price],   # Add low column
    })

    result = triple_barrier_labels(df, profit_taker=0.01, stop_loss=0.005, time_limit=3)
    assert result["target"].iloc[0] == -1
