import pandas as pd

from smarttred.models.training import prepare_training_data, train_xgboost_model


def test_prepare_training_data() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="min"),
            "close": [100 + i * 0.2 for i in range(40)],
            "open": [100 + i * 0.2 for i in range(40)],
            "high": [101 + i * 0.2 for i in range(40)],
            "low": [99 + i * 0.2 for i in range(40)],
            "volume": [1000 + i for i in range(40)],
        }
    )

    prepared = prepare_training_data(df, profit_taker=0.01, stop_loss=0.005, time_limit=3)
    assert "target" in prepared.columns
    assert not prepared.empty


def test_train_xgboost_model(tmp_path) -> None:
    price = [100 + i * 0.15 for i in range(80)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="min"),
            "close": price,
            "open": price,
            "high": [p + 1.2 for p in price],
            "low": [p - 1.2 for p in price],
            "volume": [1000 + i for i in range(80)],
        }
    )

    model, metrics = train_xgboost_model(
        df,
        output_path=tmp_path / "model.joblib",
        profit_taker=0.02,
        stop_loss=0.015,
        time_limit=10,
    )

    assert model is not None
    assert metrics["accuracy"] >= 0.0
    assert (tmp_path / "model.joblib").exists()
