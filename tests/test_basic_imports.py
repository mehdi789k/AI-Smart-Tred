from smarttred.data_pipeline import MT5DataExtractor
from smarttred.features import FeatureEngine


def test_imports_load() -> None:
    assert MT5DataExtractor is not None
    assert FeatureEngine is not None


def test_feature_engine_on_synthetic_data() -> None:
    import pandas as pd

    times = pd.date_range("2024-01-01", periods=80, freq="min")
    data = pd.DataFrame(
        {
            "timestamp": times,
            "open": 100 + pd.Series(range(80), dtype=float) / 10,
            "high": 101 + pd.Series(range(80), dtype=float) / 10,
            "low": 99 + pd.Series(range(80), dtype=float) / 10,
            "close": 100.5 + pd.Series(range(80), dtype=float) / 10,
            "volume": 1000 + pd.Series(range(80), dtype=float),
        }
    )

    engine = FeatureEngine(frac_diff_order=0.5)
    result = engine.transform(data)

    assert not result.empty
    assert "close_fd_0_5" in result.columns
    assert "returns_fd_0_5" in result.columns
    assert "rsi_14" in result.columns
