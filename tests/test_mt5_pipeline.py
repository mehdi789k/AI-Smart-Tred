from __future__ import annotations

from datetime import datetime

import pandas as pd

from smarttred.data_pipeline.mt5_extractor import MT5DataExtractor


class FakeRate:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeMT5:
    COPY_TICKS_ALL = "all"

    @staticmethod
    def initialize(**kwargs):
        return True

    @staticmethod
    def last_error():
        return "ok"

    @staticmethod
    def shutdown():
        return True

    @staticmethod
    def symbols_get():
        return []

    @staticmethod
    def copy_rates_range(symbol, timeframe, start, end):
        return [
            FakeRate(
                time=1704067200,
                open=1.0,
                high=1.1,
                low=0.9,
                close=1.05,
                tick_volume=123,
                spread=2,
                real_volume=456,
            ),
            FakeRate(
                time=1704067260,
                open=1.05,
                high=1.2,
                low=0.95,
                close=1.1,
                tick_volume=130,
                spread=2,
                real_volume=500,
            ),
        ]

    @staticmethod
    def copy_ticks_from(symbol, start, count, mode):
        return [
            {"time": 1704067200, "bid": 1.0, "ask": 1.01, "last": 1.0, "volume": 100},
            {"time": 1704067260, "bid": 1.01, "ask": 1.02, "last": 1.01, "volume": 200},
        ]


def test_download_timeframes(monkeypatch, tmp_path) -> None:
    import smarttred.data_pipeline.mt5_extractor as extractor_module

    monkeypatch.setattr(extractor_module, "mt5", FakeMT5)

    extractor = MT5DataExtractor(1, "pass", "server", "C:/MetaTrader")
    assert extractor.connect() is True

    files = extractor.download_timeframes(
        symbol="EURUSD",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
        timeframes=["M1", "M5"],
        output_dir=str(tmp_path),
    )

    assert set(files) == {"M1", "M5"}
    for path in files.values():
        assert path.endswith(".parquet")


def test_aggregated_calendar_timeframes() -> None:
    import smarttred.data_pipeline.mt5_extractor as extractor_module

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2024-01-01 00:00:00",
                "2024-01-02 00:00:00",
                "2024-03-31 00:00:00",
            ]),
            "open": [1.0, 1.5, 2.0],
            "high": [1.2, 1.7, 2.1],
            "low": [0.9, 1.3, 1.8],
            "close": [1.1, 1.6, 2.0],
            "tick_volume": [10, 20, 30],
            "real_volume": [100, 200, 300],
        }
    )

    yearly = extractor_module._aggregate_period(df, "Y1")
    quarterly = extractor_module._aggregate_period(df, "Q1")

    assert not yearly.empty
    assert not quarterly.empty
    assert {"timestamp", "open", "high", "low", "close"}.issubset(set(yearly.columns))
    assert {"timestamp", "open", "high", "low", "close"}.issubset(set(quarterly.columns))


def test_ticks_roundtrip(monkeypatch) -> None:
    import smarttred.data_pipeline.mt5_extractor as extractor_module

    monkeypatch.setattr(extractor_module, "mt5", FakeMT5)
    extractor = MT5DataExtractor(1, "pass", "server", "C:/MetaTrader")

    df = extractor.get_ticks("EURUSD")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"timestamp", "bid", "ask", "last", "volume"}.issubset(set(df.columns))
