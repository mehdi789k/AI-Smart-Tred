from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest_plugins = ("pytest_asyncio",)

from smarttred.data_pipeline.realtime_pipeline import AsyncMT5StreamPipeline


@pytest.mark.asyncio
async def test_async_pipeline_persists_ticks_to_parquet_and_sqlite(tmp_path) -> None:
    def fake_fetcher(symbol: str, timeframe: int, count: int):
        return [
            {"time": 1704067200, "bid": 1.1000, "ask": 1.1010, "last": 1.1005, "volume": 100},
            {"time": 1704067260, "bid": 1.1100, "ask": 1.1110, "last": 1.1105, "volume": 200},
        ]

    pipeline = AsyncMT5StreamPipeline(
        parquet_dir=str(tmp_path / "parquet"),
        sqlite_db=str(tmp_path / "market.db"),
    )

    result = await pipeline.consume(
        fetcher=fake_fetcher,
        symbol="EURUSD",
        timeframe=1,
        batch_size=10,
        poll_interval=0,
        iterations=1,
    )

    assert result["symbol"] == "EURUSD"
    parquet_path = tmp_path / "parquet" / "eurusd_ticks.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 2
    assert {"timestamp", "bid", "ask", "last", "volume"}.issubset(df.columns)

    with sqlite3.connect(tmp_path / "market.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM eurusd_ticks").fetchone()[0]
    assert count == 2
