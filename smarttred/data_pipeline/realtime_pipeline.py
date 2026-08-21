from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class AsyncMT5StreamPipeline:
    """Async pipeline that ingests MT5 tick data and persists it to Parquet + SQLite.

    This component is intentionally small and testable. It isolates the data
    acquisition loop from the storage layer so it can be used in a production
    trading system without coupling the live MT5 call site to the file/database
    persistence logic.
    """

    def __init__(
        self,
        parquet_dir: str = "data/raw/live",
        sqlite_db: str = "data/trades.db",
        table_prefix: str = "ticks",
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.sqlite_db = Path(sqlite_db)
        self.table_prefix = table_prefix

    def normalize_ticks(self, raw_ticks: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert raw MT5 tick objects into a consistent DataFrame schema."""
        if not raw_ticks:
            return pd.DataFrame(
                columns=["timestamp", "bid", "ask", "last", "volume"]
            )

        df = pd.DataFrame(raw_ticks)

        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")

        if "bid" in df.columns:
            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        if "ask" in df.columns:
            df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        if "last" in df.columns:
            df["last"] = pd.to_numeric(df["last"], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        columns = ["timestamp", "bid", "ask", "last", "volume"]
        existing = [col for col in columns if col in df.columns]
        df = df[existing].dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _setup_sqlite(self, table_name: str) -> sqlite3.Connection:
        """Create the SQLite database and table if they do not yet exist."""
        self.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_db)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT,
                bid REAL,
                ask REAL,
                last REAL,
                volume REAL
            )
            """
        )
        return conn

    def persist_batch(self, df: pd.DataFrame, symbol: str) -> dict[str, str]:
        """Persist a tick batch to Parquet and SQLite."""
        if df.empty:
            return {"symbol": symbol, "parquet": "", "sqlite_table": ""}

        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self.parquet_dir / f"{symbol.lower()}_{self.table_prefix}.parquet"

        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        else:
            combined = df.copy()

        combined.to_parquet(parquet_path, index=False)

        table_name = f"{symbol.lower()}_{self.table_prefix}"
        with self._setup_sqlite(table_name) as conn:
            combined.to_sql(table_name, conn, if_exists="append", index=False)

        return {
            "symbol": symbol,
            "parquet": str(parquet_path),
            "sqlite_table": table_name,
        }

    async def consume(
        self,
        fetcher: Callable[[str, int, int], list[dict[str, Any]]],
        symbol: str,
        timeframe: int,
        batch_size: int = 500,
        poll_interval: float = 1.0,
        iterations: int = 1,
    ) -> dict[str, str]:
        """Pull a tick batch, normalize it, and send it to the configured storage backends.

        Parameters
        ----------
        fetcher:
            Callable that returns raw MT5 tick data for a given symbol/timeframe/count.
        symbol:
            Symbol name such as EURUSD.
        timeframe:
            MT5 timeframe constant, e.g. mt5.TIMEFRAME_M1.
        batch_size:
            Number of ticks to request per polling loop.
        poll_interval:
            Waiting time in seconds between polling cycles.
        iterations:
            Number of cycles to run. Set to 1 for a single batch.
        """
        final_result: dict[str, str] = {"symbol": symbol, "parquet": "", "sqlite_table": ""}

        for _ in range(iterations):
            raw_ticks = await asyncio.to_thread(fetcher, symbol, timeframe, batch_size)
            normalized = self.normalize_ticks(raw_ticks)
            if not normalized.empty:
                final_result = self.persist_batch(normalized, symbol)
            if iterations > 1:
                await asyncio.sleep(poll_interval)

        return final_result
