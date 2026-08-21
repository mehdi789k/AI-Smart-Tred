from __future__ import annotations

import asyncio
import time
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional runtime dependency
    mt5 = None


TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 86400,
    "W1": getattr(mt5, "TIMEFRAME_W1", 32769) if mt5 is not None else 32769,
    "MN1": getattr(mt5, "TIMEFRAME_MN1", 49153) if mt5 is not None else 49153,
    "Q1": None,
    "Y1": None,
}


def resolve_mt5_timeframe(timeframe_name: str) -> Any:
    """Resolve a timeframe token to a MetaTrader 5 constant, if supported."""
    if timeframe_name in {"Q1", "Y1"}:
        return None
    if timeframe_name not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe '{timeframe_name}'.")

    timeframe_value = TIMEFRAME_MAP[timeframe_name]
    if timeframe_value is None:
        return None
    return timeframe_value


def _aggregate_period(df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """Aggregate an OHLCV DataFrame to coarser calendar periods like weekly/monthly/quarterly/yearly."""
    if df.empty or "timestamp" not in df.columns:
        return df

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    working = working.sort_values("timestamp").reset_index(drop=True)

    group_key = {
        "W1": working["timestamp"].dt.to_period("W-MON").astype(str),
        "MN1": working["timestamp"].dt.to_period("M").astype(str),
        "Q1": working["timestamp"].dt.to_period("Q").astype(str),
        "Y1": working["timestamp"].dt.to_period("Y").astype(str),
    }[period_name]

    agg = working.groupby(group_key, as_index=False).agg(
        timestamp=("timestamp", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        real_volume=("real_volume", "sum"),
    )
    if "spread" in working.columns:
        agg["spread"] = working.groupby(group_key)["spread"].last().reset_index(drop=True)
    agg = agg.sort_values("timestamp").reset_index(drop=True)
    return agg


def _as_record_dict(record: Any, field_names: Optional[list[str]] = None) -> dict[str, Any]:
    """Normalize MT5 records from dict-like, tuple-like, or numpy structured objects."""
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "_asdict"):
        return dict(record._asdict())
    if hasattr(record, "__dict__") and record.__dict__:
        return dict(record.__dict__)
    if isinstance(record, np.ndarray):
        return {name: value for name, value in zip(record.dtype.names or [], record.tolist())}
    if isinstance(record, np.void):
        field_names = field_names or list(record.dtype.names or [])
        values = record.tolist()
        return {name: value for name, value in zip(field_names, values)}
    if isinstance(record, (list, tuple)):
        if field_names is None:
            return {str(index): value for index, value in enumerate(record)}
        return {name: value for name, value in zip(field_names, record)}
    return dict(vars(record))


class MT5DataExtractor:
    """Extract historical and real-time market data from MetaTrader 5.

    The class keeps a retry-based connection flow and stores output in Parquet
    files so that downstream feature-engineering and ML jobs can consume the
    data without a live MT5 dependency.
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str,
        timeout: int = 10000,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or logging.getLogger("smarttred.mt5")
        self._connected = False
        self._last_error: Optional[str] = None

    def connect(self) -> bool:
        """Connect to MetaTrader 5 using retry logic."""
        if mt5 is None:
            raise ImportError("MetaTrader5 is not installed on this machine.")

        for attempt in range(1, self.max_retries + 1):
            try:
                initialized = mt5.initialize(
                    path=self.path,
                    login=self.login,
                    password=self.password,
                    server=self.server,
                    timeout=self.timeout,
                )
                if not initialized:
                    self._last_error = str(mt5.last_error())
                    self.logger.warning(
                        "MT5 initialization failed on attempt %s: %s",
                        attempt,
                        self._last_error,
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)
                    continue
                self._connected = True
                self.logger.info("Connected to MetaTrader 5 successfully.")
                return True
            except Exception as exc:  # pragma: no cover - runtime only
                self._last_error = str(exc)
                self.logger.exception("MT5 connection error")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        return False

    def disconnect(self) -> None:
        """Shutdown the MT5 terminal handle."""
        if self._connected and mt5 is not None:
            mt5.shutdown()
            self._connected = False
            self.logger.info("MetaTrader 5 connection closed.")

    def ensure_connected(self) -> None:
        """Ensure MT5 is connected before performing data extraction."""
        if not self._connected:
            connected = self.connect()
            if not connected:
                raise ConnectionError(
                    f"Unable to connect to MetaTrader 5. Last error: {self._last_error}"
                )

    def get_symbols(self) -> list[str]:
        """Return available symbols from the connected MT5 terminal."""
        self.ensure_connected()
        symbols = mt5.symbols_get()
        return [symbol.name for symbol in symbols] if symbols else []

    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: Any,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Download OHLCV data between two timestamps."""
        self.ensure_connected()
        if mt5 is None:
            raise RuntimeError("MetaTrader5 is not installed.")

        rates = mt5.copy_rates_range(symbol, timeframe, int(start.timestamp()), int(end.timestamp()))
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        rate_dtype = getattr(rates, "dtype", None)
        field_names = list(getattr(rate_dtype, "names", []) or [])
        record_list = []
        for rate in rates:
            record_list.append(_as_record_dict(rate, field_names=field_names))

        df = pd.DataFrame(record_list)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(
            columns={
                "time": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "tick_volume",
                "spread": "spread",
                "real_volume": "real_volume",
            }
        )

        for column in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")

        df = df.sort_values("timestamp").reset_index(drop=True)
        if limit is not None:
            df = df.tail(limit).reset_index(drop=True)
        return df

    def get_ticks(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        count: int = 1000,
    ) -> pd.DataFrame:
        """Download tick data from the MT5 terminal."""
        self.ensure_connected()
        if mt5 is None:
            raise RuntimeError("MetaTrader5 is not installed.")

        if start is None:
            start = datetime.now() - timedelta(minutes=5)

        ticks = mt5.copy_ticks_from(symbol, int(start.timestamp()), count, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()

        tick_dtype = getattr(ticks, "dtype", None)
        field_names = list(getattr(tick_dtype, "names", []) or [])
        record_list = []
        for tick in ticks:
            record_list.append(_as_record_dict(tick, field_names=field_names))

        df = pd.DataFrame(record_list)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(
            columns={
                "time": "timestamp",
                "bid": "bid",
                "ask": "ask",
                "last": "last",
                "volume": "volume",
            }
        )

        for column in ["bid", "ask", "last", "volume"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce", downcast="float")
        return df.sort_values("timestamp").reset_index(drop=True)

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: str,
        symbol: str,
        timeframe: str,
    ) -> str:
        """Persist a DataFrame to a Parquet file."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{symbol}_{timeframe}.parquet")
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, file_path)
        self.logger.info("Saved Parquet file: %s", file_path)
        return file_path

    def append_to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: str,
        symbol: str,
        timeframe: str,
    ) -> str:
        """Append rows to an existing Parquet dataset while avoiding duplicates.

        WARNING: The legacy single-file approach (reading the existing Parquet into memory,
        concatenating, deduplicating and rewriting) can use a lot of RAM for large datasets.
        For production-scale appends prefer one of the following strategies:
        - Write Parquet "fragments" into a dataset directory using pyarrow.dataset (no full-file reads).
        - Use DuckDB to perform fast on-disk SQL-style upserts/merges without loading full tables into RAM.

        This method attempts a safe fragment-append into a dataset directory without loading the
        entire existing dataset into memory. Deduplication across fragments is not automatic —
        downstream consumers should either read and deduplicate on load or use DuckDB to merge fragments.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{symbol}_{timeframe}.parquet")
        dataset_dir = os.path.join(output_dir, f"{symbol}_{timeframe}_dataset")

        table = pa.Table.from_pandas(df, preserve_index=False)

        try:
            # If a dataset directory already exists, write a new fragment file into it.
            # This appends without reading existing files into RAM.
            if os.path.exists(dataset_dir):
                fragment_path = os.path.join(dataset_dir, f"part-{int(time.time() * 1000)}.parquet")
                os.makedirs(dataset_dir, exist_ok=True)
                pq.write_table(table, fragment_path)
                self.logger.info("Appended Parquet fragment: %s", fragment_path)
                # Note: no dedup performed across fragments here.
                return fragment_path

            # If no dataset dir exists but a legacy single-file exists, create a dataset dir and
            # write a first fragment, then keep the legacy single-file for backward compatibility.
            if os.path.exists(file_path):
                os.makedirs(dataset_dir, exist_ok=True)
                fragment_path = os.path.join(dataset_dir, f"part-{int(time.time() * 1000)}.parquet")
                pq.write_table(table, fragment_path)
                self.logger.info("Created dataset dir and wrote fragment: %s", fragment_path)

                # Try to perform a low-memory merge using DuckDB if available. This
                # avoids loading the entire existing file into Python memory.
                try:
                    from smarttred.data_pipeline.duckdb_merge import merge_parquet_with_duckdb

                    # merge new fragment into the legacy single-file output
                    merge_parquet_with_duckdb(file_path, df, file_path, key="timestamp")
                    self.logger.info("Merged fragment into legacy Parquet using DuckDB: %s", file_path)
                except Exception:
                    # Fallback: small-data in-memory concat/dedup
                    self.logger.debug("DuckDB merge not available or failed; falling back to in-memory concat.")
                    existing = pq.read_table(file_path).to_pandas()
                    combined = pd.concat([existing, df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                    self.save_to_parquet(combined, output_dir, symbol, timeframe)

                return fragment_path

            # No existing data at all: create a dataset dir and write the first fragment and legacy file
            os.makedirs(dataset_dir, exist_ok=True)
            fragment_path = os.path.join(dataset_dir, "part-0.parquet")
            pq.write_table(table, fragment_path)
            # Also write the legacy single-file for compatibility
            pq.write_table(table, file_path)
            self.logger.info("Wrote initial Parquet dataset and file: %s, %s", fragment_path, file_path)
            return file_path

        except Exception as exc:  # pragma: no cover - runtime only
            # If anything goes wrong with the fragment approach, fall back to the older
            # read/concat/dedup approach which works for small datasets.
            self.logger.exception("Error appending to parquet dataset, falling back to in-memory concat: %s", exc)
            if os.path.exists(file_path):
                existing = pq.read_table(file_path).to_pandas()
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df = combined

            self.save_to_parquet(df, output_dir, symbol, timeframe)
            return file_path

    def download_timeframes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframes: list[str] | tuple[str, ...] | None = None,
        output_dir: str = "data/raw/forex",
    ) -> dict[str, str]:
        """Download OHLCV history for multiple timeframes and save them to Parquet.

        MT5 supports direct constants for standard intraday and daily resolutions.
        Weekly/monthly data is also available natively when supported by the broker.
        Quarterly and yearly bars are derived from D1 data when the broker does not
        expose those native periods.
        """
        if timeframes is None:
            timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1", "Q1", "Y1"]

        saved_files: dict[str, str] = {}
        for timeframe_name in timeframes:
            if timeframe_name not in TIMEFRAME_MAP:
                raise ValueError(f"Unsupported timeframe '{timeframe_name}'.")

            if timeframe_name in {"Q1", "Y1"}:
                base_df = self.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe=TIMEFRAME_MAP["D1"],
                    start=start,
                    end=end,
                )
                if base_df.empty:
                    self.logger.warning("No daily data available to aggregate %s %s", symbol, timeframe_name)
                    continue
                df = _aggregate_period(base_df, timeframe_name)
            else:
                mt5_timeframe = resolve_mt5_timeframe(timeframe_name)
                if mt5_timeframe is None:
                    self.logger.warning("Native MT5 timeframe constant not available for %s; skipping.", timeframe_name)
                    continue
                df = self.get_historical_ohlcv(
                    symbol=symbol,
                    timeframe=mt5_timeframe,
                    start=start,
                    end=end,
                )

            if df.empty:
                self.logger.warning("No data returned for %s %s", symbol, timeframe_name)
                continue
            file_path = self.save_to_parquet(df, output_dir, symbol, timeframe_name)
            saved_files[timeframe_name] = file_path
        return saved_files

    async def stream_ticks_to_parquet(
        self,
        symbol: str,
        output_dir: str,
        duration_seconds: int = 60,
        callback: Optional[Callable[[pd.DataFrame], None]] = None,
    ) -> str:
        """Collect real-time tick data for a fixed duration and save it as Parquet."""
        self.ensure_connected()
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{symbol}_ticks.parquet")
        frames: list[pd.DataFrame] = []

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration_seconds)

        while datetime.now() < end_time:
            try:
                ticks = self.get_ticks(symbol=symbol, count=500)
                if not ticks.empty:
                    frames.append(ticks)
                    if callback is not None:
                        callback(ticks)
                await asyncio.sleep(0.5)
            except Exception as exc:  # pragma: no cover - runtime only
                self._last_error = str(exc)
                self.logger.exception("Streaming tick error for %s", symbol)
                await asyncio.sleep(1.0)
                self.ensure_connected()

        if frames:
            final_df = pd.concat(frames, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            pq.write_table(pa.Table.from_pandas(final_df, preserve_index=False), file_path)
        return file_path

    async def stream_ticks(
        self,
        symbol: str,
        duration_seconds: int = 60,
        callback: Optional[Callable[[pd.DataFrame], None]] = None,
    ) -> None:
        """Stream tick data in a loop for a fixed number of seconds."""
        self.ensure_connected()
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration_seconds)

        while datetime.now() < end_time:
            try:
                ticks = self.get_ticks(symbol=symbol, count=500)
                if not ticks.empty and callback is not None:
                    callback(ticks)
                await asyncio.sleep(0.5)
            except Exception as exc:  # pragma: no cover - runtime only
                self._last_error = str(exc)
                self.logger.exception("Streaming tick error for %s", symbol)
                await asyncio.sleep(1.0)
                self.ensure_connected()
