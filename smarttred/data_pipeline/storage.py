from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class StorageManager:
    """Storage wrapper for Parquet-based market datasets."""

    def __init__(self, base_dir: str = "data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, relative_path: str) -> str:
        """Persist a DataFrame as a Parquet file in the storage directory."""
        destination = self.base_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(destination, index=False)
        return str(destination)

    def load_dataframe(self, relative_path: str) -> pd.DataFrame:
        """Load a Parquet file into a DataFrame."""
        path = self.base_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Storage file not found: {path}")
        return pd.read_parquet(path)
