"""Utilities to merge Parquet inputs using DuckDB for low-memory upsert/merge.

This helper is optional — duckdb is an optional dependency. When available it
performs a disk-based merge/dedup by a provided key (default 'timestamp') and
writes a compact Parquet output without loading entire tables into Python RAM.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


def merge_parquet_with_duckdb(existing_path: str, new_df, output_path: str, key: str = "timestamp") -> None:
    """Merge new_df into existing_path using DuckDB and write to output_path.

    Parameters
    - existing_path: path to existing parquet file (may not exist)
    - new_df: pandas DataFrame with new rows
    - output_path: destination parquet path (overwritten)
    - key: column name to deduplicate on (default 'timestamp')

    Raises RuntimeError if duckdb is not installed.
    """
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("duckdb is required for merge_parquet_with_duckdb; install via 'pip install duckdb'") from exc

    # Write incoming DataFrame to a temporary parquet file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_path = tmp.name
    tmp.close()
    try:
        # Use pyarrow for deterministic parquet output
        table = pa.Table.from_pandas(new_df, preserve_index=False)
        pq.write_table(table, tmp_path)

        con = duckdb.connect(database=":memory:")

        if os.path.exists(existing_path):
            # Use DuckDB to read both files, union them and deduplicate by key using row_number
            sql = f"""
                CREATE OR REPLACE TEMP TABLE existing AS SELECT * FROM read_parquet('{existing_path}');
                CREATE OR REPLACE TEMP TABLE incoming AS SELECT * FROM read_parquet('{tmp_path}');
                CREATE OR REPLACE TEMP TABLE merged AS
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {key} DESC) AS rn
                    FROM (
                        SELECT * FROM existing
                        UNION ALL
                        SELECT * FROM incoming
                    )
                ) WHERE rn = 1;
                COPY (SELECT * FROM merged) TO '{output_path}' (FORMAT PARQUET);
            """
            con.execute(sql)
        else:
            # No existing file: copy incoming to output
            sql = f"COPY (SELECT * FROM read_parquet('{tmp_path}')) TO '{output_path}' (FORMAT PARQUET);"
            con.execute(sql)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
