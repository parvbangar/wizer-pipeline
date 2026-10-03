"""
archiver/parquet.py
═══════════════════
Turn API rows (the JSON dicts PostgREST returns) into Parquet bytes, and check
Parquet bytes back. Pure functions — no I/O — so they are unit-tested directly.

TYPE RULES (column-wise, so every part of a day has a stable schema):
  name ends in "_at"           → timestamp[us, UTC]  (PostgREST sends ISO 8601)
  any value is a dict / list   → JSON text            (jsonb columns)
  all values bool              → bool
  all values int               → int64
  ints and floats mixed        → float64              (JSON drops ".0")
  everything else / all null   → string
Compression: zstd level 9 — measured 1,445 bytes/article on production rows vs
~4,400 bytes/article in Postgres (heap + TOAST + indexes).
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9


def _parse_ts(value):
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def _column(name: str, values: list) -> pa.Array:
    present = [v for v in values if v is not None]
    if name.endswith("_at"):
        return pa.array([_parse_ts(v) for v in values], type=pa.timestamp("us", tz="UTC"))
    if any(isinstance(v, (dict, list)) for v in present):
        return pa.array([None if v is None else json.dumps(v, ensure_ascii=False) for v in values],
                        type=pa.string())
    if present and all(isinstance(v, bool) for v in present):
        return pa.array(values, type=pa.bool_())
    if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return pa.array(values, type=pa.int64())
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return pa.array([None if v is None else float(v) for v in values], type=pa.float64())
    return pa.array([None if v is None else str(v) for v in values], type=pa.string())


def rows_to_table(rows: list[dict]) -> pa.Table:
    """Column-wise typed Arrow table; column order = first row's key order."""
    if not rows:
        return pa.table({})
    names = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in names:
                names.append(k)
    return pa.table({n: _column(n, [r.get(n) for r in rows]) for n in names})


def to_parquet_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(rows_to_table(rows), buf, compression=COMPRESSION,
                   compression_level=COMPRESSION_LEVEL)
    return buf.getvalue()


def parquet_row_count(data: bytes) -> int:
    """Row count from the Parquet footer — also proves the file is readable."""
    return pq.ParquetFile(io.BytesIO(data)).metadata.num_rows


def parquet_ids(data: bytes, column: str = "id") -> list:
    """All values of one column (used to prove an export holds exactly the rows)."""
    return pq.read_table(io.BytesIO(data), columns=[column]).column(column).to_pylist()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
