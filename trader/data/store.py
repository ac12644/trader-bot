from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger()

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}


def load_candles(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candle file not found: {path}")
    df = pl.read_parquet(path)
    df = df.sort("timestamp")
    return df


def validate_candles(df: pl.DataFrame, timeframe: str) -> list[str]:
    warnings: list[str] = []

    dupes = df.filter(pl.col("timestamp").is_duplicated())
    if len(dupes) > 0:
        warnings.append(f"Found {len(dupes)} duplicate timestamps")

    zero_vol = df.filter(pl.col("volume") == 0)
    if len(zero_vol) > 0:
        warnings.append(f"Found {len(zero_vol)} zero-volume candles")

    if timeframe in TIMEFRAME_SECONDS and len(df) > 1:
        expected_us = TIMEFRAME_SECONDS[timeframe] * 1_000_000
        # Compute diffs in microseconds
        ts_us = df["timestamp"].cast(pl.Int64)
        diffs = ts_us.diff().drop_nulls()
        # Count gaps larger than 1.5x expected interval
        gap_count = int((diffs > int(expected_us * 1.5)).sum())
        if gap_count > 0:
            warnings.append(f"Found {gap_count} gaps larger than expected interval")

    return warnings
