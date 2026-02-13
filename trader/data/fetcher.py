from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import ccxt
import polars as pl
import structlog

logger = structlog.get_logger()


def _to_futures_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' to 'BTC/USDT:USDT' for Binance futures."""
    if ":" not in symbol:
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        return f"{symbol}:{quote}"
    return symbol


def fetch_candles(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    futures_symbol = _to_futures_symbol(symbol)

    all_candles: list[list] = []
    since = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    max_retries = 5

    logger.info(
        "fetching_candles",
        symbol=futures_symbol,
        timeframe=timeframe,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    while since < end_ms:
        retries = 0
        while retries < max_retries:
            try:
                ohlcv = exchange.fetch_ohlcv(futures_symbol, timeframe, since=since, limit=1500)
                break
            except Exception as e:
                retries += 1
                logger.warning("fetch_error", error=str(e), retry=retries)
                if retries >= max_retries:
                    raise
                time.sleep(5 * retries)

        if not ohlcv:
            break

        # Filter out candles beyond end_date
        ohlcv = [c for c in ohlcv if c[0] < end_ms]
        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        logger.debug("fetched_batch", count=len(ohlcv), total=len(all_candles))
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        raise ValueError(f"No candles fetched for {symbol} {timeframe}")

    df = pl.DataFrame(
        all_candles,
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
        orient="row",
    )

    # Binance returns timestamps in milliseconds — convert to datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp"),
    )
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    safe_symbol = symbol.replace("/", "_")
    output_path = output_dir / f"{safe_symbol}_{timeframe}.parquet"
    df.write_parquet(output_path)
    logger.info("candles_saved", path=str(output_path), rows=len(df))
    return output_path
