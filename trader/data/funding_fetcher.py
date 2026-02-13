from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import ccxt
import polars as pl
import structlog

logger = structlog.get_logger()


def _to_futures_symbol(symbol: str) -> str:
    if ":" not in symbol:
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        return f"{symbol}:{quote}"
    return symbol


def fetch_funding_rates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    """Fetch historical funding rates for a perpetual contract."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    futures_symbol = _to_futures_symbol(symbol)

    all_rates: list[dict] = []
    since = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    logger.info(
        "fetching_funding_rates",
        symbol=futures_symbol,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    while since < end_ms:
        try:
            rates = exchange.fetch_funding_rate_history(
                futures_symbol, since=since, limit=1000,
            )
        except Exception as e:
            logger.warning("funding_fetch_error", error=str(e))
            time.sleep(5)
            continue

        if not rates:
            break

        for r in rates:
            ts = r.get("timestamp", 0)
            if ts >= end_ms:
                break
            all_rates.append({
                "timestamp": datetime.utcfromtimestamp(ts / 1000),
                "funding_rate": r.get("fundingRate", 0.0) or 0.0,
                "mark_price": r.get("markPrice", 0.0) or 0.0,
            })

        last_ts = rates[-1]["timestamp"]
        if last_ts >= end_ms:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rates:
        raise ValueError(f"No funding rates fetched for {symbol}")

    df = pl.DataFrame(all_rates)
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    safe_symbol = symbol.replace("/", "_")
    output_path = output_dir / f"{safe_symbol}_funding.parquet"
    df.write_parquet(output_path)
    logger.info("funding_rates_saved", path=str(output_path), rows=len(df))
    return output_path


def fetch_spot_candles(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    """Fetch spot OHLCV data from Binance spot exchange."""
    exchange = ccxt.binance({"enableRateLimit": True})

    all_candles: list[list] = []
    since = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    logger.info(
        "fetching_spot_candles",
        symbol=symbol,
        timeframe=timeframe,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1500)
        except Exception as e:
            logger.warning("spot_fetch_error", error=str(e))
            time.sleep(5)
            continue

        if not ohlcv:
            break

        ohlcv = [c for c in ohlcv if c[0] < end_ms]
        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        raise ValueError(f"No spot candles fetched for {symbol}")

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
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp"),
    )
    df = df.unique(subset=["timestamp"]).sort("timestamp")

    safe_symbol = symbol.replace("/", "_")
    output_path = output_dir / f"{safe_symbol}_spot_{timeframe}.parquet"
    df.write_parquet(output_path)
    logger.info("spot_candles_saved", path=str(output_path), rows=len(df))
    return output_path
