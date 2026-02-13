from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog
from dateutil.relativedelta import relativedelta

from trader.backtest.engine import BacktestEngine
from trader.backtest.metrics import compute_all_metrics
from trader.config.settings import Settings
from trader.strategies.base import BaseStrategy

logger = structlog.get_logger()


def walk_forward_validation(
    settings: Settings,
    strategy: BaseStrategy,
    candles_5m: dict[str, pl.DataFrame],
    candles_1h: dict[str, pl.DataFrame],
    events_calendar_path: Path | None = None,
) -> list[dict]:
    results: list[dict] = []

    # Find data range from timestamps
    all_min = []
    all_max = []
    for sym in candles_5m:
        ts = candles_5m[sym]["timestamp"]
        all_min.append(ts.min())
        all_max.append(ts.max())

    data_start = min(all_min)
    data_end = max(all_max)

    train_months = settings.backtest.walk_forward_train_months
    test_months = settings.backtest.walk_forward_test_months

    window_start = data_start

    window_num = 0
    while True:
        train_end = window_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)

        if test_end > data_end:
            break

        # Split 5m data
        train_5m = {}
        test_5m = {}
        train_1h = {}
        test_1h = {}

        for sym in candles_5m:
            df5 = candles_5m[sym]
            train_5m[sym] = df5.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < train_end)
            )
            test_5m[sym] = df5.filter(
                (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < test_end)
            )

            df1 = candles_1h[sym]
            # Include warm-up data for 1h indicators (extra 60 bars before window)
            warmup_start = window_start - relativedelta(days=3)
            train_1h[sym] = df1.filter(
                (pl.col("timestamp") >= warmup_start) & (pl.col("timestamp") < train_end)
            )
            test_warmup_start = train_end - relativedelta(days=3)
            test_1h[sym] = df1.filter(
                (pl.col("timestamp") >= test_warmup_start) & (pl.col("timestamp") < test_end)
            )

        # Skip windows with insufficient data
        skip = False
        for sym in train_5m:
            if len(train_5m[sym]) < 100 or len(test_5m[sym]) < 50:
                skip = True
                break
        if skip:
            window_start = window_start + relativedelta(months=test_months)
            continue

        # Run train
        train_engine = BacktestEngine(
            settings, strategy, train_5m, train_1h, events_calendar_path,
        )
        train_trades = train_engine.run()
        train_metrics = compute_all_metrics(train_trades, settings.starting_capital)

        # Run test
        test_engine = BacktestEngine(
            settings, strategy, test_5m, test_1h, events_calendar_path,
        )
        test_trades = test_engine.run()
        test_metrics = compute_all_metrics(test_trades, settings.starting_capital)

        window_num += 1
        result = {
            "window": window_num,
            "train_start": str(window_start),
            "train_end": str(train_end),
            "test_end": str(test_end),
            "train_trades": train_metrics.get("total_trades", 0),
            "test_trades": test_metrics.get("total_trades", 0),
            "train_profit_factor": train_metrics.get("profit_factor", 0),
            "test_profit_factor": test_metrics.get("profit_factor", 0),
            "train_sharpe": train_metrics.get("sharpe_ratio", 0),
            "test_sharpe": test_metrics.get("sharpe_ratio", 0),
            "train_return_pct": train_metrics.get("net_return_pct", 0),
            "test_return_pct": test_metrics.get("net_return_pct", 0),
        }
        results.append(result)

        logger.info(
            "walk_forward_window",
            window=window_num,
            train_pf=train_metrics.get("profit_factor", 0),
            test_pf=test_metrics.get("profit_factor", 0),
        )

        window_start = window_start + relativedelta(months=test_months)

    return results
