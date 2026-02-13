from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from trader.backtest.cost_model import SpreadCostModel
from trader.backtest.multi_asset_runner import MultiAssetFundingRunner
from trader.config.settings import BacktestConfig, FundingArbStrategyConfig


@pytest.fixture
def funding_config():
    return FundingArbStrategyConfig(
        entry_rate_threshold=0.0002,
        exit_rate_threshold=0.0,
        negative_rate_exit=True,
        basis_blowout_pct=0.02,
        max_holding_hours=720,
        min_holding_hours=72,
        notional_per_position=750.0,
    )


@pytest.fixture
def cost_model():
    bt = BacktestConfig(
        fee_taker_pct=0.05,
        fee_maker_pct=0.02,
        slippage_entry_pct=0.04,
        slippage_exit_pct=0.02,
        include_funding=False,
        monte_carlo_runs=100,
        monte_carlo_max_dd_95_pct=30.0,
        walk_forward_train_months=6,
        walk_forward_test_months=2,
        start_date="2024-06-01",
        end_date="2024-07-01",
    )
    cm = SpreadCostModel(bt)
    cm.set_spot_fees(0.10, 0.10)
    return cm


def _make_candles(n_bars: int, start_price: float, start_dt: datetime) -> pl.DataFrame:
    """Generate synthetic 1h candles."""
    timestamps = [start_dt + timedelta(hours=i) for i in range(n_bars)]
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n_bars) * 10)
    prices = np.maximum(prices, 100.0)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": prices.tolist(),
        "high": (prices + 20).tolist(),
        "low": (prices - 20).tolist(),
        "close": prices.tolist(),
        "volume": [1000.0] * n_bars,
    })


def _make_funding(n_bars: int, start_dt: datetime, rate: float = 0.0005) -> pl.DataFrame:
    """Generate synthetic funding rates at 8h intervals."""
    timestamps = []
    rates = []
    dt = start_dt
    end = start_dt + timedelta(hours=n_bars)
    while dt < end:
        if dt.hour in (0, 8, 16):
            timestamps.append(dt)
            rates.append(rate)
        dt += timedelta(hours=1)

    return pl.DataFrame({
        "timestamp": timestamps,
        "funding_rate": rates,
        "mark_price": [50000.0] * len(timestamps),
    })


class TestMultiAssetFundingRunner:
    def test_perp_symbol_conversion(self, funding_config, cost_model, tmp_path):
        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT", "ETH/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 7, 1),
        )
        assert runner._perp_symbol("BTC/USDT") == "BTC/USDT:USDT"
        assert runner._perp_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"

    def test_path_helpers(self, funding_config, cost_model, tmp_path):
        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 7, 1),
        )
        assert runner._perp_path("BTC/USDT") == tmp_path / "BTC_USDT:USDT_1h.parquet"
        assert runner._spot_path("BTC/USDT") == tmp_path / "BTC_USDT_spot_1h.parquet"
        assert runner._funding_path("BTC/USDT") == tmp_path / "BTC_USDT:USDT_funding.parquet"

    def test_run_all_skips_missing_data(self, funding_config, cost_model, tmp_path):
        """Symbols without data files should be skipped."""
        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT", "FAKE/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 7, 1),
        )
        # No data files exist
        results = runner.run_all()
        assert len(results) == 0

    def test_run_all_with_synthetic_data(self, funding_config, cost_model, tmp_path):
        """Run backtest with synthetic data for 2 symbols."""
        start = datetime(2024, 6, 1)
        n_bars = 200

        for symbol in ["BTC/USDT", "ETH/USDT"]:
            perp_sym = f"{symbol}:USDT"
            safe_perp = perp_sym.replace("/", "_")
            safe_spot = symbol.replace("/", "_")

            base_price = 50000.0 if "BTC" in symbol else 3000.0
            candles = _make_candles(n_bars, base_price, start)

            # Save perp, spot, funding
            candles.write_parquet(tmp_path / f"{safe_perp}_1h.parquet")
            candles.write_parquet(tmp_path / f"{safe_spot}_spot_1h.parquet")

            funding = _make_funding(n_bars, start, rate=0.0005)
            funding.write_parquet(tmp_path / f"{safe_perp}_funding.parquet")

        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT", "ETH/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=start,
            end_date=start + timedelta(hours=n_bars),
            leverage=20,
        )

        results = runner.run_all()
        # At least one symbol should produce trades
        total_trades = sum(len(t) for t in results.values())
        assert total_trades >= 1

    def test_run_all_handles_empty_data(self, funding_config, cost_model, tmp_path):
        """Symbols with empty data should be skipped gracefully."""
        start = datetime(2024, 6, 1)
        n_bars = 200

        # Create valid data for ETH
        candles = _make_candles(n_bars, 3000.0, start)
        safe_perp = "ETH_USDT:USDT"
        safe_spot = "ETH_USDT"
        candles.write_parquet(tmp_path / f"{safe_perp}_1h.parquet")
        candles.write_parquet(tmp_path / f"{safe_spot}_spot_1h.parquet")
        funding = _make_funding(n_bars, start, rate=0.0005)
        funding.write_parquet(tmp_path / f"{safe_perp}_funding.parquet")

        # Create empty data for BTC
        empty = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }, schema={
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })
        btc_perp = "BTC_USDT:USDT"
        empty.write_parquet(tmp_path / f"{btc_perp}_1h.parquet")
        empty.write_parquet(tmp_path / "BTC_USDT_spot_1h.parquet")
        empty_funding = pl.DataFrame({
            "timestamp": [],
            "funding_rate": [],
            "mark_price": [],
        }, schema={
            "timestamp": pl.Datetime,
            "funding_rate": pl.Float64,
            "mark_price": pl.Float64,
        })
        empty_funding.write_parquet(tmp_path / f"{btc_perp}_funding.parquet")

        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT", "ETH/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=start,
            end_date=start + timedelta(hours=n_bars),
            leverage=20,
        )

        results = runner.run_all()
        # ETH should still succeed even if BTC has empty data
        assert "ETH/USDT" in results

    def test_fetch_all_data_uses_cache(self, funding_config, cost_model, tmp_path):
        """Already-cached files should not trigger fetch calls."""
        start = datetime(2024, 6, 1)

        # Pre-create cached files
        candles = _make_candles(100, 50000.0, start)
        funding = _make_funding(100, start)
        candles.write_parquet(tmp_path / "BTC_USDT:USDT_1h.parquet")
        candles.write_parquet(tmp_path / "BTC_USDT_spot_1h.parquet")
        funding.write_parquet(tmp_path / "BTC_USDT:USDT_funding.parquet")

        runner = MultiAssetFundingRunner(
            config=funding_config,
            symbols=["BTC/USDT"],
            cost_model=cost_model,
            starting_capital=1000.0,
            data_dir=tmp_path,
            start_date=start,
            end_date=start + timedelta(hours=100),
        )

        with patch("trader.backtest.multi_asset_runner.fetch_candles") as mock_fc, \
             patch("trader.backtest.multi_asset_runner.fetch_spot_candles") as mock_fsc, \
             patch("trader.backtest.multi_asset_runner.fetch_funding_rates") as mock_ffr:
            results = runner.fetch_all_data()
            # No fetch functions should be called since files exist
            mock_fc.assert_not_called()
            mock_fsc.assert_not_called()
            mock_ffr.assert_not_called()
            assert results["BTC/USDT"] is True
