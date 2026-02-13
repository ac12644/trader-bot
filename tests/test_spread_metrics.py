from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from trader.backtest.spread_metrics import (
    compute_spread_metrics,
    run_spread_monte_carlo,
    spread_trades_to_mc_pnls,
)
from trader.models.spread_types import SpreadExitReason, SpreadTrade
from trader.models.types import Direction


def _make_trade(
    pnl_total: float,
    pnl_funding: float = 0.0,
    fees: float = 1.0,
    exit_reason: SpreadExitReason = SpreadExitReason.ZSCORE_TARGET,
    holding_mins: int = 480,
) -> SpreadTrade:
    base_time = datetime(2024, 6, 1, 12, 0)
    return SpreadTrade(
        strategy_name="test",
        symbol_a="BTC/USDT:USDT",
        symbol_b="ETH/USDT:USDT",
        direction_a=Direction.SHORT,
        direction_b=Direction.LONG,
        entry_price_a=50000.0,
        entry_price_b=3000.0,
        exit_price_a=49800.0,
        exit_price_b=3010.0,
        notional_per_leg=1500.0,
        entry_time=base_time,
        exit_time=base_time + timedelta(minutes=holding_mins),
        exit_reason=exit_reason,
        pnl_leg_a=(pnl_total - pnl_funding + fees) * 0.6,
        pnl_leg_b=(pnl_total - pnl_funding + fees) * 0.4,
        pnl_funding=pnl_funding,
        pnl_total=pnl_total,
        fees_paid=fees,
        holding_duration_minutes=holding_mins,
    )


# ─── Metrics tests ──────────────────────────────────────────────────


class TestSpreadMetrics:
    def test_empty_trades(self):
        result = compute_spread_metrics([], 1000.0)
        assert result["error"] == "no_trades"

    def test_basic_metrics(self):
        trades = [
            _make_trade(pnl_total=10.0, fees=1.0),
            _make_trade(pnl_total=-5.0, fees=1.0),
            _make_trade(pnl_total=8.0, fees=1.0),
        ]
        result = compute_spread_metrics(trades, 1000.0)

        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["win_rate_pct"] == pytest.approx(66.67, rel=0.01)
        assert result["net_profit_usd"] == pytest.approx(13.0)
        assert result["final_equity"] == pytest.approx(1013.0)
        assert result["total_fees_usd"] == pytest.approx(3.0)

    def test_profit_factor(self):
        trades = [
            _make_trade(pnl_total=20.0),
            _make_trade(pnl_total=-10.0),
        ]
        result = compute_spread_metrics(trades, 1000.0)
        assert result["profit_factor"] == pytest.approx(2.0)

    def test_funding_metrics(self):
        trades = [
            _make_trade(pnl_total=5.0, pnl_funding=3.0),
            _make_trade(pnl_total=2.0, pnl_funding=4.0),
        ]
        result = compute_spread_metrics(trades, 1000.0)
        assert result["total_funding_collected_usd"] == pytest.approx(7.0)

    def test_exit_reasons_breakdown(self):
        trades = [
            _make_trade(pnl_total=5.0, exit_reason=SpreadExitReason.ZSCORE_TARGET),
            _make_trade(pnl_total=-3.0, exit_reason=SpreadExitReason.ZSCORE_STOP),
            _make_trade(pnl_total=2.0, exit_reason=SpreadExitReason.TIME_EXIT),
            _make_trade(pnl_total=1.0, exit_reason=SpreadExitReason.ZSCORE_TARGET),
        ]
        result = compute_spread_metrics(trades, 1000.0)
        reasons = result["exit_reasons"]
        assert reasons["zscore_target"] == 2
        assert reasons["zscore_stop"] == 1
        assert reasons["time_exit"] == 1

    def test_max_consecutive_losses(self):
        trades = [
            _make_trade(pnl_total=5.0),
            _make_trade(pnl_total=-1.0),
            _make_trade(pnl_total=-2.0),
            _make_trade(pnl_total=-3.0),
            _make_trade(pnl_total=4.0),
        ]
        result = compute_spread_metrics(trades, 1000.0)
        assert result["max_consecutive_losses"] == 3

    def test_holding_hours(self):
        trades = [
            _make_trade(pnl_total=5.0, holding_mins=120),  # 2h
            _make_trade(pnl_total=3.0, holding_mins=360),  # 6h
        ]
        result = compute_spread_metrics(trades, 1000.0)
        assert result["avg_holding_hours"] == pytest.approx(4.0)


# ─── Monte Carlo tests ──────────────────────────────────────────────


class TestSpreadMonteCarlo:
    def test_mc_pnl_extraction(self):
        trades = [
            _make_trade(pnl_total=10.0),
            _make_trade(pnl_total=-5.0),
            _make_trade(pnl_total=3.0),
        ]
        pnls = spread_trades_to_mc_pnls(trades)
        np.testing.assert_array_almost_equal(pnls, [10.0, -5.0, 3.0])

    def test_mc_empty_trades(self):
        result = run_spread_monte_carlo([], 1000.0)
        assert result["error"] == "no_trades"

    def test_mc_produces_results(self):
        trades = [_make_trade(pnl_total=i * 2.0 - 5.0) for i in range(20)]
        result = run_spread_monte_carlo(trades, 1000.0, num_simulations=100)

        assert result["simulations"] == 100
        assert "max_dd_95th_pct" in result
        assert "max_dd_99th_pct" in result
        assert "final_equity_median" in result
        assert "ruin_probability_pct" in result
        assert result["max_dd_95th_pct"] >= 0

    def test_mc_all_winners(self):
        trades = [_make_trade(pnl_total=5.0) for _ in range(10)]
        result = run_spread_monte_carlo(trades, 1000.0, num_simulations=50)
        assert result["ruin_probability_pct"] == 0.0
        assert result["final_equity_median"] > 1000.0
