from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trader.backtest.multi_asset_metrics import (
    _max_concurrent_positions,
    compute_multi_asset_metrics,
)
from trader.models.spread_types import SpreadExitReason, SpreadTrade
from trader.models.types import Direction


def _make_trade(
    symbol: str,
    pnl: float,
    funding: float = 0.0,
    fees: float = 1.0,
    entry_offset_hours: int = 0,
    hold_hours: int = 72,
) -> SpreadTrade:
    """Helper to create a SpreadTrade with minimal fields."""
    entry = datetime(2024, 6, 1) + timedelta(hours=entry_offset_hours)
    exit_t = entry + timedelta(hours=hold_hours)
    return SpreadTrade(
        strategy_name="funding_arb",
        symbol_a=f"{symbol}:USDT",
        symbol_b=symbol,
        direction_a=Direction.SHORT,
        direction_b=Direction.LONG,
        entry_price_a=100.0,
        entry_price_b=100.0,
        exit_price_a=100.0,
        exit_price_b=100.0,
        notional_per_leg=750.0,
        entry_time=entry,
        exit_time=exit_t,
        exit_reason=SpreadExitReason.FUNDING_FLIP,
        pnl_leg_a=-0.5,
        pnl_leg_b=0.3,
        pnl_funding=funding,
        pnl_total=pnl,
        fees_paid=fees,
        holding_duration_minutes=hold_hours * 60,
    )


class TestMultiAssetMetrics:
    def test_empty_results(self):
        result = compute_multi_asset_metrics({}, 1000.0)
        assert result["portfolio"]["total_trades"] == 0

    def test_single_symbol_aggregation(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, funding=15.0, fees=5.0, entry_offset_hours=0),
            _make_trade("BTC/USDT", pnl=5.0, funding=8.0, fees=3.0, entry_offset_hours=100),
        ]
        result = compute_multi_asset_metrics({"BTC/USDT": trades}, 1000.0)

        port = result["portfolio"]
        assert port["total_trades"] == 2
        assert port["net_profit_usd"] == 15.0
        assert port["total_funding_collected_usd"] == 23.0
        assert port["total_fees_usd"] == 8.0
        assert port["symbols_traded"] == 1

    def test_multi_symbol_aggregation(self):
        btc_trades = [
            _make_trade("BTC/USDT", pnl=10.0, funding=15.0, entry_offset_hours=0),
        ]
        eth_trades = [
            _make_trade("ETH/USDT", pnl=-3.0, funding=2.0, entry_offset_hours=0),
            _make_trade("ETH/USDT", pnl=8.0, funding=12.0, entry_offset_hours=100),
        ]
        results = {"BTC/USDT": btc_trades, "ETH/USDT": eth_trades}
        result = compute_multi_asset_metrics(results, 1000.0)

        port = result["portfolio"]
        assert port["total_trades"] == 3
        assert port["net_profit_usd"] == 15.0  # 10 + (-3) + 8
        assert port["symbols_traded"] == 2

    def test_best_worst_symbols(self):
        results = {
            "BTC/USDT": [_make_trade("BTC/USDT", pnl=20.0, funding=25.0)],
            "ETH/USDT": [_make_trade("ETH/USDT", pnl=-5.0, funding=1.0)],
            "SOL/USDT": [_make_trade("SOL/USDT", pnl=8.0, funding=12.0)],
        }
        result = compute_multi_asset_metrics(results, 1000.0)

        best = result["best_symbols"]
        worst = result["worst_symbols"]

        assert len(best) == 2  # BTC and SOL profitable
        assert best[0][0] == "BTC/USDT"  # highest profit first
        assert best[1][0] == "SOL/USDT"

        assert len(worst) == 1  # ETH losing
        assert worst[0][0] == "ETH/USDT"

    def test_symbols_with_no_trades(self):
        results = {
            "BTC/USDT": [_make_trade("BTC/USDT", pnl=10.0, funding=15.0)],
            "FAKE/USDT": [],
        }
        result = compute_multi_asset_metrics(results, 1000.0)

        port = result["portfolio"]
        assert port["total_trades"] == 1
        assert port["symbols_traded"] == 1
        assert port["symbols_total"] == 2

        # FAKE should appear in by_symbol with 0 trades
        assert result["by_symbol"]["FAKE/USDT"]["total_trades"] == 0

    def test_portfolio_equity(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, funding=15.0),
            _make_trade("ETH/USDT", pnl=5.0, funding=8.0),
        ]
        result = compute_multi_asset_metrics(
            {"BTC/USDT": [trades[0]], "ETH/USDT": [trades[1]]},
            1000.0,
        )
        assert result["portfolio"]["final_equity"] == 1015.0

    def test_monte_carlo_included(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, funding=15.0, entry_offset_hours=0),
            _make_trade("BTC/USDT", pnl=5.0, funding=8.0, entry_offset_hours=100),
        ]
        result = compute_multi_asset_metrics({"BTC/USDT": trades}, 1000.0)
        assert "monte_carlo" in result
        assert "simulations" in result["monte_carlo"]

    def test_win_rate_calculation(self):
        results = {
            "BTC/USDT": [
                _make_trade("BTC/USDT", pnl=10.0, funding=15.0, entry_offset_hours=0),
                _make_trade("BTC/USDT", pnl=-2.0, funding=1.0, entry_offset_hours=100),
                _make_trade("BTC/USDT", pnl=5.0, funding=8.0, entry_offset_hours=200),
            ],
        }
        result = compute_multi_asset_metrics(results, 1000.0)
        port = result["portfolio"]
        assert port["winning_trades"] == 2
        assert port["losing_trades"] == 1
        assert port["win_rate_pct"] == pytest.approx(66.67, abs=0.01)

    def test_exit_reasons_aggregated(self):
        t1 = _make_trade("BTC/USDT", pnl=10.0, funding=15.0, entry_offset_hours=0)
        t2 = _make_trade("ETH/USDT", pnl=5.0, funding=8.0, entry_offset_hours=100)
        t2 = SpreadTrade(
            **{**t2.__dict__, "exit_reason": SpreadExitReason.TIME_EXIT},
        )
        results = {"BTC/USDT": [t1], "ETH/USDT": [t2]}
        result = compute_multi_asset_metrics(results, 1000.0)
        reasons = result["portfolio"]["exit_reasons"]
        assert reasons["funding_flip"] == 1
        assert reasons["time_exit"] == 1


class TestMaxConcurrentPositions:
    def test_no_overlap(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, entry_offset_hours=0, hold_hours=72),
            _make_trade("ETH/USDT", pnl=5.0, entry_offset_hours=100, hold_hours=72),
        ]
        assert _max_concurrent_positions(trades) == 1

    def test_full_overlap(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, entry_offset_hours=0, hold_hours=100),
            _make_trade("ETH/USDT", pnl=5.0, entry_offset_hours=10, hold_hours=100),
            _make_trade("SOL/USDT", pnl=3.0, entry_offset_hours=20, hold_hours=100),
        ]
        assert _max_concurrent_positions(trades) == 3

    def test_partial_overlap(self):
        trades = [
            _make_trade("BTC/USDT", pnl=10.0, entry_offset_hours=0, hold_hours=80),
            _make_trade("ETH/USDT", pnl=5.0, entry_offset_hours=50, hold_hours=80),
            _make_trade("SOL/USDT", pnl=3.0, entry_offset_hours=100, hold_hours=80),
        ]
        # BTC: 0-80, ETH: 50-130, SOL: 100-180
        # Max overlap: BTC + ETH at hour 50-80
        assert _max_concurrent_positions(trades) == 2

    def test_empty_trades(self):
        assert _max_concurrent_positions([]) == 0
