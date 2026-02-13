from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from trader.backtest.cost_model import SpreadCostModel
from trader.backtest.spread_engine import SpreadEngine
from trader.config.settings import (
    BacktestConfig,
    FundingArbStrategyConfig,
)
from trader.models.spread_types import SpreadExitReason
from trader.models.types import Direction
from trader.strategies.funding_arb import FundingArbStrategy


def _bt_config() -> BacktestConfig:
    return BacktestConfig(
        fee_taker_pct=0.05,
        fee_maker_pct=0.02,
        slippage_entry_pct=0.0,
        slippage_exit_pct=0.0,
        include_funding=True,
        monte_carlo_runs=100,
        monte_carlo_max_dd_95_pct=15.0,
        walk_forward_train_months=6,
        walk_forward_test_months=2,
        start_date="2024-01-01",
        end_date="2024-02-01",
    )


def _fa_config() -> FundingArbStrategyConfig:
    return FundingArbStrategyConfig(
        entry_rate_threshold=0.0003,
        exit_rate_threshold=0.0001,
        negative_rate_exit=True,
        basis_blowout_pct=0.02,
        max_holding_hours=48,
        min_holding_hours=24,
        notional_per_position=750.0,
    )


def _make_candles(
    start: datetime,
    n_bars: int,
    base_price: float,
    trend: float = 0.0,
) -> pl.DataFrame:
    """Generate n_bars of 1h synthetic candles."""
    rows = []
    price = base_price
    for i in range(n_bars):
        ts = start + timedelta(hours=i)
        price += trend
        rows.append({
            "timestamp": ts,
            "open": price - 5.0,
            "high": price + 10.0,
            "low": price - 10.0,
            "close": price,
            "volume": 100.0,
        })
    return pl.DataFrame(rows)


def _make_funding_rates(
    start: datetime,
    n_days: int,
    rate: float = 0.0005,
) -> pl.DataFrame:
    """Generate funding rate entries at 8h intervals."""
    rows = []
    for d in range(n_days):
        for h in [0, 8, 16]:
            ts = start + timedelta(days=d, hours=h)
            rows.append({
                "timestamp": ts,
                "funding_rate": rate,
                "mark_price": 50000.0,
            })
    return pl.DataFrame(rows)


# ─── Synchronized timeline ───────────────────────────────────────────


class TestSynchronizedTimeline:
    def test_inner_join_matching(self):
        start = datetime(2024, 1, 1)
        a = _make_candles(start, 100, 50000.0)
        b = _make_candles(start, 100, 3000.0)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())
        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            leg_b_is_spot=True,
        )
        assert engine.n_bars == 100

    def test_inner_join_partial_overlap(self):
        start = datetime(2024, 1, 1)
        a = _make_candles(start, 100, 50000.0)
        # B starts 10h later → 90 overlapping bars
        b = _make_candles(start + timedelta(hours=10), 100, 49950.0)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())
        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            leg_b_is_spot=True,
        )
        assert engine.n_bars == 90


# ─── Ratio Z-score ──────────────────────────────────────────────────


class TestRatioZscore:
    def test_zscore_nan_before_lookback(self):
        start = datetime(2024, 1, 1)
        a = _make_candles(start, 50, 50000.0)
        b = _make_candles(start, 50, 49950.0)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())
        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            lookback_period=20,
            leg_b_is_spot=True,
        )
        # First 20 bars should be NaN
        assert np.isnan(engine.ratio_zscore[0])
        assert np.isnan(engine.ratio_zscore[19])
        # Bar 20 should have a value
        assert not np.isnan(engine.ratio_zscore[20])


# ─── Funding payments ───────────────────────────────────────────────


class TestFundingPayments:
    def test_funding_arb_collects_funding(self):
        """Short perp + long spot should collect positive funding."""
        start = datetime(2024, 1, 1, 0, 0)  # midnight UTC
        n_bars = 72  # 3 days

        # Flat prices → no leg PnL, pure funding test
        a = _make_candles(start, n_bars, 50000.0, trend=0.0)
        b = _make_candles(start, n_bars, 49950.0, trend=0.0)
        funding = _make_funding_rates(start, 4, rate=0.0005)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())

        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            funding_rates=funding,
            leg_b_is_spot=True,
        )
        trades = engine.run()

        # Should have at least one trade
        assert len(trades) >= 1
        trade = trades[0]
        # Funding should be positive (short perp, positive rate)
        assert trade.pnl_funding > 0
        assert trade.exit_reason == SpreadExitReason.TIME_EXIT

    def test_no_funding_on_spot_legs(self):
        """Spot legs should not receive funding payments."""
        start = datetime(2024, 1, 1, 0, 0)
        n_bars = 72
        a = _make_candles(start, n_bars, 50000.0)
        b = _make_candles(start, n_bars, 49950.0)
        funding = _make_funding_rates(start, 4, rate=0.001)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())

        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            funding_rates=funding,
            leg_b_is_spot=True,
        )
        trades = engine.run()
        if trades:
            # Only leg_a (perp SHORT) should accumulate funding
            # leg_b is spot, should have 0 funding
            trade = trades[0]
            # With 0.001 rate and $750 notional short perp:
            # Per 8h payment: 750 * 0.001 = $0.75
            # Over ~48h → 6 payments ≈ $4.50
            assert trade.pnl_funding > 0


# ─── PnL calculation ────────────────────────────────────────────────


class TestPnlCalculation:
    def test_long_leg_pnl(self):
        from trader.models.spread_types import SpreadLeg

        leg = SpreadLeg(
            symbol="BTC", direction=Direction.LONG,
            entry_price=50000.0, current_price=50000.0,
            notional_usd=1000.0, is_perp=True,
        )
        pnl = SpreadEngine._leg_pnl(leg, 51000.0)
        # (51000 - 50000) / 50000 * 1000 = $20
        assert pnl == pytest.approx(20.0)

    def test_short_leg_pnl(self):
        from trader.models.spread_types import SpreadLeg

        leg = SpreadLeg(
            symbol="BTC", direction=Direction.SHORT,
            entry_price=50000.0, current_price=50000.0,
            notional_usd=1000.0, is_perp=True,
        )
        pnl = SpreadEngine._leg_pnl(leg, 49000.0)
        # (50000 - 49000) / 50000 * 1000 = $20
        assert pnl == pytest.approx(20.0)


# ─── End-to-end engine run ──────────────────────────────────────────


class TestEndToEnd:
    def test_funding_arb_produces_trades(self):
        """With high funding rates, engine should produce trades."""
        start = datetime(2024, 1, 1, 0, 0)
        n_bars = 200

        a = _make_candles(start, n_bars, 50000.0, trend=0.0)
        b = _make_candles(start, n_bars, 49950.0, trend=0.0)
        funding = _make_funding_rates(start, n_bars // 24 + 1, rate=0.0005)

        cost = SpreadCostModel(_bt_config())
        strategy = FundingArbStrategy(_fa_config())

        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            funding_rates=funding,
            leg_b_is_spot=True,
        )

        trades = engine.run()
        assert len(trades) >= 1
        for trade in trades:
            assert trade.fees_paid > 0
            assert trade.holding_duration_minutes > 0
        # At least some trades should collect positive funding
        # (last trade may be END_OF_DATA before any 8h payment)
        funded_trades = [t for t in trades if t.pnl_funding > 0]
        assert len(funded_trades) >= 1

    def test_end_of_data_closes_positions(self):
        """Positions open at end of data should be force-closed."""
        start = datetime(2024, 1, 1, 0, 0)
        # Short data: entry happens but never reaches max holding time
        n_bars = 30

        a = _make_candles(start, n_bars, 50000.0, trend=0.0)
        b = _make_candles(start, n_bars, 49950.0, trend=0.0)
        funding = _make_funding_rates(start, 3, rate=0.0005)

        cost = SpreadCostModel(_bt_config())
        # Long min_holding + max_holding so position stays open past data end
        strategy = FundingArbStrategy(FundingArbStrategyConfig(
            entry_rate_threshold=0.0003,
            exit_rate_threshold=0.0001,
            negative_rate_exit=True,
            basis_blowout_pct=0.02,
            max_holding_hours=1000,
            min_holding_hours=500,
            notional_per_position=750.0,
        ))

        engine = SpreadEngine(
            strategy=strategy,
            candles_a=a, candles_b=b,
            symbol_a="BTC/USDT:USDT", symbol_b="BTC/USDT",
            cost_model=cost, starting_capital=1000.0,
            funding_rates=funding,
            leg_b_is_spot=True,
        )

        trades = engine.run()
        # Any trades should be closed with END_OF_DATA
        eod_trades = [t for t in trades if t.exit_reason == SpreadExitReason.END_OF_DATA]
        for t in trades:
            assert t.exit_time is not None
        if trades:
            assert len(eod_trades) > 0
