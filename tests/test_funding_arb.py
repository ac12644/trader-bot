from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trader.config.settings import FundingArbStrategyConfig
from trader.models.spread_types import SpreadBar, SpreadExitReason, SpreadLeg, SpreadPosition
from trader.models.types import Direction
from trader.strategies.funding_arb import FundingArbStrategy


@pytest.fixture
def config() -> FundingArbStrategyConfig:
    return FundingArbStrategyConfig(
        entry_rate_threshold=0.0003,
        exit_rate_threshold=0.0001,
        negative_rate_exit=True,
        basis_blowout_pct=0.02,
        max_holding_hours=720,
        min_holding_hours=24,
        notional_per_position=750.0,
    )


@pytest.fixture
def strategy(config: FundingArbStrategyConfig) -> FundingArbStrategy:
    return FundingArbStrategy(config)


@pytest.fixture
def bar() -> SpreadBar:
    return SpreadBar(
        timestamp=datetime(2025, 1, 1, 8, 0),
        symbol_a="BTC/USDT:USDT",
        close_a=50000.0,
        high_a=50100.0,
        low_a=49900.0,
        volume_a=100.0,
        symbol_b="BTC/USDT",
        close_b=49950.0,
        high_b=50050.0,
        low_b=49850.0,
        volume_b=200.0,
    )


def _make_position(entry_time: datetime, funding_rate: float = 0.0005) -> SpreadPosition:
    return SpreadPosition(
        leg_a=SpreadLeg(
            symbol="BTC/USDT:USDT",
            direction=Direction.SHORT,
            entry_price=50000.0,
            current_price=50000.0,
            notional_usd=750.0,
            is_perp=True,
        ),
        leg_b=SpreadLeg(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            entry_price=49950.0,
            current_price=49950.0,
            notional_usd=750.0,
            is_perp=False,
        ),
        entry_time=entry_time,
        strategy_name="funding_arb",
        funding_rate_at_entry=funding_rate,
        basis_at_entry=50.0,
    )


# ─── Entry tests ─────────────────────────────────────────────────────


class TestFundingArbEntry:
    def test_no_entry_when_position_open(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {"funding_rate": 0.001}
        signal = strategy.should_enter(bar, indicators, has_open_position=True)
        assert signal is None

    def test_no_entry_below_threshold(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {"funding_rate": 0.0002}  # below 0.0003
        signal = strategy.should_enter(bar, indicators, has_open_position=False)
        assert signal is None

    def test_no_entry_at_threshold(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {"funding_rate": 0.0003}  # equal to threshold, not above
        signal = strategy.should_enter(bar, indicators, has_open_position=False)
        assert signal is None

    def test_entry_above_threshold(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {"funding_rate": 0.0005}
        signal = strategy.should_enter(bar, indicators, has_open_position=False)
        assert signal is not None
        assert signal.strategy_name == "funding_arb"
        assert signal.direction_a == Direction.SHORT  # short perp
        assert signal.direction_b == Direction.LONG  # long spot
        assert signal.entry_price_a == bar.close_a
        assert signal.entry_price_b == bar.close_b
        assert signal.notional_per_leg == 750.0
        assert signal.funding_rate_at_entry == 0.0005

    def test_entry_basis_calculated(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {"funding_rate": 0.0005}
        signal = strategy.should_enter(bar, indicators, has_open_position=False)
        assert signal is not None
        assert signal.basis_at_entry == bar.close_a - bar.close_b

    def test_entry_missing_funding_rate(self, strategy: FundingArbStrategy, bar: SpreadBar):
        indicators = {}  # no funding_rate → defaults to 0.0
        signal = strategy.should_enter(bar, indicators, has_open_position=False)
        assert signal is None


# ─── Exit tests ──────────────────────────────────────────────────────


class TestFundingArbExit:
    def test_basis_blowout_exits_immediately(self, strategy: FundingArbStrategy):
        """Basis blowout should trigger even before min hold."""
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        # Bar only 1 hour after entry (well before 24h min hold)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=1),
            symbol_a="BTC/USDT:USDT",
            close_a=51500.0,  # perp jumped
            high_a=51500.0,
            low_a=50000.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=50000.0,  # spot stayed
            high_b=50050.0,
            low_b=49950.0,
            volume_b=200.0,
        )
        # basis_pct = |51500 - 50000| / 50000 = 0.03 = 3% > 2%
        indicators = {"funding_rate": 0.0005}
        reason = strategy.should_exit(bar, position, indicators)
        assert reason == SpreadExitReason.BASIS_BLOWOUT

    def test_no_exit_before_min_hold(self, strategy: FundingArbStrategy):
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=12),  # 12h < 24h min
            symbol_a="BTC/USDT:USDT",
            close_a=50000.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49950.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        indicators = {"funding_rate": 0.00005}  # below exit threshold but min hold not met
        reason = strategy.should_exit(bar, position, indicators)
        assert reason is None

    def test_funding_flip_after_min_hold(self, strategy: FundingArbStrategy):
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=48),  # 48h > 24h min
            symbol_a="BTC/USDT:USDT",
            close_a=50000.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49950.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        indicators = {"funding_rate": 0.00005}  # below 0.0001 exit threshold
        reason = strategy.should_exit(bar, position, indicators)
        assert reason == SpreadExitReason.FUNDING_FLIP

    def test_negative_rate_exit(self, strategy: FundingArbStrategy):
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=48),
            symbol_a="BTC/USDT:USDT",
            close_a=50000.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49950.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        indicators = {"funding_rate": -0.0002}  # negative
        reason = strategy.should_exit(bar, position, indicators)
        assert reason == SpreadExitReason.FUNDING_FLIP

    def test_negative_rate_exit_disabled(self, config: FundingArbStrategyConfig):
        config_no_neg = FundingArbStrategyConfig(
            entry_rate_threshold=config.entry_rate_threshold,
            exit_rate_threshold=config.exit_rate_threshold,
            negative_rate_exit=False,
            basis_blowout_pct=config.basis_blowout_pct,
            max_holding_hours=config.max_holding_hours,
            min_holding_hours=config.min_holding_hours,
            notional_per_position=config.notional_per_position,
        )
        strat = FundingArbStrategy(config_no_neg)
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=48),
            symbol_a="BTC/USDT:USDT",
            close_a=50000.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49950.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        # -0.0002 is < exit_threshold (0.0001), so FUNDING_FLIP fires first
        indicators = {"funding_rate": -0.0002}
        reason = strat.should_exit(bar, position, indicators)
        assert reason == SpreadExitReason.FUNDING_FLIP

    def test_time_exit(self, strategy: FundingArbStrategy):
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=720),  # exactly max
            symbol_a="BTC/USDT:USDT",
            close_a=50000.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49950.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        indicators = {"funding_rate": 0.0005}  # still good, but max hold reached
        reason = strategy.should_exit(bar, position, indicators)
        assert reason == SpreadExitReason.TIME_EXIT

    def test_no_exit_healthy_position(self, strategy: FundingArbStrategy):
        entry_time = datetime(2025, 1, 1, 0, 0)
        position = _make_position(entry_time)
        bar = SpreadBar(
            timestamp=entry_time + timedelta(hours=48),
            symbol_a="BTC/USDT:USDT",
            close_a=50010.0,
            high_a=50100.0,
            low_a=49900.0,
            volume_a=100.0,
            symbol_b="BTC/USDT",
            close_b=49990.0,
            high_b=50050.0,
            low_b=49850.0,
            volume_b=200.0,
        )
        # basis_pct = |50010 - 49990| / 49990 = 0.04% < 2%, funding still good
        indicators = {"funding_rate": 0.0005}
        reason = strategy.should_exit(bar, position, indicators)
        assert reason is None

    def test_strategy_name(self, strategy: FundingArbStrategy):
        assert strategy.name == "funding_arb"
