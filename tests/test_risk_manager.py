from datetime import datetime

from trader.config.settings import RiskConfig, ScalingPhase
from trader.models.types import Direction, EquityState, Position, Signal
from trader.risk.risk_manager import RiskManager


def _make_risk_config(**overrides):
    defaults = dict(
        risk_per_trade_pct=2.0,
        daily_loss_cap_pct=5.0,
        weekly_loss_cap_pct=12.0,
        max_concurrent_positions=3,
        max_same_direction_correlated=1,
        max_total_exposure_pct=40.0,
        max_total_open_risk_pct=6.0,
        consecutive_loss_cooldown_count=3,
        consecutive_loss_cooldown_hours=2,
        drawdown_reduction_threshold_pct=15.0,
        drawdown_halt_threshold_pct=25.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _make_phases():
    return [
        ScalingPhase(min_trades=0, risk_pct=1.5, max_leverage=3),
        ScalingPhase(min_trades=50, risk_pct=2.0, max_leverage=3, min_profit_factor=1.3),
    ]


def _make_state(**overrides):
    defaults = dict(
        equity=1000.0,
        peak_equity=1000.0,
        daily_pnl=0.0,
        weekly_pnl=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        consecutive_losses=0,
    )
    defaults.update(overrides)
    return EquityState(**defaults)


def _make_signal(**overrides):
    defaults = dict(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52500.0,
        atr=500.0,
        timestamp=datetime(2024, 1, 1, 12),
        strategy_name="test",
    )
    defaults.update(overrides)
    return Signal(**defaults)


class TestRiskManager:
    def test_initial_phase(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state()
        phase = rm.get_current_phase(state)
        assert phase.risk_pct == 1.5

    def test_risk_reduction_on_drawdown(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state(equity=800.0, peak_equity=1000.0)
        risk = rm.get_risk_per_trade_pct(state)
        assert risk == 1.0  # 20% dd > 15% threshold, drops to 1%

    def test_halt_on_extreme_drawdown(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state(equity=700.0, peak_equity=1000.0)
        risk = rm.get_risk_per_trade_pct(state)
        assert risk == 0.0  # 30% dd > 25% halt

    def test_validates_signal_passes(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state()
        signal = _make_signal()
        ok, reason = rm.validate_signal(signal, state, datetime(2024, 1, 1, 12))
        assert ok

    def test_blocks_when_daily_halted(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state(is_daily_halted=True)
        signal = _make_signal()
        ok, reason = rm.validate_signal(signal, state, datetime(2024, 1, 1, 12))
        assert not ok
        assert "daily" in reason

    def test_blocks_max_positions(self):
        rm = RiskManager(
            _make_risk_config(max_concurrent_positions=1), _make_phases(), 1000.0, 3,
        )
        pos = Position(
            symbol="ETH/USDT", direction=Direction.LONG, entry_price=3000.0,
            stop_loss=2900.0, take_profit=3200.0, trailing_stop=None,
            position_size_usd=100.0, entry_time=datetime(2024, 1, 1),
            atr_at_entry=50.0, strategy_name="test",
        )
        state = _make_state()
        state.open_positions.append(pos)
        signal = _make_signal()
        ok, reason = rm.validate_signal(signal, state, datetime(2024, 1, 1, 12))
        assert not ok
        assert "max_concurrent" in reason

    def test_position_size_calculation(self):
        rm = RiskManager(_make_risk_config(), _make_phases(), 1000.0, 3)
        state = _make_state()
        signal = _make_signal(entry_price=50000.0, stop_loss=49000.0)
        size = rm.calculate_position_size(signal, state, 0.0005, 0.0008)
        assert size > 0
        # Risk is 1.5% of 1000 = $15, stop distance is 2%, size ~ 750 before fees
        assert size < 1000  # Can't exceed equity * leverage but should be reasonable

    def test_daily_loss_cap_triggers(self):
        rm = RiskManager(_make_risk_config(daily_loss_cap_pct=5.0), _make_phases(), 1000.0, 3)
        state = _make_state(daily_pnl=-60.0, peak_equity=1000.0)
        rm.check_daily_weekly_limits(state)
        assert state.is_daily_halted

    def test_consecutive_loss_cooldown(self):
        rm = RiskManager(
            _make_risk_config(consecutive_loss_cooldown_count=3), _make_phases(), 1000.0, 3,
        )
        state = _make_state(consecutive_losses=3)
        t = datetime(2024, 1, 1, 12)
        rm.check_consecutive_losses(state, t)
        assert state.cooldown_until is not None
        assert state.cooldown_until > t
