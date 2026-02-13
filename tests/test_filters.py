from datetime import datetime

from trader.config.settings import ChopFilterConfig, EventFilterConfig, SessionBlackout
from trader.filters.chop_filter import ChopFilter
from trader.filters.correlation_filter import CorrelationFilter
from trader.filters.session_filter import SessionFilter
from trader.models.types import Direction, Position


def _make_chop_config(**overrides):
    defaults = dict(
        adx_threshold=20,
        atr_percentile_threshold=30,
        atr_percentile_window_days=30,
        consecutive_failure_pause_hours=4,
        consecutive_failure_count=3,
    )
    defaults.update(overrides)
    return ChopFilterConfig(**defaults)


class TestChopFilter:
    def test_blocks_low_adx(self):
        f = ChopFilter(_make_chop_config())
        ok, reason = f.is_tradeable(15.0, 50.0, datetime(2024, 1, 1, 12))
        assert not ok
        assert "adx_too_low" in reason

    def test_blocks_low_atr_percentile(self):
        f = ChopFilter(_make_chop_config())
        ok, reason = f.is_tradeable(25.0, 10.0, datetime(2024, 1, 1, 12))
        assert not ok
        assert "atr_pct_too_low" in reason

    def test_passes_good_conditions(self):
        f = ChopFilter(_make_chop_config())
        ok, reason = f.is_tradeable(25.0, 50.0, datetime(2024, 1, 1, 12))
        assert ok

    def test_consecutive_failures_pause(self):
        f = ChopFilter(_make_chop_config(consecutive_failure_count=2))
        t = datetime(2024, 1, 1, 12)
        f.record_stop_loss_hit(t)
        f.record_stop_loss_hit(t)  # triggers pause
        ok, _ = f.is_tradeable(30.0, 60.0, t)
        assert not ok

    def test_win_resets_failures(self):
        f = ChopFilter(_make_chop_config(consecutive_failure_count=3))
        t = datetime(2024, 1, 1, 12)
        f.record_stop_loss_hit(t)
        f.record_stop_loss_hit(t)
        f.record_win()
        f.record_stop_loss_hit(t)
        # Only 1 failure after reset, shouldn't pause
        ok, _ = f.is_tradeable(30.0, 60.0, t)
        assert ok


class TestSessionFilter:
    def test_blocks_during_blackout(self):
        blackouts = [SessionBlackout(start="00:00", end="04:00", days=["mon"])]
        f = SessionFilter(blackouts)
        # Monday at 02:00 UTC
        ok, reason = f.is_tradeable(datetime(2024, 1, 1, 2, 0))  # 2024-01-01 is Monday
        assert not ok

    def test_passes_outside_blackout(self):
        blackouts = [SessionBlackout(start="00:00", end="04:00", days=["mon"])]
        f = SessionFilter(blackouts)
        ok, _ = f.is_tradeable(datetime(2024, 1, 1, 10, 0))
        assert ok


class TestCorrelationFilter:
    def _make_position(self, symbol: str, direction: Direction) -> Position:
        return Position(
            symbol=symbol,
            direction=direction,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            trailing_stop=None,
            position_size_usd=100.0,
            entry_time=datetime(2024, 1, 1),
            atr_at_entry=1.0,
            strategy_name="test",
        )

    def test_blocks_same_direction(self):
        f = CorrelationFilter(max_same_direction=1)
        positions = [self._make_position("BTC/USDT", Direction.LONG)]
        ok, _ = f.is_allowed("ETH/USDT", Direction.LONG, positions)
        assert not ok

    def test_allows_opposite_direction(self):
        f = CorrelationFilter(max_same_direction=1)
        positions = [self._make_position("BTC/USDT", Direction.LONG)]
        ok, _ = f.is_allowed("ETH/USDT", Direction.SHORT, positions)
        assert ok

    def test_allows_when_no_positions(self):
        f = CorrelationFilter(max_same_direction=1)
        ok, _ = f.is_allowed("BTC/USDT", Direction.LONG, [])
        assert ok
