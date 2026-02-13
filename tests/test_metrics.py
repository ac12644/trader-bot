from datetime import datetime

from trader.backtest.metrics import compute_all_metrics
from trader.backtest.monte_carlo import run_monte_carlo
from trader.models.types import Direction, ExitReason, Trade


def _make_trade(pnl: float, r_mult: float = 1.0, **overrides):
    defaults = dict(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        entry_price=50000.0,
        exit_price=50000.0 + pnl * 100,
        stop_loss=49000.0,
        take_profit=52500.0,
        position_size_usd=500.0,
        entry_time=datetime(2024, 1, 1, 12),
        exit_time=datetime(2024, 1, 1, 14),
        exit_reason=ExitReason.TAKE_PROFIT if pnl > 0 else ExitReason.STOP_LOSS,
        pnl_usd=pnl,
        pnl_pct=pnl / 1000 * 100,
        fees_paid=0.5,
        slippage_cost=0.0,
        r_multiple=r_mult,
        strategy_name="test",
        holding_duration_minutes=120,
    )
    defaults.update(overrides)
    return Trade(**defaults)


class TestMetrics:
    def test_basic_metrics(self):
        trades = [
            _make_trade(50.0, 2.5),
            _make_trade(-20.0, -1.0),
            _make_trade(30.0, 1.5),
            _make_trade(-20.0, -1.0),
            _make_trade(40.0, 2.0),
        ]
        m = compute_all_metrics(trades, 1000.0)
        assert m["total_trades"] == 5
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 2
        assert m["win_rate_pct"] == 60.0
        assert m["net_profit_usd"] == 80.0
        assert m["profit_factor"] > 1.0

    def test_no_trades(self):
        m = compute_all_metrics([], 1000.0)
        assert m["total_trades"] == 0

    def test_all_wins(self):
        trades = [_make_trade(10.0, 1.0) for _ in range(5)]
        m = compute_all_metrics(trades, 1000.0)
        assert m["win_rate_pct"] == 100.0
        assert m["max_consecutive_losses"] == 0

    def test_max_consecutive_losses(self):
        trades = [
            _make_trade(10.0),
            _make_trade(-5.0),
            _make_trade(-5.0),
            _make_trade(-5.0),
            _make_trade(10.0),
        ]
        m = compute_all_metrics(trades, 1000.0)
        assert m["max_consecutive_losses"] == 3


class TestMonteCarlo:
    def test_basic_monte_carlo(self):
        trades = [_make_trade(10.0) for _ in range(20)] + [_make_trade(-5.0) for _ in range(10)]
        result = run_monte_carlo(trades, 1000.0, num_simulations=100)
        assert "max_dd_95th_pct" in result
        assert result["max_dd_95th_pct"] >= 0
        assert result["final_equity_median"] > 0

    def test_empty_trades(self):
        result = run_monte_carlo([], 1000.0)
        assert "error" in result
