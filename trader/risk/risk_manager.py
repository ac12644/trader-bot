from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import structlog

from trader.config.settings import RiskConfig, ScalingPhase
from trader.models.types import EquityState, ExitReason, Position, Signal, Trade

logger = structlog.get_logger()


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        scaling_phases: list[ScalingPhase],
        starting_capital: float,
        leverage: int,
        max_entries_per_symbol_per_day: int = 2,
        max_total_daily_entries: int = 4,
    ):
        self.config = config
        self.scaling_phases = scaling_phases
        self.starting_capital = starting_capital
        self.leverage = leverage
        self.max_entries_per_symbol = max_entries_per_symbol_per_day
        self.max_daily_entries = max_total_daily_entries

    def get_current_phase(self, state: EquityState) -> ScalingPhase:
        current = self.scaling_phases[0]
        for phase in self.scaling_phases:
            if state.total_trades < phase.min_trades:
                break
            if self._phase_conditions_met(phase, state):
                current = phase

        # Drawdown demotion
        dd = self._current_drawdown_pct(state)
        if dd > self.config.drawdown_reduction_threshold_pct:
            idx = self.scaling_phases.index(current)
            if idx > 0:
                current = self.scaling_phases[idx - 1]
        return current

    def get_risk_per_trade_pct(self, state: EquityState) -> float:
        dd = self._current_drawdown_pct(state)
        if dd > self.config.drawdown_halt_threshold_pct:
            return 0.0
        if dd > 20.0:
            return 0.5
        if dd > self.config.drawdown_reduction_threshold_pct:
            return 1.0
        return self.get_current_phase(state).risk_pct

    def validate_signal(
        self,
        signal: Signal,
        state: EquityState,
        current_time: datetime,
    ) -> tuple[bool, str]:
        if state.is_daily_halted:
            return False, "daily_loss_cap_hit"

        if state.is_weekly_halted:
            return False, "weekly_loss_cap_hit"

        if state.cooldown_until and current_time < state.cooldown_until:
            return False, "cooldown_active"

        if self._current_drawdown_pct(state) > self.config.drawdown_halt_threshold_pct:
            return False, "drawdown_halt"

        if len(state.open_positions) >= self.config.max_concurrent_positions:
            return False, "max_concurrent_positions"

        symbol_entries = state.daily_entries_by_symbol.get(signal.symbol, 0)
        if symbol_entries >= self.max_entries_per_symbol:
            return False, f"max_daily_entries_{signal.symbol}"

        if state.total_daily_entries >= self.max_daily_entries:
            return False, "max_total_daily_entries"

        # Check same-symbol position already open
        for p in state.open_positions:
            if p.symbol == signal.symbol:
                return False, "position_already_open_for_symbol"

        current_open_risk = sum(
            self._position_risk_pct(p, state.equity) for p in state.open_positions
        )
        new_risk = self.get_risk_per_trade_pct(state)
        if new_risk <= 0:
            return False, "risk_is_zero"
        if current_open_risk + new_risk > self.config.max_total_open_risk_pct:
            return False, "max_total_open_risk"

        return True, "pass"

    def calculate_position_size(
        self,
        signal: Signal,
        state: EquityState,
        fee_rate: float,
        slippage_rate: float,
    ) -> float:
        risk_pct = self.get_risk_per_trade_pct(state)
        if risk_pct <= 0:
            return 0.0

        risk_budget = state.equity * (risk_pct / 100.0)
        stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

        if stop_distance_pct <= 0:
            return 0.0

        # First pass size
        position_size = risk_budget / stop_distance_pct

        # Subtract expected costs
        fee_cost = position_size * fee_rate * 2
        slippage_cost = position_size * slippage_rate
        effective_risk = risk_budget - fee_cost - slippage_cost

        if effective_risk <= 0:
            return 0.0

        position_size = effective_risk / stop_distance_pct

        # Cap by max exposure
        max_exposure = state.equity * (self.config.max_total_exposure_pct / 100.0)
        current_exposure = sum(p.position_size_usd for p in state.open_positions)
        max_new = max_exposure - current_exposure
        position_size = min(position_size, max(max_new, 0.0))

        # Cap by leverage
        max_leveraged = state.equity * self.leverage
        position_size = min(position_size, max_leveraged)

        return max(position_size, 0.0)

    def check_daily_weekly_limits(self, state: EquityState) -> None:
        if state.daily_pnl < 0:
            daily_loss_pct = abs(state.daily_pnl) / state.peak_equity * 100
            if daily_loss_pct >= self.config.daily_loss_cap_pct:
                state.is_daily_halted = True
                logger.warning("daily_loss_cap_hit", pnl=f"{state.daily_pnl:.2f}")

        if state.weekly_pnl < 0:
            weekly_loss_pct = abs(state.weekly_pnl) / state.peak_equity * 100
            if weekly_loss_pct >= self.config.weekly_loss_cap_pct:
                state.is_weekly_halted = True
                logger.warning("weekly_loss_cap_hit", pnl=f"{state.weekly_pnl:.2f}")

    def check_consecutive_losses(self, state: EquityState, current_time: datetime) -> None:
        if state.consecutive_losses >= self.config.consecutive_loss_cooldown_count:
            state.cooldown_until = current_time + timedelta(
                hours=self.config.consecutive_loss_cooldown_hours,
            )
            logger.info(
                "cooldown_activated",
                consecutive_losses=state.consecutive_losses,
                until=state.cooldown_until.isoformat(),
            )
        if state.consecutive_losses >= 5:
            state.is_daily_halted = True

    def _current_drawdown_pct(self, state: EquityState) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return ((state.peak_equity - state.equity) / state.peak_equity) * 100

    def _position_risk_pct(self, pos: Position, equity: float) -> float:
        if equity <= 0:
            return 100.0
        stop_dist = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
        return (pos.position_size_usd * stop_dist / equity) * 100

    def _phase_conditions_met(self, phase: ScalingPhase, state: EquityState) -> bool:
        if phase.min_profit_factor is not None:
            pf = self._calc_profit_factor(state.trades)
            if pf < phase.min_profit_factor:
                return False
        if phase.max_drawdown_pct is not None:
            dd = self._current_drawdown_pct(state)
            if dd > phase.max_drawdown_pct:
                return False
        if phase.min_sharpe is not None:
            sharpe = self._calc_sharpe(state.trades)
            if sharpe < phase.min_sharpe:
                return False
        return True

    def _calc_profit_factor(self, trades: list[Trade]) -> float:
        if not trades:
            return 0.0
        wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd <= 0))
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    def _calc_sharpe(self, trades: list[Trade]) -> float:
        if len(trades) < 2:
            return 0.0
        pnls = np.array([t.pnl_usd for t in trades])
        std = np.std(pnls)
        if std == 0:
            return 0.0
        return float(np.mean(pnls) / std * np.sqrt(365))
