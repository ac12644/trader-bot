from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from trader.backtest.cost_model import CostModel
from trader.config.settings import Settings
from trader.filters.chop_filter import ChopFilter
from trader.filters.correlation_filter import CorrelationFilter
from trader.filters.event_filter import EventFilter
from trader.filters.session_filter import SessionFilter
from trader.indicators import technical
from trader.models.types import (
    Candle,
    Direction,
    EquityState,
    ExitReason,
    Position,
    Trade,
)
from trader.risk.risk_manager import RiskManager
from trader.strategies.base import BaseStrategy

logger = structlog.get_logger()


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        strategy: BaseStrategy,
        candles_5m: dict[str, pl.DataFrame],
        candles_1h: dict[str, pl.DataFrame],
        events_calendar_path: Path | None = None,
    ):
        self.settings = settings
        self.strategy = strategy
        self.cost_model = CostModel(settings.backtest)
        self.risk_manager = RiskManager(
            config=settings.risk,
            scaling_phases=settings.scaling_phases,
            starting_capital=settings.starting_capital,
            leverage=settings.leverage,
            max_entries_per_symbol_per_day=settings.breakout.max_entries_per_symbol_per_day,
            max_total_daily_entries=settings.execution.max_total_daily_entries,
        )
        self.chop_filter = ChopFilter(settings.filters.chop)
        self.session_filter = SessionFilter(settings.filters.session_blackouts)
        self.event_filter = EventFilter(
            events_calendar_path or Path("events_calendar.yaml"),
            settings.filters.event,
        )
        self.correlation_filter = CorrelationFilter(
            settings.risk.max_same_direction_correlated,
        )

        self.max_holding_minutes = settings.execution.max_holding_hours * 60
        self.stale_minutes = settings.execution.stale_position_hours * 60
        self.stale_min_r = settings.execution.stale_position_min_r
        self.trailing_start_r = settings.breakout.trailing_start_r
        self.trailing_atr_mult = settings.breakout.trailing_atr_multiplier
        self.atr_window_bars_1h = settings.filters.chop.atr_percentile_window_days * 24

        # Store raw data as numpy arrays for fast access
        self.data_5m: dict[str, dict[str, np.ndarray]] = {}
        self.data_1h: dict[str, dict[str, np.ndarray]] = {}
        self.timestamps_5m: dict[str, np.ndarray] = {}
        self.timestamps_1h: dict[str, np.ndarray] = {}
        self.indicators_5m: dict[str, dict[str, np.ndarray]] = {}
        self.indicators_1h: dict[str, dict[str, np.ndarray]] = {}

        self._prepare_data(candles_5m, candles_1h)
        self._compute_indicators()

    def _prepare_data(
        self,
        candles_5m: dict[str, pl.DataFrame],
        candles_1h: dict[str, pl.DataFrame],
    ) -> None:
        for symbol in self.settings.symbols:
            df5 = candles_5m[symbol]
            self.timestamps_5m[symbol] = df5["timestamp"].to_numpy()
            self.data_5m[symbol] = {
                "open": df5["open"].to_numpy().astype(np.float64),
                "high": df5["high"].to_numpy().astype(np.float64),
                "low": df5["low"].to_numpy().astype(np.float64),
                "close": df5["close"].to_numpy().astype(np.float64),
                "volume": df5["volume"].to_numpy().astype(np.float64),
            }

            df1 = candles_1h[symbol]
            self.timestamps_1h[symbol] = df1["timestamp"].to_numpy()
            self.data_1h[symbol] = {
                "open": df1["open"].to_numpy().astype(np.float64),
                "high": df1["high"].to_numpy().astype(np.float64),
                "low": df1["low"].to_numpy().astype(np.float64),
                "close": df1["close"].to_numpy().astype(np.float64),
                "volume": df1["volume"].to_numpy().astype(np.float64),
            }

    def _compute_indicators(self) -> None:
        pb_ema_period = self.settings.breakout.pullback_ema_period
        for symbol in self.settings.symbols:
            d5 = self.data_5m[symbol]
            self.indicators_5m[symbol] = {
                "atr": technical.atr(d5["high"], d5["low"], d5["close"], 14),
                "volume_ratio": technical.volume_ratio(d5["volume"], 20),
                "ema_fast": technical.ema(d5["close"], pb_ema_period),
            }

            d1 = self.data_1h[symbol]
            ema_f = technical.ema(d1["close"], self.settings.filters.trend.ema_fast)
            ema_s = technical.ema(d1["close"], self.settings.filters.trend.ema_slow)

            self.indicators_1h[symbol] = {
                "ema_fast": ema_f,
                "ema_slow": ema_s,
                "ema_fast_slope": technical.ema_slope(ema_f),
                "adx": technical.adx(d1["high"], d1["low"], d1["close"], 14),
                "atr": technical.atr(d1["high"], d1["low"], d1["close"], 14),
                "close": d1["close"],
                "high": d1["high"],
                "low": d1["low"],
                "volume": d1["volume"],
            }

    def _find_1h_index(self, symbol: str, ts_5m: np.datetime64) -> int:
        timestamps_1h = self.timestamps_1h[symbol]
        idx = np.searchsorted(timestamps_1h, ts_5m, side="right") - 1
        if idx < 0:
            return -1
        return int(idx)

    def _ts_to_datetime(self, ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, np.datetime64):
            # Convert numpy datetime64 to Python datetime
            epoch = np.datetime64(0, "us")
            one_us = np.timedelta64(1, "us")
            us = int((ts - epoch) / one_us)
            return datetime(1970, 1, 1) + timedelta(microseconds=us)
        # polars datetime or other
        return datetime.fromisoformat(str(ts))

    def run(self) -> list[Trade]:
        state = EquityState(
            equity=self.settings.starting_capital,
            peak_equity=self.settings.starting_capital,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_losses=0,
        )

        # Build unified timeline: (timestamp, symbol, bar_index)
        timeline: list[tuple[np.datetime64, str, int]] = []
        for symbol in self.settings.symbols:
            ts_arr = self.timestamps_5m[symbol]
            for i in range(len(ts_arr)):
                timeline.append((ts_arr[i], symbol, i))
        timeline.sort(key=lambda x: x[0])

        total_bars = len(timeline)
        log_interval = max(total_bars // 20, 1)

        for bar_num, (ts, symbol, idx) in enumerate(timeline):
            if bar_num % log_interval == 0:
                pct = bar_num / total_bars * 100
                logger.info(
                    "backtest_progress",
                    pct=f"{pct:.0f}%",
                    equity=f"{state.equity:.2f}",
                    trades=state.total_trades,
                )

            current_time = self._ts_to_datetime(ts)
            self._check_day_week_reset(state, current_time)

            d5 = self.data_5m[symbol]
            candle = Candle(
                timestamp=current_time,
                open=float(d5["open"][idx]),
                high=float(d5["high"][idx]),
                low=float(d5["low"][idx]),
                close=float(d5["close"][idx]),
                volume=float(d5["volume"][idx]),
                symbol=symbol,
                timeframe="5m",
            )

            # 1. Manage open positions (check exits)
            self._manage_positions(state, candle, idx, symbol)

            # 2. Find 1h index
            h_idx = self._find_1h_index(symbol, ts)
            if h_idx < 0:
                continue

            # 3. Filters
            adx_val = self.indicators_1h[symbol]["adx"][h_idx]
            if np.isnan(adx_val):
                continue

            atr_pct = technical.atr_percentile(
                self.indicators_1h[symbol]["atr"],
                h_idx,
                self.atr_window_bars_1h,
            )

            chop_ok, _ = self.chop_filter.is_tradeable(adx_val, atr_pct, current_time)
            if not chop_ok:
                continue

            session_ok, _ = self.session_filter.is_tradeable(current_time)
            if not session_ok:
                continue

            event_ok, _ = self.event_filter.is_tradeable(current_time)
            if not event_ok:
                continue

            # 4. Strategy evaluation
            signal = self.strategy.evaluate(
                candle,
                idx,
                d5["high"],
                d5["low"],
                d5["close"],
                self.indicators_5m[symbol],
                self.indicators_1h[symbol],
                h_idx,
            )

            if signal is None:
                continue

            # 5. Correlation filter
            corr_ok, _ = self.correlation_filter.is_allowed(
                signal.symbol, signal.direction, state.open_positions,
            )
            if not corr_ok:
                continue

            # 6. Risk validation
            risk_ok, _ = self.risk_manager.validate_signal(signal, state, current_time)
            if not risk_ok:
                continue

            # 7. Position sizing
            size = self.risk_manager.calculate_position_size(
                signal,
                state,
                self.cost_model.fee_taker,
                self.cost_model.slip_entry,
            )
            if size <= 0:
                continue

            # 8. Open position
            actual_entry = self.cost_model.apply_entry_slippage(
                signal.entry_price, signal.direction,
            )
            entry_fee = self.cost_model.entry_fee(size)

            position = Position(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=actual_entry,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                trailing_stop=None,
                position_size_usd=size,
                entry_time=current_time,
                atr_at_entry=signal.atr,
                strategy_name=signal.strategy_name,
                highest_price_since_entry=actual_entry,
                lowest_price_since_entry=actual_entry,
            )

            state.open_positions.append(position)
            # Entry fee is accounted for in _close_position (included in trade PnL)
            state.daily_entries_by_symbol[symbol] = (
                state.daily_entries_by_symbol.get(symbol, 0) + 1
            )
            state.total_daily_entries += 1

        # Close remaining positions at end of data
        for pos in list(state.open_positions):
            last_idx = len(self.data_5m[pos.symbol]["close"]) - 1
            last_close = float(self.data_5m[pos.symbol]["close"][last_idx])
            last_ts = self._ts_to_datetime(self.timestamps_5m[pos.symbol][last_idx])
            self._close_position(state, pos, last_close, ExitReason.END_OF_DATA, last_ts)
        state.open_positions.clear()

        logger.info(
            "backtest_complete",
            total_trades=state.total_trades,
            final_equity=f"{state.equity:.2f}",
        )
        return state.trades

    def _manage_positions(
        self,
        state: EquityState,
        candle: Candle,
        idx: int,
        symbol: str,
    ) -> None:
        closed: list[Position] = []

        for pos in state.open_positions:
            if pos.symbol != symbol:
                continue

            exit_price: float | None = None
            exit_reason: ExitReason | None = None

            holding_mins = (candle.timestamp - pos.entry_time).total_seconds() / 60

            # Update price extremes
            pos.highest_price_since_entry = max(pos.highest_price_since_entry, candle.high)
            pos.lowest_price_since_entry = min(pos.lowest_price_since_entry, candle.low)

            # Trailing stop activation + update
            unrealized_r = self._calc_unrealized_r(pos, candle.close)
            if not pos.trailing_active and unrealized_r >= self.trailing_start_r:
                pos.trailing_active = True

            if pos.trailing_active:
                trail_dist = self.trailing_atr_mult * pos.atr_at_entry
                if pos.direction == Direction.LONG:
                    new_trail = pos.highest_price_since_entry - trail_dist
                    if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail
                else:
                    new_trail = pos.lowest_price_since_entry + trail_dist
                    if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                        pos.trailing_stop = new_trail

            # Stop loss check (conservative: check with high/low)
            if pos.direction == Direction.LONG:
                effective_stop = pos.stop_loss
                if pos.trailing_stop is not None:
                    effective_stop = max(effective_stop, pos.trailing_stop)
                if candle.low <= effective_stop:
                    exit_price = effective_stop
                    exit_reason = (
                        ExitReason.TRAILING_STOP if pos.trailing_active else ExitReason.STOP_LOSS
                    )
            else:
                effective_stop = pos.stop_loss
                if pos.trailing_stop is not None:
                    effective_stop = min(effective_stop, pos.trailing_stop)
                if candle.high >= effective_stop:
                    exit_price = effective_stop
                    exit_reason = (
                        ExitReason.TRAILING_STOP if pos.trailing_active else ExitReason.STOP_LOSS
                    )

            # Take profit check (only if stop not already hit — conservative)
            if exit_price is None:
                if pos.direction == Direction.LONG and candle.high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = ExitReason.TAKE_PROFIT
                elif pos.direction == Direction.SHORT and candle.low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = ExitReason.TAKE_PROFIT

            # Time exit
            if exit_price is None and holding_mins >= self.max_holding_minutes:
                exit_price = candle.close
                exit_reason = ExitReason.TIME_EXIT

            # Stale exit
            if exit_price is None and holding_mins >= self.stale_minutes:
                if unrealized_r < self.stale_min_r:
                    exit_price = candle.close
                    exit_reason = ExitReason.STALE_EXIT

            if exit_price is not None and exit_reason is not None:
                self._close_position(state, pos, exit_price, exit_reason, candle.timestamp)
                closed.append(pos)

        for pos in closed:
            state.open_positions.remove(pos)

    def _close_position(
        self,
        state: EquityState,
        pos: Position,
        raw_exit: float,
        reason: ExitReason,
        exit_time: datetime,
    ) -> None:
        is_stop = reason in (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP)
        actual_exit = self.cost_model.apply_exit_slippage(raw_exit, pos.direction)
        exit_fee = self.cost_model.exit_fee(pos.position_size_usd, is_stop=is_stop)
        entry_fee = self.cost_model.entry_fee(pos.position_size_usd)

        if pos.direction == Direction.LONG:
            pnl = (actual_exit - pos.entry_price) / pos.entry_price * pos.position_size_usd
        else:
            pnl = (pos.entry_price - actual_exit) / pos.entry_price * pos.position_size_usd

        pnl -= entry_fee + exit_fee

        risk_per_unit = abs(pos.entry_price - pos.stop_loss)
        if risk_per_unit > 0:
            price_move = actual_exit - pos.entry_price
            if pos.direction == Direction.SHORT:
                price_move = pos.entry_price - actual_exit
            r_multiple = price_move / risk_per_unit
        else:
            r_multiple = 0.0

        holding_mins = int((exit_time - pos.entry_time).total_seconds() / 60)

        trade = Trade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            position_size_usd=pos.position_size_usd,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            exit_reason=reason,
            pnl_usd=pnl,
            pnl_pct=(pnl / state.equity) * 100 if state.equity > 0 else 0.0,
            fees_paid=entry_fee + exit_fee,
            slippage_cost=0.0,
            r_multiple=r_multiple,
            strategy_name=pos.strategy_name,
            holding_duration_minutes=holding_mins,
        )

        state.trades.append(trade)
        state.equity += pnl
        state.peak_equity = max(state.peak_equity, state.equity)
        state.daily_pnl += pnl
        state.weekly_pnl += pnl
        state.total_trades += 1

        if pnl > 0:
            state.winning_trades += 1
            state.consecutive_losses = 0
            self.chop_filter.record_win()
        else:
            state.losing_trades += 1
            state.consecutive_losses += 1
            if reason == ExitReason.STOP_LOSS:
                self.chop_filter.record_stop_loss_hit(exit_time)

        self.risk_manager.check_daily_weekly_limits(state)
        self.risk_manager.check_consecutive_losses(state, exit_time)

    def _calc_unrealized_r(self, pos: Position, current_close: float) -> float:
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 0:
            return 0.0
        if pos.direction == Direction.LONG:
            return (current_close - pos.entry_price) / risk
        return (pos.entry_price - current_close) / risk

    def _check_day_week_reset(self, state: EquityState, current_time: datetime) -> None:
        day_str = current_time.strftime("%Y-%m-%d")
        if state.current_day != day_str:
            state.current_day = day_str
            state.daily_pnl = 0.0
            state.is_daily_halted = False
            state.daily_entries_by_symbol.clear()
            state.total_daily_entries = 0

        week_str = (current_time - timedelta(days=current_time.weekday())).strftime("%Y-%m-%d")
        if state.current_week_start != week_str:
            state.current_week_start = week_str
            state.weekly_pnl = 0.0
            state.is_weekly_halted = False
