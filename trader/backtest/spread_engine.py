from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import structlog

from trader.backtest.cost_model import SpreadCostModel
from trader.models.spread_types import (
    SpreadBar,
    SpreadEquityState,
    SpreadExitReason,
    SpreadLeg,
    SpreadPosition,
    SpreadTrade,
)
from trader.models.types import Direction
from trader.strategies.spread_base import BaseSpreadStrategy

logger = structlog.get_logger()

# Funding payment hours (UTC): 00, 08, 16
_FUNDING_HOURS = {0, 8, 16}


class SpreadEngine:
    """Backtest engine for spread / pair-trading strategies on 1h candles."""

    def __init__(
        self,
        strategy: BaseSpreadStrategy,
        candles_a: pl.DataFrame,
        candles_b: pl.DataFrame,
        symbol_a: str,
        symbol_b: str,
        cost_model: SpreadCostModel,
        starting_capital: float,
        funding_rates: pl.DataFrame | None = None,
        lookback_period: int = 168,
        leg_b_is_spot: bool = False,
    ):
        self.strategy = strategy
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.cost_model = cost_model
        self.starting_capital = starting_capital
        self.lookback = lookback_period
        self.leg_b_is_spot = leg_b_is_spot

        # Build synchronized timeline
        self.timestamps, self.close_a, self.close_b = self._build_timeline(
            candles_a, candles_b,
        )
        self.high_a, self.low_a, self.vol_a = self._extract_hlv(candles_a, self.timestamps)
        self.high_b, self.low_b, self.vol_b = self._extract_hlv(candles_b, self.timestamps)
        self.n_bars = len(self.timestamps)

        # Pre-compute ratio indicators (for mean-reversion)
        self.ratio = self.close_a / self.close_b
        self.ratio_zscore = self._rolling_zscore(self.ratio, self.lookback)

        # Funding rate lookup: timestamp → rate
        self.funding_lookup: dict[datetime, float] = {}
        self.mark_price_lookup: dict[datetime, float] = {}
        if funding_rates is not None:
            self._load_funding(funding_rates)

    # ── Data preparation ─────────────────────────────────────────────

    @staticmethod
    def _build_timeline(
        candles_a: pl.DataFrame,
        candles_b: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inner-join two candle DataFrames on timestamp."""
        joined = candles_a.select(
            pl.col("timestamp").alias("ts"),
            pl.col("close").alias("close_a"),
        ).join(
            candles_b.select(
                pl.col("timestamp").alias("ts"),
                pl.col("close").alias("close_b"),
            ),
            on="ts",
            how="inner",
        ).sort("ts")

        ts = joined["ts"].to_numpy()
        ca = joined["close_a"].to_numpy().astype(np.float64)
        cb = joined["close_b"].to_numpy().astype(np.float64)
        return ts, ca, cb

    @staticmethod
    def _extract_hlv(
        candles: pl.DataFrame,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract high/low/volume aligned to the synchronized timestamps."""
        ts_set = set(timestamps)
        mask = candles["timestamp"].to_numpy()
        indices = [i for i, t in enumerate(mask) if t in ts_set]
        h = candles["high"].to_numpy().astype(np.float64)[indices]
        lo = candles["low"].to_numpy().astype(np.float64)[indices]
        v = candles["volume"].to_numpy().astype(np.float64)[indices]
        return h, lo, v

    @staticmethod
    def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling Z-score of a 1-D array."""
        n = len(arr)
        zscores = np.full(n, np.nan)
        for i in range(window, n):
            segment = arr[i - window : i]
            mu = np.mean(segment)
            sigma = np.std(segment)
            if sigma > 0:
                zscores[i] = (arr[i] - mu) / sigma
            else:
                zscores[i] = 0.0
        return zscores

    def _load_funding(self, df: pl.DataFrame) -> None:
        """Build timestamp → funding_rate and mark_price lookups."""
        for row in df.iter_rows(named=True):
            ts = row["timestamp"]
            if isinstance(ts, datetime):
                key = ts
            else:
                key = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            self.funding_lookup[key] = row["funding_rate"]
            self.mark_price_lookup[key] = row.get("mark_price", 0.0)

    # ── Helpers ──────────────────────────────────────────────────────

    def _ts_to_datetime(self, ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, np.datetime64):
            epoch = np.datetime64(0, "us")
            one_us = np.timedelta64(1, "us")
            us = int((ts - epoch) / one_us)
            return datetime(1970, 1, 1) + timedelta(microseconds=us)
        return datetime.fromisoformat(str(ts))

    def _get_funding_rate(self, dt: datetime) -> float:
        """Get funding rate at this timestamp. Try exact match, then nearest 8h mark."""
        rate = self.funding_lookup.get(dt)
        if rate is not None:
            return rate
        # Snap to nearest previous 8h boundary
        hour = (dt.hour // 8) * 8
        snap = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        return self.funding_lookup.get(snap, 0.0)

    def _is_funding_time(self, dt: datetime) -> bool:
        return dt.hour in _FUNDING_HOURS and dt.minute == 0

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> list[SpreadTrade]:
        state = SpreadEquityState(
            equity=self.starting_capital,
            peak_equity=self.starting_capital,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_losses=0,
        )

        log_interval = max(self.n_bars // 20, 1)

        for i in range(self.n_bars):
            if i % log_interval == 0:
                pct = i / self.n_bars * 100
                logger.info(
                    "spread_backtest_progress",
                    pct=f"{pct:.0f}%",
                    equity=f"{state.equity:.2f}",
                    trades=state.total_trades,
                )

            dt = self._ts_to_datetime(self.timestamps[i])
            self._check_day_week_reset(state, dt)

            bar = SpreadBar(
                timestamp=dt,
                symbol_a=self.symbol_a,
                close_a=float(self.close_a[i]),
                high_a=float(self.high_a[i]),
                low_a=float(self.low_a[i]),
                volume_a=float(self.vol_a[i]),
                symbol_b=self.symbol_b,
                close_b=float(self.close_b[i]),
                high_b=float(self.high_b[i]),
                low_b=float(self.low_b[i]),
                volume_b=float(self.vol_b[i]),
            )

            # 1. Apply funding payments at 8h marks
            if self._is_funding_time(dt) and state.open_positions:
                self._apply_funding(state, dt)

            # 2. Build indicators for this bar
            indicators = self._bar_indicators(i, dt)

            # 3. Check exits
            self._check_exits(state, bar, indicators)

            # 4. Check entries (one position at a time)
            has_open = len(state.open_positions) > 0
            signal = self.strategy.should_enter(bar, indicators, has_open)
            if signal is not None:
                self._open_position(state, signal, bar)

        # Close remaining at end of data
        if state.open_positions:
            last_dt = self._ts_to_datetime(self.timestamps[-1])
            last_bar = SpreadBar(
                timestamp=last_dt,
                symbol_a=self.symbol_a,
                close_a=float(self.close_a[-1]),
                high_a=float(self.high_a[-1]),
                low_a=float(self.low_a[-1]),
                volume_a=float(self.vol_a[-1]),
                symbol_b=self.symbol_b,
                close_b=float(self.close_b[-1]),
                high_b=float(self.high_b[-1]),
                low_b=float(self.low_b[-1]),
                volume_b=float(self.vol_b[-1]),
            )
            for pos in list(state.open_positions):
                self._close_position(state, pos, last_bar, SpreadExitReason.END_OF_DATA)
            state.open_positions.clear()

        logger.info(
            "spread_backtest_complete",
            strategy=self.strategy.name,
            total_trades=state.total_trades,
            final_equity=f"{state.equity:.2f}",
        )
        return state.trades

    # ── Indicator construction ───────────────────────────────────────

    def _bar_indicators(self, idx: int, dt: datetime) -> dict[str, float]:
        indicators: dict[str, float] = {}
        zscore = self.ratio_zscore[idx]
        if not np.isnan(zscore):
            indicators["ratio_zscore"] = float(zscore)
        indicators["funding_rate"] = self._get_funding_rate(dt)
        indicators["ratio"] = float(self.ratio[idx])
        return indicators

    # ── Funding ──────────────────────────────────────────────────────

    def _apply_funding(self, state: SpreadEquityState, dt: datetime) -> None:
        rate = self._get_funding_rate(dt)
        if rate == 0.0:
            return

        for pos in state.open_positions:
            funding_income = 0.0
            for leg in (pos.leg_a, pos.leg_b):
                if not leg.is_perp:
                    continue
                # Short perp + positive rate → receive funding
                # Long perp + positive rate → pay funding
                if leg.direction == Direction.SHORT:
                    income = leg.notional_usd * rate
                else:
                    income = -leg.notional_usd * rate
                leg.accumulated_funding += income
                funding_income += income

            pos.accumulated_funding_total += funding_income
            state.equity += funding_income

    # ── Position management ──────────────────────────────────────────

    def _open_position(
        self,
        state: SpreadEquityState,
        signal,
        bar: SpreadBar,
    ) -> None:
        # Apply entry slippage
        price_a = self.cost_model.apply_entry_slippage(signal.entry_price_a, signal.direction_a)
        price_b = self.cost_model.apply_entry_slippage(signal.entry_price_b, signal.direction_b)

        leg_a = SpreadLeg(
            symbol=signal.symbol_a,
            direction=signal.direction_a,
            entry_price=price_a,
            current_price=price_a,
            notional_usd=signal.notional_per_leg,
            is_perp=True,
        )
        leg_b = SpreadLeg(
            symbol=signal.symbol_b,
            direction=signal.direction_b,
            entry_price=price_b,
            current_price=price_b,
            notional_usd=signal.notional_per_leg,
            is_perp=not self.leg_b_is_spot,
        )

        entry_fees = self.cost_model.entry_fee_spread(
            signal.notional_per_leg,
            signal.notional_per_leg,
            leg_b_is_spot=self.leg_b_is_spot,
        )

        position = SpreadPosition(
            leg_a=leg_a,
            leg_b=leg_b,
            entry_time=bar.timestamp,
            strategy_name=signal.strategy_name,
            zscore_at_entry=signal.zscore_at_entry,
            funding_rate_at_entry=signal.funding_rate_at_entry,
            basis_at_entry=signal.basis_at_entry,
            entry_fees_total=entry_fees,
        )

        state.open_positions.append(position)
        state.equity -= entry_fees

    def _check_exits(
        self,
        state: SpreadEquityState,
        bar: SpreadBar,
        indicators: dict[str, float],
    ) -> None:
        closed: list[SpreadPosition] = []
        for pos in state.open_positions:
            reason = self.strategy.should_exit(bar, pos, indicators)
            if reason is not None:
                self._close_position(state, pos, bar, reason)
                closed.append(pos)
        for pos in closed:
            state.open_positions.remove(pos)

    def _close_position(
        self,
        state: SpreadEquityState,
        pos: SpreadPosition,
        bar: SpreadBar,
        reason: SpreadExitReason,
    ) -> None:
        # Apply exit slippage
        exit_a = self.cost_model.apply_exit_slippage(bar.close_a, pos.leg_a.direction)
        exit_b = self.cost_model.apply_exit_slippage(bar.close_b, pos.leg_b.direction)

        # Leg PnL
        pnl_a = self._leg_pnl(pos.leg_a, exit_a)
        pnl_b = self._leg_pnl(pos.leg_b, exit_b)

        # Exit fees
        exit_fees = self.cost_model.exit_fee_spread(
            pos.leg_a.notional_usd,
            pos.leg_b.notional_usd,
            leg_b_is_spot=self.leg_b_is_spot,
        )

        total_fees = pos.entry_fees_total + exit_fees
        pnl_total = pnl_a + pnl_b + pos.accumulated_funding_total - total_fees

        holding_mins = int((bar.timestamp - pos.entry_time).total_seconds() / 60)
        basis_at_exit = bar.close_a - bar.close_b
        zscore_at_exit = 0.0
        # Try to get current zscore from latest computed value
        if hasattr(self, '_last_indicators'):
            zscore_at_exit = self._last_indicators.get("ratio_zscore", 0.0)

        trade = SpreadTrade(
            strategy_name=pos.strategy_name,
            symbol_a=pos.leg_a.symbol,
            symbol_b=pos.leg_b.symbol,
            direction_a=pos.leg_a.direction,
            direction_b=pos.leg_b.direction,
            entry_price_a=pos.leg_a.entry_price,
            entry_price_b=pos.leg_b.entry_price,
            exit_price_a=exit_a,
            exit_price_b=exit_b,
            notional_per_leg=pos.leg_a.notional_usd,
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            exit_reason=reason,
            pnl_leg_a=pnl_a,
            pnl_leg_b=pnl_b,
            pnl_funding=pos.accumulated_funding_total,
            pnl_total=pnl_total,
            fees_paid=total_fees,
            holding_duration_minutes=holding_mins,
            zscore_at_entry=pos.zscore_at_entry,
            zscore_at_exit=zscore_at_exit,
            basis_at_entry=pos.basis_at_entry,
            basis_at_exit=basis_at_exit,
        )

        state.trades.append(trade)
        state.equity += pnl_a + pnl_b - exit_fees  # funding already applied incrementally
        state.peak_equity = max(state.peak_equity, state.equity)
        state.daily_pnl += pnl_total
        state.weekly_pnl += pnl_total
        state.total_trades += 1

        if pnl_total > 0:
            state.winning_trades += 1
            state.consecutive_losses = 0
        else:
            state.losing_trades += 1
            state.consecutive_losses += 1

    @staticmethod
    def _leg_pnl(leg: SpreadLeg, exit_price: float) -> float:
        if leg.direction == Direction.LONG:
            return (exit_price - leg.entry_price) / leg.entry_price * leg.notional_usd
        return (leg.entry_price - exit_price) / leg.entry_price * leg.notional_usd

    # ── Day/week reset ───────────────────────────────────────────────

    @staticmethod
    def _check_day_week_reset(state: SpreadEquityState, dt: datetime) -> None:
        day_str = dt.strftime("%Y-%m-%d")
        if state.current_day != day_str:
            state.current_day = day_str
            state.daily_pnl = 0.0
            state.is_daily_halted = False

        week_str = (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
        if state.current_week_start != week_str:
            state.current_week_start = week_str
            state.weekly_pnl = 0.0
            state.is_weekly_halted = False
