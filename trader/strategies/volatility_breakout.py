from __future__ import annotations

import numpy as np

from trader.config.settings import BreakoutStrategyConfig
from trader.indicators import technical
from trader.models.types import Candle, Direction, Signal
from trader.strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Pullback-to-breakout strategy.

    Instead of entering on the breakout candle (which gets faked out constantly
    on 5m data), this strategy:
    1. Detects breakouts on the 1h timeframe (close > 20-period highest close)
    2. Waits for a pullback on 5m toward the breakout level
    3. Enters when momentum resumes in the breakout direction

    This gives better entries, tighter stops, and filters fake breakouts.
    """

    def __init__(self, config: BreakoutStrategyConfig):
        self.lookback_1h = config.pullback_1h_lookback
        self.vol_mult = config.volume_multiplier
        self.stop_atr_mult = config.stop_atr_multiplier
        self.tp_r = config.take_profit_r
        self.pullback_max_bars = config.pullback_max_bars
        self.retrace_pct = config.pullback_min_retrace_pct / 100.0

    @property
    def name(self) -> str:
        return "pullback_breakout"

    def evaluate(
        self,
        candle_5m: Candle,
        idx: int,
        highs_5m: np.ndarray,
        lows_5m: np.ndarray,
        closes_5m: np.ndarray,
        indicators_5m: dict[str, np.ndarray],
        indicators_1h: dict[str, np.ndarray],
        current_1h_index: int,
    ) -> Signal | None:
        h_idx = current_1h_index
        if h_idx < self.lookback_1h + 1:
            return None
        if idx < self.pullback_max_bars:
            return None

        # --- 1h trend filter ---
        ema_fast_1h = indicators_1h["ema_fast"][h_idx]
        ema_slow_1h = indicators_1h["ema_slow"][h_idx]
        slope_1h = indicators_1h["ema_fast_slope"][h_idx]
        if np.isnan(ema_fast_1h) or np.isnan(ema_slow_1h) or np.isnan(slope_1h):
            return None

        # --- Detect 1h breakout (within last 3 closed 1h bars) ---
        closes_1h = indicators_1h["close"]
        highs_1h = indicators_1h["high"]
        lows_1h = indicators_1h["low"]

        signal = self._check_long(
            candle_5m, idx, highs_5m, lows_5m, closes_5m, indicators_5m,
            h_idx, closes_1h, highs_1h, lows_1h, ema_fast_1h, ema_slow_1h, slope_1h,
        )
        if signal is not None:
            return signal

        return self._check_short(
            candle_5m, idx, highs_5m, lows_5m, closes_5m, indicators_5m,
            h_idx, closes_1h, highs_1h, lows_1h, ema_fast_1h, ema_slow_1h, slope_1h,
        )

    def _check_long(
        self, candle, idx, highs_5m, lows_5m, closes_5m, ind5m,
        h_idx, closes_1h, highs_1h, lows_1h, ema_f, ema_s, slope,
    ) -> Signal | None:
        if not (ema_f > ema_s and slope > 0):
            return None

        # Check if any of the last 3 closed 1h candles broke above channel
        breakout_level = None
        for offset in range(1, 4):  # check h_idx-0, h_idx-1, h_idx-2 (last 3 closed)
            bi = h_idx - offset + 1
            if bi < self.lookback_1h:
                continue
            channel_high = float(np.max(closes_1h[bi - self.lookback_1h : bi]))
            if closes_1h[bi] > channel_high:
                breakout_level = channel_high
                break

        if breakout_level is None:
            return None

        # --- 5m pullback detection ---
        close = candle.close
        atr_val = ind5m["atr"][idx]
        ema_fast_5m = ind5m["ema_fast"][idx]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ema_fast_5m):
            return None

        # Price must have pulled back toward the breakout level
        # Find the recent 5m swing high (peak after breakout)
        lookback_5m = min(self.pullback_max_bars, idx)
        recent_high = float(np.max(highs_5m[idx - lookback_5m : idx]))
        breakout_move = recent_high - breakout_level

        if breakout_move <= 0:
            return None

        # Current price must be below the swing high (pulled back)
        retrace = recent_high - close
        retrace_ratio = retrace / breakout_move

        if retrace_ratio < self.retrace_pct:
            return None  # hasn't pulled back enough

        # Must still be above the breakout level (not a failed breakout)
        if close <= breakout_level:
            return None

        # Momentum resumption: close > open AND close > 5m EMA AND bullish candle
        if candle.close <= candle.open:
            return None
        if close <= ema_fast_5m:
            return None

        # Candle body strength
        bar_range = candle.high - candle.low
        if bar_range <= 0:
            return None
        body_ratio = (close - candle.low) / bar_range
        if body_ratio < 0.5:
            return None

        # Volume confirmation
        vol_ratio = ind5m["volume_ratio"][idx]
        if np.isnan(vol_ratio) or vol_ratio < self.vol_mult:
            return None

        # --- Entry ---
        # Stop below pullback swing low with ATR buffer (structure-based stop)
        pullback_low = float(np.min(lows_5m[idx - lookback_5m : idx + 1]))
        stop = pullback_low - 0.3 * atr_val

        stop_distance = close - stop
        if stop_distance <= 0 or stop_distance < 0.5 * atr_val:
            return None  # skip if stop too tight or invalid

        tp = close + stop_distance * self.tp_r

        return Signal(
            symbol=candle.symbol,
            direction=Direction.LONG,
            entry_price=close,
            stop_loss=stop,
            take_profit=tp,
            atr=atr_val,
            timestamp=candle.timestamp,
            strategy_name=self.name,
        )

    def _check_short(
        self, candle, idx, highs_5m, lows_5m, closes_5m, ind5m,
        h_idx, closes_1h, highs_1h, lows_1h, ema_f, ema_s, slope,
    ) -> Signal | None:
        if not (ema_f < ema_s and slope < 0):
            return None

        # Check if any of the last 3 closed 1h candles broke below channel
        breakout_level = None
        for offset in range(1, 4):
            bi = h_idx - offset + 1
            if bi < self.lookback_1h:
                continue
            channel_low = float(np.min(closes_1h[bi - self.lookback_1h : bi]))
            if closes_1h[bi] < channel_low:
                breakout_level = channel_low
                break

        if breakout_level is None:
            return None

        # --- 5m pullback detection ---
        close = candle.close
        atr_val = ind5m["atr"][idx]
        ema_fast_5m = ind5m["ema_fast"][idx]
        if np.isnan(atr_val) or atr_val <= 0 or np.isnan(ema_fast_5m):
            return None

        lookback_5m = min(self.pullback_max_bars, idx)
        recent_low = float(np.min(lows_5m[idx - lookback_5m : idx]))
        breakout_move = breakout_level - recent_low

        if breakout_move <= 0:
            return None

        retrace = close - recent_low
        retrace_ratio = retrace / breakout_move

        if retrace_ratio < self.retrace_pct:
            return None

        if close >= breakout_level:
            return None

        # Momentum resumption: close < open AND close < 5m EMA AND bearish candle
        if candle.close >= candle.open:
            return None
        if close >= ema_fast_5m:
            return None

        bar_range = candle.high - candle.low
        if bar_range <= 0:
            return None
        body_ratio = (candle.high - close) / bar_range
        if body_ratio < 0.5:
            return None

        vol_ratio = ind5m["volume_ratio"][idx]
        if np.isnan(vol_ratio) or vol_ratio < self.vol_mult:
            return None

        # --- Entry ---
        # Stop above pullback swing high with ATR buffer (structure-based stop)
        pullback_high = float(np.max(highs_5m[idx - lookback_5m : idx + 1]))
        stop = pullback_high + 0.3 * atr_val

        stop_distance = stop - close
        if stop_distance <= 0 or stop_distance < 0.5 * atr_val:
            return None  # skip if stop too tight or invalid

        tp = close - stop_distance * self.tp_r

        return Signal(
            symbol=candle.symbol,
            direction=Direction.SHORT,
            entry_price=close,
            stop_loss=stop,
            take_profit=tp,
            atr=atr_val,
            timestamp=candle.timestamp,
            strategy_name=self.name,
        )
