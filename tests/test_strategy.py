from datetime import datetime

import numpy as np

from trader.config.settings import BreakoutStrategyConfig
from trader.models.types import Candle, Direction
from trader.strategies.volatility_breakout import VolatilityBreakoutStrategy


def _make_config(**overrides):
    defaults = dict(
        lookback_candles=20,
        volume_multiplier=1.5,
        atr_period=14,
        stop_atr_multiplier=1.5,
        take_profit_r=2.5,
        trailing_start_r=1.5,
        trailing_atr_multiplier=1.2,
        max_entries_per_symbol_per_day=2,
        pullback_1h_lookback=5,
        pullback_max_bars=12,
        pullback_min_retrace_pct=30,
        pullback_ema_period=9,
    )
    defaults.update(overrides)
    return BreakoutStrategyConfig(**defaults)


class TestPullbackBreakout:
    def _setup_data(self, n=50, trend_up=True):
        """Create synthetic data simulating a 1h breakout followed by 5m pullback."""
        # 1h data: 20 bars of channel then a breakout
        n_1h = 30
        if trend_up:
            closes_1h = np.full(n_1h, 100.0)
            closes_1h[-3] = 105.0  # breakout 3 bars ago
            closes_1h[-2] = 104.0
            closes_1h[-1] = 103.0
        else:
            closes_1h = np.full(n_1h, 100.0)
            closes_1h[-3] = 95.0  # breakdown
            closes_1h[-2] = 96.0
            closes_1h[-1] = 97.0

        highs_1h = closes_1h + 1.0
        lows_1h = closes_1h - 1.0

        # 1h indicators
        ema_fast = np.full(n_1h, 103.0 if trend_up else 97.0)
        ema_slow = np.full(n_1h, 100.0)
        slope = np.full(n_1h, 1.0 if trend_up else -1.0)
        adx = np.full(n_1h, 30.0)
        atr_1h = np.full(n_1h, 2.0)
        vol_1h = np.full(n_1h, 1000.0)

        indicators_1h = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_fast_slope": slope,
            "adx": adx,
            "atr": atr_1h,
            "close": closes_1h,
            "high": highs_1h,
            "low": lows_1h,
            "volume": vol_1h,
        }

        # 5m data: price went up after breakout, then pulled back, now resuming
        from trader.indicators import technical

        if trend_up:
            # Simulate: price rallied to ~104.5 (within 12-bar window), pulled back, now bouncing
            base = np.full(n, 100.0)
            base[-20:] = 104.0  # post-breakout area
            base[-6:] = 102.0   # pullback
            base[-3:] = [101.5, 102.0, 103.0]  # resumption
            highs_5m = base + 0.5
            lows_5m = base - 0.5
            lows_5m[-4] = 101.0  # pullback low
        else:
            base = np.full(n, 100.0)
            base[-20:] = 96.0
            base[-6:] = 98.0    # pullback up
            base[-3:] = [98.5, 98.0, 97.0]  # resumption down
            highs_5m = base + 0.5
            highs_5m[-4] = 99.0  # pullback high
            lows_5m = base - 0.5

        closes_5m = base.copy()
        if trend_up:
            closes_5m[-1] = 103.0  # bullish close
        else:
            closes_5m[-1] = 97.0

        volumes = np.full(n, 100.0)
        volumes[-1] = 200.0  # volume spike

        atr_vals = np.full(n, 1.5)  # constant ATR for simplicity
        vol_ratio = technical.volume_ratio(volumes, 5)
        ema_fast_5m = technical.ema(closes_5m, 9)

        indicators_5m = {
            "atr": atr_vals,
            "volume_ratio": vol_ratio,
            "ema_fast": ema_fast_5m,
        }

        return highs_5m, lows_5m, closes_5m, indicators_5m, indicators_1h

    def test_long_pullback_signal(self):
        strategy = VolatilityBreakoutStrategy(_make_config())
        highs, lows, closes, ind5, ind1 = self._setup_data(trend_up=True)
        idx = len(closes) - 1

        candle = Candle(
            timestamp=datetime(2024, 6, 1, 12),
            open=102.5,
            high=float(highs[idx]),
            low=float(lows[idx]),
            close=float(closes[idx]),
            volume=200.0,
            symbol="BTC/USDT",
            timeframe="5m",
        )

        signal = strategy.evaluate(candle, idx, highs, lows, closes, ind5, ind1, len(ind1["close"]) - 1)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price

    def test_short_pullback_signal(self):
        strategy = VolatilityBreakoutStrategy(_make_config())
        highs, lows, closes, ind5, ind1 = self._setup_data(trend_up=False)
        idx = len(closes) - 1

        candle = Candle(
            timestamp=datetime(2024, 6, 1, 12),
            open=97.5,
            high=float(highs[idx]),
            low=float(lows[idx]),
            close=float(closes[idx]),
            volume=200.0,
            symbol="BTC/USDT",
            timeframe="5m",
        )

        signal = strategy.evaluate(candle, idx, highs, lows, closes, ind5, ind1, len(ind1["close"]) - 1)
        assert signal is not None
        assert signal.direction == Direction.SHORT

    def test_no_signal_without_volume(self):
        strategy = VolatilityBreakoutStrategy(_make_config())
        highs, lows, closes, ind5, ind1 = self._setup_data(trend_up=True)
        ind5["volume_ratio"][-1] = 1.0  # below threshold
        idx = len(closes) - 1

        candle = Candle(
            timestamp=datetime(2024, 6, 1, 12),
            open=102.5,
            high=float(highs[idx]),
            low=float(lows[idx]),
            close=float(closes[idx]),
            volume=100.0,
            symbol="BTC/USDT",
            timeframe="5m",
        )

        signal = strategy.evaluate(candle, idx, highs, lows, closes, ind5, ind1, len(ind1["close"]) - 1)
        assert signal is None

    def test_no_signal_insufficient_data(self):
        strategy = VolatilityBreakoutStrategy(_make_config(pullback_1h_lookback=20))
        n = 5
        highs = np.array([100.0] * n)
        lows = np.array([99.0] * n)
        closes = np.array([99.5] * n)
        ind5 = {
            "atr": np.full(n, 1.0),
            "volume_ratio": np.full(n, 2.0),
            "ema_fast": np.full(n, 99.5),
        }
        ind1 = {
            "ema_fast": np.full(n, 100.0),
            "ema_slow": np.full(n, 99.0),
            "ema_fast_slope": np.full(n, 1.0),
            "adx": np.full(n, 30.0),
            "atr": np.full(n, 5.0),
            "close": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "volume": np.full(n, 1000.0),
        }

        candle = Candle(
            timestamp=datetime(2024, 1, 1),
            open=99.0, high=100.0, low=99.0, close=99.5,
            volume=100.0, symbol="BTC/USDT", timeframe="5m",
        )

        signal = strategy.evaluate(candle, 4, highs, lows, closes, ind5, ind1, 4)
        assert signal is None
