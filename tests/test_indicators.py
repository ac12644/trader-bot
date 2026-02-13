import numpy as np
import pytest

from trader.indicators import technical


class TestEMA:
    def test_basic_ema(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = technical.ema(values, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)  # SMA of first 3
        assert result[3] > result[2]  # Rising values = rising EMA

    def test_ema_short_array(self):
        values = np.array([1.0, 2.0])
        result = technical.ema(values, 5)
        assert all(np.isnan(result))


class TestATR:
    def test_basic_atr(self):
        n = 30
        high = np.linspace(102, 110, n)
        low = np.linspace(98, 90, n)
        close = np.linspace(100, 100, n)
        result = technical.atr(high, low, close, 14)
        assert np.isnan(result[0])
        assert not np.isnan(result[15])
        assert result[15] > 0

    def test_atr_constant_prices(self):
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)
        result = technical.atr(high, low, close, 14)
        # All ranges are 0, so ATR should converge to 0
        valid = result[~np.isnan(result)]
        assert all(v == pytest.approx(0.0, abs=1e-10) for v in valid)


class TestADX:
    def test_adx_trending(self):
        n = 60
        # Strong uptrend
        base = np.linspace(100, 200, n)
        high = base + 2
        low = base - 2
        close = base
        result = technical.adx(high, low, close, 14)
        # ADX should be high for strong trend
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] > 20  # Strong trend = ADX > 20


class TestBollingerBands:
    def test_basic_bands(self):
        close = np.array([10.0] * 20 + [10.0, 11.0, 9.0, 10.5, 10.2])
        upper, middle, lower = technical.bollinger_bands(close, 20, 2.0)
        assert not np.isnan(middle[19])
        assert upper[19] >= middle[19]
        assert lower[19] <= middle[19]


class TestVolumeRatio:
    def test_spike_detection(self):
        vol = np.array([100.0] * 25 + [300.0])
        result = technical.volume_ratio(vol, 20)
        assert result[-1] == pytest.approx(3.0)

    def test_normal_volume(self):
        vol = np.array([100.0] * 25)
        result = technical.volume_ratio(vol, 20)
        assert result[-1] == pytest.approx(1.0)


class TestHighestHighLowestLow:
    def test_highest_high(self):
        high = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        # highest_high(arr, 4, 3) -> arr[max(0,4-3):4] = arr[1:4] = [5,3,2] -> 5.0
        assert technical.highest_high(high, 4, 3) == 5.0
        assert technical.highest_high(high, 4, 4) == 5.0

    def test_lowest_low(self):
        low = np.array([5.0, 1.0, 3.0, 4.0, 2.0])
        assert technical.lowest_low(low, 4, 3) == 1.0
