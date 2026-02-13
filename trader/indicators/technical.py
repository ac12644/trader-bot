from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import percentileofscore


def ema(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) < period:
        return result
    alpha = 2.0 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    result = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return result
    result[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period * 2 + 1:
        return result

    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    smoothed_tr = np.zeros(n, dtype=np.float64)
    smoothed_plus = np.zeros(n, dtype=np.float64)
    smoothed_minus = np.zeros(n, dtype=np.float64)

    smoothed_tr[period] = np.sum(tr[1 : period + 1])
    smoothed_plus[period] = np.sum(plus_dm[1 : period + 1])
    smoothed_minus[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]
        smoothed_plus[i] = smoothed_plus[i - 1] - (smoothed_plus[i - 1] / period) + plus_dm[i]
        smoothed_minus[i] = (
            smoothed_minus[i - 1] - (smoothed_minus[i - 1] / period) + minus_dm[i]
        )

    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        if smoothed_tr[i] > 0:
            di_plus[i] = 100.0 * smoothed_plus[i] / smoothed_tr[i]
            di_minus[i] = 100.0 * smoothed_minus[i] / smoothed_tr[i]
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    adx_start = period * 2
    if adx_start >= n:
        return result
    result[adx_start] = np.nanmean(dx[period : adx_start + 1])
    for i in range(adx_start + 1, n):
        if not np.isnan(dx[i]) and not np.isnan(result[i - 1]):
            result[i] = (result[i - 1] * (period - 1) + dx[i]) / period
    return result


def bollinger_bands(
    close: NDArray[np.float64],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n = len(close)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        m = np.mean(window)
        s = np.std(window, ddof=0)
        middle[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return upper, middle, lower


def bollinger_bandwidth(
    upper: NDArray[np.float64],
    lower: NDArray[np.float64],
    middle: NDArray[np.float64],
) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (upper - lower) / middle
    bw[~np.isfinite(bw)] = np.nan
    return bw


def ema_slope(ema_values: NDArray[np.float64], lookback: int = 3) -> NDArray[np.float64]:
    result = np.full_like(ema_values, np.nan, dtype=np.float64)
    for i in range(lookback, len(ema_values)):
        if not np.isnan(ema_values[i]) and not np.isnan(ema_values[i - lookback]):
            result[i] = (ema_values[i] - ema_values[i - lookback]) / lookback
    return result


def volume_ratio(volume: NDArray[np.float64], period: int = 20) -> NDArray[np.float64]:
    result = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(period, len(volume)):
        avg = np.mean(volume[i - period : i])
        if avg > 0:
            result[i] = volume[i] / avg
    return result


def atr_percentile(
    atr_values: NDArray[np.float64],
    current_index: int,
    window_bars: int,
) -> float:
    start = max(0, current_index - window_bars)
    window = atr_values[start : current_index + 1]
    window = window[~np.isnan(window)]
    if len(window) < 10:
        return 0.0
    return float(percentileofscore(window, atr_values[current_index]))


def highest_high(high: NDArray[np.float64], index: int, lookback: int) -> float:
    start = max(0, index - lookback)
    return float(np.max(high[start:index]))


def lowest_low(low: NDArray[np.float64], index: int, lookback: int) -> float:
    start = max(0, index - lookback)
    return float(np.min(low[start:index]))
