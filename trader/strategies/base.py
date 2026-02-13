from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from trader.models.types import Candle, Signal


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
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
    ) -> Signal | None: ...
