from __future__ import annotations

from abc import ABC, abstractmethod

from trader.models.spread_types import (
    SpreadBar,
    SpreadExitReason,
    SpreadPosition,
    SpreadSignal,
)


class BaseSpreadStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def should_enter(
        self,
        bar: SpreadBar,
        indicators: dict[str, float],
        has_open_position: bool,
    ) -> SpreadSignal | None: ...

    @abstractmethod
    def should_exit(
        self,
        bar: SpreadBar,
        position: SpreadPosition,
        indicators: dict[str, float],
    ) -> SpreadExitReason | None: ...
