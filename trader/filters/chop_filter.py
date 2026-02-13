from __future__ import annotations

from datetime import datetime, timedelta

from trader.config.settings import ChopFilterConfig


class ChopFilter:
    def __init__(self, config: ChopFilterConfig):
        self.adx_threshold = config.adx_threshold
        self.atr_pct_threshold = config.atr_percentile_threshold
        self.failure_count_limit = config.consecutive_failure_count
        self.pause_hours = config.consecutive_failure_pause_hours
        self._consecutive_failures = 0
        self._pause_until: datetime | None = None

    def is_tradeable(
        self,
        adx_1h: float,
        atr_percentile_1h: float,
        current_time: datetime,
    ) -> tuple[bool, str]:
        if self._pause_until and current_time < self._pause_until:
            return False, f"chop_pause_until_{self._pause_until.isoformat()}"
        if adx_1h < self.adx_threshold:
            return False, f"adx_too_low_{adx_1h:.1f}"
        if atr_percentile_1h < self.atr_pct_threshold:
            return False, f"atr_pct_too_low_{atr_percentile_1h:.1f}"
        return True, "pass"

    def record_stop_loss_hit(self, current_time: datetime) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_count_limit:
            self._pause_until = current_time + timedelta(hours=self.pause_hours)
            self._consecutive_failures = 0

    def record_win(self) -> None:
        self._consecutive_failures = 0

    def reset(self) -> None:
        self._consecutive_failures = 0
        self._pause_until = None
