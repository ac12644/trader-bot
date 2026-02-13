from __future__ import annotations

from datetime import datetime

from trader.config.settings import SessionBlackout

DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


class SessionFilter:
    def __init__(self, blackout_periods: list[SessionBlackout]):
        self.blackout_periods = blackout_periods

    def is_tradeable(self, current_time: datetime) -> tuple[bool, str]:
        weekday = current_time.weekday()
        current_hhmm = current_time.strftime("%H:%M")

        for period in self.blackout_periods:
            day_indices = [DAY_MAP[d] for d in period.days if d in DAY_MAP]
            if weekday in day_indices:
                if period.start <= current_hhmm < period.end:
                    return False, f"session_blackout_{period.start}-{period.end}"
        return True, "pass"
