from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import yaml

from trader.config.settings import EventFilterConfig


class EventFilter:
    def __init__(self, calendar_path: Path, config: EventFilterConfig):
        self.pre_minutes = config.pre_event_blackout_minutes
        self.post_minutes = config.post_event_blackout_minutes
        self.event_times: list[datetime] = []

        if calendar_path.exists():
            with open(calendar_path) as f:
                raw = yaml.safe_load(f)
            for event in raw.get("events", []):
                for date_str in event.get("dates", []):
                    self.event_times.append(datetime.fromisoformat(date_str))
            self.event_times.sort()

    def is_tradeable(self, current_time: datetime) -> tuple[bool, str]:
        for event_time in self.event_times:
            blackout_start = event_time - timedelta(minutes=self.pre_minutes)
            blackout_end = event_time + timedelta(minutes=self.post_minutes)
            if blackout_start <= current_time <= blackout_end:
                return False, f"event_blackout_{event_time.isoformat()}"
            if event_time > current_time + timedelta(hours=1):
                break
        return True, "pass"
