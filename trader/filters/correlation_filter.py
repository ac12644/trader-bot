from __future__ import annotations

from trader.models.types import Direction, Position


class CorrelationFilter:
    def __init__(self, max_same_direction: int = 1):
        self.max_same_direction = max_same_direction

    def is_allowed(
        self,
        new_symbol: str,
        new_direction: Direction,
        open_positions: list[Position],
    ) -> tuple[bool, str]:
        same_dir_count = sum(1 for p in open_positions if p.direction == new_direction)
        if same_dir_count >= self.max_same_direction:
            return False, f"correlation_block_{new_direction.value}_{same_dir_count}_open"
        return True, "pass"
