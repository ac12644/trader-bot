from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from trader.models.types import Direction


class SpreadExitReason(Enum):
    ZSCORE_TARGET = "zscore_target"
    ZSCORE_STOP = "zscore_stop"
    FUNDING_FLIP = "funding_flip"
    BASIS_BLOWOUT = "basis_blowout"
    TIME_EXIT = "time_exit"
    END_OF_DATA = "end_of_data"
    DAILY_LIMIT = "daily_limit"


@dataclass(frozen=True)
class SpreadBar:
    """Synchronized data for two symbols at one timestamp."""

    timestamp: datetime
    symbol_a: str
    close_a: float
    high_a: float
    low_a: float
    volume_a: float
    symbol_b: str
    close_b: float
    high_b: float
    low_b: float
    volume_b: float


@dataclass
class SpreadSignal:
    """Signal to enter a spread/pair trade."""

    strategy_name: str
    timestamp: datetime
    symbol_a: str
    direction_a: Direction
    entry_price_a: float
    symbol_b: str
    direction_b: Direction
    entry_price_b: float
    notional_per_leg: float
    zscore_at_entry: float = 0.0
    funding_rate_at_entry: float = 0.0
    basis_at_entry: float = 0.0


@dataclass
class SpreadLeg:
    """One leg of an open spread position."""

    symbol: str
    direction: Direction
    entry_price: float
    current_price: float
    notional_usd: float
    is_perp: bool
    accumulated_funding: float = 0.0


@dataclass
class SpreadPosition:
    """Open spread/pair position with two legs."""

    leg_a: SpreadLeg
    leg_b: SpreadLeg
    entry_time: datetime
    strategy_name: str
    zscore_at_entry: float = 0.0
    funding_rate_at_entry: float = 0.0
    basis_at_entry: float = 0.0
    accumulated_funding_total: float = 0.0
    entry_fees_total: float = 0.0


@dataclass
class SpreadTrade:
    """Closed spread trade record."""

    strategy_name: str
    symbol_a: str
    symbol_b: str
    direction_a: Direction
    direction_b: Direction
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    notional_per_leg: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: SpreadExitReason
    pnl_leg_a: float
    pnl_leg_b: float
    pnl_funding: float
    pnl_total: float
    fees_paid: float
    holding_duration_minutes: int
    zscore_at_entry: float = 0.0
    zscore_at_exit: float = 0.0
    basis_at_entry: float = 0.0
    basis_at_exit: float = 0.0

    @property
    def pnl_usd(self) -> float:
        return self.pnl_total

    @property
    def pnl_pct(self) -> float:
        return 0.0  # set by engine at close time


@dataclass
class SpreadEquityState:
    """Portfolio state for spread strategies."""

    equity: float
    peak_equity: float
    daily_pnl: float
    weekly_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_losses: int
    current_day: str | None = None
    current_week_start: str | None = None
    open_positions: list[SpreadPosition] = field(default_factory=list)
    trades: list[SpreadTrade] = field(default_factory=list)
    is_daily_halted: bool = False
    is_weekly_halted: bool = False
