from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    STALE_EXIT = "stale_exit"
    DAILY_LIMIT = "daily_limit"
    WEEKLY_LIMIT = "weekly_limit"
    KILL_SWITCH = "kill_switch"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


@dataclass
class Signal:
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    timestamp: datetime
    strategy_name: str
    confidence: float = 1.0


@dataclass
class Trade:
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size_usd: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    pnl_usd: float
    pnl_pct: float
    fees_paid: float
    slippage_cost: float
    r_multiple: float
    strategy_name: str
    holding_duration_minutes: int


@dataclass
class Position:
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None
    position_size_usd: float
    entry_time: datetime
    atr_at_entry: float
    strategy_name: str
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = float("inf")
    trailing_active: bool = False


@dataclass
class EquityState:
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
    daily_entries_by_symbol: dict[str, int] = field(default_factory=dict)
    total_daily_entries: int = 0
    open_positions: list[Position] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    cooldown_until: datetime | None = None
    is_daily_halted: bool = False
    is_weekly_halted: bool = False
