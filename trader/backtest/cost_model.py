from __future__ import annotations

from trader.config.settings import BacktestConfig
from trader.models.types import Direction


class CostModel:
    def __init__(self, config: BacktestConfig):
        self.fee_taker = config.fee_taker_pct / 100.0
        self.fee_maker = config.fee_maker_pct / 100.0
        self.slip_entry = config.slippage_entry_pct / 100.0
        self.slip_exit = config.slippage_exit_pct / 100.0

    def apply_entry_slippage(self, price: float, direction: Direction) -> float:
        if direction == Direction.LONG:
            return price * (1 + self.slip_entry)
        return price * (1 - self.slip_entry)

    def apply_exit_slippage(self, price: float, direction: Direction) -> float:
        if direction == Direction.LONG:
            return price * (1 - self.slip_exit)
        return price * (1 + self.slip_exit)

    def entry_fee(self, position_size_usd: float) -> float:
        return position_size_usd * self.fee_taker

    def exit_fee(self, position_size_usd: float, is_stop: bool = False) -> float:
        rate = self.fee_taker if is_stop else self.fee_maker
        return position_size_usd * rate


class SpreadCostModel(CostModel):
    """Extended cost model for spread strategies with spot + perp legs."""

    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.spot_fee_taker = 0.001  # 0.10% default Binance spot
        self.spot_fee_maker = 0.001

    def set_spot_fees(self, taker_pct: float, maker_pct: float) -> None:
        self.spot_fee_taker = taker_pct / 100.0
        self.spot_fee_maker = maker_pct / 100.0

    def entry_fee_spread(
        self,
        notional_a: float,
        notional_b: float,
        leg_b_is_spot: bool = False,
    ) -> float:
        fee_a = notional_a * self.fee_taker
        fee_b = notional_b * (self.spot_fee_taker if leg_b_is_spot else self.fee_taker)
        return fee_a + fee_b

    def exit_fee_spread(
        self,
        notional_a: float,
        notional_b: float,
        leg_b_is_spot: bool = False,
    ) -> float:
        fee_a = notional_a * self.fee_maker
        fee_b = notional_b * (self.spot_fee_maker if leg_b_is_spot else self.fee_maker)
        return fee_a + fee_b
