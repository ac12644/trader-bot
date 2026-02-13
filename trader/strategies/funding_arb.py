from __future__ import annotations

from trader.config.settings import FundingArbStrategyConfig
from trader.models.spread_types import (
    SpreadBar,
    SpreadExitReason,
    SpreadPosition,
    SpreadSignal,
)
from trader.models.types import Direction
from trader.strategies.spread_base import BaseSpreadStrategy


class FundingArbStrategy(BaseSpreadStrategy):
    """Cash-and-carry funding rate arbitrage.

    When funding rate is high positive:
      - Short perpetual (receive funding payments)
      - Long spot (hedge directional risk)
    Profit = accumulated funding - round-trip fees.
    """

    def __init__(self, config: FundingArbStrategyConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "funding_arb"

    def should_enter(
        self,
        bar: SpreadBar,
        indicators: dict[str, float],
        has_open_position: bool,
    ) -> SpreadSignal | None:
        if has_open_position:
            return None

        funding_rate = indicators.get("funding_rate", 0.0)
        if funding_rate <= self.config.entry_rate_threshold:
            return None

        basis = bar.close_a - bar.close_b  # perp - spot

        return SpreadSignal(
            strategy_name=self.name,
            timestamp=bar.timestamp,
            symbol_a=bar.symbol_a,
            direction_a=Direction.SHORT,  # short perp → receive funding
            entry_price_a=bar.close_a,
            symbol_b=bar.symbol_b,
            direction_b=Direction.LONG,  # long spot → hedge
            entry_price_b=bar.close_b,
            notional_per_leg=self.config.notional_per_position,
            funding_rate_at_entry=funding_rate,
            basis_at_entry=basis,
        )

    def should_exit(
        self,
        bar: SpreadBar,
        position: SpreadPosition,
        indicators: dict[str, float],
    ) -> SpreadExitReason | None:
        funding_rate = indicators.get("funding_rate", 0.0)
        basis_pct = abs(bar.close_a - bar.close_b) / bar.close_b if bar.close_b > 0 else 0
        holding_hours = (bar.timestamp - position.entry_time).total_seconds() / 3600

        # Emergency: basis blowout always triggers exit
        if basis_pct > self.config.basis_blowout_pct:
            return SpreadExitReason.BASIS_BLOWOUT

        # Don't exit before min hold (let funding accumulate to cover fees)
        if holding_hours < self.config.min_holding_hours:
            return None

        # Funding rate dropped below exit threshold
        if funding_rate < self.config.exit_rate_threshold:
            return SpreadExitReason.FUNDING_FLIP

        # Funding rate went negative (we'd be paying instead of receiving)
        if self.config.negative_rate_exit and funding_rate < 0:
            return SpreadExitReason.FUNDING_FLIP

        # Max holding period
        if holding_hours >= self.config.max_holding_hours:
            return SpreadExitReason.TIME_EXIT

        return None
