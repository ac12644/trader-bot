from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog

from trader.live.exchange_client import ExchangeClient, OrderResult

logger = structlog.get_logger()


@dataclass
class ExecutionResult:
    success: bool
    perp_order: OrderResult | None
    spot_order: OrderResult | None
    error: str | None = None


class OrderExecutor:
    """Executes spread trades on exchange.

    Entry: short perp (leg_a) + buy spot (leg_b)
    Exit: close short perp (buy) + sell spot
    """

    def __init__(self, client: ExchangeClient, leverage: int = 20):
        self.client = client
        self.leverage = leverage
        self._leverage_set: set[str] = set()

    def _to_futures_symbol(self, symbol: str) -> str:
        if ":" not in symbol:
            quote = symbol.split("/")[1] if "/" in symbol else "USDT"
            return f"{symbol}:{quote}"
        return symbol

    def _spot_symbol(self, symbol: str) -> str:
        return symbol.split(":")[0] if ":" in symbol else symbol

    def _ensure_leverage(self, futures_symbol: str) -> None:
        if futures_symbol not in self._leverage_set:
            self.client.set_leverage(futures_symbol, self.leverage)
            self._leverage_set.add(futures_symbol)

    def execute_entry(
        self,
        symbol_a: str,
        symbol_b: str,
        notional_usd: float,
    ) -> ExecutionResult:
        """Execute entry: short perp + buy spot."""
        futures_sym = self._to_futures_symbol(symbol_a)
        spot_sym = self._spot_symbol(symbol_b)

        perp_order = None
        spot_order = None

        try:
            self._ensure_leverage(futures_sym)

            ticker = self.client.fetch_ticker(futures_sym, market="futures")
            amount_base = notional_usd / ticker.last_price
            amount_base = float(
                self.client.futures.amount_to_precision(futures_sym, amount_base)
            )

            logger.info(
                "executing_entry",
                futures=futures_sym,
                spot=spot_sym,
                notional=notional_usd,
                amount_base=amount_base,
                price=ticker.last_price,
            )

            # 1. Short perp
            perp_order = self.client.market_short_futures(futures_sym, amount_base)
            logger.info(
                "perp_short_filled",
                order_id=perp_order.order_id,
                price=perp_order.filled_price,
            )

            # 2. Buy spot
            spot_order = self.client.market_buy_spot(spot_sym, notional_usd)
            logger.info(
                "spot_buy_filled",
                order_id=spot_order.order_id,
                price=spot_order.filled_price,
            )

            return ExecutionResult(success=True, perp_order=perp_order, spot_order=spot_order)

        except Exception as e:
            logger.error(
                "entry_execution_failed",
                error=str(e),
                perp_filled=perp_order is not None,
                spot_filled=spot_order is not None,
            )
            if perp_order and not spot_order:
                logger.critical(
                    "PARTIAL_FILL",
                    msg="Perp short filled but spot buy failed. Manual review needed.",
                    perp_order_id=perp_order.order_id,
                )
            return ExecutionResult(
                success=False, perp_order=perp_order, spot_order=spot_order, error=str(e),
            )

    def execute_exit(
        self,
        leg_a_symbol: str,
        leg_b_symbol: str,
        leg_a_notional: float,
        leg_a_entry_price: float,
        leg_b_notional: float,
        leg_b_entry_price: float,
    ) -> ExecutionResult:
        """Execute exit: close short perp (buy back) + sell spot."""
        futures_sym = self._to_futures_symbol(leg_a_symbol)
        spot_sym = self._spot_symbol(leg_b_symbol)

        perp_order = None
        spot_order = None

        try:
            perp_amount = leg_a_notional / leg_a_entry_price
            perp_amount = float(
                self.client.futures.amount_to_precision(futures_sym, perp_amount)
            )

            spot_amount = leg_b_notional / leg_b_entry_price
            spot_amount = float(
                self.client.spot.amount_to_precision(spot_sym, spot_amount)
            )

            logger.info(
                "executing_exit",
                futures=futures_sym,
                spot=spot_sym,
                perp_amount=perp_amount,
                spot_amount=spot_amount,
            )

            # 1. Close short perp (buy back)
            perp_order = self.client.market_close_short_futures(futures_sym, perp_amount)
            logger.info(
                "perp_close_filled",
                order_id=perp_order.order_id,
                price=perp_order.filled_price,
            )

            # 2. Sell spot
            spot_order = self.client.market_sell_spot(spot_sym, spot_amount)
            logger.info(
                "spot_sell_filled",
                order_id=spot_order.order_id,
                price=spot_order.filled_price,
            )

            return ExecutionResult(success=True, perp_order=perp_order, spot_order=spot_order)

        except Exception as e:
            logger.error(
                "exit_execution_failed",
                error=str(e),
                perp_filled=perp_order is not None,
                spot_filled=spot_order is not None,
            )
            if perp_order and not spot_order:
                logger.critical(
                    "PARTIAL_EXIT",
                    msg="Perp close filled but spot sell failed. Manual review needed.",
                    perp_order_id=perp_order.order_id,
                )
            return ExecutionResult(
                success=False, perp_order=perp_order, spot_order=spot_order, error=str(e),
            )
