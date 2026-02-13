from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import ccxt
import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()


@dataclass(frozen=True)
class TickerData:
    symbol: str
    last_price: float
    bid: float
    ask: float
    timestamp: datetime


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    filled_price: float
    cost: float
    fee: float
    timestamp: datetime
    status: str


@dataclass(frozen=True)
class BalanceInfo:
    total_usdt: float
    free_usdt: float
    used_usdt: float


class ExchangeClient:
    """Wrapper around ccxt for Binance testnet spot + futures."""

    def __init__(self, testnet: bool = True):
        load_dotenv()

        api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")

        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set in .env"
            )

        # Futures client (for short perp leg)
        self.futures = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        # Spot client (for long spot leg)
        self.spot = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

        if testnet:
            # Binance futures uses demo trading (sandbox deprecated in ccxt)
            self.futures.enable_demo_trading(True)
            self.spot.enable_demo_trading(True)

        logger.info("exchange_client_initialized", testnet=testnet)

    # ── Market data ──────────────────────────────────────────────

    def fetch_ticker(self, symbol: str, market: str = "futures") -> TickerData:
        exchange = self.futures if market == "futures" else self.spot
        ticker = exchange.fetch_ticker(symbol)
        return TickerData(
            symbol=symbol,
            last_price=ticker["last"],
            bid=ticker.get("bid") or ticker["last"],
            ask=ticker.get("ask") or ticker["last"],
            timestamp=datetime.utcfromtimestamp(ticker["timestamp"] / 1000),
        )

    def fetch_ohlcv_latest(
        self, symbol: str, timeframe: str = "1h", limit: int = 2, market: str = "futures",
    ) -> list[dict]:
        exchange = self.futures if market == "futures" else self.spot
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [
            {
                "timestamp": datetime.utcfromtimestamp(c[0] / 1000),
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in ohlcv
        ]

    def fetch_funding_rate(self, symbol: str) -> float:
        result = self.futures.fetch_funding_rate(symbol)
        return result.get("fundingRate", 0.0) or 0.0

    # ── Account ──────────────────────────────────────────────────

    def fetch_balance(self, market: str = "futures") -> BalanceInfo:
        exchange = self.futures if market == "futures" else self.spot
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return BalanceInfo(
            total_usdt=usdt.get("total", 0.0) or 0.0,
            free_usdt=usdt.get("free", 0.0) or 0.0,
            used_usdt=usdt.get("used", 0.0) or 0.0,
        )

    def fetch_futures_positions(self) -> list[dict]:
        positions = self.futures.fetch_positions()
        return [
            {
                "symbol": p["symbol"],
                "side": p["side"],
                "contracts": p["contracts"],
                "notional": p["notional"],
                "unrealizedPnl": p["unrealizedPnl"],
                "entryPrice": p["entryPrice"],
            }
            for p in positions
            if p.get("contracts") and float(p["contracts"]) > 0
        ]

    # ── Orders ───────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.futures.set_leverage(leverage, symbol)
            logger.info("leverage_set", symbol=symbol, leverage=leverage)
        except Exception as e:
            logger.warning("leverage_set_failed", symbol=symbol, error=str(e))

    def market_buy_spot(self, symbol: str, amount_usdt: float) -> OrderResult:
        order = self.spot.create_market_buy_order_with_cost(symbol, amount_usdt)
        return self._parse_order(order)

    def market_sell_spot(self, symbol: str, amount_base: float) -> OrderResult:
        order = self.spot.create_market_sell_order(symbol, amount_base)
        return self._parse_order(order)

    def market_short_futures(self, symbol: str, amount_base: float) -> OrderResult:
        order = self.futures.create_market_sell_order(symbol, amount_base)
        return self._parse_order(order)

    def market_close_short_futures(self, symbol: str, amount_base: float) -> OrderResult:
        order = self.futures.create_market_buy_order(symbol, amount_base)
        return self._parse_order(order)

    def _parse_order(self, order: dict) -> OrderResult:
        fee_cost = 0.0
        if order.get("fee") and order["fee"].get("cost"):
            fee_cost = float(order["fee"]["cost"])

        return OrderResult(
            order_id=str(order["id"]),
            symbol=order["symbol"],
            side=order["side"],
            amount=float(order.get("filled") or order.get("amount") or 0.0),
            filled_price=float(order.get("average") or order.get("price") or 0.0),
            cost=float(order.get("cost") or 0.0),
            fee=fee_cost,
            timestamp=(
                datetime.utcfromtimestamp(order["timestamp"] / 1000)
                if order.get("timestamp")
                else datetime.utcnow()
            ),
            status=order.get("status", "unknown"),
        )

    # ── Health check ─────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            self.futures.fetch_time()
            self.spot.fetch_time()
            return True
        except Exception as e:
            logger.error("ping_failed", error=str(e))
            return False
