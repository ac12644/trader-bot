from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from trader.config.settings import Settings
from trader.live.exchange_client import ExchangeClient
from trader.live.order_executor import OrderExecutor
from trader.live.position_store import PositionStore
from trader.models.spread_types import (
    SpreadBar,
    SpreadExitReason,
    SpreadLeg,
    SpreadPosition,
    SpreadTrade,
)
from trader.models.types import Direction
from trader.strategies.funding_arb import FundingArbStrategy

logger = structlog.get_logger()

_FUNDING_HOURS = {0, 8, 16}


class LiveRunner:
    """Live paper trading runner for funding rate arbitrage.

    Direct analog of MultiAssetFundingRunner, but polls the exchange
    instead of iterating historical data.
    """

    def __init__(
        self,
        settings: Settings,
        client: ExchangeClient,
        executor: OrderExecutor,
        position_store: PositionStore,
    ):
        self.settings = settings
        self.client = client
        self.executor = executor
        self.store = position_store
        self.strategy = FundingArbStrategy(settings.funding_arb)

        self.symbols = (
            settings.multi_asset_funding.symbols
            if settings.multi_asset_funding
            else ["BTC/USDT"]
        )

        self.open_positions: dict[str, SpreadPosition] = {}
        self.closed_trades: list[SpreadTrade] = []
        self.leverage = settings.leverage

        self.notional = settings.funding_arb.notional_per_position
        self.margin_per_position = (2 * self.notional) / self.leverage

        poll_interval = 3600
        if settings.live:
            poll_interval = settings.live.poll_interval_seconds
        self.poll_interval = poll_interval

        self._running = False

    # ── Symbol helpers ───────────────────────────────────────────

    def _to_futures_symbol(self, symbol: str) -> str:
        if ":" not in symbol:
            quote = symbol.split("/")[1] if "/" in symbol else "USDT"
            return f"{symbol}:{quote}"
        return symbol

    def _spot_symbol(self, symbol: str) -> str:
        return symbol.split(":")[0] if ":" in symbol else symbol

    # ── Data fetching ────────────────────────────────────────────

    def _fetch_balance_usdt(self) -> float:
        try:
            balance = self.client.fetch_balance(market="futures")
            return balance.total_usdt
        except Exception as e:
            logger.error("balance_fetch_failed", error=str(e))
            return 0.0

    def _build_bar(self, symbol: str) -> SpreadBar | None:
        """Fetch latest completed 1h candle for perp + spot, build SpreadBar."""
        futures_sym = self._to_futures_symbol(symbol)
        spot_sym = self._spot_symbol(symbol)

        try:
            perp_candles = self.client.fetch_ohlcv_latest(
                futures_sym, "1h", limit=2, market="futures",
            )
            spot_candles = self.client.fetch_ohlcv_latest(
                spot_sym, "1h", limit=2, market="spot",
            )

            if not perp_candles or not spot_candles:
                return None

            # Use last completed candle (index 0 of the 2 fetched)
            pc = perp_candles[0]
            sc = spot_candles[0]

            return SpreadBar(
                timestamp=pc["timestamp"],
                symbol_a=futures_sym,
                close_a=pc["close"],
                high_a=pc["high"],
                low_a=pc["low"],
                volume_a=pc["volume"],
                symbol_b=spot_sym,
                close_b=sc["close"],
                high_b=sc["high"],
                low_b=sc["low"],
                volume_b=sc["volume"],
            )
        except Exception as e:
            logger.warning("bar_build_failed", symbol=symbol, error=str(e))
            return None

    def _fetch_funding_rate(self, symbol: str) -> float:
        futures_sym = self._to_futures_symbol(symbol)
        try:
            return self.client.fetch_funding_rate(futures_sym)
        except Exception as e:
            logger.warning("funding_fetch_failed", symbol=symbol, error=str(e))
            return 0.0

    # ── Main loop ────────────────────────────────────────────────

    def run(self) -> None:
        """Main live trading loop. Runs until interrupted."""
        self._running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Crash recovery
        self.open_positions = self.store.load()
        if self.open_positions:
            logger.info(
                "recovered_positions",
                count=len(self.open_positions),
                symbols=list(self.open_positions.keys()),
            )

        if not self.client.ping():
            logger.error("exchange_unreachable")
            return

        balance = self._fetch_balance_usdt()
        logger.info(
            "live_runner_started",
            symbols=len(self.symbols),
            balance_usdt=f"{balance:.2f}",
            leverage=self.leverage,
            notional_per_leg=self.notional,
            margin_per_position=f"{self.margin_per_position:.2f}",
            open_positions=len(self.open_positions),
        )

        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error("tick_error", error=str(e), exc_info=True)
                time.sleep(60)
                continue

            if self._running:
                self._smart_sleep()

        logger.info("live_runner_stopped")

    def _tick(self) -> None:
        """One iteration: check exits, check entries, persist."""
        now = datetime.utcnow()
        is_funding_time = now.hour in _FUNDING_HOURS and now.minute < 10

        balance = self._fetch_balance_usdt()
        margin_locked = len(self.open_positions) * self.margin_per_position

        logger.info(
            "tick",
            time=now.isoformat(),
            balance=f"{balance:.2f}",
            positions=len(self.open_positions),
            margin_locked=f"{margin_locked:.2f}",
            free_margin=f"{balance - margin_locked:.2f}",
            is_funding_time=is_funding_time,
        )

        # 1. Log funding on open positions
        if is_funding_time:
            self._log_funding()

        # 2. Check exits
        closed_symbols: list[str] = []
        for symbol, pos in list(self.open_positions.items()):
            bar = self._build_bar(symbol)
            if bar is None:
                continue

            funding_rate = self._fetch_funding_rate(symbol)
            indicators = {"funding_rate": funding_rate}
            reason = self.strategy.should_exit(bar, pos, indicators)

            if reason is not None:
                holding_hours = (now - pos.entry_time).total_seconds() / 3600
                logger.info(
                    "exit_signal",
                    symbol=symbol,
                    reason=reason.value,
                    funding_rate=funding_rate,
                    holding_hours=f"{holding_hours:.1f}",
                )
                result = self.executor.execute_exit(
                    leg_a_symbol=pos.leg_a.symbol,
                    leg_b_symbol=pos.leg_b.symbol,
                    leg_a_notional=pos.leg_a.notional_usd,
                    leg_a_entry_price=pos.leg_a.entry_price,
                    leg_b_notional=pos.leg_b.notional_usd,
                    leg_b_entry_price=pos.leg_b.entry_price,
                )
                if result.success:
                    trade = self._record_trade(pos, bar, reason, result)
                    self.closed_trades.append(trade)
                    closed_symbols.append(symbol)
                    logger.info(
                        "position_closed",
                        symbol=symbol,
                        reason=reason.value,
                        pnl_total=f"{trade.pnl_total:.2f}",
                    )
                else:
                    logger.error("exit_failed", symbol=symbol, error=result.error)

        for s in closed_symbols:
            del self.open_positions[s]

        # 3. Check entries (rank by funding rate)
        free_margin = balance - len(self.open_positions) * self.margin_per_position
        if free_margin >= self.margin_per_position:
            candidates: list[tuple[str, float]] = []
            for symbol in self.symbols:
                if symbol in self.open_positions:
                    continue
                rate = self._fetch_funding_rate(symbol)
                if rate > self.settings.funding_arb.entry_rate_threshold:
                    candidates.append((symbol, rate))

            candidates.sort(key=lambda x: x[1], reverse=True)

            for symbol, rate in candidates:
                current_free = balance - len(self.open_positions) * self.margin_per_position
                if current_free < self.margin_per_position:
                    break

                bar = self._build_bar(symbol)
                if bar is None:
                    continue

                signal_obj = self.strategy.should_enter(
                    bar, {"funding_rate": rate}, has_open_position=False,
                )
                if signal_obj is None:
                    continue

                basis_pct = (bar.close_a - bar.close_b) / bar.close_b * 100 if bar.close_b else 0
                logger.info(
                    "entry_signal",
                    symbol=symbol,
                    funding_rate=rate,
                    perp_price=bar.close_a,
                    spot_price=bar.close_b,
                    basis_pct=f"{basis_pct:.4f}%",
                )

                result = self.executor.execute_entry(
                    symbol_a=signal_obj.symbol_a,
                    symbol_b=signal_obj.symbol_b,
                    notional_usd=self.notional,
                )
                if result.success:
                    pos = self._build_position(signal_obj, result, rate)
                    self.open_positions[symbol] = pos
                    logger.info(
                        "position_opened",
                        symbol=symbol,
                        perp_price=result.perp_order.filled_price,
                        spot_price=result.spot_order.filled_price,
                        funding_rate=rate,
                    )
                else:
                    logger.error("entry_failed", symbol=symbol, error=result.error)

        # Persist after every tick
        self.store.save(self.open_positions)
        self._log_portfolio_summary(balance)

    # ── Position management ──────────────────────────────────────

    def _log_funding(self) -> None:
        for symbol, pos in self.open_positions.items():
            rate = self._fetch_funding_rate(symbol)
            funding_income = pos.leg_a.notional_usd * rate
            logger.info(
                "funding_payment",
                symbol=symbol,
                rate=rate,
                estimated_income=f"{funding_income:.4f}",
                accumulated=f"{pos.accumulated_funding_total:.4f}",
            )
            pos.accumulated_funding_total += funding_income
            pos.leg_a.accumulated_funding += funding_income

    def _build_position(self, signal, result, funding_rate: float) -> SpreadPosition:
        leg_a = SpreadLeg(
            symbol=signal.symbol_a,
            direction=Direction.SHORT,
            entry_price=result.perp_order.filled_price,
            current_price=result.perp_order.filled_price,
            notional_usd=self.notional,
            is_perp=True,
        )
        leg_b = SpreadLeg(
            symbol=signal.symbol_b,
            direction=Direction.LONG,
            entry_price=result.spot_order.filled_price,
            current_price=result.spot_order.filled_price,
            notional_usd=self.notional,
            is_perp=False,
        )
        total_fees = (result.perp_order.fee or 0.0) + (result.spot_order.fee or 0.0)

        return SpreadPosition(
            leg_a=leg_a,
            leg_b=leg_b,
            entry_time=datetime.utcnow(),
            strategy_name="funding_arb",
            funding_rate_at_entry=funding_rate,
            basis_at_entry=result.perp_order.filled_price - result.spot_order.filled_price,
            entry_fees_total=total_fees,
        )

    def _record_trade(self, pos, bar, reason, result) -> SpreadTrade:
        exit_price_a = result.perp_order.filled_price if result.perp_order else bar.close_a
        exit_price_b = result.spot_order.filled_price if result.spot_order else bar.close_b

        pnl_a = (pos.leg_a.entry_price - exit_price_a) / pos.leg_a.entry_price * pos.leg_a.notional_usd
        pnl_b = (exit_price_b - pos.leg_b.entry_price) / pos.leg_b.entry_price * pos.leg_b.notional_usd

        exit_fees = (result.perp_order.fee or 0.0) + (result.spot_order.fee or 0.0)
        total_fees = pos.entry_fees_total + exit_fees
        pnl_total = pnl_a + pnl_b + pos.accumulated_funding_total - total_fees

        holding_mins = int((datetime.utcnow() - pos.entry_time).total_seconds() / 60)

        return SpreadTrade(
            strategy_name=pos.strategy_name,
            symbol_a=pos.leg_a.symbol,
            symbol_b=pos.leg_b.symbol,
            direction_a=pos.leg_a.direction,
            direction_b=pos.leg_b.direction,
            entry_price_a=pos.leg_a.entry_price,
            entry_price_b=pos.leg_b.entry_price,
            exit_price_a=exit_price_a,
            exit_price_b=exit_price_b,
            notional_per_leg=pos.leg_a.notional_usd,
            entry_time=pos.entry_time,
            exit_time=datetime.utcnow(),
            exit_reason=reason,
            pnl_leg_a=pnl_a,
            pnl_leg_b=pnl_b,
            pnl_funding=pos.accumulated_funding_total,
            pnl_total=pnl_total,
            fees_paid=total_fees,
            holding_duration_minutes=holding_mins,
            basis_at_entry=pos.basis_at_entry,
            basis_at_exit=bar.close_a - bar.close_b,
        )

    def _log_portfolio_summary(self, balance: float) -> None:
        total_notional = len(self.open_positions) * self.notional * 2
        total_funding = sum(p.accumulated_funding_total for p in self.open_positions.values())
        total_closed_pnl = sum(t.pnl_total for t in self.closed_trades)

        logger.info(
            "portfolio_summary",
            balance_usdt=f"{balance:.2f}",
            open_positions=len(self.open_positions),
            open_symbols=list(self.open_positions.keys()),
            total_notional=f"{total_notional:.2f}",
            accumulated_funding=f"{total_funding:.4f}",
            closed_trades=len(self.closed_trades),
            closed_pnl=f"{total_closed_pnl:.2f}",
        )

    # ── Sleep / shutdown ─────────────────────────────────────────

    def _smart_sleep(self) -> None:
        """Sleep until next check, waking earlier for funding times."""
        now = datetime.utcnow()

        # Find next funding time
        next_funding = None
        for hour in sorted(_FUNDING_HOURS):
            candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                next_funding = candidate
                break
        if next_funding is None:
            tomorrow = now + timedelta(days=1)
            next_funding = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

        seconds_to_funding = (next_funding - now).total_seconds()
        sleep_seconds = min(self.poll_interval, max(seconds_to_funding - 60, 10))

        next_check = now + timedelta(seconds=sleep_seconds)
        logger.info(
            "sleeping",
            seconds=int(sleep_seconds),
            next_check=next_check.strftime("%H:%M:%S UTC"),
            next_funding=next_funding.strftime("%H:%M UTC"),
        )

        # Sleep in small chunks for responsive SIGINT
        end_time = time.time() + sleep_seconds
        while time.time() < end_time and self._running:
            time.sleep(min(5, end_time - time.time()))

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("shutdown_signal_received", signal=signum)
        self._running = False


# ── Entry point ──────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )

    project_root = Path(__file__).parent.parent.parent
    settings_path = project_root / "settings.yaml"

    logger.info("loading_settings", path=str(settings_path))
    settings = Settings.from_yaml(settings_path)

    if not settings.funding_arb:
        logger.error("funding_arb_config_missing")
        sys.exit(1)

    testnet = settings.exchange.testnet if settings.exchange else True

    client = ExchangeClient(testnet=testnet)
    executor = OrderExecutor(client, leverage=settings.leverage)

    position_file = "positions.json"
    if settings.live:
        position_file = settings.live.position_file
    store = PositionStore(project_root / position_file)

    runner = LiveRunner(settings, client, executor, store)

    print("=" * 60)
    print("  LIVE PAPER TRADING - FUNDING RATE ARBITRAGE")
    print("=" * 60)
    print(f"  Testnet: {testnet}")
    print(f"  Symbols: {len(runner.symbols)}")
    print(f"  Leverage: {settings.leverage}x")
    print(f"  Notional/leg: ${runner.notional}")
    print(f"  Margin/position: ${runner.margin_per_position:.2f}")
    print(f"  Poll interval: {runner.poll_interval}s")
    print("=" * 60)
    print("  Press Ctrl+C to stop gracefully")
    print("=" * 60)

    runner.run()


if __name__ == "__main__":
    main()
