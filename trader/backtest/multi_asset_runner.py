from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from trader.backtest.cost_model import SpreadCostModel
from trader.config.settings import FundingArbStrategyConfig
from trader.data.fetcher import fetch_candles
from trader.data.funding_fetcher import fetch_funding_rates, fetch_spot_candles
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

# Funding payment hours (UTC): 00, 08, 16
_FUNDING_HOURS = {0, 8, 16}


@dataclass
class SymbolData:
    """Pre-loaded data for one symbol."""
    symbol: str
    perp_symbol: str
    timestamps: np.ndarray
    close_perp: np.ndarray
    high_perp: np.ndarray
    low_perp: np.ndarray
    vol_perp: np.ndarray
    close_spot: np.ndarray
    high_spot: np.ndarray
    low_spot: np.ndarray
    vol_spot: np.ndarray
    funding_lookup: dict[datetime, float] = field(default_factory=dict)


class MultiAssetFundingRunner:
    """Unified portfolio-level funding arb backtest across multiple assets.

    Key difference from per-symbol approach:
      - Single shared capital pool
      - Leverage-based margin: margin_per_position = 2 * notional / leverage
      - Only opens new positions when free margin is available
      - Steps through time across ALL symbols simultaneously
    """

    def __init__(
        self,
        config: FundingArbStrategyConfig,
        symbols: list[str],
        cost_model: SpreadCostModel,
        starting_capital: float,
        data_dir: Path,
        start_date: datetime,
        end_date: datetime,
        leverage: int = 1,
    ):
        self.config = config
        self.symbols = symbols
        self.cost_model = cost_model
        self.starting_capital = starting_capital
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.leverage = leverage
        self.strategy = FundingArbStrategy(config)

        # Margin required per position = 2 legs * notional / leverage
        self.margin_per_position = (2 * config.notional_per_position) / leverage

    def _perp_symbol(self, symbol: str) -> str:
        return f"{symbol}:USDT" if ":" not in symbol else symbol

    def _safe_name(self, symbol: str) -> str:
        return symbol.replace("/", "_")

    def _perp_path(self, symbol: str) -> Path:
        return self.data_dir / f"{self._safe_name(self._perp_symbol(symbol))}_1h.parquet"

    def _spot_path(self, symbol: str) -> Path:
        return self.data_dir / f"{self._safe_name(symbol)}_spot_1h.parquet"

    def _funding_path(self, symbol: str) -> Path:
        return self.data_dir / f"{self._safe_name(self._perp_symbol(symbol))}_funding.parquet"

    def fetch_all_data(self) -> dict[str, bool]:
        """Fetch perp candles, spot candles, and funding rates for all symbols."""
        results: dict[str, bool] = {}

        for symbol in self.symbols:
            perp_sym = self._perp_symbol(symbol)
            try:
                if not self._perp_path(symbol).exists():
                    logger.info("fetching_perp_candles", symbol=perp_sym)
                    fetch_candles(perp_sym, "1h", self.start_date, self.end_date, self.data_dir)
                else:
                    logger.info("using_cached_perp", symbol=perp_sym)

                if not self._spot_path(symbol).exists():
                    logger.info("fetching_spot_candles", symbol=symbol)
                    fetch_spot_candles(symbol, "1h", self.start_date, self.end_date, self.data_dir)
                else:
                    logger.info("using_cached_spot", symbol=symbol)

                if not self._funding_path(symbol).exists():
                    logger.info("fetching_funding_rates", symbol=perp_sym)
                    fetch_funding_rates(perp_sym, self.start_date, self.end_date, self.data_dir)
                else:
                    logger.info("using_cached_funding", symbol=perp_sym)

                results[symbol] = True
            except Exception as e:
                logger.warning("data_fetch_failed", symbol=symbol, error=str(e))
                results[symbol] = False

        fetched = sum(1 for v in results.values() if v)
        logger.info("fetch_summary", total=len(self.symbols), success=fetched)
        return results

    def _load_symbol_data(self, symbol: str) -> SymbolData | None:
        """Load and synchronize perp + spot data for one symbol."""
        if not (
            self._perp_path(symbol).exists()
            and self._spot_path(symbol).exists()
            and self._funding_path(symbol).exists()
        ):
            return None

        try:
            perp_df = pl.read_parquet(self._perp_path(symbol))
            spot_df = pl.read_parquet(self._spot_path(symbol))
            funding_df = pl.read_parquet(self._funding_path(symbol))

            # Inner join on timestamp
            joined = perp_df.select(
                pl.col("timestamp").alias("ts"),
                pl.col("close").alias("close_perp"),
                pl.col("high").alias("high_perp"),
                pl.col("low").alias("low_perp"),
                pl.col("volume").alias("vol_perp"),
            ).join(
                spot_df.select(
                    pl.col("timestamp").alias("ts"),
                    pl.col("close").alias("close_spot"),
                    pl.col("high").alias("high_spot"),
                    pl.col("low").alias("low_spot"),
                    pl.col("volume").alias("vol_spot"),
                ),
                on="ts",
                how="inner",
            ).sort("ts")

            if len(joined) == 0:
                return None

            timestamps = joined["ts"].to_numpy()

            # Build funding lookup
            funding_lookup: dict[datetime, float] = {}
            for row in funding_df.iter_rows(named=True):
                ts = row["timestamp"]
                if not isinstance(ts, datetime):
                    ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                funding_lookup[ts] = row["funding_rate"]

            return SymbolData(
                symbol=symbol,
                perp_symbol=self._perp_symbol(symbol),
                timestamps=timestamps,
                close_perp=joined["close_perp"].to_numpy().astype(np.float64),
                high_perp=joined["high_perp"].to_numpy().astype(np.float64),
                low_perp=joined["low_perp"].to_numpy().astype(np.float64),
                vol_perp=joined["vol_perp"].to_numpy().astype(np.float64),
                close_spot=joined["close_spot"].to_numpy().astype(np.float64),
                high_spot=joined["high_spot"].to_numpy().astype(np.float64),
                low_spot=joined["low_spot"].to_numpy().astype(np.float64),
                vol_spot=joined["vol_spot"].to_numpy().astype(np.float64),
                funding_lookup=funding_lookup,
            )
        except Exception as e:
            logger.warning("load_failed", symbol=symbol, error=str(e))
            return None

    def _ts_to_datetime(self, ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, np.datetime64):
            epoch = np.datetime64(0, "us")
            one_us = np.timedelta64(1, "us")
            us = int((ts - epoch) / one_us)
            return datetime(1970, 1, 1) + timedelta(microseconds=us)
        return datetime.fromisoformat(str(ts))

    def _get_funding_rate(self, data: SymbolData, dt: datetime) -> float:
        rate = data.funding_lookup.get(dt)
        if rate is not None:
            return rate
        hour = (dt.hour // 8) * 8
        snap = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        return data.funding_lookup.get(snap, 0.0)

    def run_all(self) -> dict[str, list[SpreadTrade]]:
        """Run unified portfolio-level backtest across all symbols.

        Steps through every 1h bar chronologically with shared capital.
        Only opens positions when free margin >= margin_per_position.
        """
        # Load all symbol data
        all_data: dict[str, SymbolData] = {}
        for symbol in self.symbols:
            data = self._load_symbol_data(symbol)
            if data is not None:
                all_data[symbol] = data
                logger.info(
                    "loaded_symbol",
                    symbol=symbol,
                    bars=len(data.timestamps),
                    funding_points=len(data.funding_lookup),
                )

        if not all_data:
            logger.warning("no_symbols_loaded")
            return {}

        # Build master timeline (union of all timestamps)
        all_ts = set()
        for data in all_data.values():
            for ts in data.timestamps:
                all_ts.add(self._ts_to_datetime(ts))
        master_timeline = sorted(all_ts)

        logger.info(
            "unified_backtest_start",
            symbols=len(all_data),
            leverage=self.leverage,
            margin_per_pos=f"${self.margin_per_position:.2f}",
            total_bars=len(master_timeline),
        )

        # Build per-symbol index lookup: datetime → array index
        symbol_indices: dict[str, dict[datetime, int]] = {}
        for symbol, data in all_data.items():
            idx_map = {}
            for i, ts in enumerate(data.timestamps):
                idx_map[self._ts_to_datetime(ts)] = i
            symbol_indices[symbol] = idx_map

        # Portfolio state
        equity = self.starting_capital
        peak_equity = self.starting_capital
        # symbol → SpreadPosition
        open_positions: dict[str, SpreadPosition] = {}
        all_trades: dict[str, list[SpreadTrade]] = {s: [] for s in all_data}
        margin_locked = 0.0

        log_interval = max(len(master_timeline) // 20, 1)

        for bar_idx, dt in enumerate(master_timeline):
            if bar_idx % log_interval == 0:
                pct = bar_idx / len(master_timeline) * 100
                free_margin = equity - margin_locked
                logger.info(
                    "portfolio_progress",
                    pct=f"{pct:.0f}%",
                    equity=f"{equity:.2f}",
                    positions=len(open_positions),
                    free_margin=f"{free_margin:.2f}",
                )

            is_funding_time = dt.hour in _FUNDING_HOURS and dt.minute == 0

            # 1. Apply funding to all open positions
            if is_funding_time and open_positions:
                for symbol, pos in open_positions.items():
                    data = all_data[symbol]
                    rate = self._get_funding_rate(data, dt)
                    if rate == 0.0:
                        continue
                    # Short perp receives positive funding
                    for leg in (pos.leg_a, pos.leg_b):
                        if not leg.is_perp:
                            continue
                        if leg.direction == Direction.SHORT:
                            income = leg.notional_usd * rate
                        else:
                            income = -leg.notional_usd * rate
                        leg.accumulated_funding += income
                        pos.accumulated_funding_total += income
                        equity += income

            # 2. Check exits on all open positions
            closed_symbols: list[str] = []
            for symbol, pos in open_positions.items():
                data = all_data[symbol]
                idx = symbol_indices[symbol].get(dt)
                if idx is None:
                    continue  # no data for this symbol at this timestamp

                bar = SpreadBar(
                    timestamp=dt,
                    symbol_a=data.perp_symbol,
                    close_a=float(data.close_perp[idx]),
                    high_a=float(data.high_perp[idx]),
                    low_a=float(data.low_perp[idx]),
                    volume_a=float(data.vol_perp[idx]),
                    symbol_b=data.symbol,
                    close_b=float(data.close_spot[idx]),
                    high_b=float(data.high_spot[idx]),
                    low_b=float(data.low_spot[idx]),
                    volume_b=float(data.vol_spot[idx]),
                )

                indicators = {"funding_rate": self._get_funding_rate(data, dt)}
                reason = self.strategy.should_exit(bar, pos, indicators)
                if reason is not None:
                    trade = self._close_position(pos, bar, reason, equity)
                    equity += trade.pnl_leg_a + trade.pnl_leg_b - (trade.fees_paid - pos.entry_fees_total)
                    margin_locked -= self.margin_per_position
                    all_trades[symbol].append(trade)
                    closed_symbols.append(symbol)

            for s in closed_symbols:
                del open_positions[s]

            # 3. Check entries — rank by funding rate, enter highest first
            if equity - margin_locked >= self.margin_per_position:
                candidates: list[tuple[str, float, int]] = []
                for symbol, data in all_data.items():
                    if symbol in open_positions:
                        continue
                    idx = symbol_indices[symbol].get(dt)
                    if idx is None:
                        continue
                    rate = self._get_funding_rate(data, dt)
                    if rate > self.config.entry_rate_threshold:
                        candidates.append((symbol, rate, idx))

                # Sort by funding rate descending — best opportunities first
                candidates.sort(key=lambda x: x[1], reverse=True)

                for symbol, rate, idx in candidates:
                    free_margin = equity - margin_locked
                    if free_margin < self.margin_per_position:
                        break  # no more capital

                    data = all_data[symbol]
                    bar = SpreadBar(
                        timestamp=dt,
                        symbol_a=data.perp_symbol,
                        close_a=float(data.close_perp[idx]),
                        high_a=float(data.high_perp[idx]),
                        low_a=float(data.low_perp[idx]),
                        volume_a=float(data.vol_perp[idx]),
                        symbol_b=data.symbol,
                        close_b=float(data.close_spot[idx]),
                        high_b=float(data.high_spot[idx]),
                        low_b=float(data.low_spot[idx]),
                        volume_b=float(data.vol_spot[idx]),
                    )

                    pos = self._open_position(bar, data, rate, dt)
                    open_positions[symbol] = pos
                    equity -= pos.entry_fees_total
                    margin_locked += self.margin_per_position

            peak_equity = max(peak_equity, equity)

        # Close remaining positions at end of data
        for symbol, pos in list(open_positions.items()):
            data = all_data[symbol]
            last_idx = len(data.timestamps) - 1
            last_dt = self._ts_to_datetime(data.timestamps[last_idx])
            bar = SpreadBar(
                timestamp=last_dt,
                symbol_a=data.perp_symbol,
                close_a=float(data.close_perp[last_idx]),
                high_a=float(data.high_perp[last_idx]),
                low_a=float(data.low_perp[last_idx]),
                volume_a=float(data.vol_perp[last_idx]),
                symbol_b=data.symbol,
                close_b=float(data.close_spot[last_idx]),
                high_b=float(data.high_spot[last_idx]),
                low_b=float(data.low_spot[last_idx]),
                volume_b=float(data.vol_spot[last_idx]),
            )
            trade = self._close_position(pos, bar, SpreadExitReason.END_OF_DATA, equity)
            equity += trade.pnl_leg_a + trade.pnl_leg_b - (trade.fees_paid - pos.entry_fees_total)
            all_trades[symbol].append(trade)

        total_trades = sum(len(t) for t in all_trades.values())
        logger.info(
            "unified_backtest_complete",
            total_trades=total_trades,
            final_equity=f"{equity:.2f}",
            symbols_traded=sum(1 for t in all_trades.values() if t),
        )

        # Remove symbols with no trades
        return {s: trades for s, trades in all_trades.items() if trades}

    def _open_position(
        self,
        bar: SpreadBar,
        data: SymbolData,
        funding_rate: float,
        dt: datetime,
    ) -> SpreadPosition:
        notional = self.config.notional_per_position
        price_a = self.cost_model.apply_entry_slippage(bar.close_a, Direction.SHORT)
        price_b = self.cost_model.apply_entry_slippage(bar.close_b, Direction.LONG)

        leg_a = SpreadLeg(
            symbol=data.perp_symbol,
            direction=Direction.SHORT,
            entry_price=price_a,
            current_price=price_a,
            notional_usd=notional,
            is_perp=True,
        )
        leg_b = SpreadLeg(
            symbol=data.symbol,
            direction=Direction.LONG,
            entry_price=price_b,
            current_price=price_b,
            notional_usd=notional,
            is_perp=False,
        )

        entry_fees = self.cost_model.entry_fee_spread(
            notional, notional, leg_b_is_spot=True,
        )

        return SpreadPosition(
            leg_a=leg_a,
            leg_b=leg_b,
            entry_time=dt,
            strategy_name="funding_arb",
            funding_rate_at_entry=funding_rate,
            basis_at_entry=bar.close_a - bar.close_b,
            entry_fees_total=entry_fees,
        )

    def _close_position(
        self,
        pos: SpreadPosition,
        bar: SpreadBar,
        reason: SpreadExitReason,
        equity: float,
    ) -> SpreadTrade:
        exit_a = self.cost_model.apply_exit_slippage(bar.close_a, pos.leg_a.direction)
        exit_b = self.cost_model.apply_exit_slippage(bar.close_b, pos.leg_b.direction)

        pnl_a = self._leg_pnl(pos.leg_a, exit_a)
        pnl_b = self._leg_pnl(pos.leg_b, exit_b)

        exit_fees = self.cost_model.exit_fee_spread(
            pos.leg_a.notional_usd, pos.leg_b.notional_usd, leg_b_is_spot=True,
        )
        total_fees = pos.entry_fees_total + exit_fees
        pnl_total = pnl_a + pnl_b + pos.accumulated_funding_total - total_fees

        holding_mins = int((bar.timestamp - pos.entry_time).total_seconds() / 60)

        return SpreadTrade(
            strategy_name=pos.strategy_name,
            symbol_a=pos.leg_a.symbol,
            symbol_b=pos.leg_b.symbol,
            direction_a=pos.leg_a.direction,
            direction_b=pos.leg_b.direction,
            entry_price_a=pos.leg_a.entry_price,
            entry_price_b=pos.leg_b.entry_price,
            exit_price_a=exit_a,
            exit_price_b=exit_b,
            notional_per_leg=pos.leg_a.notional_usd,
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
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

    @staticmethod
    def _leg_pnl(leg: SpreadLeg, exit_price: float) -> float:
        if leg.direction == Direction.LONG:
            return (exit_price - leg.entry_price) / leg.entry_price * leg.notional_usd
        return (leg.entry_price - exit_price) / leg.entry_price * leg.notional_usd
