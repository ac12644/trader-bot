from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BacktestConfig:
    fee_taker_pct: float
    fee_maker_pct: float
    slippage_entry_pct: float
    slippage_exit_pct: float
    include_funding: bool
    monte_carlo_runs: int
    monte_carlo_max_dd_95_pct: float
    walk_forward_train_months: int
    walk_forward_test_months: int
    start_date: str
    end_date: str


@dataclass
class RiskConfig:
    risk_per_trade_pct: float
    daily_loss_cap_pct: float
    weekly_loss_cap_pct: float
    max_concurrent_positions: int
    max_same_direction_correlated: int
    max_total_exposure_pct: float
    max_total_open_risk_pct: float
    consecutive_loss_cooldown_count: int
    consecutive_loss_cooldown_hours: int
    drawdown_reduction_threshold_pct: float
    drawdown_halt_threshold_pct: float


@dataclass
class BreakoutStrategyConfig:
    lookback_candles: int
    volume_multiplier: float
    atr_period: int
    stop_atr_multiplier: float
    take_profit_r: float
    trailing_start_r: float
    trailing_atr_multiplier: float
    max_entries_per_symbol_per_day: int
    # Pullback-to-breakout params
    pullback_1h_lookback: int = 20
    pullback_max_bars: int = 36  # max 5m bars after 1h breakout to find pullback (3 hours)
    pullback_min_retrace_pct: int = 30  # min % of breakout move to retrace
    pullback_ema_period: int = 9  # fast EMA for momentum confirmation


@dataclass
class ChopFilterConfig:
    adx_threshold: int
    atr_percentile_threshold: int
    atr_percentile_window_days: int
    consecutive_failure_pause_hours: int
    consecutive_failure_count: int


@dataclass
class TrendFilterConfig:
    ema_fast: int
    ema_slow: int


@dataclass
class SessionBlackout:
    start: str
    end: str
    days: list[str]


@dataclass
class EventFilterConfig:
    pre_event_blackout_minutes: int
    post_event_blackout_minutes: int
    emergency_close_swing_pct: float


@dataclass
class ExecutionConfig:
    max_holding_hours: int
    stale_position_hours: int
    stale_position_min_r: float
    max_total_daily_entries: int


@dataclass
class FilterConfig:
    trend: TrendFilterConfig
    chop: ChopFilterConfig
    session_blackouts: list[SessionBlackout]
    event: EventFilterConfig


@dataclass
class ScalingPhase:
    min_trades: int
    risk_pct: float
    max_leverage: int
    min_profit_factor: float | None = None
    max_drawdown_pct: float | None = None
    min_sharpe: float | None = None


@dataclass
class FundingArbStrategyConfig:
    entry_rate_threshold: float
    exit_rate_threshold: float
    negative_rate_exit: bool
    basis_blowout_pct: float
    max_holding_hours: int
    min_holding_hours: int
    notional_per_position: float


@dataclass
class SpotFeeConfig:
    spot_fee_taker_pct: float
    spot_fee_maker_pct: float


@dataclass
class MultiAssetFundingConfig:
    symbols: list[str]


@dataclass
class ExchangeConfig:
    name: str
    testnet: bool


@dataclass
class LiveConfig:
    poll_interval_seconds: int = 3600
    funding_check_seconds: int = 300
    max_slippage_pct: float = 0.1
    order_timeout_seconds: int = 30
    position_file: str = "positions.json"


@dataclass
class Settings:
    symbols: list[str]
    leverage: int
    starting_capital: float
    risk: RiskConfig
    breakout: BreakoutStrategyConfig
    filters: FilterConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    scaling_phases: list[ScalingPhase]
    funding_arb: FundingArbStrategyConfig | None = None
    spot_fees: SpotFeeConfig | None = None
    multi_asset_funding: MultiAssetFundingConfig | None = None
    exchange: ExchangeConfig | None = None
    live: LiveConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        with open(path) as f:
            raw = yaml.safe_load(f)

        trading = raw["trading"]
        risk_raw = raw["risk"]
        strat_raw = raw["strategy"]["breakout"]
        filt_raw = raw["filters"]
        exec_raw = raw["execution"]
        bt_raw = raw["backtest"]
        scale_raw = raw["scaling"]["phases"]

        risk = RiskConfig(
            risk_per_trade_pct=risk_raw["risk_per_trade_pct"],
            daily_loss_cap_pct=risk_raw["daily_loss_cap_pct"],
            weekly_loss_cap_pct=risk_raw["weekly_loss_cap_pct"],
            max_concurrent_positions=risk_raw["max_concurrent_positions"],
            max_same_direction_correlated=risk_raw["max_same_direction_correlated"],
            max_total_exposure_pct=risk_raw["max_total_exposure_pct"],
            max_total_open_risk_pct=risk_raw["max_total_open_risk_pct"],
            consecutive_loss_cooldown_count=risk_raw["consecutive_loss_cooldown_count"],
            consecutive_loss_cooldown_hours=risk_raw["consecutive_loss_cooldown_hours"],
            drawdown_reduction_threshold_pct=risk_raw["drawdown_reduction_threshold_pct"],
            drawdown_halt_threshold_pct=risk_raw["drawdown_halt_threshold_pct"],
        )

        breakout = BreakoutStrategyConfig(
            lookback_candles=strat_raw["lookback_candles"],
            volume_multiplier=strat_raw["volume_multiplier"],
            atr_period=strat_raw["atr_period"],
            stop_atr_multiplier=strat_raw["stop_atr_multiplier"],
            take_profit_r=strat_raw["take_profit_r"],
            trailing_start_r=strat_raw["trailing_start_r"],
            trailing_atr_multiplier=strat_raw["trailing_atr_multiplier"],
            max_entries_per_symbol_per_day=strat_raw["max_entries_per_symbol_per_day"],
            pullback_1h_lookback=strat_raw.get("pullback_1h_lookback", 20),
            pullback_max_bars=strat_raw.get("pullback_max_bars", 36),
            pullback_min_retrace_pct=strat_raw.get("pullback_min_retrace_pct", 30),
            pullback_ema_period=strat_raw.get("pullback_ema_period", 9),
        )

        trend = TrendFilterConfig(
            ema_fast=filt_raw["trend"]["ema_fast"],
            ema_slow=filt_raw["trend"]["ema_slow"],
        )

        chop = ChopFilterConfig(
            adx_threshold=filt_raw["chop"]["adx_threshold"],
            atr_percentile_threshold=filt_raw["chop"]["atr_percentile_threshold"],
            atr_percentile_window_days=filt_raw["chop"]["atr_percentile_window_days"],
            consecutive_failure_pause_hours=filt_raw["chop"]["consecutive_failure_pause_hours"],
            consecutive_failure_count=filt_raw["chop"]["consecutive_failure_count"],
        )

        session_blackouts = [
            SessionBlackout(start=b["start"], end=b["end"], days=b["days"])
            for b in filt_raw["session"]["blackout_periods_utc"]
        ]

        event = EventFilterConfig(
            pre_event_blackout_minutes=filt_raw["event"]["pre_event_blackout_minutes"],
            post_event_blackout_minutes=filt_raw["event"]["post_event_blackout_minutes"],
            emergency_close_swing_pct=filt_raw["event"]["emergency_close_swing_pct"],
        )

        filters = FilterConfig(
            trend=trend,
            chop=chop,
            session_blackouts=session_blackouts,
            event=event,
        )

        execution = ExecutionConfig(
            max_holding_hours=exec_raw["max_holding_hours"],
            stale_position_hours=exec_raw["stale_position_hours"],
            stale_position_min_r=exec_raw["stale_position_min_r"],
            max_total_daily_entries=exec_raw["max_total_daily_entries"],
        )

        backtest_cfg = BacktestConfig(
            fee_taker_pct=bt_raw["fee_taker_pct"],
            fee_maker_pct=bt_raw["fee_maker_pct"],
            slippage_entry_pct=bt_raw["slippage_entry_pct"],
            slippage_exit_pct=bt_raw["slippage_exit_pct"],
            include_funding=bt_raw["include_funding"],
            monte_carlo_runs=bt_raw["monte_carlo_runs"],
            monte_carlo_max_dd_95_pct=bt_raw["monte_carlo_max_dd_95_pct"],
            walk_forward_train_months=bt_raw["walk_forward_train_months"],
            walk_forward_test_months=bt_raw["walk_forward_test_months"],
            start_date=bt_raw["start_date"],
            end_date=bt_raw["end_date"],
        )

        scaling_phases = [
            ScalingPhase(
                min_trades=p["min_trades"],
                risk_pct=p["risk_pct"],
                max_leverage=p["max_leverage"],
                min_profit_factor=p.get("min_profit_factor"),
                max_drawdown_pct=p.get("max_drawdown_pct"),
                min_sharpe=p.get("min_sharpe"),
            )
            for p in scale_raw
        ]

        # Optional spread strategy configs
        funding_arb_cfg = None
        fa_raw = raw.get("strategy", {}).get("funding_arb")
        if fa_raw:
            funding_arb_cfg = FundingArbStrategyConfig(
                entry_rate_threshold=fa_raw["entry_rate_threshold"],
                exit_rate_threshold=fa_raw["exit_rate_threshold"],
                negative_rate_exit=fa_raw["negative_rate_exit"],
                basis_blowout_pct=fa_raw["basis_blowout_pct"],
                max_holding_hours=fa_raw["max_holding_hours"],
                min_holding_hours=fa_raw["min_holding_hours"],
                notional_per_position=fa_raw["notional_per_position"],
            )

        spot_fees_cfg = None
        sf_raw = raw.get("spot_fees")
        if sf_raw:
            spot_fees_cfg = SpotFeeConfig(
                spot_fee_taker_pct=sf_raw["spot_fee_taker_pct"],
                spot_fee_maker_pct=sf_raw["spot_fee_maker_pct"],
            )

        multi_asset_cfg = None
        ma_raw = raw.get("strategy", {}).get("multi_asset_funding")
        if ma_raw:
            multi_asset_cfg = MultiAssetFundingConfig(
                symbols=ma_raw["symbols"],
            )

        exchange_cfg = None
        ex_raw = raw.get("exchange")
        if ex_raw:
            exchange_cfg = ExchangeConfig(
                name=ex_raw["name"],
                testnet=ex_raw.get("testnet", False),
            )

        live_cfg = None
        live_raw = raw.get("live")
        if live_raw:
            live_cfg = LiveConfig(
                poll_interval_seconds=live_raw.get("poll_interval_seconds", 3600),
                funding_check_seconds=live_raw.get("funding_check_seconds", 300),
                max_slippage_pct=live_raw.get("max_slippage_pct", 0.1),
                order_timeout_seconds=live_raw.get("order_timeout_seconds", 30),
                position_file=live_raw.get("position_file", "positions.json"),
            )

        return cls(
            symbols=trading["symbols"],
            leverage=trading["leverage"],
            starting_capital=trading["starting_capital"],
            risk=risk,
            breakout=breakout,
            filters=filters,
            execution=execution,
            backtest=backtest_cfg,
            scaling_phases=scaling_phases,
            funding_arb=funding_arb_cfg,
            spot_fees=spot_fees_cfg,
            multi_asset_funding=multi_asset_cfg,
            exchange=exchange_cfg,
            live=live_cfg,
        )
