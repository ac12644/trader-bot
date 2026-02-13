from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

import polars as pl

from trader.backtest.cost_model import SpreadCostModel
from trader.backtest.engine import BacktestEngine
from trader.backtest.metrics import compute_all_metrics
from trader.backtest.monte_carlo import run_monte_carlo
from trader.backtest.multi_asset_metrics import compute_multi_asset_metrics
from trader.backtest.multi_asset_runner import MultiAssetFundingRunner
from trader.backtest.walk_forward import walk_forward_validation
from trader.config.settings import Settings
from trader.data.fetcher import fetch_candles
from trader.data.store import load_candles, validate_candles
from trader.strategies.volatility_breakout import VolatilityBreakoutStrategy

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
)

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SETTINGS_PATH = PROJECT_ROOT / "settings.yaml"
EVENTS_PATH = PROJECT_ROOT / "events_calendar.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"


def main() -> None:
    # 1. Load config
    logger.info("loading_settings", path=str(SETTINGS_PATH))
    settings = Settings.from_yaml(SETTINGS_PATH)

    # 2. Fetch data if not cached
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    start_date = datetime.fromisoformat(settings.backtest.start_date)
    end_date = datetime.fromisoformat(settings.backtest.end_date)

    for symbol in settings.symbols:
        for tf in ["5m", "1h"]:
            safe_symbol = symbol.replace("/", "_")
            cache_path = DATA_DIR / f"{safe_symbol}_{tf}.parquet"
            if not cache_path.exists():
                logger.info("fetching_data", symbol=symbol, timeframe=tf)
                fetch_candles(symbol, tf, start_date, end_date, DATA_DIR)
            else:
                logger.info("using_cached_data", symbol=symbol, timeframe=tf)

    # 3. Load and validate
    candles_5m: dict[str, any] = {}
    candles_1h: dict[str, any] = {}

    for symbol in settings.symbols:
        safe_symbol = symbol.replace("/", "_")
        c5 = load_candles(DATA_DIR / f"{safe_symbol}_5m.parquet")
        c1 = load_candles(DATA_DIR / f"{safe_symbol}_1h.parquet")

        warnings_5m = validate_candles(c5, "5m")
        warnings_1h = validate_candles(c1, "1h")
        for w in warnings_5m + warnings_1h:
            logger.warning("data_issue", symbol=symbol, warning=w)

        candles_5m[symbol] = c5
        candles_1h[symbol] = c1
        logger.info(
            "data_loaded",
            symbol=symbol,
            candles_5m=len(c5),
            candles_1h=len(c1),
        )

    # 4. Initialize strategy
    strategy = VolatilityBreakoutStrategy(settings.breakout)

    # 5. Run full backtest
    logger.info("starting_backtest")
    engine = BacktestEngine(settings, strategy, candles_5m, candles_1h, EVENTS_PATH)
    trades = engine.run()

    # 6. Compute metrics
    metrics = compute_all_metrics(trades, settings.starting_capital)

    # 7. Gate checks
    passed = True
    gates: list[dict] = []

    pf = metrics.get("profit_factor", 0)
    if isinstance(pf, (int, float)) and pf < 1.4:
        gates.append({"gate": "profit_factor", "required": 1.4, "actual": pf, "passed": False})
        passed = False
    else:
        gates.append({"gate": "profit_factor", "required": 1.4, "actual": pf, "passed": True})

    # 8. Monte Carlo
    logger.info("running_monte_carlo", simulations=settings.backtest.monte_carlo_runs)
    mc_results = run_monte_carlo(trades, settings.starting_capital, settings.backtest.monte_carlo_runs)

    mc_dd = mc_results.get("max_dd_95th_pct", 100)
    mc_limit = settings.backtest.monte_carlo_max_dd_95_pct
    if mc_dd > mc_limit:
        gates.append({"gate": "monte_carlo_95th_dd", "required": mc_limit, "actual": mc_dd, "passed": False})
        passed = False
    else:
        gates.append({"gate": "monte_carlo_95th_dd", "required": mc_limit, "actual": mc_dd, "passed": True})

    # 9. Walk-forward (can be slow, make optional via flag)
    wf_results: list[dict] = []
    if "--skip-walkforward" not in sys.argv:
        logger.info("running_walk_forward")
        wf_results = walk_forward_validation(
            settings, strategy, candles_5m, candles_1h, EVENTS_PATH,
        )
    else:
        logger.info("walk_forward_skipped")

    # 10. Output
    output = {
        "backtest_metrics": metrics,
        "monte_carlo": mc_results,
        "walk_forward": wf_results,
        "gates": gates,
        "all_gates_passed": passed,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGS_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("results_saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        if k == "exit_reasons":
            print(f"  {k}:")
            for reason, count in v.items():
                print(f"    {reason}: {count}")
        else:
            print(f"  {k}: {v}")
    print("-" * 60)
    print("  MONTE CARLO")
    print("-" * 60)
    for k, v in mc_results.items():
        print(f"  {k}: {v}")
    print("-" * 60)
    print("  GATE CHECKS")
    print("-" * 60)
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {g['gate']}: {g['actual']} (required: {g['required']})")
    print("-" * 60)
    if wf_results:
        print("  WALK-FORWARD WINDOWS")
        print("-" * 60)
        for w in wf_results:
            print(
                f"  Window {w['window']}: "
                f"train PF={w['train_profit_factor']}, "
                f"test PF={w['test_profit_factor']}, "
                f"test return={w['test_return_pct']}%"
            )
    print("=" * 60)
    print(f"  ALL GATES PASSED: {passed}")
    print("=" * 60)


def main_spread() -> None:
    """Run multi-asset funding rate arbitrage backtest."""
    logger.info("loading_settings", path=str(SETTINGS_PATH))
    settings = Settings.from_yaml(SETTINGS_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    start_date = datetime.fromisoformat(settings.backtest.start_date)
    end_date = datetime.fromisoformat(settings.backtest.end_date)

    cost_model = SpreadCostModel(settings.backtest)
    if settings.spot_fees:
        cost_model.set_spot_fees(
            settings.spot_fees.spot_fee_taker_pct,
            settings.spot_fees.spot_fee_maker_pct,
        )

    all_results: dict[str, dict] = {}

    # ── Strategy 1: Multi-Asset Funding Rate Arbitrage ─────────────
    if settings.funding_arb:
        symbols = (
            settings.multi_asset_funding.symbols
            if settings.multi_asset_funding
            else ["BTC/USDT"]  # fallback to single-asset
        )
        logger.info("=== MULTI-ASSET FUNDING RATE ARBITRAGE ===", symbols=len(symbols))

        runner = MultiAssetFundingRunner(
            config=settings.funding_arb,
            symbols=symbols,
            cost_model=cost_model,
            starting_capital=settings.starting_capital,
            data_dir=DATA_DIR,
            start_date=start_date,
            end_date=end_date,
            leverage=20,
        )

        runner.fetch_all_data()
        fa_results = runner.run_all()
        multi_metrics = compute_multi_asset_metrics(
            fa_results, settings.starting_capital,
        )

        all_results["funding_arb"] = multi_metrics
        _print_multi_asset_results(multi_metrics)

    # Save results
    output_path = LOGS_DIR / "spread_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("spread_results_saved", path=str(output_path))


def _print_multi_asset_results(metrics: dict) -> None:
    """Pretty-print multi-asset funding arb results."""
    port = metrics["portfolio"]

    print(f"\n{'=' * 60}")
    print("  MULTI-ASSET FUNDING RATE ARBITRAGE")
    print("=" * 60)

    print("  PORTFOLIO SUMMARY:")
    for k, v in port.items():
        if k == "exit_reasons":
            print(f"    {k}:")
            for reason, count in v.items():
                print(f"      {reason}: {count}")
        else:
            print(f"    {k}: {v}")

    print("-" * 60)
    print("  PER-SYMBOL BREAKDOWN:")
    print("-" * 60)
    for symbol, sym_metrics in metrics.get("by_symbol", {}).items():
        trades = sym_metrics.get("total_trades", 0)
        profit = sym_metrics.get("net_profit_usd", 0.0)
        funding = sym_metrics.get("total_funding_collected_usd", 0.0)
        fees = sym_metrics.get("total_fees_usd", 0.0)
        print(
            f"  {symbol:15s}  trades={trades:3d}  "
            f"pnl=${profit:+8.2f}  funding=${funding:8.2f}  fees=${fees:6.2f}"
        )

    if metrics.get("best_symbols"):
        print("-" * 60)
        print("  BEST SYMBOLS:")
        for sym, profit in metrics["best_symbols"][:5]:
            print(f"    {sym}: ${profit:+.2f}")

    if metrics.get("worst_symbols"):
        print("  WORST SYMBOLS:")
        for sym, profit in metrics["worst_symbols"][:5]:
            print(f"    {sym}: ${profit:+.2f}")

    mc = metrics.get("monte_carlo", {})
    if mc:
        print("-" * 60)
        print("  MONTE CARLO")
        print("-" * 60)
        for k, v in mc.items():
            print(f"    {k}: {v}")

    print("=" * 60)


if __name__ == "__main__":
    if "--spread" in sys.argv:
        main_spread()
    else:
        main()
