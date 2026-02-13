from __future__ import annotations

from collections import defaultdict

import numpy as np

from trader.backtest.spread_metrics import compute_spread_metrics, run_spread_monte_carlo
from trader.models.spread_types import SpreadTrade


def compute_multi_asset_metrics(
    results: dict[str, list[SpreadTrade]],
    starting_capital: float,
) -> dict:
    """Aggregate funding arb results across all symbols.

    Returns:
        {
            "portfolio": {total_profit, total_funding, total_fees, total_trades, ...},
            "by_symbol": {symbol: per_symbol_metrics, ...},
            "best_symbols": [(symbol, profit), ...],
            "worst_symbols": [(symbol, profit), ...],
            "monte_carlo": {...},
        }
    """
    if not results:
        return {"error": "no_results", "portfolio": {"total_trades": 0}}

    # Per-symbol metrics
    by_symbol: dict[str, dict] = {}
    for symbol, trades in results.items():
        if trades:
            by_symbol[symbol] = compute_spread_metrics(trades, starting_capital)
        else:
            by_symbol[symbol] = {"total_trades": 0, "net_profit_usd": 0.0}

    # Aggregate all trades for portfolio-level metrics
    all_trades: list[SpreadTrade] = []
    for trades in results.values():
        all_trades.extend(trades)

    if not all_trades:
        return {
            "portfolio": {"total_trades": 0, "net_profit_usd": 0.0},
            "by_symbol": by_symbol,
            "best_symbols": [],
            "worst_symbols": [],
        }

    # Sort trades by exit time for proper equity curve
    all_trades.sort(key=lambda t: t.exit_time)

    # Portfolio-level aggregation
    total_profit = sum(t.pnl_total for t in all_trades)
    total_funding = sum(t.pnl_funding for t in all_trades)
    total_leg_pnl = sum(t.pnl_leg_a + t.pnl_leg_b for t in all_trades)
    total_fees = sum(t.fees_paid for t in all_trades)
    total_trades = len(all_trades)

    wins = [t for t in all_trades if t.pnl_total > 0]
    losses = [t for t in all_trades if t.pnl_total <= 0]

    pnls = np.array([t.pnl_total for t in all_trades])
    equity_curve = np.cumsum(pnls) + starting_capital
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak * 100
    max_dd_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    gross_profit = sum(t.pnl_total for t in wins) if wins else 0.0
    gross_loss = sum(abs(t.pnl_total) for t in losses) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_holding_hours = np.mean([t.holding_duration_minutes / 60 for t in all_trades])

    # Symbols active (had at least 1 trade)
    active_symbols = [s for s, trades in results.items() if len(trades) > 0]

    # Exit reasons aggregated
    exit_reasons: dict[str, int] = defaultdict(int)
    for t in all_trades:
        exit_reasons[t.exit_reason.value] += 1

    # Concurrent position analysis: max overlapping positions at any point
    max_concurrent = _max_concurrent_positions(all_trades)

    portfolio = {
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(len(wins) / total_trades * 100, 2) if total_trades > 0 else 0.0,
        "profit_factor": round(profit_factor, 3),
        "net_profit_usd": round(total_profit, 2),
        "net_return_pct": round(total_profit / starting_capital * 100, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_funding_collected_usd": round(total_funding, 2),
        "total_leg_pnl_usd": round(total_leg_pnl, 2),
        "total_fees_usd": round(total_fees, 2),
        "final_equity": round(starting_capital + total_profit, 2),
        "avg_holding_hours": round(float(avg_holding_hours), 1),
        "symbols_traded": len(active_symbols),
        "symbols_total": len(results),
        "max_concurrent_positions": max_concurrent,
        "exit_reasons": dict(exit_reasons),
    }

    # Rank symbols by profit
    symbol_profits = [
        (symbol, by_symbol[symbol].get("net_profit_usd", 0.0))
        for symbol in by_symbol
    ]
    symbol_profits.sort(key=lambda x: x[1], reverse=True)

    best_symbols = [(s, round(p, 2)) for s, p in symbol_profits if p > 0]
    worst_symbols = [(s, round(p, 2)) for s, p in symbol_profits if p <= 0]
    worst_symbols.reverse()  # worst first

    # Portfolio Monte Carlo on aggregated trades
    mc = run_spread_monte_carlo(all_trades, starting_capital, 1000)

    return {
        "portfolio": portfolio,
        "by_symbol": by_symbol,
        "best_symbols": best_symbols,
        "worst_symbols": worst_symbols,
        "monte_carlo": mc,
    }


def _max_concurrent_positions(trades: list[SpreadTrade]) -> int:
    """Calculate the maximum number of positions open at the same time."""
    if not trades:
        return 0

    events: list[tuple[float, int]] = []
    for t in trades:
        events.append((t.entry_time.timestamp(), 1))   # open
        events.append((t.exit_time.timestamp(), -1))    # close

    events.sort(key=lambda x: (x[0], x[1]))

    max_open = 0
    current = 0
    for _, delta in events:
        current += delta
        max_open = max(max_open, current)

    return max_open
