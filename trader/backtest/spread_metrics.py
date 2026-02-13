from __future__ import annotations

from collections import defaultdict

import numpy as np

from trader.models.spread_types import SpreadTrade


def compute_spread_metrics(
    trades: list[SpreadTrade],
    starting_capital: float,
) -> dict:
    """Compute metrics for spread/pair trades including funding income."""
    if not trades:
        return {"error": "no_trades", "total_trades": 0}

    pnls = np.array([t.pnl_total for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Equity curve and drawdown
    equity_curve = np.cumsum(pnls) + starting_capital
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak * 100

    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily returns for Sharpe
    daily_pnls = _aggregate_daily_pnls(trades)
    daily_returns = np.array(daily_pnls) if daily_pnls else np.array([0.0])
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))

    max_consec = _max_consecutive_losses(pnls)

    net_profit = float(np.sum(pnls))
    max_dd_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd_usd = float(np.max(peak - equity_curve)) if len(equity_curve) > 0 else 0.0
    recovery_factor = net_profit / max_dd_usd if max_dd_usd > 0 else float("inf")

    total_fees = sum(t.fees_paid for t in trades)
    avg_holding = np.mean([t.holding_duration_minutes for t in trades])

    # Spread-specific metrics
    total_funding = sum(t.pnl_funding for t in trades)
    total_leg_pnl = sum(t.pnl_leg_a + t.pnl_leg_b for t in trades)
    avg_holding_hours = float(avg_holding) / 60.0

    exit_reasons: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason.value] += 1

    return {
        "total_trades": total_trades,
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_consecutive_losses": max_consec,
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "avg_win_loss_ratio": round(avg_win / avg_loss, 3) if avg_loss > 0 else float("inf"),
        "recovery_factor": round(recovery_factor, 3),
        "net_profit_usd": round(net_profit, 2),
        "net_return_pct": round(net_profit / starting_capital * 100, 2),
        "total_fees_usd": round(total_fees, 2),
        "final_equity": round(starting_capital + net_profit, 2),
        "avg_holding_hours": round(avg_holding_hours, 1),
        # Spread-specific
        "total_funding_collected_usd": round(total_funding, 2),
        "total_leg_pnl_usd": round(total_leg_pnl, 2),
        "exit_reasons": dict(exit_reasons),
    }


def spread_trades_to_mc_pnls(trades: list[SpreadTrade]) -> np.ndarray:
    """Extract PnL array from spread trades for Monte Carlo simulation."""
    return np.array([t.pnl_total for t in trades])


def run_spread_monte_carlo(
    trades: list[SpreadTrade],
    starting_capital: float,
    num_simulations: int = 1000,
) -> dict:
    """Monte Carlo simulation for spread trades (same logic as directional MC)."""
    if not trades:
        return {"error": "no_trades"}

    pnls = spread_trades_to_mc_pnls(trades)
    max_drawdowns = np.zeros(num_simulations)
    final_equities = np.zeros(num_simulations)

    for i in range(num_simulations):
        shuffled = np.random.permutation(pnls)
        equity_curve = np.cumsum(shuffled) + starting_capital
        peak = np.maximum.accumulate(equity_curve)
        dd_pct = (peak - equity_curve) / peak * 100
        max_drawdowns[i] = np.max(dd_pct)
        final_equities[i] = equity_curve[-1]

    return {
        "simulations": num_simulations,
        "max_dd_95th_pct": round(float(np.percentile(max_drawdowns, 95)), 2),
        "max_dd_99th_pct": round(float(np.percentile(max_drawdowns, 99)), 2),
        "max_dd_median_pct": round(float(np.median(max_drawdowns)), 2),
        "max_dd_mean_pct": round(float(np.mean(max_drawdowns)), 2),
        "final_equity_5th_pct": round(float(np.percentile(final_equities, 5)), 2),
        "final_equity_median": round(float(np.median(final_equities)), 2),
        "final_equity_95th_pct": round(float(np.percentile(final_equities, 95)), 2),
        "ruin_probability_pct": round(
            float(np.mean(final_equities < starting_capital * 0.5)) * 100, 2,
        ),
    }


def _aggregate_daily_pnls(trades: list[SpreadTrade]) -> list[float]:
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.exit_time.strftime("%Y-%m-%d")
        daily[day] += t.pnl_total
    return list(daily.values())


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    max_consec = 0
    current = 0
    for pnl in pnls:
        if pnl <= 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec
