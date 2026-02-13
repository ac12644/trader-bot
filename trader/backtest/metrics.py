from __future__ import annotations

from collections import defaultdict

import numpy as np

from trader.models.types import Trade


def compute_all_metrics(trades: list[Trade], starting_capital: float) -> dict:
    if not trades:
        return {"error": "no_trades", "total_trades": 0}

    pnls = np.array([t.pnl_usd for t in trades])
    r_multiples = np.array([t.r_multiple for t in trades])
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

    # Exit reason breakdown
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
        "expectancy_r": round(float(np.mean(r_multiples)), 4),
        "recovery_factor": round(recovery_factor, 3),
        "net_profit_usd": round(net_profit, 2),
        "net_return_pct": round(net_profit / starting_capital * 100, 2),
        "total_fees_usd": round(total_fees, 2),
        "final_equity": round(starting_capital + net_profit, 2),
        "avg_holding_minutes": round(float(avg_holding), 1),
        "exit_reasons": dict(exit_reasons),
    }


def _aggregate_daily_pnls(trades: list[Trade]) -> list[float]:
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.exit_time.strftime("%Y-%m-%d")
        daily[day] += t.pnl_usd
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
