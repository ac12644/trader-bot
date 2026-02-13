from __future__ import annotations

import numpy as np

from trader.models.types import Trade


def run_monte_carlo(
    trades: list[Trade],
    starting_capital: float,
    num_simulations: int = 1000,
) -> dict:
    if not trades:
        return {"error": "no_trades"}

    pnls = np.array([t.pnl_usd for t in trades])
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
