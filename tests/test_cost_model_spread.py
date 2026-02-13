from __future__ import annotations

import pytest

from trader.backtest.cost_model import SpreadCostModel
from trader.config.settings import BacktestConfig


@pytest.fixture
def bt_config() -> BacktestConfig:
    return BacktestConfig(
        fee_taker_pct=0.05,
        fee_maker_pct=0.02,
        slippage_entry_pct=0.01,
        slippage_exit_pct=0.01,
        include_funding=True,
        monte_carlo_runs=1000,
        monte_carlo_max_dd_95_pct=15.0,
        walk_forward_train_months=6,
        walk_forward_test_months=2,
        start_date="2024-02-12",
        end_date="2026-02-12",
    )


@pytest.fixture
def cost_model(bt_config: BacktestConfig) -> SpreadCostModel:
    return SpreadCostModel(bt_config)


class TestSpreadCostModel:
    def test_default_spot_fees(self, cost_model: SpreadCostModel):
        assert cost_model.spot_fee_taker == pytest.approx(0.001)
        assert cost_model.spot_fee_maker == pytest.approx(0.001)

    def test_perp_fees_inherited(self, cost_model: SpreadCostModel):
        assert cost_model.fee_taker == pytest.approx(0.0005)  # 0.05% / 100
        assert cost_model.fee_maker == pytest.approx(0.0002)  # 0.02% / 100

    def test_set_spot_fees(self, cost_model: SpreadCostModel):
        cost_model.set_spot_fees(taker_pct=0.10, maker_pct=0.08)
        assert cost_model.spot_fee_taker == pytest.approx(0.001)  # 0.10 / 100
        assert cost_model.spot_fee_maker == pytest.approx(0.0008)  # 0.08 / 100

    def test_entry_fee_spread_both_perp(self, cost_model: SpreadCostModel):
        fee = cost_model.entry_fee_spread(1000.0, 1000.0, leg_b_is_spot=False)
        # Both legs: 1000 * 0.0005 + 1000 * 0.0005 = 1.0
        assert fee == pytest.approx(1.0)

    def test_entry_fee_spread_spot_leg(self, cost_model: SpreadCostModel):
        fee = cost_model.entry_fee_spread(1000.0, 1000.0, leg_b_is_spot=True)
        # Perp taker: 1000 * 0.0005 = 0.50
        # Spot taker: 1000 * 0.001 = 1.00
        assert fee == pytest.approx(1.50)

    def test_exit_fee_spread_both_perp(self, cost_model: SpreadCostModel):
        fee = cost_model.exit_fee_spread(1000.0, 1000.0, leg_b_is_spot=False)
        # Both legs: 1000 * 0.0002 + 1000 * 0.0002 = 0.40
        assert fee == pytest.approx(0.40)

    def test_exit_fee_spread_spot_leg(self, cost_model: SpreadCostModel):
        fee = cost_model.exit_fee_spread(1000.0, 1000.0, leg_b_is_spot=True)
        # Perp maker: 1000 * 0.0002 = 0.20
        # Spot maker: 1000 * 0.001 = 1.00
        assert fee == pytest.approx(1.20)

    def test_asymmetric_notionals(self, cost_model: SpreadCostModel):
        fee = cost_model.entry_fee_spread(750.0, 500.0, leg_b_is_spot=True)
        # Perp: 750 * 0.0005 = 0.375
        # Spot: 500 * 0.001 = 0.50
        assert fee == pytest.approx(0.875)
