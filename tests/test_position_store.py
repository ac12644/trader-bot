from __future__ import annotations

from datetime import datetime

import pytest

from trader.live.position_store import PositionStore
from trader.models.spread_types import SpreadLeg, SpreadPosition
from trader.models.types import Direction


def _make_position(symbol: str = "BTC/USDT") -> SpreadPosition:
    return SpreadPosition(
        leg_a=SpreadLeg(
            symbol=f"{symbol}:USDT",
            direction=Direction.SHORT,
            entry_price=50000.0,
            current_price=50100.0,
            notional_usd=750.0,
            is_perp=True,
            accumulated_funding=1.25,
        ),
        leg_b=SpreadLeg(
            symbol=symbol,
            direction=Direction.LONG,
            entry_price=49950.0,
            current_price=50050.0,
            notional_usd=750.0,
            is_perp=False,
            accumulated_funding=0.0,
        ),
        entry_time=datetime(2024, 6, 15, 8, 0, 0),
        strategy_name="funding_arb",
        funding_rate_at_entry=0.0005,
        basis_at_entry=50.0,
        accumulated_funding_total=3.75,
        entry_fees_total=1.50,
    )


class TestPositionStore:
    def test_round_trip(self, tmp_path):
        store = PositionStore(tmp_path / "positions.json")
        pos = _make_position()
        store.save({"BTC/USDT": pos})

        loaded = store.load()
        assert "BTC/USDT" in loaded
        p = loaded["BTC/USDT"]
        assert p.leg_a.entry_price == 50000.0
        assert p.leg_a.direction == Direction.SHORT
        assert p.leg_a.is_perp is True
        assert p.leg_a.accumulated_funding == 1.25
        assert p.leg_b.entry_price == 49950.0
        assert p.leg_b.direction == Direction.LONG
        assert p.leg_b.is_perp is False
        assert p.strategy_name == "funding_arb"
        assert p.funding_rate_at_entry == 0.0005
        assert p.basis_at_entry == 50.0
        assert p.accumulated_funding_total == 3.75
        assert p.entry_fees_total == 1.50
        assert p.entry_time == datetime(2024, 6, 15, 8, 0, 0)

    def test_multiple_positions(self, tmp_path):
        store = PositionStore(tmp_path / "positions.json")
        positions = {
            "BTC/USDT": _make_position("BTC/USDT"),
            "ETH/USDT": _make_position("ETH/USDT"),
        }
        store.save(positions)
        loaded = store.load()
        assert len(loaded) == 2
        assert "BTC/USDT" in loaded
        assert "ETH/USDT" in loaded

    def test_load_nonexistent_returns_empty(self, tmp_path):
        store = PositionStore(tmp_path / "doesnt_exist.json")
        loaded = store.load()
        assert loaded == {}

    def test_save_empty(self, tmp_path):
        store = PositionStore(tmp_path / "positions.json")
        store.save({})
        loaded = store.load()
        assert loaded == {}

    def test_overwrite(self, tmp_path):
        store = PositionStore(tmp_path / "positions.json")

        store.save({"BTC/USDT": _make_position()})
        assert len(store.load()) == 1

        store.save({})
        assert len(store.load()) == 0

    def test_corrupted_file_returns_empty(self, tmp_path):
        path = tmp_path / "positions.json"
        path.write_text("not valid json {{{")
        store = PositionStore(path)
        loaded = store.load()
        assert loaded == {}
