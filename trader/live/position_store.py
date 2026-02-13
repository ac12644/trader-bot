from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from trader.models.spread_types import SpreadLeg, SpreadPosition
from trader.models.types import Direction

logger = structlog.get_logger()


class PositionStore:
    """Persist open positions to JSON for crash recovery."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, positions: dict[str, SpreadPosition]) -> None:
        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "positions": {
                symbol: _serialize_position(pos)
                for symbol, pos in positions.items()
            },
        }
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.rename(self.path)
        logger.debug("positions_saved", count=len(positions))

    def load(self) -> dict[str, SpreadPosition]:
        if not self.path.exists():
            return {}

        try:
            with open(self.path) as f:
                data = json.load(f)
            positions = {}
            for symbol, pos_dict in data.get("positions", {}).items():
                positions[symbol] = _deserialize_position(pos_dict)
            logger.info(
                "positions_loaded",
                count=len(positions),
                updated_at=data.get("updated_at"),
            )
            return positions
        except Exception as e:
            logger.error("position_load_failed", error=str(e))
            return {}


def _serialize_position(pos: SpreadPosition) -> dict:
    return {
        "leg_a": _serialize_leg(pos.leg_a),
        "leg_b": _serialize_leg(pos.leg_b),
        "entry_time": pos.entry_time.isoformat(),
        "strategy_name": pos.strategy_name,
        "funding_rate_at_entry": pos.funding_rate_at_entry,
        "basis_at_entry": pos.basis_at_entry,
        "accumulated_funding_total": pos.accumulated_funding_total,
        "entry_fees_total": pos.entry_fees_total,
    }


def _serialize_leg(leg: SpreadLeg) -> dict:
    return {
        "symbol": leg.symbol,
        "direction": leg.direction.value,
        "entry_price": leg.entry_price,
        "current_price": leg.current_price,
        "notional_usd": leg.notional_usd,
        "is_perp": leg.is_perp,
        "accumulated_funding": leg.accumulated_funding,
    }


def _deserialize_position(d: dict) -> SpreadPosition:
    return SpreadPosition(
        leg_a=_deserialize_leg(d["leg_a"]),
        leg_b=_deserialize_leg(d["leg_b"]),
        entry_time=datetime.fromisoformat(d["entry_time"]),
        strategy_name=d["strategy_name"],
        funding_rate_at_entry=d.get("funding_rate_at_entry", 0.0),
        basis_at_entry=d.get("basis_at_entry", 0.0),
        accumulated_funding_total=d.get("accumulated_funding_total", 0.0),
        entry_fees_total=d.get("entry_fees_total", 0.0),
    )


def _deserialize_leg(d: dict) -> SpreadLeg:
    return SpreadLeg(
        symbol=d["symbol"],
        direction=Direction(d["direction"]),
        entry_price=d["entry_price"],
        current_price=d["current_price"],
        notional_usd=d["notional_usd"],
        is_perp=d["is_perp"],
        accumulated_funding=d.get("accumulated_funding", 0.0),
    )
