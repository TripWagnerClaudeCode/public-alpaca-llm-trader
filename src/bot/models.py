"""Shared typed records passed between pipeline stages."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class View(BaseModel):
    """A portfolio manager view: expected return relative to cash over the horizon."""

    symbol: str
    expected_return: float = Field(description="Expected total return over the horizon, e.g. 0.02 = +2%")
    confidence: float = Field(ge=0.0, le=1.0, description="0 = no confidence, 1 = certain")
    rationale: str


class OrderPlan(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    notional: float  # dollars, always positive
    current_weight: float
    target_weight: float
    close_position: bool = False  # sell the entire position (avoids fractional dust)
    qty: float | None = None          # whole-share limit order (order_style=auto, non-fractionable)
    limit_price: float | None = None


class PositionDetail(BaseModel):
    qty: float
    avg_entry_price: float
    current_price: float

    @property
    def pnl_pct(self) -> float:
        return self.current_price / self.avg_entry_price - 1 if self.avg_entry_price > 0 else 0.0


class AccountState(BaseModel):
    equity: float
    last_equity: float  # equity at previous close
    cash: float
    buying_power: float
    trading_blocked: bool = False
    positions: dict[str, float] = {}  # symbol -> market value ($)
    details: dict[str, PositionDetail] = {}  # optional: qty / avg entry / price (stops, whole-share sells)


class SubmittedOrder(BaseModel):
    symbol: str
    side: str
    notional: float | None
    broker_order_id: str | None
    status: str
    error: str | None = None
