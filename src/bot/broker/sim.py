"""Dry-run broker: no network, no orders. Lets the full pipeline run without Alpaca keys."""
from __future__ import annotations

from bot.models import AccountState, OrderPlan, PositionDetail, SubmittedOrder


class DryRunBroker:
    name = "dry-run"
    is_paper = True
    quote_feed = "sim"

    def __init__(self, equity: float = 100_000.0, positions: dict[str, float] | None = None,
                 details: dict[str, PositionDetail] | None = None, non_fractionable: set[str] | None = None,
                 quotes: dict[str, dict] | None = None, sim_spread_pct: float | None = None):
        positions = positions or {}
        self._acct = AccountState(equity=equity, last_equity=equity, cash=equity - sum(positions.values()),
                                  buying_power=equity, positions=positions, details=details or {})
        self._non_frac = set(non_fractionable or ())
        self._quotes = quotes or {}
        # sim_spread_pct: synthesize a quote of that spread around seeded reference prices (off by default)
        self._sim_spread = sim_spread_pct
        self._ref: dict[str, float] = {}
        self.submitted: list[OrderPlan] = []

    def seed_prices(self, prices: dict[str, float]) -> None:
        """Reference prices (last closes) for synthesized quotes; no effect unless sim_spread_pct is set."""
        self._ref.update({k: float(v) for k, v in prices.items() if v})

    def get_account(self) -> AccountState:
        return self._acct

    def is_market_open(self) -> bool:
        return True

    def tradable(self, symbols):
        return {s: True for s in symbols}

    def assets(self, symbols):
        return {s: {"tradable": True, "fractionable": s not in self._non_frac} for s in symbols}

    def latest_quote(self, symbol: str) -> dict | None:
        if symbol in self._quotes:
            return self._quotes[symbol]
        px = self._ref.get(symbol)
        if self._sim_spread is None or not px:
            return None
        import datetime as dt

        h = px * self._sim_spread / 2
        return {"bid": round(px - h, 4), "ask": round(px + h, 4),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    def submit(self, order: OrderPlan) -> SubmittedOrder:
        self.submitted.append(order)
        return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                              broker_order_id=None, status="simulated")

    def cancel_all(self) -> None:
        pass

    def cancel_order(self, broker_order_id: str) -> None:
        pass

    def order_status(self, broker_order_id: str) -> dict:
        return {"status": "simulated", "filled_qty": 0.0, "filled_avg_price": None}
