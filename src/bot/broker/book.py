"""Persistent simulated book: the broker behind the frozen control profiles.

No orders ever reach a real broker. Buys fill at the quote's ask, sells at the bid (falling back
to the last marked price), and the whole book (cash, positions, orders) is saved as JSON after
every change, so the control keeps a real portfolio across runs. Prices and the market clock
come from injected callables (Alpaca market data in production, fakes in tests).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

from bot.models import AccountState, OrderPlan, PositionDetail, SubmittedOrder

VERSION = 1
NY = ZoneInfo("America/New_York")


def _et_today() -> dt.date:
    return dt.datetime.now(NY).date()


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


class BookBroker:
    name = "book"
    is_paper = True
    quote_feed = "book"

    def __init__(self, book_path: Path, quotes: Callable[[str], dict | None], clock_open: Callable[[], bool],
                 starting_cash: float = 100_000.0, init: bool = False, owner: str = "control",
                 today: Callable[[], dt.date] = _et_today):
        self.path = Path(book_path)
        self._quotes, self._clock_open, self._today, self.owner = quotes, clock_open, today, owner
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError as e:
                raise ValueError(f"control book {self.path} is not valid JSON: {e}") from None
            if not isinstance(d, dict) or d.get("version") != VERSION:
                raise ValueError(f"control book {self.path} has an unknown format")
            if d.get("owner") != owner:
                raise ValueError(f"control book {self.path} belongs to {d.get('owner')!r}, not {owner!r}")
            self.state = d
        elif init:
            self.state = {"version": VERSION, "owner": owner, "cash": float(starting_cash), "positions": {},
                          "last_equity": float(starting_cash), "equity_date": None, "orders": {}}
            self._save()
        else:
            raise FileNotFoundError(f"control book {self.path} missing; run reconcile with --init-book")

    # ----------------------------------------------------------------- persistence
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def _equity(self) -> float:
        return self.state["cash"] + sum(p["qty"] * p["last_price"] for p in self.state["positions"].values())

    # ----------------------------------------------------------------- broker protocol
    def get_account(self) -> AccountState:
        today = self._today().isoformat()
        if self.state.get("equity_date") != today:
            self.state["last_equity"] = self._equity()
            self.state["equity_date"] = today
            self._save()
        pos = self.state["positions"]
        details = {s: PositionDetail(qty=p["qty"], avg_entry_price=p["avg_entry_price"], current_price=p["last_price"])
                   for s, p in pos.items()}
        eq = self._equity()
        return AccountState(equity=eq, last_equity=self.state["last_equity"], cash=self.state["cash"],
                            buying_power=self.state["cash"], positions={s: p["qty"] * p["last_price"] for s, p in pos.items()},
                            details=details)

    def is_market_open(self) -> bool:
        return bool(self._clock_open())

    def tradable(self, symbols):
        return {s: True for s in symbols}

    def assets(self, symbols):
        return {s: {"tradable": True, "fractionable": True} for s in symbols}

    def latest_quote(self, symbol: str) -> dict | None:
        return self._quotes(symbol)

    def mark(self, prices: dict[str, float]) -> None:
        for s, px in prices.items():
            if s in self.state["positions"] and _ok(px):
                self.state["positions"][s]["last_price"] = float(px)
        self._save()

    def _price(self, symbol: str, side: str) -> float | None:
        q = self._quotes(symbol) or {}
        px = q.get("ask") if side == "buy" else q.get("bid")
        if _ok(px):
            return float(px)
        last = self.state["positions"].get(symbol, {}).get("last_price")
        return float(last) if _ok(last) else None

    def submit(self, order: OrderPlan) -> SubmittedOrder:
        oid = f"book-{uuid.uuid4().hex[:16]}"

        def done(status: str, qty: float = 0.0, px: float | None = None, error: str | None = None):
            self.state["orders"][oid] = {"symbol": order.symbol, "side": order.side, "status": status,
                                         "filled_qty": qty, "filled_avg_price": px}
            self._save()
            return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                                  broker_order_id=oid, status=status, error=error)

        px = self._price(order.symbol, order.side)
        if px is None:
            return done("rejected", error="no price")
        pos = self.state["positions"]
        if order.side == "buy":
            qty = order.qty if order.qty is not None else order.notional / px
            cost = qty * px
            if cost > self.state["cash"] + 1e-6:
                return done("rejected", error="insufficient cash")
            p = pos.get(order.symbol, {"qty": 0.0, "avg_entry_price": 0.0, "last_price": px})
            new_qty = p["qty"] + qty
            p["avg_entry_price"] = (p["qty"] * p["avg_entry_price"] + cost) / new_qty
            p["qty"], p["last_price"] = new_qty, px
            pos[order.symbol] = p
            self.state["cash"] -= cost
            return done("filled", qty, px)
        held = pos.get(order.symbol)
        if not held:
            return done("rejected", error="no position")
        qty = held["qty"] if order.close_position else min(held["qty"], order.qty if order.qty is not None
                                                           else order.notional / px)
        self.state["cash"] += qty * px
        held["qty"] -= qty
        held["last_price"] = px
        if held["qty"] <= 1e-9:
            del pos[order.symbol]
        return done("filled", qty, px)

    def cancel_all(self) -> None:
        pass

    def order_status(self, broker_order_id: str) -> dict:
        o = self.state["orders"].get(broker_order_id)
        if o is None:
            return {"status": "unknown", "filled_qty": 0.0, "filled_avg_price": None}
        return {"status": o["status"], "filled_qty": o["filled_qty"], "filled_avg_price": o["filled_avg_price"]}
