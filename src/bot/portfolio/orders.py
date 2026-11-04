"""Deterministic order shaping after planning: liquidity cap and whole-share limit orders.

Every function returns (orders, reasons): each dropped or shrunk order gets a logged reason.
Full exits (close_position) are never shrunk or converted: getting out always takes priority.
"""
from __future__ import annotations

import math

import pandas as pd

from bot.models import AccountState, OrderPlan

FALLBACK_SLIPPAGE = 0.01  # last close +/- 1% when no quote is available


def avg_dollar_volume(bars: pd.DataFrame, n: int = 20) -> float | None:
    """Mean of close x volume over the last n bars; None when unusable."""
    tail = bars.tail(n)
    if len(tail) < n:
        return None
    adv = float((tail["close"] * tail["volume"]).mean())
    return adv if adv > 0 and not math.isnan(adv) else None


def cap_adv(orders: list[OrderPlan], adv: dict[str, float], participation: float,
            min_notional: float) -> tuple[list[OrderPlan], list[str]]:
    """Shrink each order to <= participation x 20-day ADV. Unknown ADV blocks buys (fail closed)."""
    out, reasons = [], []
    for o in orders:
        if o.close_position:
            out.append(o)
            continue
        a = adv.get(o.symbol)
        if a is None:
            if o.side == "buy":
                reasons.append(f"{o.symbol}: buy skipped, no 20-day ADV available")
                continue
            out.append(o)
            continue
        cap = participation * a
        if o.notional <= cap:
            out.append(o)
            continue
        if cap < min_notional:
            reasons.append(f"{o.symbol}: {o.side} skipped, ADV cap ${cap:,.2f} below minimum order")
            continue
        reasons.append(f"{o.symbol}: {o.side} ${o.notional:,.2f} shrunk to ${cap:,.2f} "
                       f"({participation:.1%} of ADV ${a:,.0f})")
        out.append(o.model_copy(update={"notional": round(cap, 2)}))
    return out, reasons


def limit_price(side: str, quote: dict | None, last_close: float | None, buffer: float) -> float | None:
    """Marketable limit: buy at ask*(1+buffer), sell at bid*(1-buffer); fallback last close +/- 1%."""
    if quote and quote.get("bid", 0) > 0 and quote.get("ask", 0) > 0:
        return quote["ask"] * (1 + buffer) if side == "buy" else quote["bid"] * (1 - buffer)
    if last_close and last_close > 0:
        return last_close * (1 + FALLBACK_SLIPPAGE) if side == "buy" else last_close * (1 - FALLBACK_SLIPPAGE)
    return None


def to_whole_share(o: OrderPlan, quote: dict | None, last_close: float | None, buffer: float,
                   held_qty: float | None, min_notional: float) -> tuple[OrderPlan | None, str | None]:
    """Convert a notional order into a whole-share marketable limit order (TIF day)."""
    px = limit_price(o.side, quote, last_close, buffer)
    if px is None:
        return None, f"{o.symbol}: skipped, no quote and no last close for a limit price"
    lp = round(px, 2 if px >= 1 else 4)
    qty = math.floor(o.notional / lp)
    if o.side == "sell" and held_qty is not None:
        qty = min(qty, math.floor(held_qty))
    if qty <= 0:
        return None, f"{o.symbol}: {o.side} ${o.notional:,.2f} skipped, rounds to 0 shares at ${lp:,.2f}"
    notional = round(qty * lp, 2)
    if notional < min_notional:
        return None, f"{o.symbol}: {o.side} {qty} sh x ${lp:,.2f} below minimum order"
    return o.model_copy(update={"qty": float(qty), "limit_price": lp, "notional": notional}), None


FRACTIONAL_QTY_DECIMALS = 6   # Alpaca accepts up to 9


def to_fractional_limit(o: OrderPlan, quote: dict | None, last_close: float | None, buffer: float,
                        min_notional: float) -> tuple[OrderPlan | None, str | None]:
    """Convert a notional buy into a fractional-qty marketable limit order (TIF day), so a gap or a
    thin book can't fill it far above the quote. Qty is rounded down: never spends more than planned."""
    px = limit_price(o.side, quote, last_close, buffer)
    if px is None:
        return None, f"{o.symbol}: skipped, no quote and no last close for a limit price"
    lp = round(px, 2 if px >= 1 else 4)
    scale = 10 ** FRACTIONAL_QTY_DECIMALS
    qty = math.floor(o.notional / lp * scale) / scale
    notional = math.floor(qty * lp * 100) / 100
    if qty <= 0 or notional < min_notional:
        return None, f"{o.symbol}: {o.side} {qty} sh x ${lp:,.2f} below minimum order"
    return o.model_copy(update={"qty": qty, "limit_price": lp, "notional": notional}), None


def shape_whole_share(orders: list[OrderPlan], non_fractionable: set[str], acct: AccountState,
                      quotes: dict[str, dict | None], closes: dict[str, float], buffer: float,
                      min_notional: float, *, limit_fractional_buys: bool = False
                      ) -> tuple[list[OrderPlan], list[str]]:
    """Non-fractionable partial orders -> whole-share limits. With limit_fractional_buys, fractionable
    buys -> fractional limits too. Full exits (close_position) and fractionable sells are left as is."""
    out, reasons = [], []
    for o in orders:
        if o.close_position:
            out.append(o)
            continue
        if o.symbol not in non_fractionable:
            if limit_fractional_buys and o.side == "buy":
                new, why = to_fractional_limit(o, quotes.get(o.symbol), closes.get(o.symbol), buffer, min_notional)
                out.append(new) if new is not None else reasons.append(why)
            else:
                out.append(o)
            continue
        d = acct.details.get(o.symbol)
        new, why = to_whole_share(o, quotes.get(o.symbol), closes.get(o.symbol), buffer,
                                  d.qty if d else None, min_notional)
        if new is None:
            reasons.append(why)
        else:
            out.append(new)
    return out, reasons
