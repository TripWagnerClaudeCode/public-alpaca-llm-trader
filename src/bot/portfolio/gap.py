"""Deterministic opening-gap guard (spec 2026-11-04, Morning Check fallback).

A planned buy is cancelled when the stock has gapped down by more than k daily standard deviations
since the analyze close: the move is against the view and the plan was made on stale prices. It
never touches a sell, never resizes an order, and keeps the order (with a note) when any input is
missing, so it can only remove risk.
"""
from __future__ import annotations

import math

from bot.models import OrderPlan


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def gap_guard(orders: list[OrderPlan], mids: dict[str, float | None], prev_close: dict[str, float],
              sigma: dict[str, float], k: float) -> tuple[list[OrderPlan], list[str]]:
    out: list[OrderPlan] = []
    why: list[str] = []
    for o in orders:
        if o.side != "buy":
            out.append(o)
            continue
        mid, prev, sig = mids.get(o.symbol), prev_close.get(o.symbol), sigma.get(o.symbol)
        if not (_ok(mid) and _ok(prev) and _ok(sig)):
            why.append(f"{o.symbol}: gap check skipped (missing quote/close/sigma), buy kept")
            out.append(o)
            continue
        gap = mid / prev - 1
        if gap <= -k * sig:
            why.append(f"{o.symbol}: buy cancelled, gapped {gap:+.2%} vs close (< -{k:g} x daily sigma {sig:.2%})")
            continue
        out.append(o)
    return out, why
