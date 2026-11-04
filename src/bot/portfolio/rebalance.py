"""Diff target weights against current holdings into an order plan."""
from __future__ import annotations

import pandas as pd

from bot.config import RiskCfg
from bot.models import AccountState, OrderPlan


def current_weights(acct: AccountState) -> pd.Series:
    if acct.equity <= 0:
        return pd.Series(dtype=float)
    return pd.Series(acct.positions, dtype=float) / acct.equity


def cap_turnover(target: pd.Series, current: pd.Series, max_turnover: float,
                 forced_exits: frozenset[str] = frozenset()) -> pd.Series:
    """Move only part of the way toward target if the full move exceeds max_turnover.
    Over several days the portfolio converges to the target (gradual ramp-in).
    `forced_exits` (stops) go to 0 in full and use up turnover budget first."""
    idx = target.index.union(current.index)
    t, c = target.reindex(idx, fill_value=0.0), current.reindex(idx, fill_value=0.0)
    if forced_exits:
        f = idx.intersection(sorted(forced_exits))
        rest = idx.difference(f)
        t[f] = 0.0
        budget = max(max_turnover - float(c[f].abs().sum()), 0.0)
        turnover = float((t[rest] - c[rest]).abs().sum())
        if turnover <= budget or turnover == 0:
            return t
        out = t.copy()
        out[rest] = c[rest] + (t[rest] - c[rest]) * (budget / turnover)
        return out
    turnover = float((t - c).abs().sum())
    if turnover <= max_turnover or turnover == 0:
        return t
    frac = max_turnover / turnover
    return c + (t - c) * frac


def plan_orders(target: pd.Series, acct: AccountState, risk: RiskCfg,
                forced_exits: frozenset[str] = frozenset()) -> tuple[list[OrderPlan], pd.Series]:
    """Returns (orders with sells first, the effective target after turnover capping).
    `forced_exits` (stop-loss exits) always sell in full and are never dropped by the order cap."""
    cur = current_weights(acct)
    eff = cap_turnover(target, cur, risk.max_daily_turnover, forced_exits)
    orders: list[OrderPlan] = []
    for sym in eff.index:
        c = float(cur.get(sym, 0.0))
        t = float(eff[sym])
        delta_usd = (t - c) * acct.equity
        if abs(delta_usd) < risk.min_order_notional:
            continue
        full_exit = t <= 1e-6 and c > 0
        orders.append(OrderPlan(
            symbol=sym, side="buy" if delta_usd > 0 else "sell", notional=round(abs(delta_usd), 2),
            current_weight=round(c, 6), target_weight=round(t, 6), close_position=full_exit,
        ))
    # Keep the most important trades within the daily order cap (largest first); the rest
    # happen on later runs as the portfolio converges.
    orders.sort(key=lambda o: (o.symbol not in forced_exits, -o.notional))
    orders = orders[: risk.max_orders_per_day]
    orders.sort(key=lambda o: (o.side != "sell", -o.notional))  # sells first to free cash
    return orders, eff
