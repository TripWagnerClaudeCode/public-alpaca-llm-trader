import datetime as dt

import pandas as pd
import pytest

from bot.journal.store import Journal
from bot.models import AccountState, PositionDetail
from bot.risk import gate


def _acct(details):
    return AccountState(equity=10_000, last_equity=10_000, cash=5_000, buying_power=5_000,
                        positions={k: d.qty * d.current_price for k, d in details.items()}, details=details)


def test_stop_hits(risk):
    acct = _acct({"A": PositionDetail(qty=10, avg_entry_price=100, current_price=85),    # -15% exactly
                  "B": PositionDetail(qty=10, avg_entry_price=100, current_price=86),    # -14%
                  "C": PositionDetail(qty=10, avg_entry_price=100, current_price=60)})
    assert gate.stop_hits(risk, acct) == []                                 # stops off by default (bot A)
    hits = gate.stop_hits(risk.model_copy(update={"stop_loss_pct": 0.15}), acct)
    assert [h.symbol for h in hits] == ["A", "C"]
    assert hits[1].pnl_pct == -0.4 and hits[1].entry == 100


def test_cooldown_business_days():
    fri = dt.date(2026, 11, 6)
    assert gate.cooldown_until(fri, 10) == dt.date(2026, 11, 20)
    assert gate.cooldown_until(fri, 0) == fri


def test_force_zero_adds_missing():
    t = gate.force_zero(pd.Series({"A": 0.1, "B": 0.2}), ["B", "Z"])
    assert t.to_dict() == {"A": 0.1, "B": 0.0, "Z": 0.0}


def test_journal_cooldowns(tmp_path):
    j = Journal(tmp_path / "j.db")
    d = dt.date(2026, 11, 3)
    j.record_stop("A", d, 100, 80, -0.2, dt.date(2026, 11, 17), 1, "paper")
    j.record_stop("B", d, 100, 80, -0.2, dt.date(2026, 11, 17), 1, "dry-run")
    assert j.active_cooldowns(d, "paper") == {"A": "2026-11-17"}
    assert j.active_cooldowns(dt.date(2026, 11, 16), "paper") == {"A": "2026-11-17"}
    assert j.active_cooldowns(dt.date(2026, 11, 17), "paper") == {}      # re-entry allowed on that date
    assert set(j.active_cooldowns(d)) == {"A", "B"}


def test_forced_exit_bypasses_turnover_ramp(risk):
    from bot.models import OrderPlan
    from bot.portfolio.rebalance import cap_turnover, plan_orders

    c = pd.Series({"A": 0.3, "B": 0.0})
    t = pd.Series({"A": 0.3, "B": 0.5})
    eff = cap_turnover(t, c, 0.4, frozenset({"A"}))
    assert eff["A"] == 0.0 and eff["B"] == pytest.approx(0.1)    # A used 0.3 of the 0.4 budget
    assert cap_turnover(t, c, 0.2, frozenset({"A"}))["B"] == 0.0
    assert cap_turnover(t, c, 0.4)["A"] == 0.3                   # unchanged without forced exits

    acct = AccountState(equity=10_000, last_equity=10_000, cash=7_000, buying_power=7_000, positions={"A": 3_000})
    cfg = risk.model_copy(update={"max_daily_turnover": 0.1, "max_orders_per_day": 1})
    orders, _ = plan_orders(pd.Series({"A": 0.3, "B": 0.5}), acct, cfg, frozenset({"A"}))
    assert len(orders) == 1 and orders[0].symbol == "A" and orders[0].close_position
    assert not gate.check_orders(cfg, orders, acct).ok                  # 30% turnover > 10%
    assert gate.check_orders(cfg, orders, acct, frozenset({"A"})).ok    # stop exits are exempt
    buy = OrderPlan(symbol="B", side="buy", notional=5_000, current_weight=0, target_weight=0.5)
    assert not gate.check_orders(cfg.model_copy(update={"max_orders_per_day": 5}), [*orders, buy], acct,
                                 frozenset({"A"})).ok                   # buys still count
