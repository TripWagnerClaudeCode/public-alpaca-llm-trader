import pandas as pd
import pytest

from bot.models import AccountState
from bot.portfolio.rebalance import cap_turnover, plan_orders


def test_cap_turnover_partial_move():
    t = pd.Series({"A": 0.5, "B": 0.5})
    c = pd.Series({"A": 0.0, "B": 0.0})
    eff = cap_turnover(t, c, 0.4)
    assert eff.sum() == pytest.approx(0.4)
    assert eff["A"] == pytest.approx(0.2)


def test_plan_sells_first_and_skips_dust(risk):
    acct = AccountState(equity=10_000, last_equity=10_000, cash=5_000, buying_power=5_000,
                        positions={"A": 3_000, "B": 2_000})
    target = pd.Series({"A": 0.1, "B": 0.2003, "C": 0.3})  # B delta is $3, below the $5 minimum
    orders, _ = plan_orders(target, acct, risk.model_copy(update={"max_daily_turnover": 10}))
    assert [o.side for o in orders] == ["sell", "buy"]
    assert {o.symbol for o in orders} == {"A", "C"}
    assert orders[0].notional == pytest.approx(2_000)


def test_full_exit_uses_close_position(risk):
    acct = AccountState(equity=10_000, last_equity=10_000, cash=7_000, buying_power=7_000, positions={"A": 3_000})
    orders, _ = plan_orders(pd.Series({"A": 0.0}), acct, risk.model_copy(update={"max_daily_turnover": 10}))
    assert len(orders) == 1 and orders[0].close_position and orders[0].side == "sell"


def test_order_count_cap_keeps_largest(risk, cash_account):
    target = pd.Series({f"S{i}": 0.01 * (i + 1) for i in range(10)})
    cfg = risk.model_copy(update={"max_orders_per_day": 3, "max_daily_turnover": 10})
    orders, _ = plan_orders(target, cash_account, cfg)
    assert {o.symbol for o in orders} == {"S9", "S8", "S7"}
