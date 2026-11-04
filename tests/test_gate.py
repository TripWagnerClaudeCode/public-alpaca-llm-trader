import pandas as pd

from bot.models import AccountState, OrderPlan
from bot.risk import gate


def _acct(**kw):
    base = dict(equity=10_000, last_equity=10_000, cash=10_000, buying_power=10_000, positions={})
    return AccountState(**(base | kw))


def test_clean_pre_trade_passes(risk):
    assert gate.pre_trade_checks(risk, _acct(), 10_000, 1, 4).ok


def test_kill_switch_file_blocks(risk):
    gate.kill_switch_path(risk).write_text("halt")
    r = gate.pre_trade_checks(risk, _acct(), None, 1, 4)
    assert not r.ok and "kill switch" in r.reasons[0]


def test_kill_switch_env_blocks(risk, monkeypatch):
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert not gate.pre_trade_checks(risk, _acct(), None, 1, 4).ok


def test_daily_loss_trips_kill_switch(risk):
    r = gate.pre_trade_checks(risk, _acct(equity=9_600, last_equity=10_000), None, 1, 4)
    assert not r.ok and r.kill_switch_tripped
    assert gate.kill_switch_path(risk).exists()


def test_drawdown_trips_kill_switch(risk):
    r = gate.pre_trade_checks(risk, _acct(equity=8_000, last_equity=8_000), 10_000, 1, 4)
    assert not r.ok and r.kill_switch_tripped


def test_stale_data_blocks(risk):
    assert not gate.pre_trade_checks(risk, _acct(), None, 9, 4).ok


def test_trading_blocked(risk):
    assert not gate.pre_trade_checks(risk, _acct(trading_blocked=True), None, 1, 4).ok


def test_target_checks(risk):
    allow = {"A", "B"}
    assert gate.check_targets(risk, pd.Series({"A": 0.1, "B": 0.1}), allow).ok
    assert not gate.check_targets(risk, pd.Series({"A": 0.5}), allow).ok          # position cap
    assert not gate.check_targets(risk, pd.Series({"A": -0.05}), allow).ok        # short
    assert not gate.check_targets(risk, pd.Series({"Z": 0.05}), allow).ok         # not allowlisted
    assert not gate.check_targets(risk, pd.Series({"A": float("nan")}), allow).ok
    many = pd.Series({f"S{i}": 0.14 for i in range(7)})                            # 98% gross
    assert not gate.check_targets(risk, many, set(many.index)).ok


def _o(sym, side, n):
    return OrderPlan(symbol=sym, side=side, notional=n, current_weight=0, target_weight=0)


def test_order_checks(risk):
    acct = _acct(cash=1_000)
    assert gate.check_orders(risk, [_o("A", "buy", 900)], acct).ok
    assert not gate.check_orders(risk, [_o("A", "buy", 1_500)], acct).ok           # no margin
    assert gate.check_orders(risk, [_o("B", "sell", 1_000), _o("A", "buy", 1_900)], acct).ok
    too_many = [_o(f"S{i}", "buy", 1) for i in range(risk.max_orders_per_day + 1)]
    assert not gate.check_orders(risk, too_many, _acct()).ok
    assert not gate.check_orders(risk, [_o("A", "buy", 5_000)], _acct()).ok         # turnover 50% > 40%


# ----------------------------------------------------------------------------- automatic vs manual kill switch
def test_trip_writes_auto_marker(risk):
    gate.trip_kill_switch(risk, "daily loss -3.5%")
    first = gate.kill_switch_path(risk).read_text().splitlines()[0]
    assert first.startswith("AUTO ") and "daily loss -3.5%" in first


def test_manual_file_is_manual(risk):
    gate.kill_switch_path(risk).write_text("")
    assert gate.kill_switch_kind(risk) == "manual"
    gate.kill_switch_path(risk).write_text("stop everything please\n")
    assert gate.kill_switch_kind(risk) == "manual"


def test_auto_file_is_auto(risk):
    gate.trip_kill_switch(risk, "drawdown")
    assert gate.kill_switch_kind(risk) == "auto"


def test_env_kill_switch_is_manual_even_with_auto_file(risk, monkeypatch):
    gate.trip_kill_switch(risk, "drawdown")
    monkeypatch.setenv("BOT_KILL_SWITCH", "yes")
    assert gate.kill_switch_kind(risk) == "manual"


def test_no_kill_switch_is_none(risk):
    assert gate.kill_switch_kind(risk) is None


def test_exits_allowed_matrix(risk, monkeypatch):
    assert gate.exits_allowed(risk)
    gate.trip_kill_switch(risk, "x")
    assert gate.exits_allowed(risk)
    gate.kill_switch_path(risk).write_text("manual halt")
    assert not gate.exits_allowed(risk)


def test_pre_trade_checks_still_blocks_on_auto(risk):
    gate.trip_kill_switch(risk, "x")
    assert not gate.pre_trade_checks(risk, _acct(), None, 1, 4).ok
