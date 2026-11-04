"""Regression tests for the final-review findings (2026-11-04): I1, I2, I3, I4, M5, M8."""
from __future__ import annotations

import datetime as dt
import json
import math

import pytest

from bot import run_daily, watch
from bot.agents.morning import apply_morning
from bot.agents.schemas import MorningCheck, MorningItem
from bot.broker.sim import DryRunBroker
from bot.journal.store import Journal
from bot.models import OrderPlan, PositionDetail, SubmittedOrder
from test_gap import _quotes_at
from test_morning import _LLM, _run_execute
from test_profile_pipeline import _write_universe, setup  # noqa: F401  (fixture)
from test_watch import _settings


def _sell(sym):
    return OrderPlan(symbol=sym, side="sell", notional=840.0, current_weight=0.08, target_weight=0.0,
                     close_position=True)


# ----------------------------------------------------------------------------- I1
def test_open_sell_symbols_ignores_never_submitted_rows(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.order(1, _sell("BLK"))                                                     # blocked execute: planned, no id
    j.order(1, _sell("OPN"), SubmittedOrder(symbol="OPN", side="sell", notional=1, broker_order_id="o1", status="new"))
    j.order(1, _sell("FIL"), SubmittedOrder(symbol="FIL", side="sell", notional=1, broker_order_id="o2",
                                            status="filled"))
    assert j.open_sell_symbols(dt.date.today()) == {"OPN"}


class _IdBroker(DryRunBroker):
    """Returns broker ids; orders stay 'new' until fill() is called, then the position is gone."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.status = {}

    def submit(self, order):
        self.submitted.append(order)
        oid = f"id{len(self.submitted)}"
        self.status[oid] = "new"
        return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                              broker_order_id=oid, status="new")

    def order_status(self, oid):
        return {"status": self.status.get(oid, "unknown"), "filled_qty": 0.0, "filled_avg_price": None}

    def fill(self, sym):
        for k in self.status:
            self.status[k] = "filled"
        self._acct.positions.pop(sym, None)
        self._acct.details.pop(sym, None)


def _idbroker(price=84.0):
    det = {"XYZ": PositionDetail(qty=10.0, avg_entry_price=100.0, current_price=price)}
    b = _IdBroker(equity=10_000, positions={"XYZ": 10 * price}, details=det)
    b.is_market_open = lambda: True
    return b


def test_watch_stops_out_after_blocked_execute(tmp_path):
    s = _settings(tmp_path)
    Journal(s.paths.resolve(s.paths.journal_db)).order(1, _sell("XYZ"))   # gate-blocked execute row
    b = _idbroker()
    assert watch.run(s, "paper", broker=b) == 0
    assert [o.symbol for o in b.submitted] == ["XYZ"]


def test_watch_open_then_filled_sells_exactly_once(tmp_path):
    s = _settings(tmp_path)
    b = _idbroker()
    assert watch.run(s, "paper", broker=b) == 0 and len(b.submitted) == 1
    assert watch.run(s, "paper", broker=b) == 0 and len(b.submitted) == 1   # sell still open
    b.fill("XYZ")
    assert watch.run(s, "paper", broker=b) == 0 and len(b.submitted) == 1   # position gone


# ----------------------------------------------------------------------------- I2
class _Moving(DryRunBroker):
    """The first quote per symbol is at the analyze close; every later one is 2% higher."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n: dict[str, int] = {}

    def latest_quote(self, sym):
        self.n[sym] = self.n.get(sym, 0) + 1
        q = dict(_quotes_at(1.0 if self.n[sym] == 1 else 1.02)[sym])
        return q


def test_limits_priced_from_quotes_fetched_after_the_morning_check(setup, monkeypatch):
    s, box, tmp = setup
    s.features.morning_check = True
    s.features.watch = False
    monkeypatch.setattr(run_daily, "make_llm", lambda s: _LLM(check=MorningCheck(items=[], summary="")))
    _write_universe(tmp)
    box["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    box["broker"] = b = _Moving()
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    later = _quotes_at(1.02)
    buys = [o for o in b.submitted if o.side == "buy" and o.limit_price]
    assert buys
    for o in buys:
        assert o.limit_price >= round(later[o.symbol]["ask"] * (1 + s.execution.limit_buffer_pct), 2) - 0.011


# ----------------------------------------------------------------------------- I3
def _emails(monkeypatch):
    import bot.notify as notify

    sent = []
    monkeypatch.setattr(notify, "send_email", lambda subj, body: sent.append(subj) or True)
    return sent


def _main_watch(s, monkeypatch, broker):
    monkeypatch.setattr(run_daily, "load_settings", lambda *a, **k: s)
    monkeypatch.setattr(run_daily, "make_broker", lambda mode, s=None, init_book=False: broker)
    return run_daily.main(["--profile", "smallcap", "--phase", "watch", "--mode", "paper"])


def test_watch_emails_when_it_exits_a_stop(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.notify.email = True
    sent = _emails(monkeypatch)
    assert _main_watch(s, monkeypatch, _idbroker()) == 0
    assert len(sent) == 1


def test_watch_quiet_when_nothing_happens(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.notify.email = True
    sent = _emails(monkeypatch)
    assert _main_watch(s, monkeypatch, _idbroker(price=95.0)) == 0
    assert sent == []


def test_watch_manual_kill_switch_does_not_email_every_30_minutes(tmp_path, monkeypatch):
    from bot.risk import gate

    s = _settings(tmp_path)
    s.notify.email = True
    gate.kill_switch_path(s.risk).write_text("halt")
    sent = _emails(monkeypatch)
    for _ in range(3):
        _main_watch(s, monkeypatch, _idbroker())
    assert sent == []


def test_watch_errors_email_once_per_day(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.notify.email = True
    sent = _emails(monkeypatch)
    b = _idbroker()
    b.get_account = lambda: (_ for _ in ()).throw(ConnectionError("alpaca down"))
    for _ in range(3):
        assert _main_watch(s, monkeypatch, b) == 1
    assert len(sent) == 1


# ----------------------------------------------------------------------------- I4
def test_morning_check_journals_inputs_and_applied_effects(setup, monkeypatch):
    llm = _LLM()

    def structured(**kw):
        return MorningCheck(items=[MorningItem(symbol="ZZZ", action="cancel", reason="r")], summary="s")

    llm.structured = structured
    _, _ = _run_execute(setup, monkeypatch, llm)
    s, _, _ = setup
    j = Journal(s.paths.resolve(s.paths.journal_db))
    rows = {r.agent: json.loads(r.content) for r in
            j.query("SELECT agent, content FROM agent_reports WHERE agent LIKE 'morning_check%'").itertuples()}
    assert rows["morning_check_input"]["buys"]
    assert any("ZZZ" in n for n in rows["morning_check_applied"])


# ----------------------------------------------------------------------------- M5
def _b(sym, notional=1000.0):
    return OrderPlan(symbol=sym, side="buy", notional=notional, current_weight=0.0, target_weight=0.1)


def test_apply_nan_fraction_keeps_order():
    out, _ = apply_morning([_b("A")], MorningCheck(items=[MorningItem(symbol="A", action="reduce",
                                                                       fraction=math.nan, reason="x")], summary=""), 5.0)
    assert len(out) == 1 and out[0].notional == 1000.0


def test_apply_duplicate_items_most_restrictive_wins():
    check = MorningCheck(items=[MorningItem(symbol="A", action="cancel", reason="x"),
                                MorningItem(symbol="A", action="keep", reason="y")], summary="")
    out, _ = apply_morning([_b("A")], check, 5.0)
    assert out == []


# ----------------------------------------------------------------------------- M8
def test_api_backend_cannot_bill_in_tests():
    from bot.agents.llm import LLM

    with pytest.raises(RuntimeError, match="disabled in tests"):
        LLM(100)
