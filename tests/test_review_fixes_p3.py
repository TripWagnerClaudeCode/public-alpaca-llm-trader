"""Phase 3 final-review fixes (trading-bot): each test reproduces one finding."""
from __future__ import annotations

import datetime as dt
import json
import sys
import textwrap
import time

import pandas as pd
import pytest

from bot import events, intraday_buy, watch
from bot.agents.llm import LLM
from bot.agents.schemas import BuyApproval, DiveFindings, EventCall, PMView
from bot.broker.sim import DryRunBroker
from bot.config import load_settings
from bot.journal.store import Journal
from bot.models import OrderPlan, PositionDetail, SubmittedOrder

NY = events.NY
IN = dt.datetime.combine(dt.date.today(), dt.time(11, 0), tzinfo=NY)
UNI = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


class EvLLM:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, analyst=("exit", 1.0), reviewer=("exit", 1.0), exc=None):
        self.analyst, self.reviewer, self.exc, self.research_calls = analyst, reviewer, exc, []

    def research(self, **k):
        self.research_calls.append(k)
        if self.exc:
            e, self.exc = self.exc, None
            raise e
        return EventCall(action=self.analyst[0], fraction=self.analyst[1], reason="a"), []

    def structured(self, **k):
        return EventCall(action=self.reviewer[0], fraction=self.reviewer[1], reason="b")


class Broker(DryRunBroker):
    def __init__(self, *a, reject=False, **k):
        super().__init__(*a, **k)
        self.canceled, self.reject = [], reject
        self.is_market_open = lambda: True

    def cancel_order(self, oid):
        self.canceled.append(oid)

    def submit(self, order):
        self.submitted.append(order)
        if self.reject:
            return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                                  broker_order_id=None, status="rejected", error="qty must be whole")
        return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                              broker_order_id=f"oid-{len(self.submitted)}", status="accepted")


def _s(tmp_path, profile="smallcap"):
    s = load_settings(profile=profile)
    s.paths.journal_db = str(tmp_path / "j.db")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    s.features.watch = s.features.event_review = True
    s.features.intraday_buys = False
    s.features.memory = False
    return s


def _seed(s, syms, universe=None):
    j = Journal(s.paths.resolve(s.paths.journal_db))
    rid = j.start_run("analyze", "dry-run")
    j.agent_report(rid, "inputs", {"as_of": "2026-11-03", "universe": universe or list(syms)})
    j.conn.executemany("INSERT INTO signals(run_id, symbol, source, value, extra) VALUES (?,?,?,?,?)",
                       [(rid, x, "adv", 1e6, json.dumps({"close": 100.0, "sigma_daily": 0.02})) for x in syms])
    j.conn.commit()
    j.finish_run(rid, "ok")
    return j


def _held(prices, qty=10.0, cash=10_000.0, **k):
    det = {x: PositionDetail(qty=qty, avg_entry_price=100.0, current_price=p) for x, p in prices.items()}
    return Broker(equity=cash + sum(qty * p for p in prices.values()), positions={x: qty * p for x, p in prices.items()},
                  details=det, quotes={x: {"bid": p - 0.05, "ask": p + 0.05, "ts": "t"} for x, p in prices.items()}, **k)


def _w(s, b, llm, now=IN, filings=None, closes=None):
    return watch.run(s, "dry-run", broker=b, llm=llm, now_fn=lambda: now,
                     filings_fn=lambda sym, since: (filings or {}).get(sym, []),
                     closes_fn=(lambda syms: closes) if closes is not None else None)


# 1 (Critical): an unfilled Event Review limit sell must not block the existing stop
def test_open_event_sell_does_not_block_stop(tmp_path):
    s = _s(tmp_path)
    _seed(s, ["XYZ"])
    b = _held({"XYZ": 93.0})
    _w(s, b, EvLLM())                                           # both agents exit: limit sell placed
    assert b.submitted[-1].limit_price is not None and not b.submitted[-1].close_position
    b2 = _held({"XYZ": 80.0})                                   # it never filled; XYZ crashes through B's -15% stop
    _w(s, b2, EvLLM(), now=IN + dt.timedelta(minutes=30))
    assert b2.canceled == ["oid-1"]
    assert [(o.symbol, o.close_position) for o in b2.submitted] == [("XYZ", True)]
    b3 = _held({"XYZ": 79.0})                                   # the stop sell is open now: never a second one
    _w(s, b3, EvLLM(), now=IN + dt.timedelta(minutes=60))
    assert b3.submitted == [] and b3.canceled == []


# 6: trims of non-fractionable names are whole shares; a rejected sell is not an executed call
def test_trim_whole_shares_and_rejection_not_executed(tmp_path):
    s = _s(tmp_path)
    _seed(s, ["XYZ"])
    b = _held({"XYZ": 93.0}, non_fractionable={"XYZ"})
    _w(s, b, EvLLM(("trim", 0.33), ("trim", 0.33)))
    assert b.submitted[0].qty == 3.0
    s2 = _s(tmp_path / "b")
    (tmp_path / "b").mkdir()
    s2.paths.journal_db = str(tmp_path / "b" / "j.db")
    _seed(s2, ["XYZ"])
    _w(s2, _held({"XYZ": 93.0}, reject=True), EvLLM())
    ev = Journal(s2.paths.resolve(s2.paths.journal_db)).query("SELECT action FROM events")
    assert ev["action"].tolist() == ["exit: rejected"]


# 5: per-watch cap and a time budget for agent work
def test_reviews_capped_per_watch(tmp_path):
    s = _s(tmp_path)
    syms = [f"S{i}" for i in range(5)]
    _seed(s, syms)
    llm = EvLLM(("hold", 1.0))
    _w(s, _held({x: 93.0 for x in syms}), llm)
    assert len(llm.research_calls) == s.events.max_reviews_per_watch == 2
    s.events.agent_budget_s = 0
    _w(s, _held({x: 93.0 for x in syms}), llm, now=IN + dt.timedelta(minutes=30))
    assert len(llm.research_calls) == 2


# 8: a failed review does not mark its filing as reviewed
def test_failed_filing_review_is_retried(tmp_path):
    s = _s(tmp_path)
    _seed(s, ["XYZ"])
    f = {"XYZ": [{"form": "8-K", "date": str(IN.date()), "url": "https://sec.gov/x.htm", "title": "8-K"}]}
    llm = EvLLM(("hold", 1.0), exc=RuntimeError("cli hiccup"))
    _w(s, _held({"XYZ": 100.0}), llm, filings=f)
    _w(s, _held({"XYZ": 100.0}), llm, now=IN + dt.timedelta(minutes=30), filings=f)
    assert len(llm.research_calls) == 2


# 2 and 3: intraday buys see cash spent by earlier buys and never re-buy a name with an open buy order
class BuyLLM:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self):
        self.pipelines = 0

    def research(self, **k):
        self.pipelines += 1
        return DiveFindings(symbol="X", question="q", answer="a", findings=[]), []

    def structured(self, **k):
        if k["schema"] is BuyApproval:
            return BuyApproval(approve=True, reason="ok")
        return PMView(symbol="X", expected_return=0.03, confidence=0.6, rationale="r")


def _buy_s(tmp_path):
    s = load_settings(profile="default")
    s.universe.symbols = list(UNI)
    s.paths.journal_db = str(tmp_path / "j.db")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    s.features.watch = s.features.intraday_buys = True
    s.features.event_review = s.features.memory = False
    s.events.max_buy_pipelines_per_watch = 2
    return s


def _buy_broker(moves, cash, held_value=0.0):
    px = {x: 100.0 for x in UNI} | moves
    det = {"FFF": PositionDetail(qty=held_value / 100, avg_entry_price=100.0, current_price=100.0)} if held_value else {}
    b = Broker(equity=cash + held_value, positions={"FFF": held_value} if held_value else {}, details=det,
               quotes={x: {"bid": p - 0.05, "ask": p + 0.05, "ts": "t"} for x, p in px.items()})
    return b


def test_second_intraday_buy_sees_spent_cash(tmp_path, monkeypatch, closes):
    s = _buy_s(tmp_path)
    _seed(s, UNI)
    monkeypatch.setattr(intraday_buy, "target_weights", lambda c, v, p, h: pd.Series({x: 0.15 for x in UNI}))
    b = _buy_broker({"AAA": 107.0, "BBB": 108.0}, cash=1_500.0, held_value=8_500.0)   # cash 15%, floor 5%
    _w(s, b, BuyLLM(), closes=closes)
    spent = sum(o.notional for o in b.submitted)
    assert len(b.submitted) == 2 and spent <= 1_500.0 - 0.05 * 10_000 + 1.0


def test_open_morning_buy_not_rebought(tmp_path, monkeypatch, closes):
    s = _buy_s(tmp_path)
    j = _seed(s, UNI)
    rid = j.start_run("execute", "dry-run")
    j.order(rid, OrderPlan(symbol="AAA", side="buy", notional=700, current_weight=0, target_weight=0.07),
            SubmittedOrder(symbol="AAA", side="buy", notional=700, broker_order_id="m1", status="accepted"))
    monkeypatch.setattr(intraday_buy, "target_weights", lambda c, v, p, h: pd.Series({x: 0.15 for x in UNI}))
    llm = BuyLLM()
    b = _buy_broker({"AAA": 107.0}, cash=10_000.0)
    _w(s, b, llm, closes=closes)
    assert b.submitted == [] and llm.pipelines == 0


# 7: the research call times out even when the CLI never reads a large stdin
def test_research_timeout_with_unread_large_stdin(tmp_path):
    p = tmp_path / "fake_claude.py"
    p.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    llm = LLM(100_000, cli_path="unused")
    llm._cli_cmd = [sys.executable, str(p)]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        llm.research(model="m", effort="e", system="s", user="x" * 3_000_000, schema=EventCall, timeout_s=2)
    assert time.time() - t0 < 15
