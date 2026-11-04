"""Phase 3: Event Reviews and intraday buys are scored from their own decision price, 10 trading days on,
against the universe average; each agent's call is a scorecard row."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bot.config import load_settings
from bot.journal.store import Journal
from bot.memory.events_score import event_calls, pending_event_symbols, score_events
from bot.memory.scorecards import build_scorecards

DAYS = pd.bdate_range("2026-10-01", periods=40)
D0 = str(DAYS[5].date())


def _bars(start, step):
    c = start * (1 + step) ** np.arange(40)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=DAYS)


BARS = {"AAA": _bars(100, -0.01), "BBB": _bars(100, 0.01), "CCC": _bars(100, 0.0), "SPY": _bars(100, 0.0)}


def _journal(tmp_path, mode="paper"):
    j = Journal(tmp_path / "j.db")
    rid = j.start_run("analyze", mode)
    j.agent_report(rid, "inputs", {"as_of": str(DAYS[4].date()), "universe": ["AAA", "BBB", "CCC"]})
    j.finish_run(rid, "ok")
    return j


def _ev(j, sym, analyst, reviewer, action, pipeline="review", mode="paper", price=None):
    return j.record_event(D0, sym, "move", "d", analyst, reviewer, action,
                          price if price is not None else float(BARS[sym]["close"].iloc[5]), 1, mode, pipeline=pipeline)


def C(a, f=1.0):
    return {"call": {"action": a, "fraction": f, "reason": "r"}}


def test_event_rows_scored(tmp_path):
    j = _journal(tmp_path)
    e1 = _ev(j, "AAA", C("exit"), C("exit"), "exit")                    # executed exit, AAA then fell: good
    e2 = _ev(j, "BBB", C("exit"), C("hold"), "hold")                    # disagreement: held, BBB rose
    e3 = _ev(j, "BBB", {"pm": {}}, {"risk": {"approve": True}}, "buy", pipeline="buy")
    _ev(j, "CCC", {"error": "x"}, None, "error")                        # failed reviews are never scored
    _ev(j, "AAA", C("exit"), C("exit"), "exit", mode="dry-run")         # dry-runs never count
    syms, start = pending_event_symbols(j)
    assert syms >= {"AAA", "BBB"} and str(start) == D0
    assert score_events(j, BARS, "SPY") == 3
    assert score_events(j, BARS, "SPY") == 0                            # once only
    rows = {r["event_id"]: r for r in j.query("SELECT * FROM event_outcomes").to_dict("records")}
    assert rows[e1]["excess_univ"] < 0 < rows[e2]["excess_univ"]
    ev = {r["id"]: r for r in j.query("SELECT * FROM events").to_dict("records")}
    calls = dict(((a, eid), d) for eid in (e1, e2, e3) for a, d in event_calls(ev[eid]))
    assert calls[("event_review", e1)] == -1 and calls[("event_analyst", e1)] == -1
    assert calls[("event_reviewer", e2)] == 1 and calls[("event_analyst", e2)] == -1
    assert ("event_review", e2) not in calls                             # a hold is not an executed call
    assert calls[("intraday_buy", e3)] == 1
    cards = build_scorecards(j)["agents"]
    assert cards["event_review"]["hit_rate"] == 1.0                      # sold before a fall
    assert cards["event_reviewer"]["hit_rate"] == 1.0                    # held before a rise
    assert cards["event_analyst"]["n"] == 2 and cards["event_analyst"]["hit_rate"] == 0.5
    assert cards["intraday_buy"]["n"] == 1


def test_event_not_matured_stays_pending(tmp_path):
    j = _journal(tmp_path)
    j.record_event(str(DAYS[35].date()), "AAA", "move", "d", C("exit"), C("exit"), "exit", 90.0, 1, "paper")
    assert score_events(j, BARS, "SPY") == 0
    assert pending_event_symbols(j)[0] == {"AAA", "BBB", "CCC"}


def _yaml(name):
    return load_settings(profile=name)


PHASE3 = ("research", "reviewers", "event_review", "intraday_buys")


def test_controls_have_no_phase3_flags():
    for p in ("control", "smallcap_control"):
        f = _yaml(p).features
        assert not any(getattr(f, x) for x in PHASE3), p


def test_live_profiles_enable_phase3():
    for p in ("default", "smallcap"):
        f = _yaml(p).features
        assert all(getattr(f, x) for x in PHASE3), p
