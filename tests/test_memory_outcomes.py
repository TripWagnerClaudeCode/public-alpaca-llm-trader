"""Phase 2: score every view and agent call against realized returns (no LLM)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bot.journal.store import Journal
from bot.memory.outcomes import agent_calls, pending_symbols, score_pending
from bot.models import View

DAYS = pd.bdate_range("2026-10-01", periods=40)
AS_OF = str(DAYS[5].date())


def _bars(start: float, step: float, n: int = 40) -> pd.DataFrame:
    close = start * (1 + step) ** np.arange(n)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
                        index=DAYS[:n])


def _analyze(j: Journal, universe=("AAA", "BBB", "CCC"), as_of=AS_OF) -> int:
    rid = j.start_run("analyze", "paper")
    j.agent_report(rid, "inputs", {"as_of": as_of, "universe": list(universe)})
    j.agent_report(rid, "technical", {"summary": "s", "notes": [
        {"symbol": "AAA", "stance": "bullish", "strength": 0.6, "key_points": []},
        {"symbol": "BBB", "stance": "bearish", "strength": 0.5, "key_points": []},
        {"symbol": "CCC", "stance": "neutral", "strength": 0.0, "key_points": []}]})
    j.agent_report(rid, "bull_r1", {"side": "bull", "overview": "o", "arguments": [{"symbol": "AAA", "argument": "a"}]})
    j.agent_report(rid, "bear_r1", {"side": "bear", "overview": "o", "arguments": [{"symbol": "BBB", "argument": "a"}]})
    j.views(rid, [View(symbol="AAA", expected_return=0.02, confidence=0.6, rationale="r"),
                  View(symbol="BBB", expected_return=-0.01, confidence=0.3, rationale="r")])
    j.finish_run(rid, "ok")
    return rid


BARS = {"AAA": _bars(100, 0.01), "BBB": _bars(100, -0.01), "CCC": _bars(100, 0.0), "SPY": _bars(100, 0.002)}


def _rows(j):
    return j.query("SELECT * FROM outcomes_v2 ORDER BY symbol")


def test_scores_matured_view_once(tmp_path):
    j = Journal(tmp_path / "j.db")
    _analyze(j)
    assert score_pending(j, BARS, "SPY") == 3
    r = _rows(j).set_index("symbol")
    ra, rb, rc = 1.01 ** 10 - 1, 0.99 ** 10 - 1, 0.0
    univ = (ra + rb + rc) / 3
    assert r.loc["AAA", "ret"] == pytest.approx(ra)
    assert r.loc["AAA", "excess_univ"] == pytest.approx(ra - univ)
    assert r.loc["BBB", "excess_bench"] == pytest.approx(rb - (1.002 ** 10 - 1))
    assert (r["horizon"] == 10).all() and (r["as_of"] == AS_OF).all()
    assert score_pending(j, BARS, "SPY") == 0


def test_not_matured_not_scored(tmp_path):
    j = Journal(tmp_path / "j.db")
    _analyze(j, as_of=str(DAYS[32].date()))           # only 7 bars after as_of
    assert score_pending(j, BARS, "SPY") == 0


def test_missing_price_stays_pending(tmp_path):
    j = Journal(tmp_path / "j.db")
    _analyze(j, universe=("AAA", "BBB", "GONE"))
    assert score_pending(j, BARS, "SPY") == 2
    assert set(_rows(j)["symbol"]) == {"AAA", "BBB"}
    syms, start = pending_symbols(j)
    assert "GONE" in syms and start == dt.date.fromisoformat(AS_OF)


def test_pending_symbols_only_unscored(tmp_path):
    j = Journal(tmp_path / "j.db")
    _analyze(j)
    assert pending_symbols(j)[0] == {"AAA", "BBB", "CCC"}
    score_pending(j, BARS, "SPY")
    assert pending_symbols(j) == (set(), None)


def test_agent_calls_extracts_every_agent(tmp_path):
    j = Journal(tmp_path / "j.db")
    rid = _analyze(j)
    ex = j.start_run("execute", "paper")
    j.agent_report(ex, "morning_check_input", {"src_run": rid, "buys": [{"symbol": "AAA", "gap_pct": 0.0},
                                                                         {"symbol": "CCC", "gap_pct": 0.0}]})
    j.agent_report(ex, "morning_check", {"items": [
        {"symbol": "AAA", "action": "cancel", "fraction": 1.0, "reason": "r"},
        {"symbol": "CCC", "action": "reduce", "fraction": 0.5, "reason": "r"}], "summary": ""})
    calls = {(c["agent"], c["symbol"]): c for c in agent_calls(j, rid)}
    assert calls[("pm", "AAA")]["direction"] == 1 and calls[("pm", "AAA")]["confidence"] == 0.6
    assert calls[("pm", "BBB")]["direction"] == -1
    assert calls[("technical", "AAA")]["direction"] == 1 and calls[("technical", "CCC")]["direction"] == 0
    assert calls[("technical", "BBB")]["direction"] == -1
    assert calls[("bull", "AAA")]["direction"] == 1 and calls[("bear", "BBB")]["direction"] == -1
    assert calls[("morning_check", "AAA")]["direction"] == -1
    assert calls[("morning_check", "CCC")]["direction"] == -0.5


def test_runs_without_inputs_ignored(tmp_path):
    j = Journal(tmp_path / "j.db")
    rid = j.start_run("analyze", "paper")      # --no-agents run: no inputs report
    j.finish_run(rid, "ok")
    assert score_pending(j, BARS, "SPY") == 0 and pending_symbols(j) == (set(), None)


# ----------------------------------------------------------------------------- reconcile hook
def _settings(tmp_path, memory=True):
    from bot.config import load_settings

    s = load_settings(profile="default")
    s.paths.journal_db = str(tmp_path / "j.db")
    s.features.memory = memory
    s.memory.dir = str(tmp_path / "mem")
    return s


def test_reconcile_memory_off_writes_nothing(tmp_path, monkeypatch):
    from bot import run_daily

    s = _settings(tmp_path, memory=False)
    _analyze(Journal(tmp_path / "j.db"))
    called = []
    monkeypatch.setattr(run_daily, "MarketData", lambda *a, **k: called.append(1))
    run_daily.memory_after_reconcile(s, Journal(tmp_path / "j.db"))
    tables = set(Journal(tmp_path / "j.db").query("SELECT name FROM sqlite_master WHERE type='table'").name)
    assert called == [] and "outcomes_v2" not in tables


def test_reconcile_memory_scores_and_error_never_raises(tmp_path, monkeypatch):
    from bot import run_daily

    s = _settings(tmp_path)
    j = Journal(tmp_path / "j.db")
    _analyze(j)

    class MD:
        def __init__(self, *a, **k):
            pass

        def daily_bars(self, symbols, days, end=None):
            return {k: v for k, v in BARS.items() if k in symbols}

    from bot.agents.schemas import PostTradeNotes

    class NoObs:
        def structured(self, **kw):
            return PostTradeNotes(observations=[], summary="")

    monkeypatch.setattr(run_daily, "MarketData", MD)
    monkeypatch.setattr(run_daily, "make_llm", lambda s: NoObs())
    out = run_daily.memory_after_reconcile(s, j)
    assert out["scored"] == 3 and out["error"] is None

    class Boom(MD):
        def daily_bars(self, *a, **k):
            raise ConnectionError("data down")

    monkeypatch.setattr(run_daily, "MarketData", Boom)
    _analyze(j)
    out = run_daily.memory_after_reconcile(s, j)            # must not raise
    assert out["error"]
    r = j.query("SELECT status, notes FROM runs WHERE phase='memory'")
    assert r["status"].tolist() == ["error"] and "data down" in r["notes"][0]
