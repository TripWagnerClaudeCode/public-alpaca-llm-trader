"""Phase 2: daily Post-Trade Reviewer (observations to staging.md) and scorecard.md each reconcile."""
from __future__ import annotations

from bot import run_daily
from bot.agents.llm import UsageLimited
from bot.agents.schemas import Observation, PostTradeNotes
from bot.journal.store import Journal
from bot.memory.files import MemoryFiles
from bot.memory.reviewer import run_post_trade
from test_memory_outcomes import BARS, _analyze, _settings


class _LLM:
    def __init__(self, n=2, fail=None):
        self.n, self.fail, self.calls = n, fail, []

    def structured(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return PostTradeNotes(observations=[Observation(text=f"obs {i}", agents=["pm"], symbols=["AAA"],
                                                        evidence=f"{i} of 3") for i in range(self.n)], summary="s")


class _MD:
    def __init__(self, *a, **k):
        pass

    def daily_bars(self, symbols, days, end=None):
        return {k: v for k, v in BARS.items() if k in symbols}


def _setup(tmp_path, monkeypatch, llm):
    s = _settings(tmp_path)
    j = Journal(tmp_path / "j.db")
    _analyze(j)
    monkeypatch.setattr(run_daily, "MarketData", _MD)
    monkeypatch.setattr(run_daily, "make_llm", lambda s: llm)
    return s, j, MemoryFiles(tmp_path / "mem")


def test_reviewer_appends_observations_to_staging(tmp_path, monkeypatch):
    llm = _LLM(n=2)
    s, j, mf = _setup(tmp_path, monkeypatch, llm)
    out = run_daily.memory_after_reconcile(s, j)
    assert out["scored"] == 3 and out["observations"] == 2 and len(llm.calls) == 1
    lines = mf.read_staging(None)
    assert len(lines) == 2 and "obs 0" in lines[0] and "[agents: pm]" in lines[0] and "evidence: 0 of 3" in lines[0]
    assert "<untrusted_data>" in llm.calls[0]["user"] and llm.calls[0]["model"] == "claude-sonnet-5-5"


def test_reviewer_caps_at_5(tmp_path, monkeypatch):
    s, j, mf = _setup(tmp_path, monkeypatch, _LLM(n=7))
    assert run_daily.memory_after_reconcile(s, j)["observations"] == 5
    assert len(mf.read_staging(None)) == 5


def test_reviewer_not_called_when_nothing_newly_scored(tmp_path, monkeypatch):
    llm = _LLM()
    s, j, mf = _setup(tmp_path, monkeypatch, llm)
    run_daily.memory_after_reconcile(s, j)
    llm.calls.clear()
    out = run_daily.memory_after_reconcile(s, j)              # nothing new to score
    assert out["scored"] == 0 and llm.calls == []


def test_scorecard_md_written_each_reconcile(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    j = Journal(tmp_path / "j.db")                           # nothing to score yet
    monkeypatch.setattr(run_daily, "make_llm", lambda s: _LLM())
    run_daily.memory_after_reconcile(s, j)
    md = (tmp_path / "mem" / "scorecard.md").read_text(encoding="utf-8")
    assert "Scorecard (large-cap)" in md and "No track record yet" in md


def test_reviewer_failure_logged_reconcile_ok(tmp_path, monkeypatch):
    s, j, mf = _setup(tmp_path, monkeypatch, _LLM(fail=UsageLimited("usage limit reached")))
    out = run_daily.memory_after_reconcile(s, j)
    assert out["scored"] == 3 and "usage limit" in out["error"]
    assert (tmp_path / "mem" / "scorecard.md").exists() and mf.read_staging(None) == []


def test_run_post_trade_direct_formats_lines(tmp_path):
    mf = MemoryFiles(tmp_path / "m")
    s = _settings(tmp_path)
    n = run_post_trade(_LLM(n=1), s, Journal(tmp_path / "j.db"), mf, [{"symbol": "AAA"}], {"n_views": 1}, "2026-11-20")
    assert n == 1 and mf.read_staging(None)[0].startswith("- (2026-11-20) obs 0")
