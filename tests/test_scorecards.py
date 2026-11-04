"""Phase 2: per-agent scorecards, PM calibration and Morning Check scoring."""
from __future__ import annotations

import pytest

from bot.journal.store import Journal
from bot.memory.outcomes import score_pending
from bot.memory.scorecards import build_scorecards, render_scorecards
from test_memory_outcomes import BARS, _analyze


def _scored(tmp_path) -> tuple[Journal, int]:
    j = Journal(tmp_path / "j.db")
    rid = _analyze(j)
    assert score_pending(j, BARS, "SPY") == 3
    return j, rid


def test_hit_rate_and_signed_excess(tmp_path):
    j, _ = _scored(tmp_path)
    c = build_scorecards(j)
    assert c["n_views"] == 2
    pm = c["agents"]["pm"]
    assert pm["n"] == 2 and pm["hit_rate"] == 1.0       # AAA long went up vs universe, BBB short went down
    ex = {r.symbol: r.excess_univ for r in j.query("SELECT * FROM outcomes_v2").itertuples()}
    assert pm["mean_signed_excess"] == pytest.approx((ex["AAA"] - ex["BBB"]) / 2)
    assert c["agents"]["bull"]["hit_rate"] == 1.0 and c["agents"]["bear"]["hit_rate"] == 1.0


def test_neutral_calls_excluded_from_hit_rate(tmp_path):
    j, _ = _scored(tmp_path)
    tech = build_scorecards(j)["agents"]["technical"]
    assert tech["n"] == 2                                 # CCC neutral excluded


def test_calibration_buckets(tmp_path):
    j, _ = _scored(tmp_path)
    cal = build_scorecards(j)["calibration"]
    assert cal["0.5-0.8"]["n"] == 1 and cal["0.3-0.5"]["n"] == 1 and cal["0.0-0.3"]["n"] == 0


def test_morning_check_cancel_scoring(tmp_path):
    j, rid = _scored(tmp_path)
    ex = j.start_run("execute", "paper")
    j.agent_report(ex, "morning_check_input", {"src_run": rid, "buys": [{"symbol": "AAA", "gap_pct": 0.0}]})
    j.agent_report(ex, "morning_check", {"items": [{"symbol": "AAA", "action": "cancel", "fraction": 1.0,
                                                    "reason": "r"}], "summary": ""})
    mc = build_scorecards(j)["morning_check"]
    a = j.query("SELECT excess_univ FROM outcomes_v2 WHERE symbol='AAA'")["excess_univ"][0]
    assert mc["cancels"] == 1 and mc["mean_excess_of_cancelled"] == pytest.approx(a)   # > 0: the cancel hurt


def test_symbol_history_last_3(tmp_path):
    j, _ = _scored(tmp_path)
    h = build_scorecards(j)["symbols"]["AAA"]
    assert h[-1]["pm_direction"] == 1 and h[-1]["as_of"] and len(h) <= 3


def test_empty_journal_zero_counts_and_render_says_no_track_record(tmp_path):
    c = build_scorecards(Journal(tmp_path / "j.db"))
    assert c["n_views"] == 0 and c["agents"] == {} and c["morning_check"]["cancels"] == 0
    md = render_scorecards(c, "large-cap")
    assert "No track record yet" in md and "10 trading days" in md


def test_render_has_agent_table(tmp_path):
    j, _ = _scored(tmp_path)
    md = render_scorecards(build_scorecards(j), "large-cap")
    assert "| pm |" in md and "hit rate" in md.lower()
