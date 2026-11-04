"""Phase 3 reviewers in analyze: an independent second PM (disagreement cuts confidence) and a Risk
Reviewer that can only lower confidence or drop a view. Neither can add or raise a view."""
import threading

from bot.agents.review import apply_risk_review, second_pm_cut
from bot.agents.schemas import (AnalystReport, DebateCase, PMDecision, PMView, RegimeReport, RiskItem,
                                RiskReview)
from bot.agents.team import AgentTeam, TeamInputs
from bot.config import AgentsCfg, PortfolioCfg, ReviewersCfg
from bot.journal.store import Journal
from bot.memory.outcomes import agent_calls
from bot.models import View


def V(sym, er, conf=0.6, why="r"):
    return View(symbol=sym, expected_return=er, confidence=conf, rationale=why)


def test_opposite_sign_cuts_confidence():
    out, notes = second_pm_cut([V("AAA", 0.03, 0.6)], [V("AAA", -0.01, 0.5)], 0.5)
    assert out[0].confidence == 0.3 and out[0].expected_return == 0.03
    assert "AAA" in notes[0]


def test_same_sign_or_missing_unchanged():
    out, notes = second_pm_cut([V("AAA", 0.03, 0.6), V("BBB", -0.02, 0.4)], [V("AAA", 0.01, 0.2)], 0.5)
    assert [(v.symbol, v.confidence) for v in out] == [("AAA", 0.6), ("BBB", 0.4)] and notes == []


def test_pm2_views_never_added():
    out, _ = second_pm_cut([V("AAA", 0.03)], [V("AAA", 0.02), V("ZZZ", 0.05, 0.9)], 0.5)
    assert [v.symbol for v in out] == ["AAA"]


def test_risk_review_never_raises():
    rv = RiskReview(summary="s", items=[
        RiskItem(symbol="AAA", action="lower", confidence=0.9, objection="o"),
        RiskItem(symbol="BBB", action="drop", confidence=0.0, objection="dilution"),
        RiskItem(symbol="CCC", action="lower", confidence=0.1, objection="o"),
        RiskItem(symbol="ZZZ", action="drop", confidence=0.0, objection="unknown")])
    out, notes = apply_risk_review([V("AAA", 0.02, 0.4), V("BBB", 0.02, 0.5), V("CCC", 0.01, 0.5), V("DDD", 0.01, 0.5)], rv)
    assert [(v.symbol, v.confidence) for v in out] == [("AAA", 0.4), ("CCC", 0.1), ("DDD", 0.5)]
    assert any("BBB" in n and "dilution" in n for n in notes)


UNI = ["AAA", "BBB"]
INP = TeamInputs(as_of="2026-11-04", universe=UNI, regime={}, technical={s: {} for s in UNI},
                 fundamentals={s: {} for s in UNI}, news={s: [] for s in UNI})


class Rec:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, pm2_exc=None, risk_exc=None, risk=None):
        self.calls, self.pm_n = [], 0
        self.pm2_exc, self.risk_exc, self.risk = pm2_exc, risk_exc, risk
        self._lock = threading.Lock()

    def structured(self, *, model, effort, system, user, schema):
        with self._lock:
            self.calls.append((schema.__name__, model, effort, system, user))
        if schema is RegimeReport:
            return RegimeReport(regime="neutral", confidence=0.5, suggested_invested_fraction=0.8, summary="s",
                                key_points=[])
        if schema is AnalystReport:
            return AnalystReport(summary="s", notes=[])
        if schema is DebateCase:
            return DebateCase(side="bull", overview="o", arguments=[])
        if schema is RiskReview:
            if self.risk_exc:
                raise self.risk_exc
            return self.risk or RiskReview(items=[], summary="ok")
        self.pm_n += 1
        if self.pm_n == 1:
            return PMDecision(market_view="m", cash_weight=0.1, views=[
                PMView(symbol="AAA", expected_return=0.03, confidence=0.6, rationale="SECRET-PM1-RATIONALE"),
                PMView(symbol="BBB", expected_return=0.02, confidence=0.5, rationale="r")])
        if self.pm2_exc:
            raise self.pm2_exc
        return PMDecision(market_view="m2", cash_weight=0.1, views=[
            PMView(symbol="AAA", expected_return=-0.01, confidence=0.4, rationale="disagree")])


def _team(llm, **kw):
    return AgentTeam(llm, AgentsCfg(), PortfolioCfg(), reviewers_cfg=ReviewersCfg(), **kw)


def test_team_runs_second_pm_then_risk_reviewer():
    llm = Rec(risk=RiskReview(summary="s", items=[RiskItem(symbol="BBB", action="drop", confidence=0, objection="x")]))
    out = _team(llm, context={"weights": {"AAA": 0.1}, "risk_limits": {"max_position_weight": 0.15}}).run(INP)
    assert [(v.symbol, v.confidence) for v in out.views] == [("AAA", 0.3)]
    names = [c[0] for c in llm.calls]
    assert names.count("PMDecision") == 2 and names.index("RiskReview") > max(i for i, n in enumerate(names) if n == "PMDecision")
    pm2 = [c for c in llm.calls if c[0] == "PMDecision"][1]
    assert pm2[1] == ReviewersCfg().second_pm_model and pm2[2] == "medium"
    risk = [c for c in llm.calls if c[0] == "RiskReview"][0]
    assert risk[2] == "high" and "max_position_weight" in risk[4] and '"AAA":0.1' in risk[4].replace(" ", "")
    assert [v["symbol"] for v in out.reports["views_before_review"]] == ["AAA", "BBB"]
    assert out.reports["second_pm"].views[0].expected_return == -0.01
    assert out.reports["risk_review"].items[0].symbol == "BBB"


def test_second_pm_independent():
    llm = Rec()
    _team(llm).run(INP)
    pm2 = [c for c in llm.calls if c[0] == "PMDecision"][1]
    assert "SECRET-PM1-RATIONALE" not in pm2[4]


def test_reviewer_failure_keeps_views():
    out = _team(Rec(pm2_exc=RuntimeError("pm2 down"), risk_exc=RuntimeError("risk down"))).run(INP)
    assert [(v.symbol, v.confidence) for v in out.views] == [("AAA", 0.6), ("BBB", 0.5)]
    assert "pm2 down" in out.reports["review_error"]["second_pm"]
    assert "risk down" in out.reports["review_error"]["risk_reviewer"]


def test_no_reviewers_cfg_means_one_pm():
    llm = Rec()
    AgentTeam(llm, AgentsCfg(), PortfolioCfg()).run(INP)
    assert [c[0] for c in llm.calls].count("PMDecision") == 1 and "RiskReview" not in [c[0] for c in llm.calls]


def test_agent_calls_include_reviewers(tmp_path):
    j = Journal(tmp_path / "j.db")
    rid = j.start_run("analyze", "paper")
    j.agent_report(rid, "inputs", {"as_of": "2026-11-03", "universe": ["AAA", "BBB", "CCC"]})
    j.agent_report(rid, "views_before_review", [
        {"symbol": "AAA", "expected_return": 0.03, "confidence": 0.6, "rationale": "r"},
        {"symbol": "BBB", "expected_return": 0.02, "confidence": 0.5, "rationale": "r"},
        {"symbol": "CCC", "expected_return": -0.01, "confidence": 0.5, "rationale": "r"}])
    j.agent_report(rid, "second_pm", {"market_view": "m", "cash_weight": 0.1, "views": [
        {"symbol": "AAA", "expected_return": -0.01, "confidence": 0.4, "rationale": "x"}]})
    j.agent_report(rid, "risk_review", {"summary": "s", "items": [
        {"symbol": "BBB", "action": "drop", "confidence": 0, "objection": "o"},
        {"symbol": "CCC", "action": "lower", "confidence": 0.2, "objection": "o"},
        {"symbol": "AAA", "action": "keep", "confidence": 0.3, "objection": ""}]})
    j.views(rid, [V("AAA", 0.03, 0.3)])     # final views after review
    j.finish_run(rid, "ok")
    calls = {(c["agent"], c["symbol"]): c for c in agent_calls(j, rid)}
    assert calls[("pm", "AAA")]["confidence"] == 0.6 and ("pm", "BBB") in calls   # the PM's own views
    assert calls[("second_pm", "AAA")]["direction"] == -1
    assert calls[("risk_reviewer", "BBB")]["direction"] == -1
    assert calls[("risk_reviewer", "CCC")]["direction"] == 1     # lowering a bearish view bets against it
    assert calls[("risk_reviewer", "AAA")]["direction"] == 1     # keep agrees with the view
