"""Phase 3 deep dives inside the agent team: planner -> capped dives -> PM sees wrapped findings."""
import threading

from bot.agents.schemas import (AnalystReport, DebateCase, DiveFindings, Finding, PMDecision, RegimeReport,
                                ResearchPlan, ResearchRequest)
from bot.agents.team import AgentTeam, TeamInputs
from bot.config import AgentsCfg, PortfolioCfg, ResearchCfg

UNI = ["AAA", "BBB", "CCC", "DDD"]
INP = TeamInputs(as_of="2026-11-04", universe=UNI, regime={}, technical={s: {"rsi": 50} for s in UNI},
                 fundamentals={s: {} for s in UNI}, news={s: [{"title": f"{s} news"}] for s in UNI})


class Rec:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, plan=None, plan_exc=None):
        self.calls, self.plan, self.plan_exc = [], plan, plan_exc
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
        if schema is ResearchPlan:
            if self.plan_exc:
                raise self.plan_exc
            return self.plan
        return PMDecision(market_view="m", cash_weight=0.1, views=[])


class FakeResearcher:
    def __init__(self, fail=False):
        self.dives, self.fail = [], fail
        self._lock = threading.Lock()

    def dive(self, symbol, question, context):
        with self._lock:
            self.dives.append((symbol, question, context))
        if self.fail:
            return None, {"symbol": symbol, "error": "RuntimeError: down", "touched": []}
        return (DiveFindings(symbol=symbol, question=question, answer=f"answer {symbol}", findings=[
            Finding(claim=f"{symbol} filed 8-K", source_url="https://sec.gov/x", date="2026-11-02",
                    support="strong")]),
                {"symbol": symbol, "error": None, "touched": [{"tool": "WebFetch", "url": "https://sec.gov/x"}]})


def _plan(*pairs):
    return ResearchPlan(requests=[ResearchRequest(symbol=s, question=q) for s, q in pairs])


def test_no_researcher_means_no_calls():
    llm = Rec()
    out = AgentTeam(llm, AgentsCfg(), PortfolioCfg()).run(INP)
    assert len(llm.calls) == 7 and "research_plan" not in out.reports


def test_planner_capped_and_universe_only():
    llm = Rec(_plan(("AAA", "q1"), ("ZZZ", "outside"), ("aaa", "dup"), ("BBB", "q2"), ("CCC", "q3"), ("DDD", "q4")))
    r = FakeResearcher()
    out = AgentTeam(llm, AgentsCfg(), PortfolioCfg(), researcher=r, research_cfg=ResearchCfg(max_dives=3)).run(INP)
    assert sorted(d[0] for d in r.dives) == ["AAA", "BBB", "CCC"]
    assert {d[0]: d[1] for d in r.dives}["AAA"] == "q1"
    assert r.dives[0][2]["news"] and "technical" in r.dives[0][2]
    planner = [c for c in llm.calls if c[0] == "ResearchPlan"][0]
    assert planner[1] == AgentsCfg().pm_model and planner[2] == "medium"
    assert set(out.reports["research"]) == {"AAA", "BBB", "CCC"}
    assert set(out.reports["research_log"]) == {"AAA", "BBB", "CCC"}
    assert out.reports["research_plan"]["requests"][0]["symbol"] == "AAA"


def test_pm_sees_wrapped_findings():
    llm = Rec(_plan(("AAA", "q1")))
    AgentTeam(llm, AgentsCfg(), PortfolioCfg(), researcher=FakeResearcher(),
              research_cfg=ResearchCfg()).run(INP)
    pm_user = [c for c in llm.calls if c[0] == "PMDecision"][0][4]
    block = pm_user[pm_user.index("Deep-dive findings:"):]
    assert block.index("<untrusted_data>") < block.index("AAA filed 8-K") < block.index("</untrusted_data>")


def test_no_dives_requested_pm_prompt_unchanged():
    llm = Rec(_plan())
    AgentTeam(llm, AgentsCfg(), PortfolioCfg(), researcher=FakeResearcher(), research_cfg=ResearchCfg()).run(INP)
    pm_user = [c for c in llm.calls if c[0] == "PMDecision"][0][4]
    assert "Deep-dive findings" not in pm_user


def test_dive_failures_dont_block_pm():
    llm = Rec(_plan(("AAA", "q1"), ("BBB", "q2")))
    out = AgentTeam(llm, AgentsCfg(), PortfolioCfg(), researcher=FakeResearcher(fail=True),
                    research_cfg=ResearchCfg()).run(INP)
    assert [c for c in llm.calls if c[0] == "PMDecision"]
    assert out.reports["research"] == {}
    assert out.reports["research_log"]["AAA"]["error"]


def test_planner_failure_dont_block_pm():
    llm = Rec(plan_exc=RuntimeError("planner down"))
    out = AgentTeam(llm, AgentsCfg(), PortfolioCfg(), researcher=FakeResearcher(),
                    research_cfg=ResearchCfg()).run(INP)
    assert [c for c in llm.calls if c[0] == "PMDecision"]
    assert "planner down" in out.reports["research_error"]["error"]
