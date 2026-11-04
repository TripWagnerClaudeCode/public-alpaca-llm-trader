"""The agent team: 4 analysts (parallel) -> bull/bear debate -> portfolio manager.

All market data is fetched in code and passed in as compact JSON, so each agent call
is a single structured-output request (no tool loop). That keeps runs cheap and
reproducible, and every input is journaled.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from bot.agents.llm import LLM, load_prompt
from bot.agents.schemas import AnalystReport, DebateCase, PMDecision, RegimeReport, ResearchPlan, RiskReview
from bot.config import AgentsCfg, PortfolioCfg
from bot.models import View

log = logging.getLogger(__name__)


@dataclass
class TeamInputs:
    as_of: str
    universe: list[str]
    regime: dict
    technical: dict[str, dict]
    fundamentals: dict[str, dict]
    news: dict[str, list[dict]]


@dataclass
class TeamOutput:
    views: list[View]
    cash_weight: float
    reports: dict[str, object] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)


def _j(obj) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"), sort_keys=True)


_OPEN, _CLOSE = "<untrusted_data>", "</untrusted_data>"
_TAG = re.compile(r"<\s*(/?)\s*untrusted_data\s*>", re.IGNORECASE)


def wrap(text: str) -> str:
    """Wrap third-party text (headlines, fundamentals, market data) so the model treats it as data.
    Any tag-like variant inside it, in any case or spacing, is defanged so it cannot close the wrapper early."""
    s = _TAG.sub(lambda m: f"[{m.group(1)}untrusted_data]", text)
    return f"{_OPEN}\n{s}\n{_CLOSE}"


def validate_views(decision: PMDecision, universe: list[str], cfg: AgentsCfg,
                   min_cash: float) -> tuple[list[View], float, list[str]]:
    """Deterministic post-checks on the PM's output. Bad views are dropped, never 'fixed'."""
    allowed = set(universe)
    out, rejected, seen = [], [], set()
    for v in decision.views:
        sym = v.symbol.upper().strip()
        if sym not in allowed:
            rejected.append(f"{sym}: not in universe")
            continue
        if sym in seen:
            rejected.append(f"{sym}: duplicate view")
            continue
        if abs(v.expected_return) > cfg.max_abs_view_return:
            rejected.append(f"{sym}: |{v.expected_return:+.2%}| exceeds max {cfg.max_abs_view_return:.0%}")
            continue
        seen.add(sym)
        out.append(View(symbol=sym, expected_return=v.expected_return,
                        confidence=min(v.confidence, 0.8), rationale=v.rationale))
    cash = min(max(decision.cash_weight, min_cash), 0.5)
    return out, cash, rejected


class AgentTeam:
    def __init__(self, llm: LLM, cfg: AgentsCfg, pcfg: PortfolioCfg, memory: dict[str, str] | None = None,
                 researcher=None, research_cfg=None, reviewers_cfg=None, context: dict | None = None):
        """memory: {agent: wrapped memory block} (features.memory). None = prompts exactly as before.
        researcher (features.research): the PM model plans up to research_cfg.max_dives deep dives after
        the debate; findings reach the PM wrapped as untrusted data. None = no planner, no dives."""
        self.llm, self.cfg, self.pcfg, self.memory = llm, cfg, pcfg, memory
        self.researcher, self.research_cfg = researcher, research_cfg
        # reviewers_cfg (features.reviewers): independent second PM + Risk Reviewer after the PM.
        # context: {"weights": current portfolio weights, "risk_limits": the risk: settings} for the reviewer.
        self.reviewers_cfg, self.context = reviewers_cfg, context or {}

    def _review(self, inp: TeamInputs, views: list[View], pm_system: str, pm_user: str, reports: dict) -> list[View]:
        """Second PM (same inputs, own memory, no access to PM1's output) cuts disagreeing confidences, then
        the Risk Reviewer may lower or drop. A failing reviewer leaves the views as they were."""
        from bot.agents.review import apply_risk_review, second_pm_cut

        rc = self.reviewers_cfg
        reports["views_before_review"] = [v.model_dump() for v in views]
        errors: dict[str, str] = {}
        try:
            system, user = self._with_memory("second_pm", pm_system, pm_user)
            pm2 = self.llm.structured(model=rc.second_pm_model, effort=rc.second_pm_effort, system=system,
                                      user=user, schema=PMDecision)
            reports["second_pm"] = pm2
            pm2_views, _, _ = validate_views(pm2, inp.universe, self.cfg, self.pcfg.min_cash_weight)
            views, notes = second_pm_cut(views, pm2_views, rc.disagree_factor)
            reports["second_pm_cut"] = notes
        except Exception as e:  # noqa: BLE001
            log.warning("second PM failed: %s", e)
            errors["second_pm"] = f"{type(e).__name__}: {e}"[:500]
        if views:
            try:
                system = load_prompt("risk_reviewer", self.cfg.prompt_dir)
                user = (f"As of {inp.as_of}. Review every view below.\n\nViews:\n{_j([v.model_dump() for v in views])}"
                        f"\n\nCurrent portfolio weights:\n{_j(self.context.get('weights', {}))}"
                        f"\n\nRisk limits:\n{_j(self.context.get('risk_limits', {}))}"
                        f"\n\nDeep-dive findings:\n{wrap(_j(reports.get('research') or {}))}")
                system, user = self._with_memory("risk_reviewer", system, user)
                rv = self.llm.structured(model=rc.risk_model, effort=rc.risk_effort, system=system, user=user,
                                         schema=RiskReview)
                reports["risk_review"] = rv
                views, notes = apply_risk_review(views, rv)
                reports["risk_review_applied"] = notes
            except Exception as e:  # noqa: BLE001
                log.warning("Risk Reviewer failed: %s", e)
                errors["risk_reviewer"] = f"{type(e).__name__}: {e}"[:500]
        if errors:
            reports["review_error"] = errors
        return views

    def _research(self, inp: TeamInputs, analyst_blob: str, bull, bear, reports: dict) -> dict:
        """Planner -> capped, universe-only dives in parallel. Returns {symbol: findings dict}. Never raises."""
        rc = self.research_cfg
        try:
            system = load_prompt("research_planner", self.cfg.prompt_dir)
            user = (f"As of {inp.as_of}. Universe: {', '.join(inp.universe)}. You may request at most "
                    f"{rc.max_dives} deep dives.\n\nAnalyst reports:\n{analyst_blob}\n\n"
                    f"Bull case:\n{_j(bull.model_dump())}\n\nBear case:\n{_j(bear.model_dump())}")
            system, user = self._with_memory("portfolio_manager", system, user)
            plan = self.llm.structured(model=self.cfg.pm_model, effort=rc.planner_effort, system=system,
                                       user=user, schema=ResearchPlan)
        except Exception as e:  # noqa: BLE001 - no plan, no dives; the PM still decides
            log.warning("research planner failed: %s", e)
            reports["research_error"] = {"error": f"{type(e).__name__}: {e}"[:500]}
            return {}
        reports["research_plan"] = plan.model_dump()
        allowed, picked = set(inp.universe), {}
        for r in plan.requests:
            sym = r.symbol.upper().strip()
            if sym in allowed and sym not in picked and len(picked) < rc.max_dives:
                picked[sym] = r.question
        jobs = {}
        with ThreadPoolExecutor(max_workers=max(1, min(3, len(picked)))) as ex:
            for sym, q in picked.items():
                ctx = {"technical": inp.technical.get(sym), "fundamentals": inp.fundamentals.get(sym),
                       "news": inp.news.get(sym)}
                jobs[sym] = ex.submit(self.researcher.dive, sym, q, ctx)
            results = {sym: f.result() for sym, f in jobs.items()}
        reports["research_log"] = {sym: rec for sym, (_, rec) in results.items()}
        found = {sym: res.model_dump() for sym, (res, _) in results.items() if res is not None}
        reports["research"] = found
        return found

    def _with_memory(self, agent: str, system: str, user: str) -> tuple[str, str]:
        if not self.memory or agent not in self.memory:
            return system, user
        from bot.memory.block import MEMORY_SENTENCE

        return f"{system}\n\n{MEMORY_SENTENCE}", f"{self.memory[agent]}\n\n{user}"

    def _analyst(self, name: str, payload: dict, schema):
        user = f"As of {payload.pop('as_of')}. Data:\n{wrap(_j(payload))}"
        system, user = self._with_memory(name, load_prompt(name, self.cfg.prompt_dir), user)
        return self.llm.structured(model=self.cfg.analyst_model, effort=self.cfg.analyst_effort,
                                   system=system, user=user, schema=schema)

    def run(self, inp: TeamInputs) -> TeamOutput:
        jobs = {
            "regime": (RegimeReport, {"as_of": inp.as_of, "market": inp.regime}),
            "technical": (AnalystReport, {"as_of": inp.as_of, "symbols": inp.technical}),
            "fundamentals": (AnalystReport, {"as_of": inp.as_of, "symbols": inp.fundamentals}),
            "news": (AnalystReport, {"as_of": inp.as_of, "symbols": inp.news}),
        }
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {k: ex.submit(self._analyst, k, payload, schema) for k, (schema, payload) in jobs.items()}
            reports = {k: f.result() for k, f in futs.items()}
        analyst_blob = _j({k: v.model_dump() for k, v in reports.items()})

        bull = bear = None
        for rnd in range(max(1, self.cfg.debate_rounds)):
            ctx = f"Analyst reports:\n{analyst_blob}"
            bull_ctx = ctx + (f"\n\nBear case from previous round:\n{_j(bear.model_dump())}" if bear else "")
            bear_ctx = ctx + (f"\n\nBull case:\n{_j(bull.model_dump())}" if bull else "")
            bull_sys, bull_ctx = self._with_memory("bull", load_prompt("bull", self.cfg.prompt_dir), bull_ctx)
            bear_sys, bear_ctx = self._with_memory("bear", load_prompt("bear", self.cfg.prompt_dir), bear_ctx)
            with ThreadPoolExecutor(max_workers=2) as ex:
                fb = ex.submit(self.llm.structured, model=self.cfg.debate_model, effort=self.cfg.debate_effort,
                               system=bull_sys, user=bull_ctx, schema=DebateCase)
                fr = ex.submit(self.llm.structured, model=self.cfg.debate_model, effort=self.cfg.debate_effort,
                               system=bear_sys, user=bear_ctx, schema=DebateCase)
                bull, bear = fb.result(), fr.result()
            reports[f"bull_r{rnd + 1}"], reports[f"bear_r{rnd + 1}"] = bull, bear

        pm_system = load_prompt("portfolio_manager", self.cfg.prompt_dir).format(
            horizon=self.cfg.horizon_days, max_view=self.cfg.max_abs_view_return,
            min_cash=self.pcfg.min_cash_weight, universe=", ".join(inp.universe),
        )
        pm_user = (f"As of {inp.as_of}.\n\nAnalyst reports:\n{analyst_blob}\n\n"
                   f"Bull case:\n{_j(bull.model_dump())}\n\nBear case:\n{_j(bear.model_dump())}")
        if self.researcher is not None and self.research_cfg is not None:
            found = self._research(inp, analyst_blob, bull, bear, reports)
            if found:
                pm_user += f"\n\nDeep-dive findings:\n{wrap(_j(found))}"
        base_system, base_user = pm_system, pm_user
        pm_system, pm_user = self._with_memory("portfolio_manager", pm_system, pm_user)
        decision = self.llm.structured(model=self.cfg.pm_model, effort=self.cfg.pm_effort,
                                       system=pm_system, user=pm_user, schema=PMDecision)
        reports["portfolio_manager"] = decision
        views, cash, rejected = validate_views(decision, inp.universe, self.cfg, self.pcfg.min_cash_weight)
        if rejected:
            log.warning("Rejected PM views: %s", rejected)
        if self.reviewers_cfg is not None:
            views = self._review(inp, views, base_system, base_user, reports)
        return TeamOutput(views=views, cash_weight=cash, reports=reports, rejected=rejected)
