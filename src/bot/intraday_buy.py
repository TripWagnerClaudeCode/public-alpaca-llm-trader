"""Intraday buys (spec 2026-11-04 Phase 3, "Intraday trading rules"): the bot's EXISTING strategy, sooner.

Only an event trigger (big move or new filing) on an unheld symbol of the bot's own universe (A: its
symbol list; B: the current screener list) can start one. A single-symbol pipeline then runs:
deep dive -> PM view (positive and within max_abs_view_return) -> an independent second PM view AND the
Risk Reviewer must both approve -> the optimizer sizes it against the latest analyze views -> caps ->
risk gate -> marketable limit buy. The LLM never picks the size.

Locked caps (risk:): at most intraday_max_weight_frac x max_position_weight per name, at most
intraday_max_buys_per_day buys, today's turnover counts toward max_daily_turnover, no re-buy of a name
sold today, the stop cooldown applies, min cash is kept. Any kill switch (manual or automatic) blocks.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math

from bot.agents.llm import UsageLimited, load_prompt
from bot.agents.schemas import BuyApproval, PMDecision, PMView
from bot.agents.team import validate_views, wrap
from bot.events import (BACKOFF_KEY, NY, Trigger, _default_filings, _memory, _with_mem, backing_off,
                        find_triggers, in_window, latest_analyze_run, market_refs, new_filings)
from bot.models import OrderPlan, View
from bot.portfolio.optimizer import target_weights

log = logging.getLogger("bot.intraday_buy")
DONE = ("planned", "rejected", "canceled", "cancelled", "expired", "blocked")


def universe_for(s, day: dt.date) -> list[str]:
    if s.universe.source == "screener":
        from bot.run_daily import screener_universe

        return screener_universe(s, day) or []
    return list(s.universe.symbols)


def buy_candidates(s, universe, held, triggers: list[Trigger], today_events: list[dict]) -> list[Trigger]:
    """Unheld universe symbols with a trigger and no buy pipeline yet today; one per symbol, by |z|."""
    piped = {e["symbol"] for e in today_events if e.get("pipeline") == "buy"}
    uni, best = set(universe), {}
    for t in triggers:
        if t.symbol in uni and t.symbol not in held and t.symbol not in piped:
            best.setdefault(t.symbol, t)
    return sorted(best.values(), key=lambda t: -abs(t.z))


def size_intraday(closes, base_views: list[View], view: View, pcfg, horizon: int, cur_weights: dict,
                  risk) -> float:
    """Weight to buy: the optimizer's target for the symbol with the new view replacing any analyze view,
    capped at intraday_max_weight_frac x max_position_weight, minus what is already held. Never negative."""
    views = [v for v in base_views if v.symbol != view.symbol] + [view]
    pc = pcfg.model_copy(update={"min_cash_weight": max(pcfg.min_cash_weight, 1 - risk.max_gross_exposure)})
    tw = target_weights(closes, views, pc, horizon)
    target = float(tw.get(view.symbol, 0.0)) if hasattr(tw, "get") else 0.0
    if not math.isfinite(target):
        return 0.0
    cap = risk.intraday_max_weight_frac * risk.max_position_weight
    return max(0.0, min(target, cap) - float(cur_weights.get(view.symbol, 0.0)))


def _orders_today(journal, day: dt.date, mode: str):
    nxt = day + dt.timedelta(days=1)
    return journal.conn.execute(
        "SELECT o.symbol, o.side, o.notional, COALESCE(o.status,'') FROM orders o JOIN runs r ON r.id = o.run_id "
        "WHERE r.mode=? AND o.created_at >= ? AND o.created_at < ?", (mode, str(day), str(nxt))).fetchall()


def turnover_today(journal, day: dt.date, mode: str, equity: float) -> float:
    """Sum of today's submitted order notionals (any phase) as a share of equity."""
    if equity <= 0:
        return 0.0
    return sum(n or 0.0 for _, _, n, st in _orders_today(journal, day, mode) if st not in DONE) / equity


def exited_today(journal, day: dt.date, mode: str) -> set[str]:
    return {sym for sym, side, _, st in _orders_today(journal, day, mode) if side == "sell" and st not in DONE}


def _base_views(journal, rid: int | None) -> tuple[list[View], list[str]]:
    if rid is None:
        return [], []
    views = [View(symbol=a, expected_return=b, confidence=c, rationale=d or "") for a, b, c, d in
             journal.conn.execute("SELECT symbol, expected_return, confidence, rationale FROM views WHERE run_id=?",
                                  (rid,))]
    row = journal.conn.execute("SELECT content FROM agent_reports WHERE run_id=? AND agent='inputs'", (rid,)).fetchone()
    uni = []
    if row:
        try:
            uni = list(json.loads(row[0]).get("universe") or [])
        except (TypeError, ValueError):
            uni = []
    return views, uni


def _default_closes(s):
    def fetch(symbols):
        from bot.data.market import MarketData

        md = MarketData(s.paths.resolve(s.paths.cache_dir))
        return md.closes(md.daily_bars(list(symbols), s.data.history_days))
    return fetch


def _pm_view(llm, model: str, effort: str, system: str, user: str, s, sym: str) -> View | None:
    v = llm.structured(model=model, effort=effort, system=system, user=user, schema=PMView)
    v = v.model_copy(update={"symbol": sym})
    ok, _, _ = validate_views(PMDecision(market_view="", cash_weight=s.portfolio.min_cash_weight, views=[v]),
                              [sym], s.agents, s.portfolio.min_cash_weight)
    return ok[0] if ok else None


def run_intraday_buys(s, mode: str, broker, journal, run_id: int, acct, llm_factory, now_fn,
                      filings_fn=None, closes_fn=None, deadline: float | None = None) -> list[str]:
    """Intraday buy pipelines (features.intraday_buys). Returns notes; never raises past the watch."""
    from bot.agents.research import Researcher
    from bot.portfolio.costs import normalize_quote
    from bot.risk import gate

    if not s.features.intraday_buys:
        return []
    now = now_fn()
    ec, rk, day = s.events, s.risk, now.astimezone(NY).date()
    if not in_window(now, ec.window):
        return ["intraday buys: outside the trading window"]
    if gate.kill_switch_active(rk):
        return ["intraday buys: kill switch active"]
    if backing_off(journal, now):
        return ["intraday buys: agents backing off (usage limit)"]
    today = journal.events_today(day, mode)
    pipes = sum(1 for e in today if e["pipeline"] == "buy")
    bought = sum(1 for e in today if e["pipeline"] == "buy" and e["action"] == "buy")
    if pipes >= ec.max_buy_pipelines_per_day or bought >= rk.intraday_max_buys_per_day:
        return []
    eq = acct.equity
    floor = max(s.portfolio.min_cash_weight, 1 - rk.max_gross_exposure)
    rows = _orders_today(journal, day, mode)
    open_buys = {sym for sym, side, _, st in rows if side == "buy" and st not in DONE}
    # buys still working (not filled yet) have not left the account's cash: count them as spent
    cash = acct.cash - sum(n or 0.0 for _, side, n, st in rows if side == "buy" and st not in DONE and st != "filled")
    if eq <= 0 or cash / eq - floor <= rk.min_order_notional / eq:
        return ["intraday buys: no cash above the minimum cash weight"]
    held = ({x for x, d in acct.details.items() if d.qty > 0} | {x for x, v in acct.positions.items() if v > 0}
            | open_buys)   # a name with an open buy order today is never bought again intraday
    blocked = exited_today(journal, day, mode) | set(journal.active_cooldowns(day, mode))
    uni = [x for x in universe_for(s, day) if x not in held and x not in blocked]
    if not uni:
        return []
    close, sig = market_refs(journal, mode)
    quotes = {x: normalize_quote(broker.latest_quote(x)) for x in uni}
    mids = {x: q["mid"] for x, q in quotes.items() if q}
    filings, notes = new_filings(journal, uni, day - dt.timedelta(days=1), filings_fn or _default_filings(s))
    cands = buy_candidates(s, uni, held, find_triggers(uni, mids, close, sig, filings, ec.k_sigma), today)
    if not cands:
        return notes
    rid = latest_analyze_run(journal, mode)
    base, base_uni = _base_views(journal, rid)
    base_of = {v.symbol: v for v in base}
    rv = s.reviewers
    cur_w = {x: v / eq for x, v in acct.positions.items()}
    import time

    here = 0
    for t in cands:
        if pipes >= ec.max_buy_pipelines_per_day or bought >= rk.intraday_max_buys_per_day:
            break
        if here >= ec.max_buy_pipelines_per_watch or (deadline is not None and time.monotonic() >= deadline):
            notes.append("intraday buys: per-watch limit reached")
            break
        pipes += 1
        here += 1
        sym, price, furl = t.symbol, mids.get(t.symbol), (t.filing or {}).get("url")
        arec: dict = {}
        brec: dict = {}
        try:
            llm = llm_factory()
            mem = _memory(s, journal, ["portfolio_manager", "second_pm", "risk_reviewer"], [sym])
            researcher = Researcher(llm, s.research, s.agents.prompt_dir, cache_dir=s.paths.resolve(s.paths.cache_dir),
                                    filings_fn=filings_fn)
            prior = base_of.get(sym)
            q = (f"What caused today's {t.kind} in {sym} ({t.detail}), and does it change the outlook for the "
                 f"next {s.agents.horizon_days} trading days?")
            dive, drec = researcher.dive(sym, q, {"trigger": t.detail,
                                                  "last_analyze_view": prior.model_dump() if prior else None})
            arec["dive"] = drec | {"findings": dive.model_dump() if dive else None}
            evidence = {"trigger": t.detail, "mid": price, "prior_close": close.get(sym),
                        "last_analyze_view": prior.model_dump() if prior else None,
                        "findings": dive.model_dump() if dive else None}
            user = (f"Symbol: {sym}. Horizon: {s.agents.horizon_days} trading days. Expected return must stay "
                    f"within +/-{s.agents.max_abs_view_return:.0%}.\n\nEvidence:\n"
                    f"{wrap(json.dumps(evidence, default=str, indent=1))}")
            system = load_prompt("intraday_pm", s.agents.prompt_dir)
            psys, puser = _with_mem(mem, "portfolio_manager", system, user)
            view = _pm_view(llm, s.agents.pm_model, s.agents.pm_effort, psys, puser, s, sym)
            arec["pm"] = view.model_dump() if view else None
            if view is None or view.expected_return <= 0:
                journal.record_event(day, sym, t.kind, t.detail, arec, None, "pass: no positive view", price,
                                     run_id, mode, pipeline="buy", filing_url=furl)
                continue
            ssys, suser = _with_mem(mem, "second_pm", system, user)   # independent: never sees the first view
            v2 = _pm_view(llm, rv.second_pm_model, rv.second_pm_effort, ssys, suser, s, sym)
            brec["second_pm"] = v2.model_dump() if v2 else None
            ruser = (f"Approve or reject this single intraday buy of {sym}.\n\nProposed view:\n"
                     f"{json.dumps(view.model_dump(), default=str)}\n\nCurrent portfolio weights:\n"
                     f"{json.dumps({k: round(v, 4) for k, v in cur_w.items()})}\n\nRisk limits:\n"
                     f"{json.dumps(rk.model_dump(), default=str)}\n\nEvidence:\n"
                     f"{wrap(json.dumps(evidence, default=str, indent=1))}")
            rsys, ruser = _with_mem(mem, "risk_reviewer", load_prompt("risk_reviewer", s.agents.prompt_dir), ruser)
            ra = llm.structured(model=rv.risk_model, effort=rv.risk_effort, system=rsys, user=ruser, schema=BuyApproval)
            brec["risk"] = ra.model_dump()
            if not (v2 is not None and v2.expected_return > 0 and ra.approve):
                journal.record_event(day, sym, t.kind, t.detail, arec, brec, "rejected: approvals", price,
                                     run_id, mode, pipeline="buy", filing_url=furl)
                continue
            closes = (closes_fn or _default_closes(s))(base_uni or uni)
            w = size_intraday(closes, base, view, s.portfolio, s.agents.horizon_days, cur_w, rk)
            avail = cash / eq - floor
            w = min(w, avail)
            q_ = quotes.get(sym)
            action = "buy"
            plan = None
            if w * eq < rk.min_order_notional or not q_:
                action = "blocked: size below minimum" if q_ else "blocked: no quote"
            elif turnover_today(journal, day, mode, eq) + w > rk.max_daily_turnover + 0.01:
                action = "blocked: daily turnover"
            else:
                limit = q_["ask"] * (1 + s.execution.limit_buffer_pct)
                notional = w * eq
                info = broker.assets([sym]).get(sym, {"tradable": True, "fractionable": True})
                qty = notional / limit if info.get("fractionable", True) else math.floor(notional / limit)
                if not info.get("tradable", True) or qty <= 0:
                    action = "blocked: not tradable / below one share"
                else:
                    qty = round(qty, 6)
                    plan = OrderPlan(symbol=sym, side="buy", notional=round(qty * limit, 2),
                                     current_weight=cur_w.get(sym, 0.0), target_weight=cur_w.get(sym, 0.0) + w,
                                     qty=qty, limit_price=limit)
                    chk = gate.check_orders(rk, [plan], acct.model_copy(update={"cash": cash}))
                    if not chk.ok:
                        action, plan = "blocked: risk gate " + "; ".join(chk.reasons), None
                    elif not in_window(now_fn(), ec.window) or gate.kill_switch_active(rk):
                        action, plan = "blocked: window or kill switch", None
            if plan is not None:
                sub = broker.submit(plan)
                journal.order(run_id, plan, sub)
                journal.views(run_id, [view])
                if sub.status == "rejected":
                    action = "buy: rejected"
                else:
                    bought += 1
                    cash -= plan.notional                 # the next candidate sees what this buy spent
                    cur_w[sym] = cur_w.get(sym, 0.0) + w
                notes.append(f"intraday buy {sym} {w:.1%} ({t.kind})")
            journal.record_event(day, sym, t.kind, t.detail, arec, brec, action, price, run_id, mode,
                                 pipeline="buy", filing_url=furl)
        except UsageLimited as e:
            until = now + dt.timedelta(hours=ec.backoff_hours)
            journal.set_meta(BACKOFF_KEY, until.isoformat())
            journal.record_event(day, sym, t.kind, t.detail, arec | {"error": str(e)[:300]}, brec or None, "error",
                                 price, run_id, mode, pipeline="buy", filing_url=furl)
            notes.append(f"intraday buys: usage limit, agents off until {until:%H:%M} ET")
            break
        except Exception as e:  # noqa: BLE001 - one failed pipeline never stops the watch
            log.warning("intraday buy pipeline %s failed: %s", sym, e)
            journal.record_event(day, sym, t.kind, t.detail, arec | {"error": f"{type(e).__name__}: {e}"[:300]},
                                 brec or None, "error", price, run_id, mode, pipeline="buy", filing_url=furl)
    return notes
