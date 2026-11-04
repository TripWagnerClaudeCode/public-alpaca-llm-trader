"""Intraday Event Review (spec 2026-11-04 Phase 3, "Intraday trading rules").

Each 30-minute watch looks for triggers on HELD names, in code:
- a move: |mid / prior close - 1| >= k x daily sigma (both from the latest analyze run's market refs);
- a filing: a new 8-K/10-Q/10-K on EDGAR (each filing URL reviewed once, ever).
Each trigger opens a review: the Event analyst (Sonnet, web research) proposes hold/trim/exit, then the
Event reviewer (Opus) answers on its own. A trim or exit executes only when BOTH agree (the less severe
call wins), only inside the time window, as a marketable limit sell. Every review is journaled.
Profile A has no stop: its exits come from here (views turning negative intraday), not new stop levels.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from bot.agents.llm import UsageLimited, load_prompt
from bot.agents.schemas import EventCall
from bot.agents.team import wrap
from bot.models import OrderPlan

log = logging.getLogger("bot.events")
NY = ZoneInfo("America/New_York")
SEVERITY = {"hold": 0, "trim": 1, "exit": 2}
BACKOFF_KEY = "agents_backoff_until"


@dataclass
class Trigger:
    symbol: str
    kind: str          # "move" | "filing"
    detail: str
    z: float
    filing: dict | None = None


def now_et() -> dt.datetime:
    return dt.datetime.now(NY)


def in_window(now: dt.datetime, window: tuple[str, str]) -> bool:
    """No LLM-initiated order in the first or last 15 minutes of the session (bounds inclusive)."""
    t = now.astimezone(NY).strftime("%H:%M")
    return window[0] <= t <= window[1]


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def find_triggers(symbols, mids: dict, prev_close: dict, sigma: dict, filings: dict, k: float) -> list[Trigger]:
    """Filings first (most specific news), then moves by |z|. Missing inputs never trigger a move."""
    out_f, out_m = [], []
    for sym in symbols:
        mid, prev, sig = mids.get(sym), prev_close.get(sym), sigma.get(sym)
        z = (mid / prev - 1) / sig if _ok(mid) and _ok(prev) and _ok(sig) else 0.0
        for f in filings.get(sym) or []:
            out_f.append(Trigger(sym, "filing", f"{f.get('form')} filed {f.get('date')}: {f.get('url')}", z, f))
        if abs(z) >= k:
            out_m.append(Trigger(sym, "move", f"{mid / prev - 1:+.2%} vs prior close ({z:+.1f} sigma)", z))
    return out_f + sorted(out_m, key=lambda t: -abs(t.z))


def _frac(c: EventCall) -> float:
    return 1.0 if c.action == "exit" else c.fraction


def combine(a: EventCall, b: EventCall) -> EventCall:
    """Both must agree to act; the less severe call wins (exit+trim -> trim; trims -> the smaller trim)."""
    for c in (a, b):
        if c.action == "trim" and not (isinstance(c.fraction, (int, float)) and math.isfinite(c.fraction)
                                       and 0 < c.fraction <= 1):
            return EventCall(action="hold", fraction=0.0, reason="invalid trim fraction")
    if SEVERITY[a.action] == 0 or SEVERITY[b.action] == 0:
        return EventCall(action="hold", fraction=0.0, reason="agents did not both call for a sale")
    if a.action == b.action == "exit":
        return EventCall(action="exit", fraction=1.0, reason="both agents: exit")
    f = min(_frac(a), _frac(b))
    if f >= 0.999:
        return EventCall(action="exit", fraction=1.0, reason="both agents: sell all")
    return EventCall(action="trim", fraction=f, reason=f"both agents: trim {f:.0%}")


def backing_off(journal, now: dt.datetime) -> bool:
    v = journal.get_meta(BACKOFF_KEY)
    if not v:
        return False
    try:
        return now < dt.datetime.fromisoformat(v)
    except ValueError:
        return False


def latest_analyze_run(journal, mode: str) -> int | None:
    row = journal.conn.execute("SELECT id FROM runs WHERE phase='analyze' AND status='ok' AND mode=? "
                               "ORDER BY id DESC LIMIT 1", (mode,)).fetchone()
    return int(row[0]) if row else None


def market_refs(journal, mode: str) -> tuple[dict[str, float], dict[str, float]]:
    """(prior close, daily sigma) per symbol from the latest analyze run (the gap guard's source)."""
    from bot.run_daily import load_cost_refs, load_market_refs

    rid = latest_analyze_run(journal, mode)
    if rid is None:
        return {}, {}
    _, close = load_market_refs(journal, rid)
    sig = {k: v[1] for k, v in load_cost_refs(journal, rid).items() if v[1]}
    return close, sig


def _default_filings(s):
    def fetch(sym: str, since: dt.date) -> list[dict]:
        from bot.research import edgar

        return edgar.recent_filings(sym, since, forms=("8-K", "10-Q", "10-K"),
                                    cache_dir=s.paths.resolve(s.paths.cache_dir))
    return fetch


def new_filings(journal, symbols, since: dt.date, fetch) -> tuple[dict[str, list], list[str]]:
    """Filings not yet reviewed (by URL, across all days). EDGAR errors are notes, never failures."""
    seen = {r[0] for r in journal.conn.execute(   # a failed review does not count: the filing is retried
        "SELECT filing_url FROM events WHERE filing_url IS NOT NULL AND action <> 'error'")}
    out, notes = {}, []
    for sym in symbols:
        try:
            fl = [f for f in fetch(sym, since) if f.get("url") and f["url"] not in seen]
        except Exception as e:  # noqa: BLE001
            notes.append(f"EDGAR check {sym} failed: {type(e).__name__}")
            continue
        if fl:
            out[sym] = fl[:1]      # newest only: one filing review per symbol per watch
    return out, notes


def _memory(s, journal, agents, symbols) -> dict[str, str]:
    try:
        from bot.memory.block import blocks_for_run

        mem, _ = blocks_for_run(s, journal, agents, symbols)
        return mem or {}
    except Exception:  # noqa: BLE001
        return {}


def _with_mem(mem: dict, agent: str, system: str, user: str) -> tuple[str, str]:
    if agent not in mem:
        return system, user
    from bot.memory.block import MEMORY_SENTENCE

    return f"{system}\n\n{MEMORY_SENTENCE}", f"{mem[agent]}\n\n{user}"


def _position(acct, sym: str) -> dict:
    d = acct.details[sym]
    value = acct.positions.get(sym, d.qty * d.current_price)
    return {"qty": d.qty, "avg_entry": d.avg_entry_price, "price": d.current_price,
            "pnl_pct": round(d.current_price / d.avg_entry_price - 1, 4) if d.avg_entry_price else None,
            "weight": round(value / acct.equity, 4) if acct.equity > 0 else None}


def review_one(s, llm, t: Trigger, pos: dict, mem: dict, filing_text: str | None) -> tuple[EventCall, dict, EventCall | None, dict]:
    ec = s.events
    evidence = {"trigger": t.detail, "position": pos}
    if t.filing:
        evidence["filing"] = {k: t.filing.get(k) for k in ("form", "date", "url", "title")}
        if filing_text:
            evidence["filing_text"] = filing_text
    user = (f"Held position: {t.symbol}. Trigger: {t.kind}.\n\nEvidence:\n"
            f"{wrap(json.dumps(evidence, default=str, indent=1))}")
    system, user = _with_mem(mem, "event_analyst", load_prompt("event_analyst", s.agents.prompt_dir), user)
    a, touched = llm.research(model=ec.analyst_model, effort=ec.analyst_effort, system=system, user=user,
                              schema=EventCall, max_turns=ec.max_turns, timeout_s=ec.timeout_s)
    arec = {"call": a.model_dump(), "touched": touched}
    if a.action == "hold":
        return a, arec, None, {}
    ruser = (f"Held position: {t.symbol}. Trigger: {t.kind}. The Event analyst proposes {a.action}"
             + (f" ({a.fraction:.0%} of the position)" if a.action == "trim" else "")
             + f".\n\nEvidence and the analyst's reasoning:\n"
             f"{wrap(json.dumps(evidence | {'analyst_reason': a.reason}, default=str, indent=1))}")
    rsys, ruser = _with_mem(mem, "event_reviewer", load_prompt("event_reviewer", s.agents.prompt_dir), ruser)
    b = llm.structured(model=ec.reviewer_model, effort=ec.reviewer_effort, system=rsys, user=ruser, schema=EventCall)
    return a, arec, b, {"call": b.model_dump()}


def sell_plan(sym: str, acct, quote: dict, call: EventCall, buffer: float,
              fractionable: bool = True) -> OrderPlan | None:
    """Marketable limit sell of the whole position (exit) or a fraction (trim); never Alpaca's market close.
    Non-fractionable names trim in whole shares (none if under one share)."""
    d = acct.details.get(sym)
    if d is None or d.qty <= 0 or not quote:
        return None
    qty = d.qty if call.action == "exit" else round(d.qty * call.fraction, 6)
    if call.action == "trim" and not fractionable:
        qty = float(math.floor(qty))
    if qty <= 0:
        return None
    limit = quote["bid"] * (1 - buffer)
    value = acct.positions.get(sym, d.qty * d.current_price)
    cur = value / acct.equity if acct.equity > 0 else 0.0
    return OrderPlan(symbol=sym, side="sell", notional=qty * quote["bid"], current_weight=cur,
                     target_weight=0.0 if call.action == "exit" else cur * (1 - call.fraction),
                     qty=qty, limit_price=limit)


def run_event_reviews(s, mode: str, broker, journal, run_id: int, acct, llm_factory, now_fn,
                      skip: set[str], filings_fn=None, deadline: float | None = None) -> list[str]:
    """Event Reviews for held names; returns notes for the watch run row. now_fn is re-read at submit
    time (a review takes minutes), so the window holds when the order is actually sent."""
    from bot.portfolio.costs import normalize_quote

    if not s.features.event_review:
        return []
    now = now_fn()
    ec, day = s.events, now.astimezone(NY).date()
    if not in_window(now, ec.window):
        return ["event review: outside the trading window"]
    if backing_off(journal, now):
        return ["event review: agents backing off (usage limit)"]
    held = sorted(x for x, d in acct.details.items() if d.qty > 0 and x not in skip)
    if not held:
        return []
    notes: list[str] = []
    close, sig = market_refs(journal, mode)
    quotes = {x: normalize_quote(broker.latest_quote(x)) for x in held}
    mids = {x: q["mid"] for x, q in quotes.items() if q}
    filings, fnotes = new_filings(journal, held, day - dt.timedelta(days=1), filings_fn or _default_filings(s))
    notes += fnotes
    today = journal.events_today(day, mode)
    done = {(e["symbol"], e["kind"]) for e in today}
    left = ec.max_reviews_per_day - sum(1 for e in today if e["pipeline"] == "review")
    triggers = [t for t in find_triggers(held, mids, close, sig, filings, ec.k_sigma)
                if (t.symbol, t.kind) not in done or t.kind == "filing"]
    reviewed: set[str] = set()
    llm = None
    import time

    for t in triggers:
        if left <= 0:
            notes.append("event review: daily cap reached")
            break
        if len(reviewed) >= ec.max_reviews_per_watch:
            notes.append("event review: per-watch cap reached (the rest wait for the next watch)")
            break
        if deadline is not None and time.monotonic() >= deadline:
            notes.append("event review: agent time budget for this watch used up")
            break
        if t.symbol in reviewed:
            continue
        reviewed.add(t.symbol)
        left -= 1
        llm = llm or llm_factory()
        mem = _memory(s, journal, ["event_analyst", "event_reviewer"], [t.symbol])
        text = None
        if t.filing:
            try:
                from bot.research import edgar

                text = edgar.filing_text(t.filing["url"], s.research.max_filing_chars)
            except Exception:  # noqa: BLE001 - the analyst can still fetch it itself
                text = None
        price = mids.get(t.symbol)
        try:
            a, arec, b, brec = review_one(s, llm, t, _position(acct, t.symbol), mem, text)
        except UsageLimited as e:
            until = now + dt.timedelta(hours=ec.backoff_hours)
            journal.set_meta(BACKOFF_KEY, until.isoformat())
            journal.record_event(day, t.symbol, t.kind, t.detail, {"error": str(e)[:300]}, None, "error",
                                 price, run_id, mode, filing_url=(t.filing or {}).get("url"))
            notes.append(f"event review: usage limit, agents off until {until:%H:%M} ET")
            break
        except Exception as e:  # noqa: BLE001 - one failed review never stops the watch
            log.warning("event review %s failed: %s", t.symbol, e)
            journal.record_event(day, t.symbol, t.kind, t.detail, {"error": f"{type(e).__name__}: {e}"[:300]},
                                 None, "error", price, run_id, mode, filing_url=(t.filing or {}).get("url"))
            continue
        final = combine(a, b) if b is not None else a
        action = final.action
        if action in ("trim", "exit"):
            from bot.risk import gate

            try:
                frac_ok = bool(broker.assets([t.symbol]).get(t.symbol, {}).get("fractionable", True))
            except Exception:  # noqa: BLE001 - unknown: assume whole shares (always valid)
                frac_ok = False
            plan = sell_plan(t.symbol, acct, quotes.get(t.symbol), final, s.execution.limit_buffer_pct, frac_ok)
            if plan is None:
                action = f"{action}: no quote"
            elif not in_window(now_fn(), ec.window) or not gate.exits_allowed(s.risk):
                action = f"{action}: blocked"
            elif t.symbol in journal.open_sell_symbols(day):
                action = f"{action}: sell already open"
            else:
                sub = broker.submit(plan)
                journal.order(run_id, plan, sub)
                if sub.status == "rejected":
                    action = f"{action}: rejected"
                notes.append(f"event {action} {t.symbol} ({t.kind})")
        journal.record_event(day, t.symbol, t.kind, t.detail, arec, brec or None, action, price, run_id, mode,
                             filing_url=(t.filing or {}).get("url"))
    return notes
