"""Morning Check (spec 2026-11-04, Phase 1): one agent reviews the planned BUYS at execute time
against what happened since the analyze run (overnight headlines, radar posts, the opening gap).

It can keep, reduce or cancel a buy. It can never add an order, enlarge one, or touch a sell:
apply_morning() enforces that in code, whatever the model returns.
"""
from __future__ import annotations

import json
import math

from bot.agents.llm import load_prompt
from bot.agents.schemas import MorningCheck
from bot.agents.team import wrap
from bot.config import MorningCheckCfg
from bot.models import OrderPlan


def build_user_prompt(as_of: str, context: dict) -> str:
    """Everything in the context comes from third parties or is derived from them: all of it is wrapped."""
    data = json.dumps(context, default=str, separators=(",", ":"), sort_keys=True)
    return (f"As of {as_of}. Review each planned buy below. Return one item per symbol.\n"
            f"Planned buys and new evidence:\n{wrap(data)}")


def run_morning_check(llm, cfg: MorningCheckCfg, prompt_dir: str, as_of: str, context: dict,
                      memory: str = "") -> MorningCheck:
    system, user = load_prompt("morning_check", prompt_dir), build_user_prompt(as_of, context)
    if memory:
        from bot.memory.block import MEMORY_SENTENCE

        system, user = f"{system}\n\n{MEMORY_SENTENCE}", f"{memory}\n\n{user}"
    return llm.structured(model=cfg.model, effort=cfg.effort, system=system, user=user, schema=MorningCheck)


def apply_morning(orders: list[OrderPlan], check: MorningCheck, min_notional: float) -> tuple[list[OrderPlan], list[str]]:
    """Apply the agent's decisions to BUYS only. Never enlarges an order; never adds one."""
    buys = {o.symbol for o in orders if o.side == "buy"}
    decisions = {}
    notes: list[str] = []
    rank = {"cancel": 0, "reduce": 1, "keep": 2}
    for it in check.items:
        sym = it.symbol.upper().strip()
        if sym not in buys:
            notes.append(f"{sym}: morning check item ignored (not a planned buy)")
            continue
        if it.action == "reduce" and not math.isfinite(float(it.fraction)):
            notes.append(f"{sym}: morning check reduce ignored (fraction is not a number)")
            continue
        prev = decisions.get(sym)   # duplicate items: the most restrictive wins
        if prev is None or rank[it.action] < rank[prev.action] or (
                it.action == prev.action == "reduce" and it.fraction < prev.fraction):
            decisions[sym] = it
    out: list[OrderPlan] = []
    for o in orders:
        it = decisions.get(o.symbol) if o.side == "buy" else None
        if it is None or it.action == "keep":
            out.append(o)
            continue
        if it.action == "cancel":
            notes.append(f"{o.symbol}: buy cancelled by morning check ({it.reason})")
            continue
        frac = min(max(float(it.fraction), 0.0), 1.0)
        if o.qty is not None:
            qty = float(int(o.qty * frac))
            notional = qty * (o.limit_price or (o.notional / o.qty if o.qty else 0.0))
            new = o.model_copy(update={"qty": qty, "notional": round(notional, 2)})
        else:
            new = o.model_copy(update={"notional": round(o.notional * frac, 2)})
        if new.notional < min_notional or (new.qty is not None and new.qty < 1):
            notes.append(f"{o.symbol}: buy dropped, reduced below the minimum order ({it.reason})")
            continue
        notes.append(f"{o.symbol}: buy reduced to {frac:.0%} by morning check ({it.reason})")
        out.append(new)
    return out, notes
