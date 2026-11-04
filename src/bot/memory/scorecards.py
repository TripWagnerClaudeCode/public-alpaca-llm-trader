"""Scorecards built from outcomes_v2 and every agent's calls (no LLM).

hit = sign(direction) == sign(excess vs universe), over non-zero calls only.
mean_signed_excess = mean(direction * excess vs universe): above 0 means the agent's calls helped.
"""
from __future__ import annotations

import datetime as dt

from bot.journal.store import Journal
from bot.memory.outcomes import _ensure, agent_calls

BUCKETS = (("0.0-0.3", 0.0, 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-0.8", 0.5, 0.81))
NO_RECORD = "No track record yet: views are scored 10 trading days after they are made."


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def build_scorecards(journal: Journal, since: dt.date | None = None, horizon: int = 10) -> dict:
    _ensure(journal)
    sql = "SELECT run_id, symbol, as_of, excess_univ, ret FROM outcomes_v2 WHERE horizon=?"
    params: tuple = (horizon,)
    if since is not None:
        sql += " AND as_of >= ?"
        params += (str(since),)
    outs: dict[int, dict[str, tuple[str, float, float]]] = {}
    for rid, sym, as_of, ex, ret in journal.conn.execute(sql, params):
        outs.setdefault(int(rid), {})[sym] = (as_of, ex, ret)
    agents: dict[str, list[tuple[float, float]]] = {}
    cal: dict[str, list[float]] = {k: [] for k, _, _ in BUCKETS}
    cancels: list[float] = []
    symbols: dict[str, list[dict]] = {}
    n_views = 0
    for rid in sorted(outs):
        o = outs[rid]
        for c in agent_calls(journal, rid):
            sym, d = c["symbol"], c["direction"]
            if sym not in o:
                continue
            as_of, ex, ret = o[sym]
            if c["agent"] == "morning_check":   # scored from its own decision price (after the gap)
                ex = (1 + ret) / (1 + c["gap_pct"]) - 1 - (ret - ex)
            if c["agent"] == "pm":
                n_views += 1
                symbols.setdefault(sym, []).append({"as_of": as_of, "pm_direction": d, "excess_univ": ex})
                conf = c.get("confidence") or 0.0
                for key, lo, hi in BUCKETS:
                    if lo <= conf < hi:
                        cal[key].append(d * ex)
            if c["agent"] == "morning_check" and d == -1:
                cancels.append(ex)
            if d:
                agents.setdefault(c["agent"], []).append((d, ex))
    from bot.memory.events_score import scored_event_calls

    for agent, d, ex in scored_event_calls(journal, since, horizon):   # Phase 3 intraday decisions
        agents.setdefault(agent, []).append((d, ex))
    return {
        "n_views": n_views,
        "agents": {a: {"n": len(v), "hit_rate": sum(1 for d, ex in v if (d > 0) == (ex > 0) and ex != 0) / len(v),
                       "mean_signed_excess": _mean([d * ex for d, ex in v])} for a, v in agents.items()},
        "calibration": {k: {"n": len(v), "mean_signed_excess": _mean(v)} for k, v in cal.items()},
        "morning_check": {"cancels": len(cancels), "mean_excess_of_cancelled": _mean(cancels)},
        "symbols": {s: h[-3:] for s, h in symbols.items()},
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def render_scorecards(cards: dict, profile: str) -> str:
    L = [f"# Scorecard ({profile})", "", f"*Regenerated {dt.date.today()} by the bot. Edits here are overwritten.*", ""]
    if not cards["n_views"] and not cards["agents"]:
        return "\n".join(L + [NO_RECORD, ""])
    L += [f"Scored views: {cards['n_views']} (10 trading days, excess vs the universe average)", "",
          "| agent | calls | hit rate | mean signed excess |", "|---|---|---|---|"]
    for a, v in sorted(cards["agents"].items()):
        L.append(f"| {a} | {v['n']} | {v['hit_rate']:.0%} | {_pct(v['mean_signed_excess'])} |")
    L += ["", "**PM calibration** (does higher confidence mean better results?)", "",
          "| confidence | views | mean signed excess |", "|---|---|---|"]
    for k, v in cards["calibration"].items():
        L.append(f"| {k} | {v['n']} | {_pct(v['mean_signed_excess'])} |")
    mc = cards["morning_check"]
    L += ["", f"**Morning Check cancels:** {mc['cancels']}; average excess of the cancelled buys "
              f"{_pct(mc['mean_excess_of_cancelled'])} (below 0 means the cancels helped)", ""]
    return "\n".join(L)
