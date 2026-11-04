"""Score every analyze run's universe against realized returns, and extract every agent's calls.

No LLM anywhere: entry is the close on the run's as_of date, exit the close `h` trading days later
(benchmark index). excess_univ = symbol return - equal-weight universe mean (same window, symbols
with both prices); excess_bench = symbol return - benchmark return. A symbol without prices stays
unscored and is retried next time (never scored as 0).
"""
from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd

from bot.journal.store import Journal

SCHEMA = """CREATE TABLE IF NOT EXISTS outcomes_v2 (
    run_id INTEGER, symbol TEXT, as_of TEXT, horizon INTEGER, ret REAL, excess_univ REAL,
    excess_bench REAL, PRIMARY KEY (run_id, symbol, horizon))"""
LOOKBACK_DAYS = 120    # runs older than this are no longer retried
STANCE = {"bullish": 1, "neutral": 0, "bearish": -1}
MORNING = {"cancel": -1.0, "reduce": -0.5, "keep": 0.0}


def _ensure(j: Journal) -> None:
    j.conn.execute(SCHEMA)


def _reports(j: Journal, run_id: int) -> dict[str, dict]:
    out = {}
    for agent, content in j.conn.execute("SELECT agent, content FROM agent_reports WHERE run_id=?", (run_id,)):
        try:
            out[agent] = json.loads(content)
        except (TypeError, ValueError):
            continue
    return out


def _runs(j: Journal) -> list[tuple[int, str, list[str]]]:
    """(run_id, as_of, universe) for successful analyze runs that had agents (an inputs report)."""
    since = str(dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS))
    rows = j.conn.execute(
        "SELECT r.id, r.mode, a.content FROM runs r JOIN agent_reports a ON a.run_id = r.id AND a.agent = 'inputs' "
        "WHERE r.phase = 'analyze' AND r.status = 'ok' AND r.started_at >= ? AND r.mode IN ('paper', 'live', 'control') "
        "ORDER BY r.id", (since,)).fetchall()
    latest: dict[tuple[str, str], tuple[int, str, list[str]]] = {}
    for rid, mode, content in rows:      # dry-runs never count; a same-day re-run replaces the earlier one
        try:
            d = json.loads(content)
            latest[(mode, str(d["as_of"])[:10])] = (int(rid), str(d["as_of"])[:10], [str(x) for x in d["universe"]])
        except (TypeError, ValueError, KeyError):
            continue
    return sorted(latest.values())


def _scored(j: Journal, horizon: int) -> dict[int, set[str]]:
    _ensure(j)
    out: dict[int, set[str]] = {}
    for rid, sym in j.conn.execute("SELECT run_id, symbol FROM outcomes_v2 WHERE horizon=?", (horizon,)):
        out.setdefault(int(rid), set()).add(sym)
    return out


def pending_symbols(journal: Journal, horizon: int = 10) -> tuple[set[str], dt.date | None]:
    done = _scored(journal, horizon)
    syms: set[str] = set()
    start = None
    for rid, as_of, univ in _runs(journal):
        missing = set(univ) - done.get(rid, set())
        if missing:
            syms |= missing
            d = dt.date.fromisoformat(as_of)
            start = d if start is None or d < start else start
    return syms, start


def _close(df: pd.DataFrame | None, day: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None
    idx = pd.DatetimeIndex(df.index).normalize()
    hit = df["close"].to_numpy()[idx == day]
    if not len(hit):
        return None
    v = float(hit[-1])
    return v if math.isfinite(v) and v > 0 else None


def score_pending(journal: Journal, bars: dict[str, pd.DataFrame], benchmark: str,
                  horizons: tuple[int, ...] = (10,)) -> int:
    bench = bars.get(benchmark)
    if bench is None or bench.empty:
        return 0
    days = pd.DatetimeIndex(bench.index).normalize()
    n = 0
    for h in horizons:
        done = _scored(journal, h)
        for rid, as_of, univ in _runs(journal):
            todo = [s for s in univ if s not in done.get(rid, set())]
            if not todo:
                continue
            d0 = pd.Timestamp(as_of)
            pos = days.get_indexer([d0])[0]
            if pos < 0 or pos + h >= len(days):
                continue
            d1 = days[pos + h]
            b0, b1 = _close(bench, d0), _close(bench, d1)
            if b0 is None or b1 is None:
                continue
            rets = {}
            for s in univ:
                c0, c1 = _close(bars.get(s), d0), _close(bars.get(s), d1)
                if c0 is not None and c1 is not None:
                    rets[s] = c1 / c0 - 1
            if not rets:
                continue
            univ_mean = sum(rets.values()) / len(rets)
            bench_ret = b1 / b0 - 1
            for s in todo:
                if s in rets:
                    journal.conn.execute("INSERT OR IGNORE INTO outcomes_v2 VALUES (?,?,?,?,?,?,?)",
                                         (rid, s, as_of, h, rets[s], rets[s] - univ_mean, rets[s] - bench_ret))
                    n += 1
    journal.conn.commit()
    return n


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def agent_calls(journal: Journal, run_id: int) -> list[dict]:
    """Every agent's directional call on each symbol for one analyze run (morning checks that
    reviewed this run's targets included). direction: +1 bullish/long, -1 bearish, 0 neutral."""
    calls: list[dict] = []
    reps = _reports(journal, run_id)
    # The PM's own views: before the Phase 3 reviewers changed them, when they ran; else the final views.
    before = reps.get("views_before_review")
    pm_views = ([(v.get("symbol"), v.get("expected_return"), v.get("confidence")) for v in before]
                if isinstance(before, list) else journal.conn.execute(
                    "SELECT symbol, expected_return, confidence FROM views WHERE run_id=?", (run_id,)).fetchall())
    sign_of = {}
    for sym, er, conf in pm_views:
        if not sym or not isinstance(er, (int, float)):
            continue
        sign_of[sym] = _sign(er)
        calls.append({"agent": "pm", "symbol": sym, "direction": _sign(er), "strength": abs(er),
                      "confidence": conf, "expected_return": er})
    for v in (reps.get("second_pm") or {}).get("views", []) or []:
        er = v.get("expected_return")
        if v.get("symbol") and isinstance(er, (int, float)):
            calls.append({"agent": "second_pm", "symbol": str(v["symbol"]).upper(), "direction": _sign(er),
                          "strength": abs(er), "confidence": v.get("confidence"), "expected_return": er})
    for it in (reps.get("risk_review") or {}).get("items", []) or []:
        sym = str(it.get("symbol") or "").upper()
        if sign_of.get(sym):   # keep agrees with the view's direction; lower/drop bets against it
            d = sign_of[sym] if it.get("action") == "keep" else -sign_of[sym]
            calls.append({"agent": "risk_reviewer", "symbol": sym, "direction": d, "strength": None,
                          "confidence": None, "expected_return": None})
    for agent in ("technical", "fundamentals", "news"):
        for note in (reps.get(agent) or {}).get("notes", []) or []:
            d = STANCE.get(str(note.get("stance")))
            if d is not None and note.get("symbol"):
                calls.append({"agent": agent, "symbol": note["symbol"], "direction": d,
                              "strength": note.get("strength"), "confidence": None, "expected_return": None})
    for name, rep in reps.items():
        side = "bull" if name.startswith("bull_r") else "bear" if name.startswith("bear_r") else None
        if side:
            for arg in (rep or {}).get("arguments", []) or []:
                if arg.get("symbol"):
                    calls.append({"agent": side, "symbol": arg["symbol"], "direction": 1 if side == "bull" else -1,
                                  "strength": None, "confidence": None, "expected_return": None})
    for ex_run, content in journal.conn.execute(
            "SELECT run_id, content FROM agent_reports WHERE agent='morning_check_input'").fetchall():
        try:
            inp = json.loads(content)
            if int(inp.get("src_run", -1)) != run_id:
                continue
        except (TypeError, ValueError, AttributeError):
            continue
        gaps = {b.get("symbol"): b.get("gap_pct") for b in inp.get("buys", []) or []}
        rep = _reports(journal, ex_run).get("morning_check") or {}
        for it in rep.get("items", []) or []:
            d = MORNING.get(str(it.get("action")))
            gap = gaps.get(it.get("symbol"))
            # the check decided after the overnight gap: without that price it cannot be scored fairly
            if d is not None and it.get("symbol") and isinstance(gap, (int, float)) and math.isfinite(gap):
                calls.append({"agent": "morning_check", "symbol": it["symbol"], "direction": d,
                              "strength": it.get("fraction"), "confidence": None, "expected_return": None,
                              "gap_pct": float(gap)})
    return calls
