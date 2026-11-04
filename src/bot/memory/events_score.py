"""Phase 3 scoring of intraday decisions (no LLM): each Event Review and intraday buy pipeline row is
scored from its own decision price (the mid when it decided) to the close h trading days after its day,
against the equal-weight universe mean over the same days (close of the event day -> close h days on).
Only paper/live rows count; failed reviews (action "error") never do. Unpriced rows are retried.

Calls (direction: +1 = bet the stock does well, -1 = bet it does badly):
  event_review    an executed trim/exit -> -1 (holds are not executed calls)
  event_analyst   its own trim/exit proposal -> -1, whether or not the reviewer agreed
  event_reviewer  agreed trim/exit -> -1; a hold against the analyst's proposal -> +1
  intraday_buy    an executed intraday buy -> +1
"""
from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd

from bot.journal.store import Journal
from bot.memory.outcomes import LOOKBACK_DAYS, _close, _runs

SCHEMA = """CREATE TABLE IF NOT EXISTS event_outcomes (
    event_id INTEGER, horizon INTEGER, day TEXT, symbol TEXT, ret REAL, excess_univ REAL,
    PRIMARY KEY (event_id, horizon))"""


def _ensure(j: Journal) -> None:
    j.conn.execute(SCHEMA)


def _events(j: Journal) -> list[dict]:
    since = str(dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS))
    cur = j.conn.execute("SELECT * FROM events WHERE mode IN ('paper','live') AND action <> 'error' "
                         "AND price > 0 AND day >= ? ORDER BY id", (since,))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _universe_at(runs, day: str, sym: str) -> list[str]:
    uni = [u for _, as_of, u in runs if as_of <= day]
    return sorted(set(uni[-1]) | {sym}) if uni else [sym]


def _done(j: Journal, h: int) -> set[int]:
    _ensure(j)
    return {int(r[0]) for r in j.conn.execute("SELECT event_id FROM event_outcomes WHERE horizon=?", (h,))}


def pending_event_symbols(j: Journal, horizon: int = 10) -> tuple[set[str], dt.date | None]:
    done, runs = _done(j, horizon), _runs(j)
    syms: set[str] = set()
    start = None
    for e in _events(j):
        if e["id"] in done:
            continue
        syms |= set(_universe_at(runs, e["day"], e["symbol"]))
        d = dt.date.fromisoformat(e["day"])
        start = d if start is None or d < start else start
    return syms, start


def score_events(j: Journal, bars: dict[str, pd.DataFrame], benchmark: str, horizons=(10,)) -> int:
    bench = bars.get(benchmark)
    if bench is None or bench.empty:
        return 0
    days = pd.DatetimeIndex(bench.index).normalize()
    runs, n = _runs(j), 0
    for h in horizons:
        done = _done(j, h)
        for e in _events(j):
            if e["id"] in done:
                continue
            d0 = pd.Timestamp(e["day"])
            pos = days.get_indexer([d0])[0]
            if pos < 0 or pos + h >= len(days):
                continue
            d1 = days[pos + h]
            c1 = _close(bars.get(e["symbol"]), d1)
            if c1 is None:
                continue
            rets = []
            for u in _universe_at(runs, e["day"], e["symbol"]):
                a, b = _close(bars.get(u), d0), _close(bars.get(u), d1)
                if a is not None and b is not None:
                    rets.append(b / a - 1)
            if not rets:
                continue
            ret = c1 / float(e["price"]) - 1
            if not math.isfinite(ret):
                continue
            j.conn.execute("INSERT OR IGNORE INTO event_outcomes VALUES (?,?,?,?,?,?)",
                           (e["id"], h, e["day"], e["symbol"], ret, ret - sum(rets) / len(rets)))
            n += 1
    j.conn.commit()
    return n


def _call(blob) -> dict | None:
    try:
        d = json.loads(blob) if isinstance(blob, str) else blob
        c = (d or {}).get("call")
        return c if isinstance(c, dict) else None
    except (TypeError, ValueError, AttributeError):
        return None


def event_calls(e: dict) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    action = str(e.get("action") or "")
    if e.get("pipeline") == "buy":
        if action == "buy":
            out.append(("intraday_buy", 1))
        return out
    a, r = _call(e.get("analyst")), _call(e.get("reviewer"))
    if action in ("trim", "exit"):
        out.append(("event_review", -1))
    if a and a.get("action") in ("trim", "exit"):
        out.append(("event_analyst", -1))
        if r:
            out.append(("event_reviewer", -1 if r.get("action") in ("trim", "exit") else 1))
    return out


def scored_event_calls(j: Journal, since: dt.date | None = None, horizon: int = 10) -> list[tuple[str, int, float]]:
    """[(agent, direction, excess_univ)] for every scored intraday decision."""
    _ensure(j)
    sql = ("SELECT e.*, o.excess_univ AS _ex FROM event_outcomes o JOIN events e ON e.id = o.event_id "
           "WHERE o.horizon=?")
    params: tuple = (horizon,)
    if since is not None:
        sql += " AND o.day >= ?"
        params += (str(since),)
    cur = j.conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    out = []
    for row in cur.fetchall():
        e = dict(zip(cols, row))
        out += [(agent, d, float(e["_ex"])) for agent, d in event_calls(e)]
    return out
