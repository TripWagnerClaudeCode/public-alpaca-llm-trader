"""Daily per-symbol radar features, frozen at analyze time (no lookahead).

Rules (all times UTC; W = window_hours, default 24h):
- Only posts with ts <= freeze_ts are used. With point_in_time=True (default) a post must ALSO have been
  fetched by the collector at or before freeze_ts, so re-computing an old date later gives exactly what
  the bot could have known then (late-arriving / backfilled posts don't leak in).
- mention_count = rows for the symbol with ts in (freeze - W, freeze].
- Baseline = the `baseline_days` W-wide buckets immediately before the window:
  bucket i covers (freeze - (i+1)W, freeze - iW], i = 1..baseline_days.
  With require_live_coverage=True (default) a bucket only counts if a collector run started inside it,
  because a bucket the collector never saw has an unknown count, not a zero count (the PC was asleep,
  or the radar didn't exist yet). Backfilled history from a first run therefore doesn't form a baseline.
- baseline_mean / baseline_std = mean and sample std (ddof=1) of the counted buckets.
- mention_z = (mention_count - mean) / max(std, sqrt(mean), std_floor)
  The sqrt(mean) term is the Poisson noise floor, std_floor (1.0) stops a quiet name with a flat
  baseline of 0 from exploding to z = inf on its first mention.
- mention_z is None when fewer than `min_baseline_days` buckets count, or (with live coverage required)
  when no collector run started inside the window itself.
- n_sources = distinct `origin` values in the window (publisher domain, feed host, or source:author).
- fringe_only = (>= fringe_min_sources distinct non-mainstream origins) AND (no mainstream row in the window).
- summary = None. No LLM is called (user decision: numeric features only). See summary_hook.
"""
from __future__ import annotations

import datetime as dt
import math
import sqlite3
from pathlib import Path

from bot.radar.store import iso, parse_iso


def summary_hook(symbol: str, posts: list[dict]) -> str | None:
    """HOOK for an optional <= 200-char summary. Intentionally returns None: no LLM is called.

    If this is ever enabled, it must only see sanitized, length-capped text (posts[*]['text'] is already
    stripped of HTML and capped at 1000 chars), output <= 200 chars, and stay out of the agents' input until
    the radar passes evaluation (spec: shadow mode).
    """
    return None


def _empty(baseline_days: int = 0) -> dict:
    return {"mention_count": 0, "mention_z": None, "n_sources": 0, "fringe_only": False, "n_mainstream": 0,
            "baseline_mean": None, "baseline_std": None, "baseline_days": baseline_days, "summary": None}


def compute_features(
    db_path: Path,
    symbols: list[str],
    freeze_ts: dt.datetime,
    *,
    window_hours: int = 24,
    baseline_days: int = 30,
    min_baseline_days: int = 14,
    std_floor: float = 1.0,
    require_live_coverage: bool = True,
    fringe_min_sources: int = 3,
    point_in_time: bool = True,
) -> dict[str, dict]:
    """Pure read-only function: {symbol: {mention_count, mention_z, n_sources, fringe_only, n_mainstream,
    baseline_mean, baseline_std, baseline_days, summary}} for every requested symbol."""
    if freeze_ts.tzinfo is None:
        freeze_ts = freeze_ts.replace(tzinfo=dt.timezone.utc)
    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    db_path = Path(db_path)
    if not db_path.exists() or not symbols:
        return {s: _empty() for s in symbols}

    W = dt.timedelta(hours=window_hours)
    start = freeze_ts - W * (baseline_days + 1)
    fz = iso(freeze_ts)
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        q = ("SELECT symbol, origin, ts, is_mainstream FROM posts WHERE ts > ? AND ts <= ? "
             f"AND symbol IN ({','.join('?' * len(symbols))})")
        params: list = [iso(start), fz, *symbols]
        if point_in_time:
            q += " AND fetched_at <= ?"
            params.append(fz)
        rows = con.execute(q, params).fetchall()
        runs = [parse_iso(r[0]) for r in con.execute(
            "SELECT started_at FROM runs WHERE started_at > ? AND started_at <= ?", (iso(start), fz))]
    finally:
        con.close()

    def bucket(t: dt.datetime) -> int:
        # 0 = the window (freeze - W, freeze]; i = baseline bucket i
        return max(0, (freeze_ts - t) // W)

    covered = set(range(baseline_days + 1))
    if require_live_coverage:
        covered = {bucket(t) for t in runs}

    counts = {s: [0] * (baseline_days + 1) for s in symbols}
    window_rows: dict[str, list] = {s: [] for s in symbols}
    for sym, origin, ts, mainstream in rows:
        b = bucket(parse_iso(ts))
        if b > baseline_days:
            continue
        counts[sym][b] += 1
        if b == 0:
            window_rows[sym].append((origin, bool(mainstream)))

    base_idx = [i for i in range(1, baseline_days + 1) if i in covered]
    out: dict[str, dict] = {}
    for s in symbols:
        n = counts[s][0]
        wr = window_rows[s]
        n_main = sum(1 for _, m in wr if m)
        fringe_origins = {o for o, m in wr if not m}
        f = _empty(len(base_idx))
        f.update(mention_count=n, n_sources=len({o for o, _ in wr}), n_mainstream=n_main,
                 fringe_only=(len(fringe_origins) >= fringe_min_sources and n_main == 0))
        if len(base_idx) >= max(2, min_baseline_days) and (0 in covered):
            base = [counts[s][i] for i in base_idx]
            mean = sum(base) / len(base)
            std = math.sqrt(sum((x - mean) ** 2 for x in base) / (len(base) - 1))
            f.update(baseline_mean=round(mean, 4), baseline_std=round(std, 4),
                     mention_z=round((n - mean) / max(std, math.sqrt(mean), std_floor), 4))
        f["summary"] = summary_hook(s, [])  # always None (no LLM)
        out[s] = f
    return out
