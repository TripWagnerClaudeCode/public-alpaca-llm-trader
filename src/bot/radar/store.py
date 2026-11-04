"""SQLite storage for the radar: posts (one row per post x matched symbol), runs, features, meta."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    source TEXT NOT NULL,          -- collector: hn, gdelt, google_news, substack, bluesky, youtube
    origin TEXT NOT NULL,          -- independent voice: publisher domain, feed host, or source:author
    symbol TEXT NOT NULL,
    title TEXT,
    text TEXT,                     -- sanitized, <= 1000 chars
    ts TEXT NOT NULL,              -- post time, UTC ISO 'YYYY-MM-DDTHH:MM:SSZ'
    fetched_at TEXT NOT NULL,      -- when the collector stored it (point-in-time guard)
    text_hash TEXT NOT NULL,
    is_mainstream INTEGER NOT NULL DEFAULT 0,
    UNIQUE (url, symbol),
    UNIQUE (text_hash, symbol)
);
CREATE INDEX IF NOT EXISTS posts_symbol_ts ON posts (symbol, ts);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    stats TEXT
);
CREATE TABLE IF NOT EXISTS features (
    as_of TEXT NOT NULL,
    freeze_ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    mention_count INTEGER NOT NULL,
    mention_z REAL,
    n_sources INTEGER NOT NULL,
    fringe_only INTEGER NOT NULL,
    baseline_mean REAL,
    baseline_std REAL,
    baseline_days INTEGER,
    n_mainstream INTEGER,
    summary TEXT,                  -- always NULL for now: no LLM (see features.summary_hook)
    computed_at TEXT NOT NULL,
    PRIMARY KEY (as_of, symbol)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def iso(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


@dataclass
class Post:
    url: str
    source: str
    origin: str
    title: str
    text: str
    ts: dt.datetime
    is_mainstream: bool = False
    symbols: set[str] = field(default_factory=set)


class RadarStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---------- posts ----------
    def insert_posts(self, posts: list[Post], fetched_at: dt.datetime, hasher) -> tuple[int, int]:
        """Insert one row per (post, symbol); duplicates by URL or text hash are ignored. Returns (new, dupes)."""
        new = dupes = 0
        f_at = iso(fetched_at)
        with self.connect() as c:
            for p in posts:
                ts = min(p.ts, fetched_at)  # no future-dated posts (clock skew / bad feeds)
                h = hasher(p.title, p.text)
                for sym in sorted(p.symbols):
                    cur = c.execute(
                        "INSERT OR IGNORE INTO posts (url, source, origin, symbol, title, text, ts, fetched_at,"
                        " text_hash, is_mainstream) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (p.url, p.source, p.origin, sym, p.title, p.text, iso(ts), f_at, h, int(p.is_mainstream)),
                    )
                    if cur.rowcount:
                        new += 1
                    else:
                        dupes += 1
        return new, dupes

    def posts_since(self, symbols: list[str], since: dt.datetime, limit_per_symbol: int) -> dict[str, list[dict]]:
        """Newest posts per symbol with post time after `since` (title, url, source, ts)."""
        syms = sorted(set(symbols))
        if not syms:
            return {}
        q = ("SELECT symbol, title, url, source, ts FROM posts WHERE ts > ? AND symbol IN (%s) "
             "ORDER BY ts DESC" % ",".join("?" * len(syms)))
        out: dict[str, list[dict]] = {}
        with self.connect() as c:
            for r in c.execute(q, (iso(since), *syms)):
                lst = out.setdefault(r["symbol"], [])
                if len(lst) < limit_per_symbol:
                    lst.append({"title": r["title"], "url": r["url"], "source": r["source"], "ts": r["ts"]})
        return out

    # ---------- runs ----------
    def start_run(self, started_at: dt.datetime) -> int:
        with self.connect() as c:
            return int(c.execute("INSERT INTO runs (started_at) VALUES (?)", (iso(started_at),)).lastrowid)

    def finish_run(self, run_id: int, finished_at: dt.datetime, stats: dict) -> None:
        with self.connect() as c:
            c.execute("UPDATE runs SET finished_at=?, stats=? WHERE id=?",
                      (iso(finished_at), json.dumps(stats, default=str), run_id))

    # ---------- meta (round-robin cursors, last-run times) ----------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    # ---------- features ----------
    def write_features(self, as_of: dt.date, freeze_ts: dt.datetime, feats: dict[str, dict]) -> None:
        now = iso(dt.datetime.now(dt.timezone.utc))
        with self.connect() as c:
            for sym, f in feats.items():
                c.execute(
                    "INSERT OR REPLACE INTO features (as_of, freeze_ts, symbol, mention_count, mention_z, n_sources,"
                    " fringe_only, baseline_mean, baseline_std, baseline_days, n_mainstream, summary, computed_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(as_of), iso(freeze_ts), sym, f["mention_count"], f["mention_z"], f["n_sources"],
                     int(f["fringe_only"]), f["baseline_mean"], f["baseline_std"], f["baseline_days"],
                     f["n_mainstream"], f.get("summary"), now),
                )
