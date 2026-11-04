"""SQLite journal: every input, agent output, view, target, order and fill is recorded."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    phase TEXT NOT NULL,            -- analyze | execute
    mode TEXT NOT NULL,             -- dry-run | paper | live
    status TEXT NOT NULL,           -- running | ok | blocked | error
    notes TEXT,
    prompt_version TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    est_cost_usd REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_reports (
    run_id INTEGER, agent TEXT, content TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS views (
    run_id INTEGER, symbol TEXT, expected_return REAL, confidence REAL, rationale TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    run_id INTEGER, symbol TEXT, source TEXT, value REAL, extra TEXT
);
CREATE TABLE IF NOT EXISTS target_weights (
    run_id INTEGER, symbol TEXT, weight REAL
);
CREATE TABLE IF NOT EXISTS orders (
    run_id INTEGER, symbol TEXT, side TEXT, notional REAL, current_weight REAL, target_weight REAL,
    broker_order_id TEXT, status TEXT, error TEXT, filled_qty REAL, filled_avg_price REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    date TEXT PRIMARY KEY, equity REAL, cash REAL, benchmark_close REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, ts TEXT, symbol TEXT, kind TEXT, detail TEXT,
    analyst TEXT, reviewer TEXT, action TEXT, price REAL, run_id INTEGER, mode TEXT,
    pipeline TEXT DEFAULT 'review', filing_url TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Journal:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _now(self) -> str:
        return dt.datetime.now().isoformat(timespec="seconds")

    def start_run(self, phase: str, mode: str, prompt_version: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at, phase, mode, status, prompt_version) VALUES (?,?,?,?,?)",
            (self._now(), phase, mode, "running", prompt_version),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, notes: str = "", input_tokens: int = 0,
                   output_tokens: int = 0, est_cost_usd: float = 0.0) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, notes=?, input_tokens=?, output_tokens=?, est_cost_usd=? WHERE id=?",
            (status, notes, input_tokens, output_tokens, est_cost_usd, run_id),
        )
        self.conn.commit()

    def agent_report(self, run_id: int, agent: str, content) -> None:
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        self.conn.execute("INSERT INTO agent_reports VALUES (?,?,?,?)", (run_id, agent, text, self._now()))
        self.conn.commit()

    def views(self, run_id: int, views) -> None:
        self.conn.executemany("INSERT INTO views VALUES (?,?,?,?,?)",
                              [(run_id, v.symbol, v.expected_return, v.confidence, v.rationale) for v in views])
        self.conn.commit()

    def signals(self, run_id: int, source: str, rows: dict[str, tuple[float, dict]]) -> None:
        self.conn.executemany("INSERT INTO signals VALUES (?,?,?,?,?)",
                              [(run_id, s, source, val, json.dumps(extra)) for s, (val, extra) in rows.items()])
        self.conn.commit()

    def target_weights(self, run_id: int, weights: pd.Series) -> None:
        self.conn.executemany("INSERT INTO target_weights VALUES (?,?,?)",
                              [(run_id, s, float(w)) for s, w in weights.items()])
        self.conn.commit()

    def order(self, run_id: int, plan, submitted=None) -> None:
        self.conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, plan.symbol, plan.side, plan.notional, plan.current_weight, plan.target_weight,
             getattr(submitted, "broker_order_id", None), getattr(submitted, "status", "planned"),
             getattr(submitted, "error", None), None, None, self._now()),
        )
        self.conn.commit()

    def update_fill(self, broker_order_id: str, status: str, filled_qty: float, filled_avg_price: float | None):
        self.conn.execute("UPDATE orders SET status=?, filled_qty=?, filled_avg_price=? WHERE broker_order_id=?",
                          (status, filled_qty, filled_avg_price, broker_order_id))
        self.conn.commit()

    def record_equity(self, date: dt.date, equity: float, cash: float, benchmark_close: float | None) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", (str(date), equity, cash, benchmark_close))
        self.conn.commit()

    def peak_equity(self) -> float | None:
        row = self.conn.execute("SELECT MAX(equity) FROM equity").fetchone()
        return row[0] if row and row[0] else None

    def latest_targets(self, mode: str, max_age_days: int = 3) -> tuple[int, pd.Series] | None:
        """Most recent successful analyze run's targets for this mode, if recent enough.
        Mode-scoped so a dry-run never changes what paper/live trades."""
        row = self.conn.execute(
            "SELECT id, started_at FROM runs WHERE phase='analyze' AND status='ok' AND mode=? "
            "ORDER BY id DESC LIMIT 1", (mode,)
        ).fetchone()
        if not row:
            return None
        run_id, started = row
        if (dt.datetime.now() - dt.datetime.fromisoformat(started)).days > max_age_days:
            return None
        rows = self.conn.execute("SELECT symbol, weight FROM target_weights WHERE run_id=?", (run_id,)).fetchall()
        return run_id, pd.Series({s: w for s, w in rows}, dtype=float)

    def open_order_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT broker_order_id FROM orders WHERE broker_order_id IS NOT NULL "
            "AND status NOT IN ('filled','canceled','cancelled','rejected','expired')"
        ).fetchall()
        return [r[0] for r in rows]

    def open_sell_symbols(self, day: dt.date) -> set[str]:
        """Symbols with a sell SUBMITTED on `day` (not a gate-blocked 'planned' row) still in a non-terminal state (filled,
        canceled, rejected, expired). The intraday watch uses this to never sell a position twice."""
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM orders WHERE side='sell' AND created_at >= ? "
            "AND COALESCE(status,'') <> 'planned' "   # a gate-blocked (never submitted) row is not an open sell
            "AND COALESCE(status,'') NOT IN ('filled','canceled','cancelled','rejected','expired')",
            (str(day),)).fetchall()
        return {r[0] for r in rows}

    # ---------- Phase 3: intraday events (Event Reviews, intraday buy pipelines) and small key/value state ----------
    def record_event(self, day, symbol: str, kind: str, detail: str, analyst, reviewer, action: str,
                     price: float | None, run_id: int, mode: str, pipeline: str = "review",
                     filing_url: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(day, ts, symbol, kind, detail, analyst, reviewer, action, price, run_id, mode, "
            "pipeline, filing_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(day), self._now(), symbol, kind, detail,
             None if analyst is None else json.dumps(analyst, default=str),
             None if reviewer is None else json.dumps(reviewer, default=str),
             action, price, run_id, mode, pipeline, filing_url))
        self.conn.commit()
        return int(cur.lastrowid)

    def events_today(self, day, mode: str) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM events WHERE day=? AND mode=? ORDER BY id", (str(day), mode))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def open_sell_orders(self, day: dt.date) -> list[tuple[int, str, str | None]]:
        """(run_id, symbol, broker_order_id) of sells submitted on `day` still non-terminal (as open_sell_symbols)."""
        return [(int(r[0]), r[1], r[2]) for r in self.conn.execute(
            "SELECT run_id, symbol, broker_order_id FROM orders WHERE side='sell' AND created_at >= ? "
            "AND COALESCE(status,'') <> 'planned' "
            "AND COALESCE(status,'') NOT IN ('filled','canceled','cancelled','rejected','expired')", (str(day),))]

    def event_sell_keys(self, day) -> set[tuple[int, str]]:
        """(run_id, symbol) of Event Review trims/exits submitted on `day`."""
        return {(int(r[0]), r[1]) for r in self.conn.execute(
            "SELECT run_id, symbol FROM events WHERE day=? AND action IN ('trim','exit')", (str(day),))}

    def mark_sell_canceled(self, run_id: int, symbol: str) -> None:
        self.conn.execute("UPDATE orders SET status='canceled' WHERE run_id=? AND symbol=? AND side='sell'",
                          (run_id, symbol))
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
        self.conn.commit()

    INTRADAY_SCHEMA = "CREATE TABLE IF NOT EXISTS intraday_equity (ts TEXT, equity REAL, cash REAL)"

    def record_intraday_equity(self, ts: str, equity: float, cash: float) -> None:
        self.conn.execute(self.INTRADAY_SCHEMA)
        self.conn.execute("INSERT INTO intraday_equity VALUES (?,?,?)", (ts, equity, cash))
        self.conn.commit()

    def query(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    # ---------- per-position stops (profile B) ----------
    STOPS_SCHEMA = """CREATE TABLE IF NOT EXISTS stops (
        symbol TEXT, triggered_at TEXT, entry REAL, price REAL, pnl_pct REAL, cooldown_until TEXT,
        run_id INTEGER, mode TEXT)"""

    def record_stop(self, symbol: str, triggered_at: dt.date, entry: float, price: float, pnl_pct: float,
                    cooldown_until: dt.date, run_id: int | None = None, mode: str = "") -> None:
        self.conn.execute(self.STOPS_SCHEMA)
        self.conn.execute("INSERT INTO stops VALUES (?,?,?,?,?,?,?,?)",
                          (symbol, str(triggered_at), entry, price, pnl_pct, str(cooldown_until), run_id, mode))
        self.conn.commit()

    def active_cooldowns(self, as_of: dt.date, mode: str | None = None) -> dict[str, str]:
        """{symbol: cooldown_until} for stops whose cooldown has not ended (re-entry allowed on that date).
        Mode-scoped like latest_targets, so a dry-run never blocks paper/live re-entry."""
        self.conn.execute(self.STOPS_SCHEMA)
        sql = "SELECT symbol, MAX(cooldown_until) FROM stops WHERE cooldown_until > ?"
        params: tuple = (str(as_of),)
        if mode is not None:
            sql += " AND mode = ?"
            params += (mode,)
        rows = self.conn.execute(sql + " GROUP BY symbol", params).fetchall()
        return {s: u for s, u in rows}

    # ---------- weekly screener candidates (profile B) ----------
    SCREEN_SCHEMA = """CREATE TABLE IF NOT EXISTS screen_candidates (
        screen_date TEXT NOT NULL, run_ts TEXT NOT NULL, symbol TEXT NOT NULL, passed INTEGER NOT NULL,
        reason TEXT, rank INTEGER, score REAL, features TEXT);
    CREATE INDEX IF NOT EXISTS ix_screen_candidates_date ON screen_candidates(screen_date, run_ts);"""

    def log_screen(self, screen_date: dt.date, run_ts: str, rows: list[dict]) -> int:
        """Every screened candidate (passed or failed with reason). rows: symbol, passed, reason, rank, score, features."""
        self.conn.executescript(self.SCREEN_SCHEMA)
        self.conn.executemany(
            "INSERT INTO screen_candidates VALUES (?,?,?,?,?,?,?,?)",
            [(str(screen_date), run_ts, r["symbol"], int(bool(r.get("passed"))), r.get("reason"), r.get("rank"),
              r.get("score"), json.dumps(r.get("features") or {}, default=str)) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    def screen_candidates(self, screen_date: dt.date | str | None = None, passed_only: bool = False) -> pd.DataFrame:
        """Candidates of the latest run per screen_date (or of one screen_date), features as JSON text."""
        self.conn.executescript(self.SCREEN_SCHEMA)
        sql = ("SELECT c.* FROM screen_candidates c JOIN (SELECT screen_date, MAX(run_ts) AS run_ts "
               "FROM screen_candidates GROUP BY screen_date) l "
               "ON c.screen_date = l.screen_date AND c.run_ts = l.run_ts")
        where, params = [], []
        if screen_date is not None:
            where.append("c.screen_date = ?")
            params.append(str(screen_date))
        if passed_only:
            where.append("c.passed = 1")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self.query(sql + " ORDER BY c.screen_date, c.rank IS NULL, c.rank, c.symbol", tuple(params))

    # ---------- honest fills (profile B, stage 2) ----------
    # Created lazily (only when execution.record_quotes is on), so bot A's journal never gets these tables.
    QUOTES_SCHEMA = """CREATE TABLE IF NOT EXISTS order_quotes (
        run_id INTEGER, src_run_id INTEGER, symbol TEXT, side TEXT, notional REAL, qty REAL, limit_price REAL,
        action TEXT, broker_order_id TEXT, bid REAL, ask REAL, mid REAL, spread_pct REAL, quote_ts TEXT,
        feed TEXT, adv_shares REAL, sigma_daily REAL, recorded_at TEXT);
    CREATE INDEX IF NOT EXISTS ix_order_quotes_oid ON order_quotes(broker_order_id);
    CREATE TABLE IF NOT EXISTS fill_costs (
        broker_order_id TEXT PRIMARY KEY, run_id INTEGER, symbol TEXT, side TEXT, fill_date TEXT,
        filled_qty REAL, fill_price REAL, notional REAL, mid REAL, spread_pct REAL, spread_estimated INTEGER,
        sigma_daily REAL, adv_shares REAL, impact_pct REAL, modeled_cost_pct REAL, modeled_cost_usd REAL,
        slippage_vs_mid REAL, slippage_usd REAL, flags TEXT, computed_at TEXT);"""

    def _honest_schema(self) -> None:
        self.conn.executescript(self.QUOTES_SCHEMA)

    def record_order_quote(self, run_id: int, src_run_id: int | None, plan, action: str,
                           broker_order_id: str | None, quote: dict | None, feed: str | None,
                           adv_shares: float | None, sigma_daily: float | None) -> None:
        """One row per planned order (submitted, skipped or blocked): quote at execute + ADV/sigma refs."""
        self._honest_schema()
        q = quote or {}
        self.conn.execute(
            "INSERT INTO order_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, src_run_id, plan.symbol, plan.side, plan.notional, plan.qty, plan.limit_price, action,
             broker_order_id, q.get("bid"), q.get("ask"), q.get("mid"), q.get("spread_pct"),
             None if q.get("ts") is None else str(q.get("ts")), feed, adv_shares, sigma_daily, self._now()))
        self.conn.commit()

    def _latest_refs(self, symbol: str, before_run_id: int) -> tuple[float | None, float | None]:
        row = self.conn.execute("SELECT extra FROM signals WHERE symbol=? AND source='adv' AND run_id<=? "
                                "ORDER BY run_id DESC LIMIT 1", (symbol, before_run_id)).fetchone()
        e = json.loads(row[0] or "{}") if row else {}
        return e.get("adv_shares20"), e.get("sigma_daily")

    def update_fill_costs(self, default_spread_pct: float) -> int:
        """(Re)compute modeled cost and slippage for every order with a fill. Idempotent (keyed by
        broker_order_id). Fills without a stored quote use default_spread_pct and are flagged."""
        from bot.portfolio.costs import fill_cost

        self._honest_schema()
        rows = self.conn.execute(
            "SELECT o.broker_order_id, o.run_id, o.symbol, o.side, o.filled_qty, o.filled_avg_price, o.created_at, "
            "q.mid, q.spread_pct, q.adv_shares, q.sigma_daily, q.rowid FROM orders o "
            "LEFT JOIN order_quotes q ON q.broker_order_id = o.broker_order_id "
            "WHERE o.broker_order_id IS NOT NULL AND o.filled_qty > 0 AND o.filled_avg_price > 0").fetchall()
        n = 0
        for oid, run_id, sym, side, qty, px, created, mid, spread, adv_sh, sigma, qrow in rows:
            if adv_sh is None or sigma is None:
                ra, rs = self._latest_refs(sym, run_id)
                adv_sh, sigma = adv_sh or ra, sigma or rs
            quote = None if qrow is None or spread is None else {"mid": mid, "spread_pct": spread}
            c = fill_cost(side, float(qty), float(px), quote, sigma, adv_sh, default_spread_pct)
            self.conn.execute(
                "INSERT OR REPLACE INTO fill_costs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (oid, run_id, sym, side, str(created)[:10], float(qty), float(px), c["notional"], c["mid"],
                 c["spread_pct"], c["spread_estimated"], c["sigma_daily"], c["adv_shares"], c["impact_pct"],
                 c["modeled_cost_pct"], c["modeled_cost_usd"], c["slippage_vs_mid"], c["slippage_usd"],
                 c["flags"], self._now()))
            n += 1
        self.conn.commit()
        return n

    def pnl_table(self) -> pd.DataFrame:
        """Per equity date: equity, paper P&L vs the first equity row, cumulative modeled cost of fills
        up to that date, realistic P&L = paper - cost. Empty when there is no equity history."""
        self._honest_schema()
        cols = ["date", "equity", "paper_pnl", "paper_pnl_pct", "cum_cost", "realistic_pnl", "realistic_pnl_pct"]
        eq = self.query("SELECT date, equity FROM equity WHERE equity IS NOT NULL ORDER BY date")
        if eq.empty:
            return pd.DataFrame(columns=cols)
        costs = self.query("SELECT fill_date, SUM(modeled_cost_usd) AS c FROM fill_costs GROUP BY fill_date")
        start = float(eq["equity"].iloc[0])
        eq["paper_pnl"] = eq["equity"] - start
        eq["paper_pnl_pct"] = eq["paper_pnl"] / start if start else 0.0
        eq["cum_cost"] = [float(costs.loc[costs["fill_date"] <= d, "c"].sum()) for d in eq["date"]]
        eq["realistic_pnl"] = eq["paper_pnl"] - eq["cum_cost"]
        eq["realistic_pnl_pct"] = eq["realistic_pnl"] / start if start else 0.0
        return eq[cols]

    def pnl_summary(self) -> dict | None:
        t = self.pnl_table()
        if t.empty:
            return None
        last = t.iloc[-1]
        n = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(spread_estimated), 0) FROM fill_costs").fetchone()
        return {"start_date": str(t["date"].iloc[0]), "start_equity": float(t["equity"].iloc[0]),
                "date": str(last["date"]), "equity": float(last["equity"]), "paper_pnl": float(last["paper_pnl"]),
                "paper_pnl_pct": float(last["paper_pnl_pct"]), "cum_cost": float(last["cum_cost"]),
                "realistic_pnl": float(last["realistic_pnl"]),
                "realistic_pnl_pct": float(last["realistic_pnl_pct"]),
                "n_fills": int(n[0]), "n_estimated": int(n[1])}


def format_pnl(p: dict) -> list[str]:
    """Text lines for the report/email: paper vs realistic P&L side by side."""
    est = f", {p['n_estimated']} with estimated spread" if p["n_estimated"] else ""
    return [f"Paper P&L since {p['start_date']}: ${p['paper_pnl']:+,.2f} ({p['paper_pnl_pct']:+.2%})  |  "
            f"Realistic P&L: ${p['realistic_pnl']:+,.2f} ({p['realistic_pnl_pct']:+.2%})",
            f"Modeled trading cost (half spread + sqrt impact): ${p['cum_cost']:,.2f} "
            f"over {p['n_fills']} fills{est}"]
