"""Profile B stage 2: quote capture, spread guard, cost model, realistic P&L, screener features, radar shadow."""
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import pytest

from bot import run_daily
from bot.agents.team import TeamOutput
from bot.broker.sim import DryRunBroker
from bot.config import FeaturesCfg, load_settings
from bot.data.market import MarketData
from bot.journal.report import write_report
from bot.journal.store import Journal
from bot.models import OrderPlan, SubmittedOrder
from bot.notify import summarize_runs
from bot.portfolio.costs import fill_cost, modeled_cost_pct, normalize_quote, slippage_vs_mid, spread_guard

UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]


def _bars(i: int, n: int = 300, volume: float = 1e6) -> pd.DataFrame:
    rng = np.random.default_rng(i)
    idx = pd.bdate_range(end=pd.Timestamp(dt.date.today()), periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": np.full(n, volume)}, index=idx)


class FakeMD:
    def __init__(self, cache_dir):
        pass

    def daily_bars(self, symbols, days, end=None):
        return {s: _bars(i) for i, s in enumerate(sorted(symbols))}

    closes = staticmethod(MarketData.closes)

    def fundamentals(self, symbol):
        return {}

    def news(self, symbol, limit=5):
        return []


def _plan(sym, side, notional=1_000.0, **kw):
    return OrderPlan(symbol=sym, side=side, notional=notional, current_weight=0.0, target_weight=0.05, **kw)


def _settings(tmp_path, profile="smallcap", **risk):
    s = load_settings(profile=profile)
    upd = {
        "paths": s.paths.model_copy(update={"journal_db": str(tmp_path / "j.db"),
                                            "reports_dir": str(tmp_path / "reports"),
                                            "cache_dir": str(tmp_path / "cache")}),
        "risk": s.risk.model_copy(update={"kill_switch_file": str(tmp_path / "KS"), **risk}),
        "radar": s.radar.model_copy(update={"db_path": str(tmp_path / "radar.db")}),
    }
    if profile == "smallcap":
        upd["universe"] = s.universe.model_copy(update={"screener_dir": str(tmp_path / "universe")})
    else:
        upd["universe"] = s.universe.model_copy(update={"symbols": UNIVERSE})
    return s.model_copy(update=upd)


def _write_universe(tmp_path, symbols=UNIVERSE):
    d = tmp_path / "universe"
    d.mkdir(exist_ok=True)
    cands = [{"symbol": x, "rank": i + 1, "score": 2.0 - i * 0.1, "mom": 0.5, "mom_window": "12-1",
              "ret_3m": 0.2, "vol_surge": 0.3, "prox_high": 0.95, "adv20": 1e8, "sector": "Tech"}
             for i, x in enumerate(symbols)]
    (d / f"{dt.date.today()}.json").write_text(
        json.dumps({"as_of": str(dt.date.today()), "symbols": symbols, "candidates": cands}), encoding="utf-8")


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    monkeypatch.setattr(run_daily, "MarketData", FakeMD)
    box = {}
    monkeypatch.setattr(run_daily, "make_broker", lambda mode, s=None: box["broker"])
    return box


# ----------------------------------------------------------------------------- pure functions
def test_normalize_quote_rejects_bad_quotes():
    assert normalize_quote(None) is None
    assert normalize_quote({"bid": 0, "ask": 10}) is None          # one-sided
    assert normalize_quote({"bid": 10, "ask": 0}) is None
    assert normalize_quote({"bid": 10.2, "ask": 10.0}) is None     # crossed
    q = normalize_quote({"bid": 9.95, "ask": 10.05, "ts": "t"})
    assert q["mid"] == pytest.approx(10.0) and q["spread_pct"] == pytest.approx(0.01) and q["ts"] == "t"


def test_spread_guard_skips_wide_and_unquoted_buys_only():
    qs = {"WIDE": normalize_quote({"bid": 9.8, "ask": 10.2}),      # 4% spread
          "TIGHT": normalize_quote({"bid": 9.99, "ask": 10.01}),
          "SWIDE": normalize_quote({"bid": 9.0, "ask": 11.0})}
    orders = [_plan("WIDE", "buy"), _plan("TIGHT", "buy"), _plan("NOQ", "buy"),
              _plan("SWIDE", "sell"), _plan("SNOQ", "sell", close_position=True)]
    out, why, skipped = spread_guard(orders, qs, 0.015)
    assert [o.symbol for o in out] == ["TIGHT", "SWIDE", "SNOQ"]
    assert set(skipped) == {"WIDE", "NOQ"} and len(why) == 2
    assert "spread 4.00% > max 1.50%" in skipped["WIDE"] and "no usable quote" in skipped["NOQ"]
    # guard off: everything passes, even without quotes
    assert spread_guard(orders, {}, None) == (orders, [], {})


def test_cost_math():
    assert slippage_vs_mid("buy", 10.05, 10.0) == pytest.approx(0.005)
    assert slippage_vs_mid("sell", 9.95, 10.0) == pytest.approx(0.005)
    assert slippage_vs_mid("sell", 10.05, 10.0) == pytest.approx(-0.005)   # price improvement
    pct, impact = modeled_cost_pct(0.01, 0.03, 100, 10_000)
    assert impact == pytest.approx(0.03 * 0.1) and pct == pytest.approx(0.005 + 0.003)
    c = fill_cost("buy", 100, 10.05, {"mid": 10.0, "spread_pct": 0.01}, 0.03, 10_000, 0.003)
    assert c["modeled_cost_usd"] == pytest.approx(0.008 * 1005) and c["spread_estimated"] == 0
    assert c["slippage_usd"] == pytest.approx(0.005 * 1005) and c["flags"] == ""
    # no quote: default spread, flagged; no ADV: half spread only, flagged
    c = fill_cost("sell", 100, 10.0, None, None, None, 0.003)
    assert c["spread_pct"] == 0.003 and c["spread_estimated"] == 1 and c["slippage_vs_mid"] is None
    assert c["modeled_cost_pct"] == pytest.approx(0.0015) and c["flags"] == "spread_estimated,no_impact_inputs"


# ----------------------------------------------------------------------------- journal cost model + P&L
def _journal_with_fills(tmp_path):
    j = Journal(tmp_path / "j.db")
    a = j.start_run("analyze", "paper")
    j.signals(a, "adv", {"QQ": (1e6, {"adv20": 1e6, "adv_shares20": 10_000, "sigma_daily": 0.03}),
                         "NQ": (1e6, {"adv20": 1e6, "adv_shares20": 40_000, "sigma_daily": 0.02})})
    e = j.start_run("execute", "paper")
    p1, p2 = _plan("QQ", "buy", 1_000.0), _plan("NQ", "sell", 500.0)
    j.order(e, p1, SubmittedOrder(symbol="QQ", side="buy", notional=1000, broker_order_id="o1", status="new"))
    j.order(e, p2, SubmittedOrder(symbol="NQ", side="sell", notional=500, broker_order_id="o2", status="new"))
    j.record_order_quote(e, a, p1, "new", "o1", normalize_quote({"bid": 9.95, "ask": 10.05, "ts": "t"}),
                         "iex", 10_000, 0.03)
    j.update_fill("o1", "filled", 100, 10.05)
    j.update_fill("o2", "filled", 100, 5.0)          # no stored quote: estimated spread + refs from signals
    return j


def test_update_fill_costs(tmp_path):
    j = _journal_with_fills(tmp_path)
    assert j.update_fill_costs(0.003) == 2
    assert j.update_fill_costs(0.003) == 2        # idempotent
    fc = j.query("SELECT * FROM fill_costs ORDER BY broker_order_id").set_index("broker_order_id")
    assert len(fc) == 2
    assert fc.loc["o1", "modeled_cost_usd"] == pytest.approx(0.008 * 1005)
    assert fc.loc["o1", "slippage_vs_mid"] == pytest.approx(0.005) and fc.loc["o1", "spread_estimated"] == 0
    imp = 0.02 * (100 / 40_000) ** 0.5
    assert fc.loc["o2", "spread_estimated"] == 1 and "spread_estimated" in fc.loc["o2", "flags"]
    assert fc.loc["o2", "modeled_cost_pct"] == pytest.approx(0.0015 + imp)


def test_realistic_pnl(tmp_path):
    j = _journal_with_fills(tmp_path)
    j.update_fill_costs(0.003)
    today = dt.date.today()
    j.record_equity(today - dt.timedelta(days=1), 100_000, 100_000, None)
    j.record_equity(today, 101_000, 50_000, None)
    t = j.pnl_table()
    cost = float(j.query("SELECT SUM(modeled_cost_usd) AS c FROM fill_costs").c[0])
    assert list(t.cum_cost) == pytest.approx([0.0, cost])        # fills dated today
    assert t.paper_pnl.iloc[-1] == 1_000 and t.realistic_pnl.iloc[-1] == pytest.approx(1_000 - cost)
    p = j.pnl_summary()
    assert p["n_fills"] == 2 and p["n_estimated"] == 1

    _, body = summarize_runs(j, 0, "paper", "reconcile", False, realistic_pnl=True)
    assert "Paper P&L since" in body and "Realistic P&L" in body and "1 with estimated spread" in body
    _, body_a = summarize_runs(j, 0, "paper", "reconcile", False)
    assert "Realistic" not in body_a and "Paper P&L" not in body_a

    rpt = write_report(tmp_path / "r", 1, "paper", None, pd.Series({"QQ": 0.1}), {}, [], 0.0, pnl=p)
    assert "## P&L: paper vs realistic" in rpt.read_text(encoding="utf-8")
    rpt = write_report(tmp_path / "r2", 1, "paper", None, pd.Series({"QQ": 0.1}), {}, [], 0.0)
    assert "P&L" not in rpt.read_text(encoding="utf-8")


def test_pnl_table_empty(tmp_path):
    assert Journal(tmp_path / "j.db").pnl_table().empty
    assert Journal(tmp_path / "j.db").pnl_summary() is None


# ----------------------------------------------------------------------------- pipeline (B)
def test_execute_records_quotes_and_guards_spread(tmp_path, pipe):
    s = _settings(tmp_path)
    _write_universe(tmp_path)
    pipe["broker"] = DryRunBroker(positions={"OLD": 3_000.0})
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    j = Journal(tmp_path / "j.db")
    adv = {r.symbol: json.loads(r.extra) for r in j.query("SELECT symbol, extra FROM signals WHERE source='adv'")
           .itertuples()}
    assert adv["AAA"]["adv_shares20"] == pytest.approx(1e6) and adv["AAA"]["sigma_daily"] > 0

    quotes = {"AAA": {"bid": 99.0, "ask": 101.0},          # 2% spread > 1.5% -> buy skipped
              "BBB": {"bid": 0.0, "ask": 101.0},           # one-sided -> no quote -> buy skipped
              "OLD": {"bid": 0.0, "ask": 0.0}}             # sell without quote still goes through
    broker = DryRunBroker(positions={"OLD": 3_000.0}, quotes=quotes, sim_spread_pct=0.003)
    pipe["broker"] = broker
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    sent = {o.symbol: o for o in broker.submitted}
    assert "AAA" not in sent and "BBB" not in sent
    assert sent["OLD"].side == "sell" and {"CCC", "DDD"} & set(sent)
    q = j.query("SELECT * FROM order_quotes").set_index("symbol")
    assert q.loc["AAA", "action"].startswith("skipped:") and q.loc["AAA", "spread_pct"] == pytest.approx(0.02)
    assert q.loc["BBB", "action"].startswith("skipped:") and pd.isna(q.loc["BBB", "bid"])
    assert q.loc["OLD", "action"] == "simulated" and pd.isna(q.loc["OLD", "mid"])
    c = q.loc[sorted({"CCC", "DDD"} & set(sent))[0]]
    assert c.feed == "sim" and c.spread_pct == pytest.approx(0.003, rel=1e-3)
    assert c.adv_shares == pytest.approx(1e6) and c.sigma_daily > 0 and c.src_run_id is not None


def test_default_profile_unchanged(tmp_path, pipe):
    s = _settings(tmp_path, profile="default")
    s.features = FeaturesCfg()   # stage-2 cost features must not leak into A (upgrade features: test_gap/morning/watch)
    s.execution = s.execution.model_copy(update={"order_style": "notional", "limit_fractional_buys": False})
    assert s.execution.record_quotes is False and s.risk.max_spread_pct is None and s.radar.enabled is False
    broker = DryRunBroker(quotes={"AAA": {"bid": 1.0, "ask": 200.0}})   # absurd spread: must not matter for A
    pipe["broker"] = broker
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    assert "AAA" in {o.symbol for o in broker.submitted}
    j = Journal(tmp_path / "j.db")
    tables = set(j.query("SELECT name FROM sqlite_master WHERE type='table'").name)
    assert not {"order_quotes", "fill_costs"} & tables
    assert j.query("SELECT COUNT(*) AS n FROM signals").n[0] == 0
    rpt = sorted((tmp_path / "reports").glob("*.md"))[-1].read_text(encoding="utf-8")
    assert "P&L" not in rpt
    assert run_daily.make_broker("dry-run", load_settings()).latest_quote("SPY") is None


# ----------------------------------------------------------------------------- screener features + radar
class _FakeLLM:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, *a, **k):
        pass


def _fake_team(seen):
    class Team:
        def __init__(self, *a, **k):
            pass

        def run(self, inp):
            seen["inputs"] = inp
            return TeamOutput(views=[], cash_weight=0.05, reports={})
    return Team


def _patch_agents(monkeypatch, seen):
    import bot.agents.llm as llm_mod
    import bot.agents.team as team_mod

    monkeypatch.setattr(llm_mod, "LLM", _FakeLLM)
    monkeypatch.setattr(team_mod, "AgentTeam", _fake_team(seen))


def test_screener_features_in_technical_payload_and_radar_shadow(tmp_path, pipe, monkeypatch):
    import bot.radar.features as rf

    calls = {}

    def fake_features(db_path, symbols, freeze_ts):
        calls.update(db=db_path, syms=list(symbols), ts=freeze_ts)
        return {x: {"mention_z": 2.5 if x == "AAA" else None, "n_sources": 1, "fringe_only": False}
                for x in symbols}

    monkeypatch.setattr(rf, "compute_features", fake_features)
    seen = {}
    _patch_agents(monkeypatch, seen)
    s = _settings(tmp_path)
    _write_universe(tmp_path)
    pipe["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=True, use_kronos=False) == 0

    tech = seen["inputs"].technical
    assert tech["AAA"]["screener"] == {"rank": 1, "score": 2.0, "momentum_12_1": 0.5, "momentum_window": "12-1",
                                       "ret_3m": 0.2, "volume_surge_log_5d_vs_50d": 0.3,
                                       "close_vs_52w_high": 0.95}
    assert "last_close" in tech["AAA"]
    # radar: called with tradable symbols, journaled, never passed to the agents
    assert sorted(calls["syms"]) == UNIVERSE and calls["ts"].tzinfo is not None
    assert str(calls["db"]).endswith("radar.db")
    j = Journal(tmp_path / "j.db")
    r = j.query("SELECT symbol, value, extra FROM signals WHERE source='radar'").set_index("symbol")
    assert r.loc["AAA", "value"] == 2.5 and r.loc["BBB", "value"] == 0.0
    assert json.loads(r.loc["AAA", "extra"])["n_sources"] == 1
    blob = json.dumps(seen["inputs"].__dict__, default=str)
    assert "mention_z" not in blob and "fringe_only" not in blob
    inputs_row = j.query("SELECT content FROM agent_reports WHERE agent='inputs'").content[0]
    assert "mention_z" not in inputs_row


def test_radar_failure_never_blocks_analyze(tmp_path, pipe, monkeypatch):
    import bot.radar.features as rf

    def boom(*a, **k):
        raise RuntimeError("radar db locked")

    monkeypatch.setattr(rf, "compute_features", boom)
    s = _settings(tmp_path)
    _write_universe(tmp_path)
    pipe["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    j = Journal(tmp_path / "j.db")
    assert j.query("SELECT COUNT(*) AS n FROM signals WHERE source='radar'").n[0] == 0
    targets_fail = j.latest_targets("dry-run")[1]

    # missing module (ImportError) is tolerated too, and targets do not depend on the radar
    monkeypatch.setitem(sys.modules, "bot.radar.features", None)
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    pd.testing.assert_series_equal(j.latest_targets("dry-run")[1].sort_index(), targets_fail.sort_index())


def test_radar_off_not_called(tmp_path, pipe, monkeypatch):
    import bot.radar.features as rf

    monkeypatch.setattr(rf, "compute_features", lambda *a, **k: pytest.fail("radar called while disabled"))
    s = _settings(tmp_path)
    s = s.model_copy(update={"radar": s.radar.model_copy(update={"enabled": False})})
    _write_universe(tmp_path)
    pipe["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0


def test_default_profile_has_no_screener_key(tmp_path, pipe, monkeypatch):
    seen = {}
    _patch_agents(monkeypatch, seen)
    s = _settings(tmp_path, profile="default")
    pipe["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=True, use_kronos=False) == 0
    assert all("screener" not in v for v in seen["inputs"].technical.values())
