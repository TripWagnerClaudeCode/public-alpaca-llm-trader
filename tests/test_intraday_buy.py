"""Phase 3 intraday buys: an event trigger on an unheld universe symbol runs a single-symbol pipeline
(deep dive -> PM view -> second PM AND Risk Reviewer must approve -> the optimizer sizes it -> caps ->
risk gate -> marketable limit). The LLM never picks the size."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from bot import events, intraday_buy, watch
from bot.agents.schemas import BuyApproval, DiveFindings, PMView
from bot.broker.sim import DryRunBroker
from bot.config import load_settings
from bot.journal.store import Journal
from bot.models import OrderPlan, PositionDetail, View

NY = events.NY
IN = dt.datetime.combine(dt.date.today(), dt.time(11, 0), tzinfo=NY)   # order rows use the real clock
UNI = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def T(sym, kind="move", z=-3.0):
    return events.Trigger(sym, kind, "d", z)


def test_candidates_universe_only(tmp_path):
    s = load_settings(profile="default")
    trig = [T("AAA", z=2.6), T("ZZZ", z=-5), T("BBB", z=-4), T("CCC", z=3.0), T("BBB", "filing", 0.0)]
    got = intraday_buy.buy_candidates(s, UNI, {"CCC"}, trig, [{"symbol": "DDD", "pipeline": "buy"}])
    assert [t.symbol for t in got] == ["BBB", "AAA"]          # ZZZ outside, CCC held; one per symbol, |z| order


def test_screener_universe_for_b(tmp_path, monkeypatch):
    s = load_settings(profile="smallcap")
    monkeypatch.setattr("bot.run_daily.screener_universe", lambda s, d: ["XX", "YY"])
    assert intraday_buy.universe_for(s, dt.date(2026, 11, 4)) == ["XX", "YY"]
    assert intraday_buy.universe_for(load_settings(profile="default"), dt.date(2026, 11, 4)) == \
        list(load_settings(profile="default").universe.symbols)


def test_size_from_optimizer_capped_half_max_weight(monkeypatch, closes):
    s = load_settings(profile="default")
    monkeypatch.setattr(intraday_buy, "target_weights", lambda c, v, p, h: pd.Series({"AAA": 0.15, "BBB": 0.1}))
    v = View(symbol="AAA", expected_return=0.03, confidence=0.6, rationale="r")
    assert intraday_buy.size_intraday(closes, [], v, s.portfolio, 10, {}, s.risk) == pytest.approx(0.075)
    monkeypatch.setattr(intraday_buy, "target_weights", lambda c, v, p, h: pd.Series({"AAA": 0.02}))
    assert intraday_buy.size_intraday(closes, [], v, s.portfolio, 10, {}, s.risk) == pytest.approx(0.02)
    assert intraday_buy.size_intraday(closes, [], v, s.portfolio, 10, {"AAA": 0.05}, s.risk) == 0.0


def test_size_replaces_base_view_on_symbol(monkeypatch, closes):
    s = load_settings(profile="default")
    seen = {}
    monkeypatch.setattr(intraday_buy, "target_weights",
                        lambda c, v, p, h: seen.setdefault("v", v) and pd.Series({"AAA": 0.05}))
    base = [View(symbol="AAA", expected_return=-0.02, confidence=0.5, rationale="old"),
            View(symbol="BBB", expected_return=0.01, confidence=0.5, rationale="b")]
    new = View(symbol="AAA", expected_return=0.03, confidence=0.6, rationale="new")
    intraday_buy.size_intraday(closes, base, new, s.portfolio, 10, {}, s.risk)
    assert sorted((x.symbol, x.rationale) for x in seen["v"]) == [("AAA", "new"), ("BBB", "b")]


# ------------------------------------------------------------------ watch integration
class FakeLLM:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, er=0.03, er2=0.02, risk_ok=True):
        self.er, self.er2, self.risk_ok = er, er2, risk_ok
        self.calls = []

    def research(self, **k):
        self.calls.append(("research", k))
        return DiveFindings(symbol="X", question="q", answer="beat and raise", findings=[]), []

    def structured(self, **k):
        self.calls.append((k["schema"].__name__, k))
        if k["schema"] is BuyApproval:
            return BuyApproval(approve=self.risk_ok, reason="risk says")
        er = self.er if k["effort"] == "high" else self.er2       # first PM high effort, second PM medium
        return PMView(symbol="AAA", expected_return=er, confidence=0.6, rationale="pm")


def _settings(tmp_path):
    s = load_settings(profile="default")
    s.universe.symbols = list(UNI)
    s.paths.journal_db = str(tmp_path / "j.db")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    s.features.watch = True
    s.features.event_review = False
    s.features.intraday_buys = True
    s.features.memory = False
    return s


def _seed(s, views=()):
    j = Journal(s.paths.resolve(s.paths.journal_db))
    rid = j.start_run("analyze", "dry-run")
    j.agent_report(rid, "inputs", {"as_of": "2026-11-03", "universe": UNI})
    j.conn.executemany("INSERT INTO signals(run_id, symbol, source, value, extra) VALUES (?,?,?,?,?)",
                       [(rid, x, "adv", 1e6, json.dumps({"close": 100.0, "sigma_daily": 0.02})) for x in UNI])
    j.conn.commit()
    j.views(rid, list(views))
    j.finish_run(rid, "ok")
    return j


def _broker(moves: dict[str, float], cash=10_000.0, held=None):
    px = {x: 100.0 for x in UNI} | moves
    held = held or {}
    det = {x: PositionDetail(qty=q, avg_entry_price=100.0, current_price=px[x]) for x, q in held.items()}
    b = DryRunBroker(equity=cash + sum(q * px[x] for x, q in held.items()),
                     positions={x: q * px[x] for x, q in held.items()}, details=det,
                     quotes={x: {"bid": p - 0.05, "ask": p + 0.05, "ts": "t"} for x, p in px.items()})
    b.is_market_open = lambda: True
    return b


def _run(s, b, llm, closes, now=IN, target=0.15):
    import bot.intraday_buy as ib

    ib.target_weights = lambda c, v, p, h: pd.Series({"AAA": target, "BBB": target})
    return watch.run(s, "dry-run", broker=b, llm=llm, now_fn=lambda: now,
                     filings_fn=lambda sym, since: [], closes_fn=lambda syms: closes)


@pytest.fixture(autouse=True)
def _restore_tw(monkeypatch):
    monkeypatch.setattr(intraday_buy, "target_weights", intraday_buy.target_weights)


def _ev(s):
    return Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM events ORDER BY id")


def test_buy_is_marketable_limit(tmp_path, closes):
    s = _settings(tmp_path)
    _seed(s)
    b, llm = _broker({"AAA": 107.0}), FakeLLM()                   # +7%, 2% sigma -> 3.5 sigma
    assert _run(s, b, llm, closes) == 0
    [o] = b.submitted
    lim = 107.05 * (1 + s.execution.limit_buffer_pct)
    assert (o.symbol, o.side) == ("AAA", "buy") and o.limit_price == pytest.approx(lim)
    assert o.notional == pytest.approx(0.075 * 10_000) and o.qty == pytest.approx(o.notional / lim, rel=1e-4)
    ev = _ev(s)
    assert ev["pipeline"].tolist() == ["buy"] and ev["action"].tolist() == ["buy"]
    j = Journal(s.paths.resolve(s.paths.journal_db))
    wr = j.query("SELECT r.phase, v.symbol FROM views v JOIN runs r ON r.id=v.run_id WHERE r.phase='watch'")
    assert wr["symbol"].tolist() == ["AAA"]
    names = [c[0] for c in llm.calls]
    assert names[0] == "research" and names.count("PMView") == 2 and "BuyApproval" in names


@pytest.mark.parametrize("kw,why", [({"risk_ok": False}, "risk"), ({"er2": -0.01}, "second")])
def test_needs_both_approvals(tmp_path, closes, kw, why):
    s = _settings(tmp_path)
    _seed(s)
    b = _broker({"AAA": 107.0})
    _run(s, b, FakeLLM(**kw), closes)
    assert b.submitted == []
    ev = _ev(s)
    assert ev["action"][0].startswith("rejected")
    rec = json.loads(ev["reviewer"][0])
    assert "risk" in rec and "second_pm" in rec


@pytest.mark.parametrize("er", [-0.01, 0.0, 0.5])               # negative, flat, or over max_abs_view_return
def test_failed_view_or_negative_view_no_buy(tmp_path, closes, er):
    s = _settings(tmp_path)
    _seed(s)
    b, llm = _broker({"AAA": 107.0}), FakeLLM(er=er)
    _run(s, b, llm, closes)
    assert b.submitted == [] and "BuyApproval" not in [c[0] for c in llm.calls]


def _buy_row(j, sym, side="buy", status="simulated"):
    rid = j.start_run("watch", "dry-run")
    j.order(rid, OrderPlan(symbol=sym, side=side, notional=500, current_weight=0, target_weight=0.05),
            type("S", (), {"broker_order_id": None, "status": status, "error": None})())
    return rid


def test_intraday_caps_third_buy(tmp_path, closes):
    s = _settings(tmp_path)
    j = _seed(s)
    for x in ("EEE", "FFF"):
        j.record_event(IN.date(), x, "move", "d", {}, {}, "buy", 100.0, 1, "dry-run", pipeline="buy")
    b = _broker({"AAA": 107.0})
    llm = FakeLLM()
    _run(s, b, llm, closes)
    assert b.submitted == [] and llm.calls == []


def test_intraday_caps_exited_today_and_cooldown(tmp_path, closes):
    s = _settings(tmp_path)
    j = _seed(s)
    _buy_row(j, "AAA", side="sell")
    j.record_stop("BBB", IN.date(), 100, 80, -0.2, IN.date() + dt.timedelta(days=5), 1, "dry-run")
    b, llm = _broker({"AAA": 107.0, "BBB": 93.0}), FakeLLM()
    _run(s, b, llm, closes)
    assert b.submitted == [] and llm.calls == []


def test_intraday_caps_turnover(tmp_path, closes):
    s = _settings(tmp_path)
    j = _seed(s)
    s.risk.max_daily_turnover = 0.10
    _buy_row(j, "CCC")                                            # $500 = 5% of equity already traded today
    b = _broker({"AAA": 107.0})
    _run(s, b, FakeLLM(), closes)
    assert b.submitted == []                                       # 5% + 7.5% > 10% (+1% gate tolerance)
    assert _ev(s)["action"][0].startswith("blocked")


def test_intraday_caps_cash(tmp_path, closes):
    s = _settings(tmp_path)
    _seed(s)
    b = _broker({"AAA": 107.0}, cash=400.0, held={"CCC": 96.0})  # cash 4% of equity, below the 5% min cash
    llm = FakeLLM()
    _run(s, b, llm, closes)
    assert b.submitted == [] and llm.calls == []                  # checked before any agent call


def test_pipeline_cap_two_per_day(tmp_path, closes):
    s = _settings(tmp_path)
    _seed(s)
    b = _broker({"AAA": 107.0, "BBB": 108.0, "CCC": 109.0})
    llm = FakeLLM(risk_ok=False)                                   # every pipeline runs and is rejected
    _run(s, b, llm, closes)
    _run(s, b, llm, closes, now=IN + dt.timedelta(minutes=30))
    assert [c[0] for c in llm.calls].count("research") == 2 == s.events.max_buy_pipelines_per_day


def test_flag_off_or_outside_window_or_kill_switch(tmp_path, closes):
    s = _settings(tmp_path)
    _seed(s)
    llm = FakeLLM()
    _run(s, _broker({"AAA": 107.0}), llm, closes, now=IN.replace(hour=15, minute=50))
    s.features.intraday_buys = False
    _run(s, _broker({"AAA": 107.0}), llm, closes)
    s.features.intraday_buys = True
    from bot.risk import gate

    gate.trip_kill_switch(s.risk, "auto: daily loss")             # even an automatic kill switch blocks buys
    b = _broker({"AAA": 107.0})
    _run(s, b, llm, closes)
    assert llm.calls == [] and b.submitted == []
