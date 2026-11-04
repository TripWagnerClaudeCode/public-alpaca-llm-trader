"""Profile B pipeline (analyze -> execute) in dry-run with fake market data and a fake screener file."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from bot import run_daily
from bot.broker.sim import DryRunBroker
from bot.config import FeaturesCfg, load_settings
from bot.data.market import MarketData
from bot.journal.store import Journal
from bot.models import PositionDetail
from bot.risk import gate

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
        return {s: _bars(i, volume=1_000.0 if s == "AAA" else 1e6) for i, s in enumerate(sorted(symbols))}

    closes = staticmethod(MarketData.closes)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    s = load_settings(profile="smallcap")
    s = s.model_copy(update={
        "paths": s.paths.model_copy(update={"journal_db": str(tmp_path / "j.db"),
                                            "reports_dir": str(tmp_path / "reports"),
                                            "cache_dir": str(tmp_path / "cache")}),
        # Stage-1 tests use a quote-less sim broker; the spread guard (stage 2) is tested in test_honest_fills.
        "risk": s.risk.model_copy(update={"kill_switch_file": str(tmp_path / "KILL_SWITCH_SMALLCAP"),
                                          "max_spread_pct": None}),
        "universe": s.universe.model_copy(update={"screener_dir": str(tmp_path / "universe")}),
        "radar": s.radar.model_copy(update={"db_path": str(tmp_path / "radar.db")}),
    })
    monkeypatch.setattr(run_daily, "MarketData", FakeMD)
    box = {}
    monkeypatch.setattr(run_daily, "make_broker", lambda mode, s=None: box["broker"])
    return s, box, tmp_path


def _write_universe(tmp_path, days_old=0, symbols=UNIVERSE):
    d = tmp_path / "universe"
    d.mkdir(exist_ok=True)
    as_of = dt.date.today() - dt.timedelta(days=days_old)
    (d / f"{as_of}.json").write_text(json.dumps({"as_of": str(as_of), "symbols": symbols}), encoding="utf-8")


def _last_run(s):
    return Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM runs ORDER BY id DESC LIMIT 1").iloc[0]


def test_missing_screener_universe_blocks(setup):
    s, box, tmp = setup
    box["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 2
    r = _last_run(s)
    assert r.status == "blocked" and "screener universe" in r.notes


def test_stale_screener_universe_blocks(setup):
    s, box, tmp = setup
    box["broker"] = DryRunBroker()
    _write_universe(tmp, days_old=9)
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 2


def test_held_names_outside_screener_are_sold(setup):
    s, box, tmp = setup
    _write_universe(tmp)
    box["broker"] = DryRunBroker(positions={"OLD": 5_000.0}, non_fractionable={"BBB"})
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    j = Journal(s.paths.resolve(s.paths.journal_db))
    _, targets = j.latest_targets("dry-run")
    assert targets["OLD"] == 0.0 and set(UNIVERSE) <= set(targets.index)
    adv = j.query("SELECT symbol, extra FROM signals WHERE source='adv'")
    assert set(adv.symbol) == {*UNIVERSE, "OLD", "IWM"}

    assert run_daily.execute(s, "dry-run", approve=False) == 0
    by = {o.symbol: o for o in box["broker"].submitted}
    assert by["OLD"].side == "sell" and by["OLD"].target_weight < 0.05   # ramps out under the turnover cap
    # liquidity cap: AAA has $100k ADV -> at most 2% = $2,000 per order
    assert by["AAA"].notional <= 2_000 + 1e-6
    # non-fractionable: whole-share limit at last close + 1% (no quote in the sim)
    last = FakeMD(None).daily_bars(UNIVERSE, 0)["BBB"]["close"].iloc[-1]
    assert by["BBB"].qty == int(by["BBB"].qty) and by["BBB"].limit_price == pytest.approx(last * 1.01, abs=0.01)
    # fractionable buy: fractional-qty marketable limit at last close + 1% (finding 003)
    last_c = FakeMD(None).daily_bars(UNIVERSE, 0)["CCC"]["close"].iloc[-1]
    assert by["CCC"].limit_price == pytest.approx(last_c * 1.01, abs=0.01) and by["CCC"].qty > 0


def test_stop_triggers_exit_and_cooldown(setup):
    s, box, tmp = setup
    _write_universe(tmp)
    det = {"CCC": PositionDetail(qty=100, avg_entry_price=100.0, current_price=80.0)}   # -20%
    box["broker"] = DryRunBroker(positions={"CCC": 8_000.0}, details=det)
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    j = Journal(s.paths.resolve(s.paths.journal_db))
    assert j.latest_targets("dry-run")[1]["CCC"] > 0       # optimizer still wants it

    assert run_daily.execute(s, "dry-run", approve=False) == 0
    by = {o.symbol: o for o in box["broker"].submitted}
    assert by["CCC"].side == "sell" and by["CCC"].close_position
    stops = j.query("SELECT * FROM stops")
    assert list(stops.symbol) == ["CCC"]
    assert stops.cooldown_until[0] == str(gate.cooldown_until(dt.date.today(), 10))
    assert stops.pnl_pct[0] == pytest.approx(-0.2)

    # a repeat trigger during cooldown is not journaled twice
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    assert len(j.query("SELECT * FROM stops")) == 1

    # during the cooldown analyze forces it to 0 (even if no longer held)
    box["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    assert j.latest_targets("dry-run")[1]["CCC"] == 0.0


def test_default_profile_has_no_stops_or_refs(tmp_path, monkeypatch):
    s = load_settings()
    s = s.model_copy(update={
        "universe": s.universe.model_copy(update={"symbols": UNIVERSE, "benchmark": "SPY"}),
        "paths": s.paths.model_copy(update={"journal_db": str(tmp_path / "j.db"),
                                            "reports_dir": str(tmp_path / "reports")}),
        "risk": s.risk.model_copy(update={"kill_switch_file": str(tmp_path / "KS")}),
        "features": FeaturesCfg(),   # B's stops/refs must not leak into A; upgrade features tested elsewhere
        "execution": s.execution.model_copy(update={"order_style": "notional", "limit_fractional_buys": False}),
    })
    monkeypatch.setattr(run_daily, "MarketData", FakeMD)
    det = {"CCC": PositionDetail(qty=100, avg_entry_price=100.0, current_price=50.0)}
    broker = DryRunBroker(positions={"CCC": 5_000.0}, details=det, non_fractionable={"BBB"})
    monkeypatch.setattr(run_daily, "make_broker", lambda mode, s=None: broker)
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    j = Journal(tmp_path / "j.db")
    assert j.query("SELECT COUNT(*) AS n FROM signals").n[0] == 0
    assert j.active_cooldowns(dt.date.today()) == {}
    assert all(o.qty is None for o in broker.submitted)
    assert not any(o.symbol == "CCC" and o.close_position for o in broker.submitted)


def test_execute_rechecks_stored_targets_against_risk_caps(setup):
    """Targets are read back from the journal at execute; a weight over the cap (edited DB, or a
    cap tightened since analyze) must block the run before any order is sent."""
    s, box, tmp = setup
    _write_universe(tmp)
    box["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    j = Journal(s.paths.resolve(s.paths.journal_db))
    run_id, _ = j.latest_targets("dry-run")
    j.conn.execute("UPDATE target_weights SET weight=0.5 WHERE run_id=? AND symbol='CCC'", (run_id,))
    j.conn.commit()

    assert run_daily.execute(s, "dry-run", approve=False) == 2
    assert box["broker"].submitted == []
    r = _last_run(s)
    assert r.status == "blocked" and "position weight cap" in r.notes


def test_default_profile_limit_buys_same_trades_never_larger(tmp_path, monkeypatch):
    """Strategy freeze: A's limit buys change how orders are sent, not what or how much is traded."""
    def run(limits, sub):
        s = load_settings()
        s = s.model_copy(update={
            "universe": s.universe.model_copy(update={"symbols": UNIVERSE, "benchmark": "SPY"}),
            "paths": s.paths.model_copy(update={"journal_db": str(tmp_path / f"{sub}.db"),
                                                "reports_dir": str(tmp_path / sub)}),
            "risk": s.risk.model_copy(update={"kill_switch_file": str(tmp_path / "KS")}),
            "features": FeaturesCfg(),
            "execution": s.execution.model_copy(update={"order_style": "auto" if limits else "notional",
                                                        "limit_fractional_buys": limits}),
        })
        monkeypatch.setattr(run_daily, "MarketData", FakeMD)
        quotes = {}
        for i, sym in enumerate(sorted([*UNIVERSE, "SPY"])):
            px = float(_bars(i)["close"].iloc[-1])
            quotes[sym] = {"bid": px * 0.9995, "ask": px * 1.0005, "ts": "t"}
        broker = DryRunBroker(positions={"CCC": 5_000.0}, quotes=quotes)
        monkeypatch.setattr(run_daily, "make_broker", lambda mode, s=None: broker)
        assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
        assert run_daily.execute(s, "dry-run", approve=False) == 0
        return {(o.symbol, o.side): o for o in broker.submitted}

    plain, lim = run(False, "n"), run(True, "l")
    assert set(lim) == set(plain) and any(side == "buy" for _, side in plain)
    for k, o in lim.items():
        assert o.notional <= plain[k].notional + 0.01
        if k[1] == "buy":
            assert o.qty and o.limit_price and o.qty * o.limit_price <= plain[k].notional + 0.01
        else:
            assert o.qty is None    # sells and full exits stay market orders
