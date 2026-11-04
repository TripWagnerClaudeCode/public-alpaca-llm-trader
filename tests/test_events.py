"""Phase 3 intraday Event Review in watch: code triggers (big move, new filing) on held names open a
two-agent review; a trim or exit executes only when both agents agree, inside the time window."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from bot import events, watch
from bot.agents.llm import UsageLimited
from bot.agents.schemas import EventCall
from bot.broker.sim import DryRunBroker
from bot.config import load_settings
from bot.journal.store import Journal
from bot.models import PositionDetail
from bot.risk import gate

NY = events.NY
IN = dt.datetime.combine(dt.date.today(), dt.time(11, 0), tzinfo=NY)   # order rows use the real clock
LATE = IN.replace(hour=15, minute=50)


def C(action, fraction=1.0, reason="r"):
    return EventCall(action=action, fraction=fraction, reason=reason)


def test_triggers_move_and_filing():
    tr = events.find_triggers(["AAA", "BBB", "CCC"], mids={"AAA": 94.0, "BBB": 98.0, "CCC": 100.0},
                              prev_close={"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
                              sigma={"AAA": 0.02, "BBB": 0.01, "CCC": 0.02},
                              filings={"CCC": [{"form": "8-K", "date": "2026-11-04", "url": "u1"}]}, k=2.5)
    got = {(t.symbol, t.kind) for t in tr}
    assert got == {("AAA", "move"), ("CCC", "filing")}          # BBB: -2 sigma, below k
    assert [t.kind for t in tr][0] == "filing"                   # filings first, then moves by |z|
    assert next(t for t in tr if t.symbol == "AAA").z == pytest.approx(-3.0)
    assert events.find_triggers(["AAA"], {"AAA": None}, {"AAA": 100.0}, {"AAA": 0.02}, {}, 2.5) == []


def test_combine_least_severe():
    assert events.combine(C("exit"), C("hold")).action == "hold"
    assert events.combine(C("hold"), C("exit")).action == "hold"
    assert events.combine(C("exit"), C("exit")).action == "exit"
    t = events.combine(C("exit"), C("trim", 0.4))
    assert (t.action, t.fraction) == ("trim", 0.4)
    assert events.combine(C("trim", 0.5), C("trim", 0.3)).fraction == 0.3
    assert events.combine(C("trim", float("nan")), C("trim", 0.3)).action == "hold"


def test_in_window():
    w = ("09:45", "15:45")
    assert events.in_window(IN, w) and events.in_window(dt.datetime(2026, 11, 4, 15, 45, tzinfo=NY), w)
    assert not events.in_window(LATE, w) and not events.in_window(dt.datetime(2026, 11, 4, 9, 40, tzinfo=NY), w)


# ------------------------------------------------------------------ watch integration
class FakeLLM:
    input_tokens = output_tokens = 0
    cost_usd = 0.0

    def __init__(self, analyst=("exit", 1.0), reviewer=("exit", 1.0), exc=None):
        self.analyst, self.reviewer, self.exc = analyst, reviewer, exc
        self.research_calls, self.structured_calls = [], []

    def research(self, **k):
        self.research_calls.append(k)
        if self.exc:
            raise self.exc
        self.output_tokens += 10
        return C(*self.analyst, reason="analyst reason"), [{"tool": "WebFetch", "url": "https://x.com/a"}]

    def structured(self, **k):
        self.structured_calls.append(k)
        return C(*self.reviewer, reason="reviewer reason")


def _settings(tmp_path, profile="smallcap", on=True):
    s = load_settings(profile=profile)
    s.paths.journal_db = str(tmp_path / "j.db")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    s.features.watch = True
    s.features.event_review = on
    s.features.memory = False
    return s


def _seed_analyze(s, syms, close=100.0, sigma=0.02):
    j = Journal(s.paths.resolve(s.paths.journal_db))
    rid = j.start_run("analyze", "dry-run")
    j.conn.executemany("INSERT INTO signals(run_id, symbol, source, value, extra) VALUES (?,?,?,?,?)",
                       [(rid, x, "adv", 1e6, json.dumps({"close": close, "sigma_daily": sigma})) for x in syms])
    j.conn.commit()
    j.finish_run(rid, "ok")
    return j


def _broker(prices: dict[str, float], qty=10.0):
    det = {x: PositionDetail(qty=qty, avg_entry_price=100.0, current_price=p) for x, p in prices.items()}
    b = DryRunBroker(equity=10_000, positions={x: qty * p for x, p in prices.items()}, details=det,
                     quotes={x: {"bid": p - 0.05, "ask": p + 0.05, "ts": "t"} for x, p in prices.items()})
    b.is_market_open = lambda: True
    return b


def _run(s, b, llm, now=IN, filings=None):
    return watch.run(s, "dry-run", broker=b, llm=llm, now_fn=lambda: now,
                     filings_fn=(lambda sym, since: (filings or {}).get(sym, [])))


def _events(s):
    return Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM events ORDER BY id")


def test_flag_off_no_agent_calls(tmp_path):
    s = _settings(tmp_path, on=False)
    _seed_analyze(s, ["AAA"])
    llm = FakeLLM()
    assert _run(s, _broker({"AAA": 93.0}), llm) == 0
    assert llm.research_calls == [] and llm.structured_calls == []


def test_agree_exit_places_limit_sell(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    b, llm = _broker({"AAA": 93.0}), FakeLLM()        # -7% vs close, sigma 2% -> -3.5 sigma; inside B's -15% stop
    assert _run(s, b, llm) == 0
    [o] = b.submitted
    assert (o.symbol, o.side, o.qty, o.close_position) == ("AAA", "sell", 10.0, False)
    assert o.limit_price == pytest.approx(92.95 * (1 - s.execution.limit_buffer_pct))
    ev = _events(s)
    assert ev["action"].tolist() == ["exit"] and ev["kind"].tolist() == ["move"]
    assert json.loads(ev["analyst"][0])["touched"][0]["url"] == "https://x.com/a"
    assert ev["price"][0] == pytest.approx(93.0)
    assert llm.research_calls[0]["model"] == "claude-sonnet-5-5"
    assert llm.structured_calls[0]["model"] == "claude-opus-5-5"


def test_trim_both_uses_min_fraction(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    b = _broker({"AAA": 93.0})
    _run(s, b, FakeLLM(analyst=("trim", 0.5), reviewer=("trim", 0.3)))
    [o] = b.submitted
    assert o.qty == pytest.approx(3.0) and o.target_weight < o.current_weight


def test_disagreement_holds(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    b = _broker({"AAA": 93.0})
    _run(s, b, FakeLLM(analyst=("exit", 1.0), reviewer=("hold", 1.0)))
    assert b.submitted == []
    ev = _events(s)
    assert ev["action"].tolist() == ["hold"]
    assert json.loads(ev["analyst"][0])["call"]["action"] == "exit"
    assert json.loads(ev["reviewer"][0])["call"]["action"] == "hold"


def test_analyst_hold_skips_reviewer(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    llm = FakeLLM(analyst=("hold", 1.0))
    _run(s, _broker({"AAA": 93.0}), llm)
    assert llm.structured_calls == [] and _events(s)["action"].tolist() == ["hold"]


def test_dedupe_one_review_per_symbol_kind_per_day(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    llm = FakeLLM(analyst=("hold", 1.0))
    b = _broker({"AAA": 93.0})
    _run(s, b, llm)
    _run(s, b, llm)
    assert len(llm.research_calls) == 1


def test_filing_trigger_dedupes_by_url(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    llm = FakeLLM(analyst=("hold", 1.0))
    f = {"AAA": [{"form": "8-K", "date": "2026-11-04", "url": "https://sec.gov/a8k.htm", "title": "8-K"}]}
    b = _broker({"AAA": 100.0})
    _run(s, b, llm, filings=f)
    assert _events(s)["kind"].tolist() == ["filing"]
    assert "https://sec.gov/a8k.htm" in llm.research_calls[0]["user"]
    _run(s, b, llm, now=IN + dt.timedelta(days=1), filings=f)    # same filing seen again tomorrow
    assert len(llm.research_calls) == 1


def test_daily_cap_six(tmp_path):
    s = _settings(tmp_path)
    syms = [f"S{i}" for i in range(8)]
    _seed_analyze(s, syms)
    llm = FakeLLM(analyst=("hold", 1.0))
    for i in range(5):                      # 2 per watch (per-watch cap), 6 per day
        _run(s, _broker({x: 93.0 for x in syms}), llm, now=IN + dt.timedelta(minutes=30 * i))
    assert len(llm.research_calls) == 6 == s.events.max_reviews_per_day


def test_outside_window_no_llm_order(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA", "XYZ"])
    b, llm = _broker({"AAA": 93.0, "XYZ": 84.0}), FakeLLM()      # XYZ -16%: B's existing stop
    _run(s, b, llm, now=LATE)
    assert [o.symbol for o in b.submitted] == ["XYZ"] and b.submitted[0].close_position
    assert llm.research_calls == []


def test_usage_limit_backs_off_but_stops_still_run(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA", "XYZ"])
    llm = FakeLLM(exc=UsageLimited("usage limit reached"))
    b = _broker({"AAA": 93.0})
    t0 = IN.replace(hour=10)
    _run(s, b, llm, now=t0)
    j = Journal(s.paths.resolve(s.paths.journal_db))
    until = dt.datetime.fromisoformat(j.get_meta("agents_backoff_until"))
    assert until == t0 + dt.timedelta(hours=5)
    llm2 = FakeLLM()
    b2 = _broker({"BBB": 93.0, "XYZ": 84.0})
    _seed_analyze(s, ["BBB", "XYZ"])
    assert _run(s, b2, llm2, now=t0 + dt.timedelta(hours=1)) == 0
    assert [o.symbol for o in b2.submitted] == ["XYZ"]           # the deterministic stop still fired
    assert llm2.research_calls == []
    _run(s, b2, llm2, now=t0 + dt.timedelta(hours=5, minutes=30))
    assert llm2.research_calls                                    # the backoff expired


def test_manual_kill_switch_blocks_event_orders(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    (tmp_path / "KILL_SWITCH").write_text("manual")
    b, llm = _broker({"AAA": 93.0}), FakeLLM()
    assert _run(s, b, llm) == 2
    assert b.submitted == [] and llm.research_calls == []


def test_auto_kill_switch_still_allows_event_exit(tmp_path):
    s = _settings(tmp_path)
    _seed_analyze(s, ["AAA"])
    gate.trip_kill_switch(s.risk, "daily loss")
    b = _broker({"AAA": 93.0})
    _run(s, b, FakeLLM())
    assert [o.side for o in b.submitted] == ["sell"]


def test_profile_a_event_exit(tmp_path):
    s = _settings(tmp_path, profile="default")
    assert s.risk.stop_loss_pct is None                           # A still has no stop
    _seed_analyze(s, ["AAPL"])
    b = _broker({"AAPL": 93.0})
    _run(s, b, FakeLLM())
    assert [(o.symbol, o.side) for o in b.submitted] == [("AAPL", "sell")]
