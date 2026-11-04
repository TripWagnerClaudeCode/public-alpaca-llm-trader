"""Deterministic opening-gap guard: cancel a planned buy that gaps down hard against its view."""
from __future__ import annotations

from bot.config import load_settings
from bot.models import OrderPlan
from bot.portfolio.gap import gap_guard
from bot.run_daily import needs_market_refs
from test_profile_pipeline import setup  # noqa: F401  (fixture)


def _o(sym, side="buy", notional=1000.0):
    return OrderPlan(symbol=sym, side=side, notional=notional, current_weight=0.0, target_weight=0.1)


PREV = {"A": 100.0, "B": 100.0}
SIG = {"A": 0.02, "B": 0.02}


def test_gap_down_beyond_k_sigma_cancels_buy():
    out, why = gap_guard([_o("A")], {"A": 93.0}, PREV, SIG, k=3.0)    # -7% < -6%
    assert out == [] and len(why) == 1 and "A" in why[0]


def test_gap_within_band_kept():
    out, why = gap_guard([_o("A")], {"A": 95.0}, PREV, SIG, k=3.0)    # -5% > -6%
    assert [o.symbol for o in out] == ["A"] and why == []


def test_gap_up_kept():
    out, _ = gap_guard([_o("A")], {"A": 110.0}, PREV, SIG, k=3.0)
    assert [o.symbol for o in out] == ["A"]


def test_sells_never_dropped():
    out, _ = gap_guard([_o("A", side="sell")], {"A": 50.0}, PREV, SIG, k=3.0)
    assert [o.side for o in out] == ["sell"]


def test_missing_quote_kept_with_note():
    out, why = gap_guard([_o("A"), _o("C")], {"A": None}, PREV, SIG, k=3.0)
    assert [o.symbol for o in out] == ["A", "C"]
    assert len(why) == 2 and all("kept" in w for w in why)


def test_order_never_enlarged():
    o = _o("B", notional=1234.5)
    (kept,), _ = gap_guard([o], {"B": 99.0}, PREV, SIG, k=3.0)
    assert kept.notional == 1234.5


def test_market_refs_saved_when_morning_check_on():
    s = load_settings(profile="default")
    s.features.morning_check = False
    s.execution.order_style = "notional"
    base = needs_market_refs(s)
    s.features.morning_check = True
    assert needs_market_refs(s) and not base


def _quotes_at(factor):
    from test_profile_pipeline import UNIVERSE, _bars

    out = {}
    for i, sym in enumerate(sorted([*UNIVERSE, "IWM"])):   # FakeMD numbers series over universe + benchmark
        px = float(_bars(i)["close"].iloc[-1]) * factor
        out[sym] = {"bid": round(px * 0.9995, 4), "ask": round(px * 1.0005, 4), "ts": "t"}
    return out


def _execute_with_gap(setup, factor, morning=True):
    from bot import run_daily
    from bot.broker.sim import DryRunBroker
    from test_profile_pipeline import _write_universe

    s, box, tmp = setup
    s.features.morning_check = morning
    s.features.watch = False
    _write_universe(tmp)
    box["broker"] = DryRunBroker()
    assert run_daily.analyze(s, "dry-run", use_agents=False, use_kronos=False) == 0
    box["broker"] = DryRunBroker(quotes=_quotes_at(factor))
    assert run_daily.execute(s, "dry-run", approve=False) == 0
    return box["broker"].submitted


def test_execute_gap_down_cancels_all_buys(setup):
    assert [o for o in _execute_with_gap(setup, 0.85) if o.side == "buy"] == []


def test_execute_no_gap_keeps_buys(setup):
    assert [o for o in _execute_with_gap(setup, 1.0) if o.side == "buy"]


def test_execute_gap_ignored_when_feature_off(setup):
    assert [o for o in _execute_with_gap(setup, 0.85, morning=False) if o.side == "buy"]
