import numpy as np
import pandas as pd
import pytest

from bot.models import AccountState, OrderPlan, PositionDetail
from bot.portfolio.orders import (avg_dollar_volume, cap_adv, limit_price, shape_whole_share,
                                  to_fractional_limit, to_whole_share)


def _o(sym, side, n, close=False):
    return OrderPlan(symbol=sym, side=side, notional=n, current_weight=0, target_weight=0, close_position=close)


def test_whole_share_buy_uses_ask_plus_buffer():
    o, why = to_whole_share(_o("X", "buy", 1_000), {"bid": 9.98, "ask": 10.00}, 9.5, 0.005, None, 25)
    assert why is None
    assert o.limit_price == 10.05 and o.qty == 99 and o.notional == pytest.approx(994.95)
    assert o.notional <= 1_000   # never spends more than planned


def test_whole_share_sell_uses_bid_and_held_qty():
    o, _ = to_whole_share(_o("X", "sell", 1_000), {"bid": 10.00, "ask": 10.02}, None, 0.005, 40, 25)
    assert o.limit_price == 9.95 and o.qty == 40     # capped at shares held


def test_whole_share_fallback_to_last_close():
    assert limit_price("buy", None, 20.0, 0.005) == pytest.approx(20.2)
    assert limit_price("sell", {"bid": 0, "ask": 0}, 20.0, 0.005) == pytest.approx(19.8)
    assert limit_price("buy", None, None, 0.005) is None
    o, why = to_whole_share(_o("X", "buy", 100), None, None, 0.005, None, 5)
    assert o is None and "no quote" in why


def test_whole_share_skips_zero_qty_and_tiny():
    o, why = to_whole_share(_o("X", "buy", 30), {"bid": 49.9, "ask": 50.0}, None, 0.005, None, 25)
    assert o is None and "0 shares" in why
    o, why = to_whole_share(_o("X", "buy", 40), {"bid": 19.9, "ask": 20.0}, None, 0.005, None, 25)
    assert o is None and "minimum" in why        # 1 share x $20.10 < $25


def test_shape_only_touches_non_fractionable_partial_orders():
    acct = AccountState(equity=10_000, last_equity=10_000, cash=0, buying_power=0, positions={"Y": 500},
                        details={"Y": PositionDetail(qty=50, avg_entry_price=10, current_price=10)})
    orders = [_o("F", "buy", 500), _o("N", "buy", 500), _o("Y", "sell", 500, close=True)]
    out, why = shape_whole_share(orders, {"N", "Y"}, acct, {"N": {"bid": 9.9, "ask": 10.0}}, {}, 0.005, 25)
    by = {o.symbol: o for o in out}
    assert by["F"].qty is None                       # fractionable: notional market order
    assert by["N"].qty == 49 and by["N"].limit_price == 10.05
    assert by["Y"].qty is None and by["Y"].close_position   # full exits stay close_position
    assert not why


def test_adv_cap_shrinks_and_fails_closed():
    orders = [_o("A", "buy", 5_000), _o("B", "buy", 500), _o("C", "buy", 1_000), _o("D", "sell", 9_000, close=True),
              _o("E", "sell", 9_000), _o("T", "buy", 1_000)]
    adv = {"A": 100_000.0, "B": 100_000.0, "D": 1_000.0, "E": 100_000.0, "T": 1_000.0}
    out, why = cap_adv(orders, adv, 0.02, 25)
    by = {o.symbol: o for o in out}
    assert by["A"].notional == 2_000                 # shrunk to 2% of ADV
    assert by["B"].notional == 500                   # already within
    assert "C" not in by                             # no ADV -> buy skipped (fail closed)
    assert by["D"].notional == 9_000                 # full exits are never shrunk
    assert by["E"].notional == 2_000
    assert "T" not in by                             # cap $20 below the $25 minimum
    assert len(why) == 4 and any("shrunk" in w and w.startswith("A:") for w in why)


def test_avg_dollar_volume():
    idx = pd.bdate_range("2026-01-01", periods=30)
    b = pd.DataFrame({"close": np.full(30, 10.0), "volume": np.full(30, 1_000.0)}, index=idx)
    assert avg_dollar_volume(b) == pytest.approx(10_000)
    assert avg_dollar_volume(b.tail(5)) is None


# ------------------------------------------------- finding 003: limit orders for fractionable buys
def test_fractional_limit_buy_uses_ask_plus_buffer_and_never_overspends():
    o, why = to_fractional_limit(_o("F", "buy", 1_000), {"bid": 9.98, "ask": 10.00}, 9.5, 0.005, 25)
    assert why is None
    assert o.limit_price == 10.05 and o.qty == pytest.approx(99.502487, abs=1e-6)
    assert o.qty != int(o.qty)                       # still fractional
    assert o.notional <= 1_000


def test_fractional_limit_falls_back_to_last_close_or_skips():
    o, _ = to_fractional_limit(_o("F", "buy", 100), None, 20.0, 0.005, 5)
    assert o.limit_price == pytest.approx(20.2)
    o, why = to_fractional_limit(_o("F", "buy", 100), None, None, 0.005, 5)
    assert o is None and "no quote" in why


def test_shape_limits_fractionable_buys_only_when_enabled():
    acct = AccountState(equity=10_000, last_equity=10_000, cash=0, buying_power=0, positions={"Y": 500, "S": 500},
                        details={"Y": PositionDetail(qty=50, avg_entry_price=10, current_price=10)})
    orders = [_o("F", "buy", 500), _o("S", "sell", 200), _o("Y", "sell", 500, close=True)]
    quotes = {"F": {"bid": 9.9, "ask": 10.0}}
    out, why = shape_whole_share(orders, set(), acct, quotes, {}, 0.005, 25, limit_fractional_buys=True)
    by = {o.symbol: o for o in out}
    assert by["F"].limit_price == 10.05 and by["F"].qty == pytest.approx(49.751243, abs=1e-6)
    assert by["S"].qty is None and by["S"].limit_price is None     # fractionable sells unchanged
    assert by["Y"].close_position and by["Y"].limit_price is None  # full exits stay market
    assert not why
    off, _ = shape_whole_share(orders, set(), acct, quotes, {}, 0.005, 25)
    assert {o.symbol: o for o in off}["F"].limit_price is None     # default: unchanged behaviour
