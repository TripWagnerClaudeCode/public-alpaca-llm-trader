"""BookBroker: the persistent simulated book used by the frozen control profiles."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from bot.broker.book import BookBroker
from bot.models import OrderPlan

DAY1, DAY2 = dt.date(2026, 11, 5), dt.date(2026, 11, 6)


def _quotes(table):
    return lambda sym: table.get(sym)


def _book(tmp_path, quotes=None, init=True, owner="control", today=DAY1, cash=100_000.0):
    return BookBroker(tmp_path / "book.json", _quotes(quotes or {}), lambda: True, starting_cash=cash,
                      init=init, owner=owner, today=lambda: today)


def _buy(sym, notional):
    return OrderPlan(symbol=sym, side="buy", notional=notional, current_weight=0.0, target_weight=0.01)


def test_buy_fills_at_ask_and_persists(tmp_path):
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}})
    sub = b.submit(_buy("AAPL", 1000.0))
    assert sub.status == "filled" and sub.broker_order_id
    again = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}}, init=False)
    acct = again.get_account()
    assert acct.details["AAPL"].qty == pytest.approx(5.0)
    assert acct.cash == pytest.approx(99_000.0)
    assert again.order_status(sub.broker_order_id)["status"] == "filled"
    assert again.order_status(sub.broker_order_id)["filled_avg_price"] == pytest.approx(200.0)


def test_sell_close_position_fills_at_bid(tmp_path):
    q = {"AAPL": {"bid": 210.0, "ask": 211.0}}
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}})
    b.submit(_buy("AAPL", 1000.0))
    b = _book(tmp_path, q, init=False)
    sub = b.submit(OrderPlan(symbol="AAPL", side="sell", notional=1050.0, current_weight=0.01,
                             target_weight=0.0, close_position=True))
    assert sub.status == "filled"
    acct = b.get_account()
    assert "AAPL" not in acct.positions
    assert acct.cash == pytest.approx(99_000.0 + 5 * 210.0)


def test_buy_more_than_cash_rejected(tmp_path):
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}}, cash=500.0)
    sub = b.submit(_buy("AAPL", 1000.0))
    assert sub.status == "rejected"
    acct = b.get_account()
    assert acct.cash == pytest.approx(500.0) and acct.positions == {}


def test_buy_without_any_price_rejected(tmp_path):
    b = _book(tmp_path, {})
    assert b.submit(_buy("ZZZ", 100.0)).status == "rejected"


def test_missing_book_without_init_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _book(tmp_path, init=False)


def test_corrupt_book_raises(tmp_path):
    (tmp_path / "book.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _book(tmp_path, init=False)


def test_owner_mismatch_raises(tmp_path):
    _book(tmp_path, owner="control")
    with pytest.raises(ValueError):
        _book(tmp_path, init=False, owner="smallcap_control")


def test_init_does_not_overwrite_existing_book(tmp_path):
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}})
    b.submit(_buy("AAPL", 1000.0))
    again = _book(tmp_path, init=True)
    assert "AAPL" in again.get_account().positions


def test_get_account_marks_to_last_price(tmp_path):
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}})
    b.submit(_buy("AAPL", 1000.0))
    b.mark({"AAPL": 220.0})                       # day-1 close
    day2 = _book(tmp_path, init=False, today=DAY2)
    acct = day2.get_account()
    assert acct.equity == pytest.approx(99_000.0 + 5 * 220.0)
    assert acct.last_equity == pytest.approx(99_000.0 + 5 * 220.0)  # rolled from the prior day's mark
    day2.mark({"AAPL": 230.0})
    acct = day2.get_account()
    assert acct.equity == pytest.approx(99_000.0 + 5 * 230.0)
    assert acct.last_equity == pytest.approx(99_000.0 + 5 * 220.0)  # same day: not rolled again


def test_write_is_atomic_json(tmp_path):
    b = _book(tmp_path, {"AAPL": {"bid": 199.0, "ask": 200.0}})
    b.submit(_buy("AAPL", 1000.0))
    d = json.loads((tmp_path / "book.json").read_text(encoding="utf-8"))
    assert d["version"] == 1 and d["owner"] == "control"
    assert not list(tmp_path.glob("*.tmp"))
