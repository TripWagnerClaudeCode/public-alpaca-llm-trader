"""AlpacaBroker.submit request shapes (no network: the trading client is a recorder)."""
from types import SimpleNamespace

from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from bot.broker.alpaca_broker import AlpacaBroker
from bot.models import OrderPlan


class _Client:
    def __init__(self):
        self.reqs = []

    def submit_order(self, req):
        self.reqs.append(req)
        return SimpleNamespace(id="oid-1", status="accepted")


def _broker():
    b = object.__new__(AlpacaBroker)   # skip __init__: no keys, no network
    b.client = _Client()
    return b


def _plan(**kw):
    base = dict(symbol="F", side="buy", notional=500.0, current_weight=0, target_weight=0.05)
    return OrderPlan(**{**base, **kw})


def test_fractional_limit_buy_keeps_its_fractional_qty():
    b = _broker()
    b.submit(_plan(qty=49.751243, limit_price=10.05, notional=499.999992))
    req = b.client.reqs[0]
    assert isinstance(req, LimitOrderRequest)
    assert req.qty == 49.751243 and req.limit_price == 10.05


def test_whole_share_limit_and_notional_market_unchanged():
    b = _broker()
    b.submit(_plan(qty=49.0, limit_price=10.05))
    b.submit(_plan())
    assert isinstance(b.client.reqs[0], LimitOrderRequest) and b.client.reqs[0].qty == 49
    assert isinstance(b.client.reqs[1], MarketOrderRequest) and b.client.reqs[1].notional == 500.0
