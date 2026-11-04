"""Alpaca implementation of the Broker protocol (alpaca-py)."""
from __future__ import annotations

import logging
import uuid

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from bot.config import secret
from bot.models import AccountState, OrderPlan, PositionDetail, SubmittedOrder

log = logging.getLogger(__name__)


class AlpacaBroker:
    name = "alpaca"
    quote_feed = "iex"

    def __init__(self, paper: bool = True, api_key_env: str = "ALPACA_API_KEY",
                 secret_key_env: str = "ALPACA_SECRET_KEY"):
        if not paper and secret("ALLOW_LIVE_TRADING", required=False).lower() != "yes":
            raise RuntimeError("Live trading requested but ALLOW_LIVE_TRADING=yes is not set in .env")
        self.is_paper = paper
        self._key, self._secret = secret(api_key_env), secret(secret_key_env)
        self.client = TradingClient(self._key, self._secret, paper=paper)
        self._data = None

    def get_account(self) -> AccountState:
        a = self.client.get_account()
        raw = self.client.get_all_positions()
        positions = {p.symbol: float(p.market_value) for p in raw}
        details = {}
        for p in raw:
            try:
                details[p.symbol] = PositionDetail(qty=float(p.qty), avg_entry_price=float(p.avg_entry_price),
                                                   current_price=float(p.current_price))
            except (TypeError, ValueError):
                log.warning("Incomplete position data for %s; no stop check possible", p.symbol)
        return AccountState(
            equity=float(a.equity), last_equity=float(a.last_equity), cash=float(a.cash),
            buying_power=float(a.buying_power), trading_blocked=bool(a.trading_blocked), positions=positions,
            details=details,
        )

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def tradable(self, symbols: list[str]) -> dict[str, bool]:
        return {s: a["tradable"] and a["fractionable"] for s, a in self.assets(symbols).items()}

    def assets(self, symbols: list[str]) -> dict[str, dict[str, bool]]:
        out = {}
        for s in symbols:
            try:
                asset = self.client.get_asset(s)
                out[s] = {"tradable": bool(asset.tradable), "fractionable": bool(asset.fractionable)}
            except APIError:
                out[s] = {"tradable": False, "fractionable": False}
        return out

    def latest_quote(self, symbol: str) -> dict | None:
        """Latest IEX quote (free feed). None on any error or an empty side."""
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            if self._data is None:
                self._data = StockHistoricalDataClient(self._key, self._secret)
            q = self._data.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))[symbol]
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            if bid <= 0 or ask <= 0 or ask < bid:
                return None
            return {"bid": bid, "ask": ask, "ts": str(q.timestamp)}
        except Exception as e:
            log.warning("No quote for %s: %s", symbol, e)
            return None

    def submit(self, order: OrderPlan) -> SubmittedOrder:
        coid = f"bot-{uuid.uuid4().hex[:16]}"
        try:
            if order.close_position:
                o = self.client.close_position(order.symbol)
            elif order.qty is not None and order.limit_price is not None:
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=int(order.qty) if float(order.qty).is_integer() else order.qty,  # fractional: day TIF
                    limit_price=round(order.limit_price, 2 if order.limit_price >= 1 else 4),
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=coid,
                )
                o = self.client.submit_order(req)
            else:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    notional=round(order.notional, 2),
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,  # notional/fractional orders must be DAY
                    client_order_id=coid,
                )
                o = self.client.submit_order(req)
            return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                                  broker_order_id=str(o.id), status=str(o.status.value if hasattr(o.status, "value") else o.status))
        except APIError as e:
            log.error("Order rejected %s %s $%.2f: %s", order.side, order.symbol, order.notional, e)
            return SubmittedOrder(symbol=order.symbol, side=order.side, notional=order.notional,
                                  broker_order_id=None, status="rejected", error=str(e))

    def cancel_all(self) -> None:
        self.client.cancel_orders()

    def cancel_order(self, broker_order_id: str, wait_s: float = 10.0) -> None:
        """Cancel one order and wait (briefly) until it is terminal, so the shares are free to sell."""
        import time

        try:
            self.client.cancel_order_by_id(broker_order_id)
        except APIError as e:   # already filled/canceled: the status poll below tells
            log.warning("cancel %s: %s", broker_order_id, e)
        end = time.time() + wait_s
        while time.time() < end:
            if self.order_status(broker_order_id)["status"] in ("canceled", "filled", "expired", "rejected"):
                return
            time.sleep(1)

    def order_status(self, broker_order_id: str) -> dict:
        o = self.client.get_order_by_id(broker_order_id)
        return {
            "status": str(o.status.value if hasattr(o.status, "value") else o.status),
            "filled_qty": float(o.filled_qty or 0),
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
        }
