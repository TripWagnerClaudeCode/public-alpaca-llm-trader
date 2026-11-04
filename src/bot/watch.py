"""Intraday watch (spec 2026-11-04, Phase 1): runs every 30 minutes during market hours.

1. refresh the status of open orders
2. record intraday equity
3. run the profile's EXISTING stop (risk.stop_loss_pct) now instead of at the next morning's execute

Strategy freeze: no new exit levels. Profile A has no stop (stop_loss_pct None), so its watch only
records. Exits run under an automatic kill switch (loss-limit breach) but never under a manual one.
"""
from __future__ import annotations

import datetime as dt
import logging

from bot.config import Settings
from bot.journal.store import Journal
from bot.models import OrderPlan
from bot.risk import gate

log = logging.getLogger("bot.watch")


def run(s: Settings, mode: str, broker=None, llm=None, now_fn=None, filings_fn=None, closes_fn=None) -> int:
    """0 = ok / skipped / disabled, 1 = error, 2 = exits blocked by a manual kill switch.
    llm / now_fn / filings_fn: injectable for tests (Event Review, features.event_review)."""
    if not s.features.watch:
        return 0
    journal = Journal(s.paths.resolve(s.paths.journal_db))
    if broker is None:
        from bot.run_daily import make_broker

        broker = make_broker(mode, s)
    run_id = journal.start_run("watch", mode)
    import time as _time

    agent_deadline = _time.monotonic() + s.events.agent_budget_s
    holder: dict = {"llm": llm}

    def llm_factory():
        if holder["llm"] is None:
            from bot.run_daily import make_llm

            holder["llm"] = make_llm(s)
        return holder["llm"]

    def tokens() -> tuple[int, int]:
        x = holder["llm"]
        return (getattr(x, "input_tokens", 0), getattr(x, "output_tokens", 0)) if x is not None else (0, 0)

    try:
        if not broker.is_market_open():
            journal.finish_run(run_id, "skipped", "market closed")
            return 0
        for oid in journal.open_order_ids():
            st = broker.order_status(oid)
            journal.update_fill(oid, st["status"], st["filled_qty"], st["filled_avg_price"])
        acct = broker.get_account()
        journal.record_intraday_equity(dt.datetime.now().isoformat(timespec="seconds"), acct.equity, acct.cash)
        if not gate.exits_allowed(s.risk):
            journal.finish_run(run_id, "blocked", "manual kill switch: no exits")
            log.error("Manual kill switch active: intraday exits blocked")
            return 2
        today = dt.date.today()
        open_sells = journal.open_sell_symbols(today)
        open_rows = journal.open_sell_orders(today)
        event_keys = journal.event_sell_keys(today)
        active = journal.active_cooldowns(today, mode)
        sold = []
        for h in gate.stop_hits(s.risk, acct):
            d = acct.details.get(h.symbol)
            if d is None or d.qty <= 0:
                continue
            if h.symbol in open_sells:
                rows = [r for r in open_rows if r[1] == h.symbol]
                if not rows or any((r[0], r[1]) not in event_keys for r in rows):
                    continue          # a stop (or planned) sell is already working: never a second one
                for rid_, sym_, oid in rows:   # only unfilled Event Review limit sells: cancel, then the stop fires
                    if oid:
                        broker.cancel_order(oid)
                    journal.mark_sell_canceled(rid_, sym_)
                log.warning("WATCH STOP %s: cancelled the unfilled Event Review sell first", h.symbol)
            value = acct.positions.get(h.symbol, d.qty * d.current_price)
            plan = OrderPlan(symbol=h.symbol, side="sell", notional=value,
                             current_weight=value / acct.equity if acct.equity > 0 else 0.0,
                             target_weight=0.0, close_position=True)
            log.warning("WATCH STOP %s: %.1f%% vs entry $%.2f (now $%.2f) -> exit", h.symbol, h.pnl_pct * 100,
                        h.entry, h.price)
            sub = broker.submit(plan)
            journal.order(run_id, plan, sub)
            if h.symbol not in active:
                journal.record_stop(h.symbol, today, h.entry, h.price, h.pnl_pct,
                                    gate.cooldown_until(today, s.risk.stop_cooldown_days), run_id, mode)
            sold.append(h.symbol)
        notes = [f"stops exited: {', '.join(sold)}" if sold else "no stop exits"]
        if s.features.event_review:
            from bot import events

            try:
                notes += events.run_event_reviews(s, mode, broker, journal, run_id, broker.get_account(),
                                                  llm_factory, now_fn or events.now_et,
                                                  set(sold) | open_sells, filings_fn, deadline=agent_deadline)
            except Exception as e:  # noqa: BLE001 - stops already ran; an Event Review error is only a note
                log.exception("Event review failed")
                notes.append(f"event review error: {e!r}"[:300])
        if s.features.intraday_buys:
            from bot import events, intraday_buy

            try:
                notes += intraday_buy.run_intraday_buys(s, mode, broker, journal, run_id, broker.get_account(),
                                                        llm_factory, now_fn or events.now_et, filings_fn, closes_fn,
                                                        deadline=agent_deadline)
            except Exception as e:  # noqa: BLE001 - an intraday-buy error is only a note
                log.exception("Intraday buys failed")
                notes.append(f"intraday buy error: {e!r}"[:300])
        journal.finish_run(run_id, "ok", "; ".join(notes), *tokens())
        return 0
    except Exception as e:  # noqa: BLE001 - logged and journaled; the next watch retries
        log.exception("Watch failed")
        journal.finish_run(run_id, "error", repr(e)[:500], *tokens())
        return 1
