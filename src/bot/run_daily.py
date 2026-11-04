"""Daily pipeline.

  analyze  (after the close): data -> Kronos -> agent team -> skfolio targets -> journal + report
  execute  (after the open):  latest targets -> order plan -> risk gate -> (approval) -> submit -> fills
  watch    (every 30 min, market hours): refresh orders, intraday equity, existing stops (bot/watch.py)

Usage:
  uv run python -m bot.run_daily --phase analyze --mode dry-run
  uv run python -m bot.run_daily --profile smallcap --phase analyze --mode dry-run   # profile B
  uv run python -m bot.run_daily --phase all --mode dry-run --no-agents     # quant baseline, no LLM cost
  uv run python -m bot.run_daily --phase execute --mode paper
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time

import pandas as pd

from bot.config import PROFILES, Settings, load_settings
from bot.data.features import market_regime_snapshot, technical_snapshot
from bot.data.market import MarketData, latest_bar_age_days
from bot.journal.report import write_report
from bot.journal.store import Journal
from bot.portfolio.costs import adv_shares, daily_sigma, normalize_quote, spread_guard
from bot.portfolio.optimizer import target_weights
from bot.portfolio.orders import avg_dollar_volume, cap_adv, shape_whole_share
from bot.portfolio.rebalance import plan_orders
from bot.risk import gate

log = logging.getLogger("bot")
PROFILE_LABEL = {"default": "large-cap", "smallcap": "small-cap"}
TEAM_AGENTS = ["regime", "technical", "fundamentals", "news", "bull", "bear", "portfolio_manager",
               "second_pm", "risk_reviewer"]

# Screener candidate fields passed to the technical analyst (profile B), renamed for the prompt.
SCREENER_FEATURES = {"rank": "rank", "score": "score", "mom": "momentum_12_1", "mom_window": "momentum_window",
                     "ret_3m": "ret_3m", "vol_surge": "volume_surge_log_5d_vs_50d",
                     "prox_high": "close_vs_52w_high"}


def make_llm(s: Settings):
    """The agents' LLM for this profile's backend (default claude-cli: your Claude subscription via the claude CLI, no API key)."""
    from bot.agents.llm import LLM, find_claude_cli

    if s.agents.backend == "claude-cli":
        return LLM(s.agents.max_daily_output_tokens, cli_path=find_claude_cli(s.agents.cli_path))
    if s.agents.backend == "api":
        return LLM(s.agents.max_daily_output_tokens)
    raise ValueError(f"unknown agents.backend {s.agents.backend!r}")


def make_broker(mode: str, s: Settings | None = None, init_book: bool = False):
    if mode == "control":
        # Frozen v1 control: a persistent simulated book. Alpaca is used for market data and the
        # clock only (the live profile's paper keys); no order ever reaches a broker.
        from bot.broker import alpaca_broker
        from bot.broker.book import BookBroker

        if s is None:
            raise ValueError("control mode needs settings")
        data = alpaca_broker.AlpacaBroker(paper=True, api_key_env=s.broker.api_key_env,
                                          secret_key_env=s.broker.secret_key_env)
        return BookBroker(s.paths.resolve(s.control.book_path), data.latest_quote, data.is_market_open,
                          starting_cash=s.control.starting_cash, init=init_book, owner=s.profile)
    if mode == "dry-run":
        from bot.broker.sim import DryRunBroker

        # With record_quotes on, the sim quotes last close +/- half of default_spread_pct.
        sim = s.execution.default_spread_pct if s is not None and s.execution.record_quotes else None
        return DryRunBroker(sim_spread_pct=sim)
    from bot.broker.alpaca_broker import AlpacaBroker

    if s is None:
        return AlpacaBroker(paper=(mode == "paper"))
    return AlpacaBroker(paper=(mode == "paper"), api_key_env=s.broker.api_key_env,
                        secret_key_env=s.broker.secret_key_env)


def load_screener(s: Settings, as_of: dt.date) -> dict | None:
    """Newest fresh screener file (parsed), or None (missing/stale/empty => run blocked)."""
    from bot.screen.universe import load_universe

    data = load_universe(s.paths.resolve(s.universe.screener_dir), as_of, s.universe.max_age_days)
    return data if data and data.get("symbols") else None


def screener_symbols(data: dict) -> list[str]:
    return list(dict.fromkeys(str(x).upper().strip() for x in data["symbols"]))


def screener_universe(s: Settings, as_of: dt.date) -> list[str] | None:
    """Symbols from the newest fresh screener file, or None (missing/stale/empty => run blocked)."""
    data = load_screener(s, as_of)
    return screener_symbols(data) if data else None


def screener_features(data: dict | None) -> dict[str, dict]:
    """{symbol: {rank, score, momentum_12_1, ret_3m, volume_surge..., close_vs_52w_high}} from the
    screener file's candidates (missing fields are left out)."""
    out: dict[str, dict] = {}
    for c in (data or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("symbol"):
            continue
        f = {}
        for k, name in SCREENER_FEATURES.items():
            v = c.get(k)
            if v is None or (isinstance(v, float) and v != v):
                continue
            f[name] = round(v, 4) if isinstance(v, float) else v
        out[str(c["symbol"]).upper().strip()] = f
    return out


def needs_market_refs(s: Settings) -> bool:
    return (s.risk.max_adv_participation is not None or s.execution.order_style in ("auto", "limit")
            or s.execution.record_quotes or s.features.morning_check)


def save_market_refs(journal: Journal, run_id: int, bars: dict[str, pd.DataFrame]) -> None:
    """20-day ADV ($ and shares), daily sigma and last close per symbol, stored with the targets so
    execute (order caps) and reconcile (cost model) need no refetch."""
    rows = []
    for sym, b in sorted(bars.items()):
        adv = avg_dollar_volume(b)
        close = float(b["close"].iloc[-1]) if len(b) else None
        rows.append((run_id, sym, "adv", adv, json.dumps({"adv20": adv, "close": close,
                                                          "adv_shares20": adv_shares(b),
                                                          "sigma_daily": daily_sigma(b)})))
    journal.conn.executemany("INSERT INTO signals(run_id, symbol, source, value, extra) VALUES (?,?,?,?,?)", rows)
    journal.conn.commit()


def load_cost_refs(journal: Journal, run_id: int) -> dict[str, tuple[float | None, float | None]]:
    """{symbol: (adv_shares20, sigma_daily)} from an analyze run's adv rows."""
    rows = journal.conn.execute("SELECT symbol, extra FROM signals WHERE run_id=? AND source='adv'",
                                (run_id,)).fetchall()
    out = {}
    for sym, extra in rows:
        e = json.loads(extra or "{}")
        out[sym] = (e.get("adv_shares20"), e.get("sigma_daily"))
    return out


def radar_shadow(s: Settings, journal: Journal, run_id: int, symbols: list[str]) -> int:
    """Shadow mode: journal radar features for the tradable symbols. Never passed to the agents and
    never allowed to fail analyze (any error, including a missing module, is only a warning)."""
    try:
        from bot.radar.features import compute_features

        freeze = dt.datetime.now(dt.timezone.utc)
        feats = compute_features(s.paths.resolve(s.radar.db_path), list(symbols), freeze)
        rows = {sym: (float(f.get("mention_z") or 0.0), {**f, "freeze_ts": freeze.isoformat(timespec="seconds")})
                for sym, f in (feats or {}).items()}
        journal.signals(run_id, "radar", rows)
        log.info("Radar (shadow): %d symbols journaled, %d with mention_z", len(rows),
                 sum(1 for f in (feats or {}).values() if f.get("mention_z") is not None))
        return len(rows)
    except Exception as e:  # noqa: BLE001 - shadow mode must never block analyze
        log.warning("Radar shadow features failed (ignored): %r", e)
        return 0


def load_market_refs(journal: Journal, run_id: int) -> tuple[dict[str, float], dict[str, float]]:
    rows = journal.conn.execute("SELECT symbol, extra FROM signals WHERE run_id=? AND source='adv'",
                                (run_id,)).fetchall()
    adv, close = {}, {}
    for sym, extra in rows:
        e = json.loads(extra or "{}")
        if e.get("adv20"):
            adv[sym] = float(e["adv20"])
        if e.get("close"):
            close[sym] = float(e["close"])
    return adv, close


def kronos_relative(forecasts: dict) -> dict[str, dict]:
    """Kronos raw forecasts have a strong common bias; use them relative to the universe mean."""
    if not forecasts:
        return {}
    mean = sum(f.exp_return for f in forecasts.values()) / len(forecasts)
    return {s: {"raw": round(f.exp_return, 4), "relative": round(f.exp_return - mean, 4),
                "dispersion": round(f.dispersion, 4), "horizon_days": f.horizon_days}
            for s, f in forecasts.items()}


# ----------------------------------------------------------------------------- analyze
def analyze(s: Settings, mode: str, use_agents: bool, use_kronos: bool) -> int:
    journal = Journal(s.paths.resolve(s.paths.journal_db))
    from bot.agents.llm import prompt_version

    run_id = journal.start_run("analyze", mode, prompt_version(s.agents.prompt_dir) if use_agents else "no-agents")
    llm = None
    try:
        broker = make_broker(mode, s)
        acct = broker.get_account()
        today = dt.date.today()
        screened = s.universe.source == "screener"
        screen_feats: dict[str, dict] = {}
        if screened:
            sdata = load_screener(s, today)
            universe = screener_symbols(sdata) if sdata else None
            screen_feats = screener_features(sdata)
            if universe is None:
                msg = (f"screener universe missing or stale (none <= {s.universe.max_age_days} days old in "
                       f"{s.universe.screener_dir})")
                journal.finish_run(run_id, "blocked", msg)
                log.error("BLOCKED before analysis: %s", msg)
                return 2
            log.info("Screener universe: %d symbols", len(universe))
        else:
            universe = list(s.universe.symbols)
        # Held names outside the screener list are priced too, then get target 0 (sold).
        extra_held = sorted(x for x in acct.positions if x not in universe) if screened else []
        cooldown = journal.active_cooldowns(today, mode) if s.risk.stop_loss_pct is not None else {}
        if cooldown:
            log.info("Stop cooldown, forced to target 0: %s", cooldown)

        md = MarketData(s.paths.resolve(s.paths.cache_dir))
        fetch = s.all_symbols if not screened else list(dict.fromkeys([*universe, *extra_held, s.universe.benchmark]))
        bars = md.daily_bars(fetch, s.data.history_days)
        age = latest_bar_age_days(bars)
        pre = gate.pre_trade_checks(s.risk, acct, journal.peak_equity(), age, s.data.max_staleness_days)
        if not pre.ok:
            journal.finish_run(run_id, "blocked", "; ".join(pre.reasons))
            log.error("BLOCKED before analysis: %s", pre.reasons)
            return 2

        tradable_syms = [x for x in universe if x in bars and x not in cooldown]
        if not tradable_syms:
            journal.finish_run(run_id, "blocked", "no universe symbols with price data")
            log.error("BLOCKED: no universe symbols with price data")
            return 2
        closes = md.closes(bars)
        if needs_market_refs(s):
            save_market_refs(journal, run_id, bars)
        if s.radar.enabled:
            radar_shadow(s, journal, run_id, tradable_syms)   # journaled only; agents never see it

        kronos = {}
        if use_kronos and s.kronos.enabled:
            from bot.signals.kronos_signal import KronosSignal

            log.info("Running Kronos on %d symbols...", len(tradable_syms))
            kronos = kronos_relative(KronosSignal(s.kronos).forecast({k: bars[k] for k in tradable_syms}))
            journal.signals(run_id, "kronos", {k: (v["relative"], v) for k, v in kronos.items()})

        team_out, regime, views, min_cash = None, None, [], s.portfolio.min_cash_weight
        if use_agents:
            from bot.agents.llm import LLM
            from bot.agents.team import AgentTeam, TeamInputs

            log.info("Fetching fundamentals and news...")
            tech = {x: technical_snapshot(bars[x]) | ({"kronos": kronos[x]} if x in kronos else {})
                    | ({"screener": screen_feats[x]} if x in screen_feats else {})
                    for x in tradable_syms}
            fund = {x: md.fundamentals(x) for x in tradable_syms}
            news = {x: md.news(x, s.data.news_per_symbol) for x in tradable_syms}
            inputs = TeamInputs(as_of=str(closes.index[-1].date()), universe=tradable_syms,
                                regime=market_regime_snapshot(closes, s.universe.benchmark),
                                technical=tech, fundamentals=fund, news=news)
            journal.agent_report(run_id, "inputs", inputs.__dict__)
            llm = make_llm(s)
            log.info("Running agent team...")
            from bot.memory.block import blocks_for_run

            memory, merr = blocks_for_run(s, journal, TEAM_AGENTS, tradable_syms)
            if merr:
                journal.agent_report(run_id, "memory_error", {"error": merr})
            if memory:
                journal.agent_report(run_id, "memory_block", memory)
            researcher = None
            if s.features.research:
                from bot.agents.research import Researcher

                researcher = Researcher(llm, s.research, s.agents.prompt_dir,
                                        cache_dir=s.paths.resolve(s.paths.cache_dir))
            review_ctx = None
            if s.features.reviewers:
                eq = acct.equity if acct.equity > 0 else 1.0
                review_ctx = {"weights": {k: round(v / eq, 4) for k, v in acct.positions.items()},
                              "risk_limits": s.risk.model_dump()}
            team_out = AgentTeam(llm, s.agents, s.portfolio, memory=memory, researcher=researcher,
                                 research_cfg=s.research,
                                 reviewers_cfg=s.reviewers if s.features.reviewers else None,
                                 context=review_ctx).run(inputs)
            for name, rep in team_out.reports.items():
                journal.agent_report(run_id, name, rep.model_dump() if hasattr(rep, "model_dump") else rep)
            journal.views(run_id, team_out.views)
            views, regime = team_out.views, team_out.reports.get("regime")
            min_cash = team_out.cash_weight

        pcfg = s.portfolio.model_copy(update={"min_cash_weight": max(min_cash, 1 - s.risk.max_gross_exposure)})
        targets = target_weights(closes[tradable_syms], views, pcfg, s.agents.horizon_days)
        forced = [*extra_held, *(c for c in cooldown if c in universe or c in acct.positions)]
        targets = gate.force_zero(targets, forced)
        chk = gate.check_targets(s.risk, targets, set(tradable_syms))
        journal.target_weights(run_id, targets)
        cost = llm.cost_usd if llm else 0.0
        pnl = journal.pnl_summary() if s.execution.record_quotes else None
        rpt = write_report(s.paths.resolve(s.paths.reports_dir), run_id, mode, team_out, targets, kronos,
                           chk.reasons, cost, regime, pnl=pnl)
        status = "ok" if chk.ok else "blocked"
        journal.finish_run(run_id, status, "; ".join(chk.reasons),
                           llm.input_tokens if llm else 0, llm.output_tokens if llm else 0, cost)
        log.info("Analyze %s. Invested %.1f%% across %d names. Report: %s  LLM cost $%.2f",
                 status, targets.sum() * 100, int((targets > 0).sum()), rpt, cost)
        return 0 if chk.ok else 2
    except Exception as e:
        log.exception("Analyze failed")
        journal.finish_run(run_id, "error", repr(e)[:500],
                           llm.input_tokens if llm else 0, llm.output_tokens if llm else 0,
                           llm.cost_usd if llm else 0.0)
        return 1


# ----------------------------------------------------------------------------- execute
def execute(s: Settings, mode: str, approve: bool) -> int:
    journal = Journal(s.paths.resolve(s.paths.journal_db))
    run_id = journal.start_run("execute", mode)
    try:
        broker = make_broker(mode, s)
        acct = broker.get_account()
        pre = gate.pre_trade_checks(s.risk, acct, journal.peak_equity(), 0, s.data.max_staleness_days)
        if not pre.ok:
            journal.finish_run(run_id, "blocked", "; ".join(pre.reasons))
            log.error("BLOCKED: %s", pre.reasons)
            return 2
        latest = journal.latest_targets(mode, max_age_days=3)
        if latest is None:
            journal.finish_run(run_id, "blocked", "no recent successful analyze run")
            log.error("No recent targets; run --phase analyze first")
            return 2
        src_run, targets = latest
        # Re-check the stored targets against today's caps (the journal DB or the config may have
        # changed since analyze). The symbol allowlist was checked at analyze time.
        tchk = gate.check_targets(s.risk, targets, set(targets.index))
        if not tchk.ok:
            journal.finish_run(run_id, "blocked", "; ".join(tchk.reasons))
            log.error("BLOCKED (stored targets): %s", tchk.reasons)
            return 2
        if not broker.is_market_open():
            journal.finish_run(run_id, "blocked", "market closed")
            log.error("Market is closed; market orders are only sent during regular hours")
            return 2
        targets, stopped = apply_stops(s, journal, acct, targets, run_id, mode)

        auto = s.execution.order_style == "auto"
        info: dict[str, dict[str, bool]] = {}
        if auto:  # non-fractionable names trade in whole shares; only non-tradable ones are zeroed
            info = broker.assets(list(targets[targets > 0].index))
            not_ok = [k for k, v in info.items() if not v["tradable"]]
        else:
            tradable = broker.tradable(list(targets[targets > 0].index))
            not_ok = [k for k, v in tradable.items() if not v]
        if not_ok:
            log.warning("Not tradable%s on %s, zeroing: %s", "" if auto else "/fractionable", broker.name, not_ok)
            targets = targets.drop(not_ok)

        orders, _eff = plan_orders(targets, acct, s.risk, stopped)
        adjustments: list[str] = []
        qcache: dict[str, dict | None] = {}

        def quote(sym: str) -> dict | None:   # one fetch per symbol per run; bad/one-sided quotes -> None
            if sym not in qcache:
                qcache[sym] = normalize_quote(broker.latest_quote(sym))
            return qcache[sym]

        if needs_market_refs(s):   # seed sim quotes before ANY quote lookup (quote() caches)
            adv, last_close = load_market_refs(journal, src_run)
            if s.execution.record_quotes and hasattr(broker, "seed_prices"):
                broker.seed_prices(last_close)   # dry-run sim quotes only

        if s.features.morning_check:   # Morning Check: may only cancel or shrink buys
            orders, why = morning_check_orders(s, journal, run_id, src_run, orders, quote)
            adjustments += why
            for o in orders:   # the agent call can take minutes: price limits and spreads from fresh quotes
                if o.side == "buy":
                    qcache.pop(o.symbol, None)

        if needs_market_refs(s):
            if s.risk.max_adv_participation is not None:
                orders, why = cap_adv(orders, adv, s.risk.max_adv_participation, s.risk.min_order_notional)
                adjustments += why
            if auto:
                need = [o.symbol for o in orders if not o.close_position]
                missing = [x for x in need if x not in info]
                if missing:
                    info |= broker.assets(missing)
                non_frac = {x for x in need if not info[x]["fractionable"]}
                limit_frac = s.execution.limit_fractional_buys
                priced = non_frac | ({o.symbol for o in orders if o.side == "buy" and not o.close_position}
                                     if limit_frac else set())
                quotes = {x: quote(x) for x in sorted(priced)}
                orders, why = shape_whole_share(orders, non_frac, acct, quotes, last_close,
                                                s.execution.limit_buffer_pct, s.risk.min_order_notional,
                                                limit_fractional_buys=limit_frac)
                adjustments += why
        record = s.execution.record_quotes
        refs = load_cost_refs(journal, src_run) if record else {}
        feed = getattr(broker, "quote_feed", None)

        def journal_quote(o, action: str, oid: str | None = None) -> None:
            if record:
                a, sg = refs.get(o.symbol, (None, None))
                journal.record_order_quote(run_id, src_run, o, action, oid, qcache.get(o.symbol), feed, a, sg)

        if record or s.risk.max_spread_pct is not None:
            planned = list(orders)
            for o in planned:
                q = quote(o.symbol)
                log.info("  quote %-5s %s", o.symbol,
                         f"bid ${q['bid']:,.4f} ask ${q['ask']:,.4f} spread {q['spread_pct']:.2%} ({feed}, {q['ts']})"
                         if q else f"no usable quote ({feed})")
            orders, why, skipped = spread_guard(orders, {o.symbol: quote(o.symbol) for o in planned},
                                                s.risk.max_spread_pct)
            adjustments += why
            for o in planned:
                if o.symbol in skipped:
                    journal_quote(o, "skipped: " + skipped[o.symbol])
        for w in adjustments:
            log.warning("Order adjusted: %s", w)
        chk = gate.check_orders(s.risk, orders, acct, stopped)
        if not chk.ok:
            for o in orders:
                journal.order(run_id, o)
                journal_quote(o, "blocked by risk gate")
            journal.finish_run(run_id, "blocked", "; ".join(chk.reasons))
            log.error("Order plan BLOCKED by risk gate: %s", chk.reasons)
            return 2

        log.info("Order plan from analyze run %d (%d orders):", src_run, len(orders))
        for o in orders:
            log.info("  %-4s %-5s $%10.2f  %5.1f%% -> %5.1f%%%s", o.side.upper(), o.symbol, o.notional,
                     o.current_weight * 100, o.target_weight * 100,
                     "  (close)" if o.close_position else
                     (f"  ({o.qty:.0f} sh limit ${o.limit_price:,.2f})" if o.qty is not None else ""))
        adj_note = f"; {len(adjustments)} order adjustments" if adjustments else ""
        if not orders:
            journal.finish_run(run_id, "ok", "no trades needed" + adj_note)
            return 0
        if approve:
            ans = input(f"Submit {len(orders)} orders to {broker.name} ({mode})? Type 'yes': ").strip().lower()
            if ans != "yes":
                for o in orders:
                    journal_quote(o, "not approved")
                journal.finish_run(run_id, "blocked", "not approved by user")
                log.warning("Not approved; nothing submitted")
                return 2

        submitted = []
        for o in orders:
            sub = broker.submit(o)
            journal.order(run_id, o, sub)
            journal_quote(o, sub.status, sub.broker_order_id)
            submitted.append(sub)
            if o.side == "sell" and sub.broker_order_id:
                time.sleep(0.5)
        # Poll fills briefly; anything still open is reconciled on the next run.
        deadline = time.time() + (60 if mode != "dry-run" else 0)
        pending = [x.broker_order_id for x in submitted if x.broker_order_id]
        while pending and time.time() < deadline:
            time.sleep(5)
            still = []
            for oid in pending:
                st = broker.order_status(oid)
                journal.update_fill(oid, st["status"], st["filled_qty"], st["filled_avg_price"])
                if st["status"] not in ("filled", "canceled", "rejected", "expired"):
                    still.append(oid)
            pending = still
        rejected = [x for x in submitted if x.status == "rejected"]
        if record:
            journal.update_fill_costs(s.execution.default_spread_pct)
        if mode != "dry-run":  # simulated equity must not overwrite the paper/live equity curve
            acct2 = broker.get_account()
            journal.record_equity(dt.date.today(), acct2.equity, acct2.cash, None)
        status = "ok" if not rejected else "error"
        journal.finish_run(run_id, status,
                           f"{len(submitted)} submitted, {len(rejected)} rejected, {len(pending)} pending" + adj_note)
        log.info("Execute %s: %d submitted, %d rejected, %d still pending", status, len(submitted), len(rejected), len(pending))
        return 0 if not rejected else 1
    except Exception as e:
        log.exception("Execute failed")
        journal.finish_run(run_id, "error", repr(e)[:500])
        return 1


def morning_check_orders(s: Settings, journal: Journal, run_id: int, src_run: int, orders: list,
                         quote) -> tuple[list, list[str]]:
    """Deterministic gap guard on planned buys (always, when the feature is on), then the Morning
    Check agent, which may only cancel or shrink buys. Any agent failure (including a subscription
    usage limit) is journaled and the gap-guarded plan proceeds."""
    from bot.agents.morning import apply_morning, run_morning_check
    from bot.portfolio.gap import gap_guard

    _adv, prev = load_market_refs(journal, src_run)
    sigma = {k: v[1] for k, v in load_cost_refs(journal, src_run).items() if v[1]}
    mids = {o.symbol: (quote(o.symbol) or {}).get("mid") for o in orders if o.side == "buy"}
    orders, notes = gap_guard(orders, mids, prev, sigma, s.morning_check.gap_sigma_k)
    buys = [o for o in orders if o.side == "buy"]
    if not buys:
        journal.agent_report(run_id, "morning_check_applied", notes)
        return orders, notes
    try:
        ctx = morning_context(s, journal, src_run, buys, mids, prev, sigma)
        journal.agent_report(run_id, "morning_check_input", ctx)
        from bot.memory.block import blocks_for_run

        mem, merr = blocks_for_run(s, journal, ["morning_check"], [o.symbol for o in buys])
        if merr:
            journal.agent_report(run_id, "memory_error", {"error": merr})
        if mem:
            journal.agent_report(run_id, "memory_block", mem)
        check = run_morning_check(make_llm(s), s.morning_check, s.agents.prompt_dir,
                                  dt.datetime.now().isoformat(timespec="minutes"), ctx,
                                  memory=(mem or {}).get("morning_check", ""))
        journal.agent_report(run_id, "morning_check", check.model_dump())
        orders, why = apply_morning(orders, check, s.risk.min_order_notional)
        notes = notes + why
    except Exception as e:  # noqa: BLE001 - fall back to the gap-guarded plan
        log.warning("Morning check unavailable (%s); proceeding with the gap-guarded plan", e)
        journal.agent_report(run_id, "morning_check", {"error": f"{type(e).__name__}: {e}"[:500]})
        notes = notes + [f"morning check unavailable ({type(e).__name__}); gap guard only"]
    journal.agent_report(run_id, "morning_check_applied", notes)   # gap-guard drops + agent effects
    return orders, notes


def morning_context(s: Settings, journal: Journal, src_run: int, buys: list, mids: dict, prev: dict,
                    sigma: dict) -> dict:
    """Per planned buy: the analyze rationale, the gap, and what appeared since the analyze run."""
    row = journal.conn.execute("SELECT started_at FROM runs WHERE id=?", (src_run,)).fetchone()
    since = dt.datetime.fromisoformat(row[0]).astimezone() if row else dt.datetime.now().astimezone()
    views = {r[0]: r[1] for r in journal.conn.execute(
        "SELECT symbol, rationale FROM views WHERE run_id=?", (src_run,)).fetchall()}
    syms = [o.symbol for o in buys]
    n = s.morning_check.max_items_per_symbol
    radar: dict[str, list[dict]] = {}
    rp = s.paths.resolve(s.radar.db_path)
    if rp.exists():
        try:
            from bot.radar.store import RadarStore

            radar = RadarStore(rp).posts_since(syms, since, n)
        except Exception as e:  # noqa: BLE001 - extra evidence only
            log.warning("radar posts unavailable: %s", e)
    md = MarketData(s.paths.resolve(s.paths.cache_dir))
    out = []
    for o in buys:
        try:
            news = [h for h in (getattr(md, "news", lambda *_: [])(o.symbol, n) or [])
                    if str(h.get("date", "")) >= str(since.date())]
        except Exception:  # noqa: BLE001
            news = []
        p, m = prev.get(o.symbol), mids.get(o.symbol)
        out.append({"symbol": o.symbol, "planned_notional": round(o.notional, 2),
                    "view_rationale": views.get(o.symbol, "(no explicit view: optimizer prior)"),
                    "gap_pct": round(m / p - 1, 4) if p and m else None, "daily_sigma": sigma.get(o.symbol),
                    "overnight_headlines": news, "radar_posts": radar.get(o.symbol, [])})
    return {"analysis_time": since.isoformat(timespec="minutes"), "src_run": src_run, "buys": out}


def apply_stops(s: Settings, journal: Journal, acct, targets: pd.Series, run_id: int,
                mode: str) -> tuple[pd.Series, frozenset[str]]:
    """Per-position stop: positions at <= -stop_loss_pct vs avg entry, and names still in cooldown,
    are forced to target 0. New stops are journaled with their cooldown end.
    Returns (targets, held symbols that must be exited in full this run)."""
    if s.risk.stop_loss_pct is None:
        return targets, frozenset()
    today = dt.date.today()
    active = journal.active_cooldowns(today, mode)
    forced = []
    for h in gate.stop_hits(s.risk, acct):
        log.warning("STOP %s: %.1f%% vs entry $%.2f (now $%.2f) -> exit", h.symbol, h.pnl_pct * 100, h.entry, h.price)
        if h.symbol not in active:
            until = gate.cooldown_until(today, s.risk.stop_cooldown_days)
            journal.record_stop(h.symbol, today, h.entry, h.price, h.pnl_pct, until, run_id, mode)
        forced.append(h.symbol)
    forced += [c for c in active if c in targets.index or c in acct.positions]
    if forced:
        log.info("Forced to target 0 (stop/cooldown): %s", sorted(set(forced)))
    return gate.force_zero(targets, forced), frozenset(x for x in forced if x in acct.positions)


def reconcile(s: Settings, mode: str, init_book: bool = False) -> None:
    """Update fills for orders left pending by earlier runs and record today's equity.
    Control mode also marks the simulated book to the latest quote mid (falls back to the last price)."""
    if mode == "dry-run":
        return
    journal = Journal(s.paths.resolve(s.paths.journal_db))
    broker = make_broker(mode, s, init_book=init_book) if mode == "control" else make_broker(mode, s)
    if mode == "control":
        mids = {}
        for sym in list(broker.get_account().positions):
            q = normalize_quote(broker.latest_quote(sym))
            if q:
                mids[sym] = (q["bid"] + q["ask"]) / 2
        broker.mark(mids)
    for oid in journal.open_order_ids():
        st = broker.order_status(oid)
        journal.update_fill(oid, st["status"], st["filled_qty"], st["filled_avg_price"])
    if s.execution.record_quotes:
        n = journal.update_fill_costs(s.execution.default_spread_pct)
        log.info("Modeled costs updated for %d fills", n)
    acct = broker.get_account()
    journal.record_equity(dt.date.today(), acct.equity, acct.cash, None)
    memory_after_reconcile(s, journal)


def memory_after_reconcile(s: Settings, journal: Journal) -> dict:
    """Phase 2: score matured views (no LLM). Never raises: a failure is logged and journaled as a
    'memory' run with status error, and reconcile carries on."""
    out: dict = {"scored": 0, "observations": 0, "error": None}
    if not s.features.memory:
        return out
    try:
        from bot.memory.outcomes import _ensure, agent_calls, pending_symbols, score_pending

        _ensure(journal)
        mark = journal.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM outcomes_v2").fetchone()[0]
        from bot.memory.events_score import pending_event_symbols, score_events

        syms, start = pending_symbols(journal)
        esyms, estart = pending_event_symbols(journal)
        syms = syms | esyms
        start = min(x for x in (start, estart) if x is not None) if (start or estart) else None
        if syms:
            md = MarketData(s.paths.resolve(s.paths.cache_dir))
            days = (dt.date.today() - start).days + 30
            bars = md.daily_bars(sorted(syms | {s.universe.benchmark}), days)
            out["scored"] = score_pending(journal, bars, s.universe.benchmark)
            out["events_scored"] = score_events(journal, bars, s.universe.benchmark)
        if s.memory.dir:
            from bot.memory import memory_root
            from bot.memory.files import MemoryFiles
            from bot.memory.reviewer import run_post_trade
            from bot.memory.scorecards import build_scorecards, render_scorecards

            files = MemoryFiles(memory_root(s), title=f"Lessons ({PROFILE_LABEL.get(s.profile, s.profile)})")
            cards = build_scorecards(journal)
            files.write_scorecard(render_scorecards(cards, PROFILE_LABEL.get(s.profile, s.profile)))
            if out["scored"]:
                rows = journal.query("SELECT run_id, symbol, as_of, excess_univ, excess_bench FROM outcomes_v2 "
                                     "WHERE rowid > ?", (mark,)).to_dict("records")
                calls: dict[int, list[dict]] = {}
                for r in rows:
                    rid = int(r["run_id"])
                    if rid not in calls:
                        calls[rid] = agent_calls(journal, rid)
                    r["calls"] = [{k: c[k] for k in ("agent", "direction", "confidence")}
                                  for c in calls[rid] if c["symbol"] == r["symbol"]]
                out["observations"] = run_post_trade(make_llm(s), s, journal, files, rows, cards,
                                                     str(dt.date.today()))
    except Exception as e:  # noqa: BLE001 - memory must never break reconcile
        log.warning("memory step failed: %s", e)
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        rid = journal.start_run("memory", "reconcile")
        journal.finish_run(rid, "error", out["error"])
    return out


def curate_phase(s: Settings, mode: str) -> int:
    """Weekly (Sunday): the Curator turns this week's staged observations into lessons."""
    journal = Journal(s.paths.resolve(s.paths.journal_db))
    run_id = journal.start_run("curate", mode)
    try:
        if not s.features.memory or not s.memory.dir:
            journal.finish_run(run_id, "skipped", "memory is off")
            return 0
        from bot.memory import memory_root
        from bot.memory.curator import curate
        from bot.memory.files import MemoryFiles
        from bot.memory.scorecards import build_scorecards

        label = PROFILE_LABEL.get(s.profile, s.profile)
        files = MemoryFiles(memory_root(s), title=f"Lessons ({label})")
        if not files.read_staging(None):
            journal.finish_run(run_id, "ok", "nothing to curate")
            return 0
        out = curate(make_llm(s), s, files, build_scorecards(journal), str(dt.date.today()))
        journal.finish_run(run_id, "ok", json.dumps(out))
        return 0
    except Exception as e:  # noqa: BLE001 - files are untouched when the Curator call fails
        log.exception("Curate failed")
        journal.finish_run(run_id, "error", repr(e)[:500])
        return 1


def watch_should_email(journal: Journal, before: int) -> bool:
    """A watch emails when it submitted exits, or on its first error of the day. A manual kill switch
    (blocked) or a quiet watch stays silent, so a halted bot does not email every 30 minutes."""
    runs = journal.query("SELECT id, status, started_at FROM runs WHERE id > ? AND phase='watch'", (before,))
    if runs.empty:
        return False
    ids = tuple(int(x) for x in runs["id"])
    n = journal.conn.execute(f"SELECT COUNT(*) FROM orders WHERE run_id IN ({','.join('?' * len(ids))})",
                             ids).fetchone()[0]
    if n:
        return True
    if "error" in set(runs["status"]):
        prior = journal.conn.execute(
            "SELECT COUNT(*) FROM runs WHERE phase='watch' AND status='error' AND id <= ? AND started_at >= ?",
            (before, str(dt.date.today()))).fetchone()[0]
        return prior == 0
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agent-team trading bot daily run")
    ap.add_argument("--phase", choices=["analyze", "execute", "all", "reconcile", "watch", "curate"], default=None)
    ap.add_argument("--mode", choices=["dry-run", "paper", "live", "control"], default=None,
                    help="defaults to execution.mode in settings.yaml")
    ap.add_argument("--no-agents", action="store_true", help="skip LLM agents (quant baseline only)")
    ap.add_argument("--no-kronos", action="store_true")
    ap.add_argument("--approve", action="store_true", help="ask for confirmation before submitting orders")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="default",
                    help="default = config/settings.yaml, smallcap = config/settings_smallcap.yaml")
    ap.add_argument("--config", default=None, help="settings file; overrides --profile")
    ap.add_argument("--init-book", action="store_true",
                    help="control mode, reconcile only: create the simulated book if it does not exist")
    ap.add_argument("--test-email", action="store_true", help="send a test summary email and exit")
    args = ap.parse_args(argv)
    if args.test_email and not args.phase:
        args.phase = "reconcile"
    if not args.phase:
        ap.error("--phase is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "httpx2", "urllib3", "openbb", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = load_settings(args.config, args.profile)
    if not args.config and s.profile != args.profile:
        log.error("Settings file declares profile %r but --profile is %r", s.profile, args.profile)
        return 1
    if s.profile != "default":
        log.info("Profile %s: journal %s, kill switch %s", s.profile, s.paths.journal_db, s.risk.kill_switch_file)
    tag = "trading-bot" if s.profile == "default" else f"trading-bot {s.profile}"
    mode = args.mode or s.execution.mode
    approve = args.approve or s.execution.require_approval or (mode == "live" and not s.execution.live_auto_execute)

    if args.test_email:
        from bot.notify import send_email

        ok = send_email(f"[{tag} {mode}] test email", "If you can read this, run summaries will reach you.")
        print("sent" if ok else "NOT sent: check SMTP_USER / SMTP_PASSWORD in .env and the log above")
        return 0 if ok else 1

    journal = Journal(s.paths.resolve(s.paths.journal_db))
    before = int(journal.query("SELECT COALESCE(MAX(id), 0) AS m FROM runs")["m"].iloc[0])
    rc = 0
    try:
        if args.phase == "reconcile":
            reconcile(s, mode, init_book=args.init_book)
        elif args.phase == "watch":
            from bot import watch

            rc = watch.run(s, mode)
        elif args.phase == "curate":
            rc = curate_phase(s, mode)
        else:
            if args.phase in ("analyze", "all"):
                rc = analyze(s, mode, use_agents=not args.no_agents, use_kronos=not args.no_kronos)
            if args.phase in ("execute", "all") and rc == 0:
                rc = execute(s, mode, approve)
    except Exception:
        log.exception("Unhandled error in %s", args.phase)
        rc = 1
    finally:
        if s.notify.email and mode in s.notify.modes:
            from bot.notify import send_email, summarize_runs

            summary = summarize_runs(journal, before, mode, args.phase, gate.kill_switch_active(s.risk),
                                     realistic_pnl=s.execution.record_quotes)
            if summary and tag != "trading-bot":
                summary = (summary[0].replace("[trading-bot", f"[{tag}", 1), summary[1])
            if rc and summary is None:
                summary = (f"[{tag} {mode}] {args.phase} ERROR",
                           f"The {args.phase} run crashed before writing to the journal. See data\\logs\\.")
            if args.phase == "watch":   # every 30 min: email exits and the day's first error only
                if summary and watch_should_email(journal, before):
                    send_email(*summary)
            elif summary and (rc or args.phase in s.notify.phases or args.phase == "curate"):
                send_email(*summary)
    return rc


if __name__ == "__main__":
    sys.exit(main())
