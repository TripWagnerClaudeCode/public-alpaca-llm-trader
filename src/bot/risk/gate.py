"""Hard risk limits. Plain deterministic code: the LLMs cannot override anything here.

Every check fails closed: any violation blocks the whole run's orders.
Severe conditions (daily loss / drawdown) also write the kill-switch file, which
blocks all future runs until a human deletes it.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import pandas as pd

from bot.config import ROOT, RiskCfg
from bot.models import AccountState, OrderPlan

TOL = 1e-4


@dataclass
class GateResult:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    kill_switch_tripped: bool = False

    def fail(self, reason: str):
        self.ok = False
        self.reasons.append(reason)


def kill_switch_path(cfg: RiskCfg) -> Path:
    p = Path(cfg.kill_switch_file)
    return p if p.is_absolute() else ROOT / p


def kill_switch_active(cfg: RiskCfg) -> bool:
    return kill_switch_path(cfg).exists() or os.environ.get("BOT_KILL_SWITCH", "").lower() in ("1", "yes", "true")


AUTO_MARKER = "AUTO "


def trip_kill_switch(cfg: RiskCfg, reason: str) -> None:
    """Written by the bot itself on a loss-limit breach. The AUTO marker lets intraday exits keep running."""
    kill_switch_path(cfg).write_text(f"{AUTO_MARKER}{dt.datetime.now().isoformat()} {reason}\n"
                                     "Delete this file to resume trading.\n")


def kill_switch_kind(cfg: RiskCfg) -> Literal["manual", "auto"] | None:
    """"auto": the bot wrote the file on a loss-limit breach (new risk blocked, exits allowed).
    "manual": any other file, or BOT_KILL_SWITCH set (nothing trades, exits included). None: inactive."""
    if os.environ.get("BOT_KILL_SWITCH", "").lower() in ("1", "yes", "true"):
        return "manual"
    p = kill_switch_path(cfg)
    if not p.exists():
        return None
    try:
        first = p.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError:
        return "manual"
    return "auto" if first and first[0].startswith(AUTO_MARKER) else "manual"


def exits_allowed(cfg: RiskCfg) -> bool:
    """Deterministic exits (stops) may run unless a human halted the bot."""
    return kill_switch_kind(cfg) != "manual"


def pre_trade_checks(cfg: RiskCfg, acct: AccountState, peak_equity: float | None,
                     data_age_days: int, max_staleness_days: int) -> GateResult:
    """Run before any agent work, so a halted bot spends no LLM tokens."""
    r = GateResult()
    if kill_switch_active(cfg):
        r.fail(f"kill switch active ({kill_switch_path(cfg).name} present or BOT_KILL_SWITCH set)")
        return r
    if acct.trading_blocked:
        r.fail("broker reports trading_blocked")
    if data_age_days > max_staleness_days:
        r.fail(f"market data stale: latest bar is {data_age_days} days old (max {max_staleness_days})")
    if acct.last_equity > 0:
        day_ret = acct.equity / acct.last_equity - 1
        if day_ret <= -cfg.max_daily_loss_pct:
            msg = f"daily loss {day_ret:.2%} breached limit -{cfg.max_daily_loss_pct:.2%}"
            trip_kill_switch(cfg, msg)
            r.kill_switch_tripped = True
            r.fail(msg)
    if peak_equity and peak_equity > 0:
        dd = acct.equity / peak_equity - 1
        if dd <= -cfg.max_drawdown_pct:
            msg = f"drawdown {dd:.2%} from peak ${peak_equity:,.2f} breached limit -{cfg.max_drawdown_pct:.2%}"
            trip_kill_switch(cfg, msg)
            r.kill_switch_tripped = True
            r.fail(msg)
    return r


def check_targets(cfg: RiskCfg, target: pd.Series, allowlist: set[str]) -> GateResult:
    r = GateResult()
    if target.isna().any():
        r.fail("target weights contain NaN")
        return r
    if (target < -TOL).any():
        r.fail(f"negative (short) weights not allowed: {target[target < -TOL].to_dict()}")
    over = target[target > cfg.max_position_weight + TOL]
    if len(over):
        r.fail(f"position weight cap {cfg.max_position_weight:.0%} exceeded: {over.round(4).to_dict()}")
    gross = float(target.clip(lower=0).sum())
    if gross > cfg.max_gross_exposure + TOL:
        r.fail(f"gross exposure {gross:.2%} exceeds {cfg.max_gross_exposure:.0%}")
    unknown = set(target[target > TOL].index) - allowlist
    if unknown:
        r.fail(f"symbols not in allowlist: {sorted(unknown)}")
    return r


def check_orders(cfg: RiskCfg, orders: list[OrderPlan], acct: AccountState,
                 forced_exits: frozenset[str] = frozenset()) -> GateResult:
    """`forced_exits`: full stop-loss exits, which do not count toward the turnover limit
    (reducing risk must never be blocked by it)."""
    r = GateResult()
    if len(orders) > cfg.max_orders_per_day:
        r.fail(f"{len(orders)} orders exceeds max_orders_per_day={cfg.max_orders_per_day}")
    buys = sum(o.notional for o in orders if o.side == "buy")
    sells = sum(o.notional for o in orders if o.side == "sell")
    # Sells may fill at slightly worse prices; require 1% headroom. No margin is ever used.
    if buys > acct.cash + sells * 0.99 + TOL:
        r.fail(f"buys ${buys:,.2f} exceed cash ${acct.cash:,.2f} + sell proceeds ${sells:,.2f} (no margin allowed)")
    if acct.equity > 0:
        exempt = sum(o.notional for o in orders if o.side == "sell" and o.close_position and o.symbol in forced_exits)
        turnover = (buys + sells - exempt) / acct.equity
        if turnover > cfg.max_daily_turnover + 0.01:
            r.fail(f"turnover {turnover:.2%} exceeds max_daily_turnover {cfg.max_daily_turnover:.0%}")
    for o in orders:
        if o.notional <= 0:
            r.fail(f"non-positive notional for {o.symbol}")
    return r


# ----------------------------------------------------------------------------- per-position stops
@dataclass
class StopHit:
    symbol: str
    entry: float
    price: float
    pnl_pct: float


def stop_hits(cfg: RiskCfg, acct: AccountState) -> list[StopHit]:
    """Held positions whose unrealized P&L vs avg entry is <= -stop_loss_pct. Empty when stops are off."""
    if cfg.stop_loss_pct is None:
        return []
    hits = []
    for sym, d in sorted(acct.details.items()):
        if sym in acct.positions and d.avg_entry_price > 0 and d.pnl_pct <= -cfg.stop_loss_pct + 1e-12:
            hits.append(StopHit(sym, d.avg_entry_price, d.current_price, round(d.pnl_pct, 6)))
    return hits


def cooldown_until(today: dt.date, trading_days: int) -> dt.date:
    """Approximate trading days with business days (exchange holidays ignored)."""
    return (pd.Timestamp(today) + pd.offsets.BDay(trading_days)).date() if trading_days > 0 else today


def force_zero(target: pd.Series, symbols) -> pd.Series:
    """Force the given symbols to target 0 (added if missing, so held ones get sold)."""
    syms = sorted(set(symbols))
    if not syms:
        return target
    out = target.copy()
    for s in syms:
        out[s] = 0.0
    return out
