"""Pure screening logic: exclusions, filters, features, scoring.

Network access lives in `bot.screen.sources`; `run_screen` takes the fetchers as callables
so the whole pipeline can be tested with synthetic data.

Pipeline (docs/PROFILE_B_SMALLCAP.md):
  Alpaca assets -> exclusions -> bulk daily prices -> stale / price >= $3 / 20d ADV >= $5M
  -> >= 126 trading days of history -> market cap (survivors only) in $300M-$5B
  -> features -> winsorized cross-sectional z-scores -> weighted score -> top N.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")

# Radar chatter (mention_z) stays at 0 until it passes evaluation (hypothesis H2).
RADAR_WEIGHT = 0.0
WEIGHTS = {"mom": 0.40, "ret_3m": 0.20, "vol_surge": 0.20, "prox_high": 0.20, "mention_z": RADAR_WEIGHT}


@dataclass(frozen=True)
class ScreenParams:
    min_price: float = 3.0
    min_adv: float = 5_000_000.0        # 20-day average dollar volume
    adv_window: int = 20
    min_mcap: float = 300_000_000.0
    max_mcap: float = 5_000_000_000.0
    min_history: int = 126              # trading days
    max_stale_bars: int = 2             # symbol's last bar may lag the panel's last date by at most this many bars
    winsor: float = 3.0
    top_n: int = 20
    exclude_adrs: bool = True
    exclude_spacs: bool = True
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))


# ---------------------------------------------------------------- exclusions
_SPECIAL_SYMBOL = re.compile(r"[./\-\s+^=]")
_NAME_RULES: list[tuple[str, re.Pattern]] = [
    ("warrant", re.compile(r"\bwarrants?\b", re.I)),
    ("unit", re.compile(r"\bunits?\b", re.I)),
    ("right", re.compile(r"\brights?\b", re.I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpfd\b|\bperpetual\b", re.I)),
    ("etf", re.compile(r"\bet[fn]s?\b|exchange[- ]traded", re.I)),
    ("fund", re.compile(r"\bfunds?\b", re.I)),
    ("trust", re.compile(r"\btrust\b", re.I)),
    ("debt", re.compile(r"\bnotes?\s+due\b|\bdebentures?\b|\bsenior\s+notes?\b|%", re.I)),
]
_SPAC = re.compile(
    r"\bacquisitions?\s+(corp|corporation|company|co|inc|ltd|limited|holdings?)\b|\bblank\s+check\b"
    r"|\b(corp|corporation|co|inc|ltd|limited)\.?\s+[IVX]+\b",  # serial SPACs: "Churchill Capital Corp XI"
    re.I)
_ADR = re.compile(r"\bamerican\s+depositary\b|\bdepositary\s+(shares?|receipts?)\b|\bADRs?\b|\bADSs?\b", re.I)


def exclusion_reason(asset: dict, params: ScreenParams = ScreenParams()) -> str | None:
    """Why an Alpaca asset (raw /v2/assets dict) is not screenable, or None if it is."""
    if asset.get("class", "us_equity") != "us_equity":
        return "not_us_equity"
    if asset.get("status", "active") != "active":
        return "inactive"
    if not asset.get("tradable", False):
        return "not_tradable"
    ex = str(asset.get("exchange") or "")
    if ex not in EXCHANGES:
        return f"exchange:{ex or 'none'}"
    sym = str(asset.get("symbol") or "")
    if not sym or _SPECIAL_SYMBOL.search(sym):
        return "special_symbol"
    name = str(asset.get("name") or "")
    for reason, rx in _NAME_RULES:
        if rx.search(name):
            return reason
    if params.exclude_spacs and _SPAC.search(name):
        return "spac"
    if params.exclude_adrs and _ADR.search(name):
        return "adr"
    return None


# ---------------------------------------------------------------- price stats & features
def price_stats(close: pd.DataFrame, volume: pd.DataFrame, adv_window: int = 20) -> pd.DataFrame:
    """Per-symbol last price, ADV, history length and staleness (bars behind the panel's last date)."""
    rows = {}
    idx = close.index
    for s in close.columns:
        c = close[s].dropna()
        c = c[c > 0]
        if c.empty:
            rows[s] = {"price": np.nan, "adv20": np.nan, "n_hist": 0, "last_date": None, "bars_behind": np.inf}
            continue
        v = volume[s].reindex(c.index).fillna(0.0) if s in volume else pd.Series(0.0, index=c.index)
        dv = (c * v).iloc[-adv_window:]
        rows[s] = {
            "price": float(c.iloc[-1]),
            "adv20": float(dv.mean()),
            "n_hist": int(len(c)),
            "last_date": c.index[-1].date().isoformat(),
            "bars_behind": int(len(idx) - 1 - idx.get_loc(c.index[-1])),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_features(close: pd.DataFrame, volume: pd.DataFrame, high: pd.DataFrame | None = None,
                     min_history: int = 126) -> pd.DataFrame:
    """Raw momentum/volume/high-proximity features per symbol (NaN when not computable).

    mom       : 12-1 momentum, close[t-21]/close[t-252]-1; 6-1 (t-126) when history < 253 bars
    ret_3m    : close[t]/close[t-63]-1
    vol_surge : log(5-day avg dollar volume / 50-day avg dollar volume)
    prox_high : close[t] / max(high over last 252 bars)
    """
    out = {}
    for s in close.columns:
        c = close[s].dropna()
        c = c[c > 0]
        n = len(c)
        f = {"mom": np.nan, "mom_window": None, "ret_3m": np.nan, "vol_surge": np.nan, "prox_high": np.nan}
        if n >= 253:
            f["mom"], f["mom_window"] = c.iloc[-22] / c.iloc[-253] - 1, "12-1"
        elif n >= min(min_history, 127) and n >= 22:
            f["mom"], f["mom_window"] = c.iloc[-22] / c.iloc[max(-127, -n)] - 1, "6-1"
        if n >= 64:
            f["ret_3m"] = c.iloc[-1] / c.iloc[-64] - 1
        if n >= 50:
            v = volume[s].reindex(c.index).fillna(0.0) if s in volume else None
            if v is not None:
                dv = c * v
                a5, a50 = dv.iloc[-5:].mean(), dv.iloc[-50:].mean()
                if a5 > 0 and a50 > 0:
                    f["vol_surge"] = math.log(a5 / a50)
        if n >= 1:
            hi = c.iloc[-252:].max()
            if high is not None and s in high:
                h = high[s].reindex(c.index).iloc[-252:].max()
                if pd.notna(h) and h > 0:
                    hi = max(hi, h)
            f["prox_high"] = c.iloc[-1] / hi
        out[s] = f
    return pd.DataFrame.from_dict(out, orient="index")


def zscore_winsor(x: pd.Series, limit: float = 3.0) -> pd.Series:
    """Cross-sectional z-score clipped to +/-limit. Missing values (and degenerate inputs) score 0."""
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = x.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=x.index)
    sd = valid.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=x.index)
    return ((x - valid.mean()) / sd).clip(-limit, limit).fillna(0.0)


def score(features: pd.DataFrame, weights: dict = WEIGHTS, limit: float = 3.0) -> tuple[pd.Series, pd.DataFrame]:
    """Weighted sum of winsorized z-scores. Returns (score, z-scores)."""
    z = pd.DataFrame(index=features.index)
    total = pd.Series(0.0, index=features.index)
    for k, w in weights.items():
        col = features[k] if k in features else pd.Series(np.nan, index=features.index)
        z[k] = zscore_winsor(col, limit)
        total = total + w * z[k]
    return total, z


# ---------------------------------------------------------------- pipeline
@dataclass
class ScreenResult:
    as_of: dt.date
    rows: list[dict]                 # every candidate: symbol, passed, reason, rank, score, features
    counts: dict                     # survivors after each stage, in order
    top: list[dict]                  # top-N passing candidates (flattened features)
    params: ScreenParams
    prices_through: str | None = None  # date of the last price bar used

    @property
    def symbols(self) -> list[str]:
        return [c["symbol"] for c in self.top]

    def payload(self, generated_at: str) -> dict:
        p = asdict(self.params)
        return {
            "as_of": self.as_of.isoformat(),
            "generated_at": generated_at,
            "prices_through": self.prices_through,
            "symbols": self.symbols,
            "candidates": self.top,
            "filters": {**{k: v for k, v in p.items() if k != "weights"}, "exchanges": list(EXCHANGES),
                        "weights": p["weights"], "counts": self.counts},
        }


def _clean(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def run_screen(
    assets: list[dict],
    fetch_prices: Callable[[list[str]], dict[str, pd.DataFrame]],
    fetch_mcaps: Callable[[list[str]], dict[str, dict]],
    as_of: dt.date,
    params: ScreenParams = ScreenParams(),
    mention_z: dict[str, float] | None = None,
) -> ScreenResult:
    """Run the full screen. `fetch_prices(symbols)` -> {"close","volume","high"} wide frames
    (date index, symbol columns), already truncated to <= as_of. `fetch_mcaps(symbols)` ->
    {symbol: {"market_cap": float|None, "quote_type": str|None, "currency": str|None, "reason": str|None, ...}}."""
    rows: dict[str, dict] = {}
    counts: dict[str, int] = {"assets": 0}

    def fail(sym, reason):
        rows[sym]["passed"] = False
        rows[sym]["reason"] = reason

    # 1. exclusions
    eligible = []
    for a in assets:
        sym = str(a.get("symbol") or "")
        if not sym or sym in rows:
            continue
        counts["assets"] += 1
        rows[sym] = {"symbol": sym, "passed": None, "reason": None, "rank": None, "score": None,
                     "features": {"name": a.get("name"), "exchange": a.get("exchange"),
                                  "fractionable": bool(a.get("fractionable", False)),
                                  "shortable": bool(a.get("shortable", False))}}
        r = exclusion_reason(a, params)
        if r:
            fail(sym, f"excluded:{r}")
        else:
            eligible.append(sym)
    counts["eligible"] = len(eligible)

    # 2. prices
    panel = fetch_prices(eligible) if eligible else {}
    close = panel.get("close", pd.DataFrame())
    volume = panel.get("volume", pd.DataFrame())
    high = panel.get("high")
    close = close.loc[close.index.date <= as_of] if len(close) else close
    stats = price_stats(close, volume, params.adv_window) if len(close.columns) else pd.DataFrame()
    feats = compute_features(close, volume, high, params.min_history) if len(close.columns) else pd.DataFrame()

    def st(sym, k):
        return stats.at[sym, k] if sym in stats.index else None

    alive = []
    for sym in eligible:
        if sym not in stats.index or not st(sym, "n_hist"):
            fail(sym, "no_price_data")
            continue
        rows[sym]["features"].update({k: _clean(st(sym, k)) for k in ("price", "adv20", "n_hist", "last_date")})
        if sym in feats.index:
            rows[sym]["features"].update({k: _clean(feats.at[sym, k]) for k in feats.columns})
        alive.append(sym)
    counts["price_data"] = len(alive)

    def stage(name, syms, test):
        keep = []
        for s in syms:
            r = test(s)
            if r:
                fail(s, r)
            else:
                keep.append(s)
        counts[name] = len(keep)
        return keep

    alive = stage("fresh", alive, lambda s: "stale_price" if st(s, "bars_behind") > params.max_stale_bars else None)
    alive = stage("price", alive, lambda s: f"price<{params.min_price:g}" if st(s, "price") < params.min_price else None)
    alive = stage("adv", alive, lambda s: f"adv20<{params.min_adv / 1e6:g}M" if not st(s, "adv20") >= params.min_adv else None)
    alive = stage("history", alive, lambda s: f"history<{params.min_history}" if st(s, "n_hist") < params.min_history else None)

    # 3. market cap, survivors only
    caps = fetch_mcaps(alive) if alive else {}

    def cap_test(s):
        info = caps.get(s) or {"reason": "no_response"}
        f = rows[s]["features"]
        f["market_cap"] = _clean(info.get("market_cap"))
        for k in ("quote_type", "currency", "country", "sector", "industry"):
            if info.get(k) is not None:
                f[k] = info[k]
        qt = info.get("quote_type")
        if qt and str(qt).upper() != "EQUITY":
            return f"not_equity:{qt}"
        cur = info.get("currency")
        if cur and str(cur).upper() != "USD":
            return f"non_usd:{cur}"
        m = info.get("market_cap")
        if m is None or not np.isfinite(m) or m <= 0:
            return f"mcap_unavailable:{info.get('reason') or 'missing'}"
        if m < params.min_mcap:
            return f"mcap<{params.min_mcap / 1e6:g}M"
        if m > params.max_mcap:
            return f"mcap>{params.max_mcap / 1e9:g}B"
        return None

    alive = stage("mcap", alive, cap_test)

    # 4. score passing candidates
    if alive:
        fx = feats.reindex(alive).copy()
        fx["mention_z"] = pd.Series(mention_z or {}, dtype=float).reindex(alive)
        sc, z = score(fx, params.weights, params.winsor)
        order = sorted(alive, key=lambda s: (-sc[s], s))
        for rank, s in enumerate(order, 1):
            rows[s].update(passed=True, reason=None, rank=rank, score=round(float(sc[s]), 6))
            rows[s]["features"]["mention_z"] = _clean(fx.at[s, "mention_z"])
            rows[s]["features"].update({f"z_{k}": round(float(z.at[s, k]), 4) for k in z.columns})
    else:
        order = []
    counts["passed"] = len(order)
    top = []
    for s in order[: params.top_n]:
        r = rows[s]
        top.append({"symbol": s, "rank": r["rank"], "score": r["score"], **r["features"]})
    counts["top"] = len(top)
    through = close.index[-1].date().isoformat() if len(close) else None
    return ScreenResult(as_of=as_of, rows=list(rows.values()), counts=counts, top=top, params=params,
                        prices_through=through)
