"""Radar ticker universe: profile A symbols + newest small-cap screener file, plus company names."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path

import yaml

from bot.config import ROOT
from bot.radar.config import RadarConfig
from bot.radar.matching import is_fund_name, name_variants

log = logging.getLogger(__name__)
_DATED = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def profile_a_symbols(settings_path: Path) -> list[str]:
    try:
        with open(settings_path, encoding="utf-8") as f:
            return [str(s).upper() for s in (yaml.safe_load(f) or {}).get("universe", {}).get("symbols", [])]
    except Exception as e:
        log.warning("radar: cannot read profile A symbols from %s: %s", settings_path, e)
        return []


def newest_universe_file(universe_dir: Path, as_of: dt.date | None = None) -> Path | None:
    """Newest data/universe/smallcap/YYYY-MM-DD.json (dated <= as_of when given), or None."""
    if not universe_dir.is_dir():
        return None
    files = []
    for p in universe_dir.iterdir():
        m = _DATED.match(p.name)
        if m:
            try:
                d = dt.date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if as_of is None or d <= as_of:
                files.append((d, p))
    return max(files)[1] if files else None


def smallcap_symbols(universe_dir: Path, as_of: dt.date | None = None) -> list[str]:
    p = newest_universe_file(universe_dir, as_of)
    if p is None:
        log.info("radar: no small-cap universe file in %s yet; using profile A symbols only", universe_dir)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [str(s).upper() for s in data.get("symbols", []) if isinstance(s, str)]
    except Exception as e:
        log.warning("radar: bad universe file %s: %s", p, e)
        return []


def all_smallcap_symbols_ever(universe_dir: Path) -> set[str]:
    """Union over every screener file (for splitting evaluation results A vs small cap)."""
    out: set[str] = set()
    if universe_dir.is_dir():
        for p in universe_dir.glob("*.json"):
            if _DATED.match(p.name):
                try:
                    out.update(str(s).upper() for s in json.loads(p.read_text(encoding="utf-8")).get("symbols", []))
                except Exception:
                    pass
    return out


def load_universe(cfg: RadarConfig, as_of: dt.date | None = None) -> tuple[list[str], list[str]]:
    """(profile A symbols, small-cap symbols not already in A)."""
    a = profile_a_symbols(cfg.resolve(cfg.settings_path))
    b = [s for s in smallcap_symbols(cfg.resolve(cfg.universe_dir), as_of) if s not in a]
    return a, b


# ---------- company names ----------
def _fetch_alpaca_names() -> dict[str, str]:
    """All active US equity names from Alpaca (raw HTTP: alpaca-py's enum rejects newer exchanges like TXSE)."""
    import httpx
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key, sec = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not sec:
        return {}
    r = httpx.get("https://paper-api.alpaca.markets/v2/assets",
                  params={"status": "active", "asset_class": "us_equity"},
                  headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}, timeout=30)
    r.raise_for_status()
    return {a["symbol"]: a.get("name") or "" for a in r.json() if a.get("symbol")}


def _fetch_yf_name(sym: str) -> str:
    import yfinance as yf

    info = yf.Ticker(sym).info or {}
    return info.get("longName") or info.get("shortName") or ""


def load_raw_names(symbols: list[str], cache_path: Path, max_age_days: int = 7, allow_network: bool = True) -> dict[str, str]:
    """{symbol: raw company name}, cached in data/cache/radar_names.json. Fails soft to whatever is cached."""
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    names: dict[str, str] = cache.get("names", {})
    fetched = cache.get("fetched_at")
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(fetched)).days if fetched else 10**6
    missing = [s for s in symbols if s not in names]
    if allow_network and (age >= max_age_days or (missing and age >= 1)):
        try:
            fresh = _fetch_alpaca_names()
            if fresh:
                names = {**names, **fresh}
                fetched = dt.datetime.now(dt.timezone.utc).isoformat()
        except Exception as e:
            log.warning("radar: Alpaca asset names unavailable: %s", e)
        for s in [s for s in symbols if s not in names][:20]:
            try:
                names[s] = _fetch_yf_name(s)
            except Exception as e:
                log.info("radar: yfinance name unavailable for %s: %s", s, e)
                names[s] = ""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"fetched_at": fetched or dt.datetime.now(dt.timezone.utc).isoformat(),
                                              "names": names}, indent=0), encoding="utf-8")
        except Exception as e:
            log.warning("radar: cannot write names cache: %s", e)
    return {s: names.get(s, "") for s in symbols}


def build_names(symbols: list[str], raw: dict[str, str], aliases: dict[str, list[str]]) -> dict[str, list[str]]:
    """{symbol: [name variants]} for matching. Funds/ETFs get only configured aliases."""
    out = {}
    for s in symbols:
        r = raw.get(s, "")
        out[s] = name_variants("" if is_fund_name(r) else r, aliases.get(s, []), symbol=s)
    return out
