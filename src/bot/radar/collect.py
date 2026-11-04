"""One collector run: fetch every enabled source, match tickers, store with dedupe."""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import httpx
from dotenv import load_dotenv

from bot.config import ROOT
from bot.radar.config import RadarConfig
from bot.radar.matching import AMBIGUOUS_NAMES, STOPLIST, TickerMatcher
from bot.radar.sources import SOURCES
from bot.radar.store import RadarStore, iso, parse_iso
from bot.radar.text import MAX_TEXT, text_hash
from bot.radar.universe import build_names, load_raw_names, load_universe

log = logging.getLogger(__name__)
UTC = dt.timezone.utc
ORDER = ["google_news", "hn", "substack", "bluesky", "youtube", "gdelt"]  # slow GDELT last


# Machine-generated pages that say nothing about attention: option-chain quote pages, 13F holding-change
# bots, "seasonal returns chart" pages. Dropped before matching (plus config junk_title_patterns).
JUNK_TITLES = [
    r"\b\d+(\.\d+)?\s+(call|put)\b.*\(\w{1,6}\d{6}[CP]\d{8}\)",
    r"\w{1,6}\d{6}[CP]\d{8}",
    r"stock price, news, quote (&|and) history",
    r"seasonal returns? chart",
    r"interactive stock chart",
    r"profile, company faqs",
    r"\b(shares|stake|position|holdings?) (of|in) .{1,80}\b(acquired|sold|purchased|bought|raised|lowered|"
    r"cut|trimmed|boosted|increased|decreased|reduced|grown)\b",
    r"\b(acquires|sells|buys|purchases|trims|boosts|raises|lowers|cuts|increases|decreases|reduces|grows) "
    r"(new )?(\d[\d,]* )?(shares|stake|position|holdings)\b",
    r"\b(takes|initiates|establishes) (a )?(new )?(position|stake)\b",
    r"\b(shares|stock|stake|position|holdings?) (purchased|acquired|sold|bought|raised|lowered|trimmed|"
    r"boosted|increased|decreased|reduced|cut|grown) by\b",
    r"\bform 144\b",
]


def is_junk(post, patterns: list[re.Pattern], junk_origins: list[str]) -> bool:
    o = post.origin.lower()
    if any(o == d or o.endswith("." + d) for d in junk_origins):
        return True
    return any(p.search(post.title) for p in patterns)


def build_matcher(cfg: RadarConfig, symbols: list[str], names: dict[str, list[str]]) -> TickerMatcher:
    return TickerMatcher(
        symbols, names,
        stoplist=STOPLIST | {w.upper() for w in cfg.stoplist_extra},
        ambiguous=AMBIGUOUS_NAMES | {w.lower() for w in cfg.ambiguous_names},
        max_symbols=cfg.max_symbols_per_post,
    )


def run_collect(cfg: RadarConfig, client: httpx.Client | None = None, now: dt.datetime | None = None,
                allow_network_names: bool = True, sleep=time.sleep) -> dict:
    load_dotenv(ROOT / ".env")  # optional BSKY_HANDLE / BSKY_APP_PASSWORD
    t0 = time.monotonic()
    now = now or dt.datetime.now(UTC)
    deadline = t0 + cfg.max_runtime_seconds
    store = RadarStore(cfg.resolve(cfg.db_path))
    run_id = store.start_run(now)

    a_syms, b_syms = load_universe(cfg, now.date())
    symbols = list(dict.fromkeys(a_syms + b_syms))
    raw = load_raw_names(symbols, cfg.resolve(cfg.names_cache), cfg.names_max_age_days, allow_network_names)
    names = build_names(symbols, raw, cfg.aliases)
    matcher = build_matcher(cfg, symbols, names)
    since = now - dt.timedelta(hours=cfg.lookback_hours)
    junk = [re.compile(p, re.I) for p in JUNK_TITLES + cfg.junk_title_patterns]

    own_client = client is None
    client = client or httpx.Client(timeout=cfg.http_timeout_seconds, follow_redirects=True,
                                    headers={"User-Agent": cfg.user_agent})
    stats: dict = {"symbols": len(symbols), "profile_a": len(a_syms), "smallcap": len(b_syms), "sources": {}}
    try:
        for name in ORDER:
            scfg = cfg.source(name)
            st = {"enabled": scfg.enabled, "requests": 0, "fetched": 0, "junk": 0, "matched_posts": 0, "new_rows": 0,
                  "dupes": 0, "errors": [], "skipped": None, "seconds": 0.0}
            stats["sources"][name] = st
            if not scfg.enabled:
                st["skipped"] = "disabled"
                continue
            last = store.get_meta(f"last_run:{name}")
            if last and scfg.min_interval_minutes and now - parse_iso(last) < dt.timedelta(minutes=scfg.min_interval_minutes):
                st["skipped"] = f"ran < {scfg.min_interval_minutes} min ago"
                continue
            if time.monotonic() > deadline:
                st["skipped"] = "run deadline"
                continue
            s0 = time.monotonic()
            src = SOURCES[name](scfg, client, cfg.mainstream_domains, deadline=deadline, sleep=sleep)
            src.cursor = int(store.get_meta(f"cursor:{name}", "0") or 0)
            posts = src.fetch(symbols, names, since)
            matched, n_junk = [], 0
            for p in posts:
                if is_junk(p, junk, cfg.junk_origins):
                    n_junk += 1
                    continue
                p.symbols = matcher.match(f"{p.title}\n{p.text}")
                if p.symbols:
                    p.text = p.text[:MAX_TEXT]
                    matched.append(p)
            new, dupes = store.insert_posts(matched, now, text_hash)
            store.set_meta(f"cursor:{name}", str(src.cursor))
            if src.requests:
                store.set_meta(f"last_run:{name}", iso(now))
            st.update(requests=src.requests, fetched=len(posts), junk=n_junk, matched_posts=len(matched), new_rows=new,
                      dupes=dupes, errors=src.errors[:5], seconds=round(time.monotonic() - s0, 1))
            log.info("radar %-11s req=%d fetched=%d matched=%d new=%d dupes=%d errors=%d (%.1fs)", name,
                     src.requests, len(posts), len(matched), new, dupes, len(src.errors), st["seconds"])
    finally:
        if own_client:
            client.close()
        stats["seconds"] = round(time.monotonic() - t0, 1)
        store.finish_run(run_id, dt.datetime.now(UTC), stats)
    return stats
