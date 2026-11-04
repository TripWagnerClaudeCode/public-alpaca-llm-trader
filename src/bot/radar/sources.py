"""Radar sources. Each one: fetch(symbols, names, since) -> list[Post], failing SOFT (warning + []).

Politeness: a per-run request cap (round-robin over symbols across runs, cursor kept in radar.db meta),
a pause between requests to the same API, a shared run deadline, timeouts, and a plain UA string.
All text is sanitized by clean_text (HTML stripped, whitespace collapsed, capped at 1000 chars).
Posts are returned UNMATCHED; the collector assigns symbols with TickerMatcher.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time
from urllib.parse import urlparse

import httpx

from bot.radar.config import SourceCfg
from bot.radar.matching import AMBIGUOUS_NAMES, STOPLIST
from bot.radar.store import Post
from bot.radar.text import clean_text

log = logging.getLogger(__name__)
UTC = dt.timezone.utc


class Budget(Exception):
    """Request cap or run deadline reached: stop this source quietly."""


def domain_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_mainstream_domain(domain: str, mainstream: list[str] | set[str]) -> bool:
    d = domain.lower()
    return any(d == m or d.endswith("." + m) for m in mainstream)


def _struct_to_dt(st) -> dt.datetime | None:
    if not st:
        return None
    try:
        return dt.datetime(*st[:6], tzinfo=UTC)
    except Exception:
        return None


class Source:
    name = "base"
    default_mainstream = False

    def __init__(self, cfg: SourceCfg, client: httpx.Client, mainstream_domains: list[str] | None = None,
                 deadline: float | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.client = client
        self.mainstream_domains = set(mainstream_domains or [])
        self.deadline = deadline
        self.sleep = sleep
        self.cursor = 0          # round-robin position (loaded/saved by the collector)
        self.requests = 0
        self.errors: list[str] = []
        self._last_req: float | None = None

    # ---- public ----
    def fetch(self, symbols: list[str], names: dict[str, list[str]], since: dt.datetime) -> list[Post]:
        try:
            return self._fetch(symbols, names, since)
        except Exception as e:  # fail soft, always
            log.warning("radar source %s failed: %s", self.name, e)
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            return []

    def _fetch(self, symbols, names, since) -> list[Post]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- helpers ----
    def _get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        if self.requests >= self.cfg.max_requests:
            raise Budget("request cap")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise Budget("deadline")
        if self._last_req is not None and self.cfg.pause_seconds > 0:
            wait = self.cfg.pause_seconds - (time.monotonic() - self._last_req)
            if wait > 0:
                if self.deadline is not None and time.monotonic() + wait > self.deadline:
                    raise Budget("deadline")
                self.sleep(wait)
        self.requests += 1
        try:
            r = self.client.get(url, params=params, headers=headers)
        finally:
            self._last_req = time.monotonic()
        r.raise_for_status()
        return r

    def _each(self, queries: list, run) -> list[Post]:
        """Run `run(query)` for up to max_requests queries starting at the round-robin cursor.
        A failing query is logged and skipped (soft); the cursor advances past attempted queries."""
        out: list[Post] = []
        if not queries:
            return out
        n = len(queries)
        start = self.cursor % n
        attempted = 0
        for k in range(min(n, self.cfg.max_requests)):
            q = queries[(start + k) % n]
            try:
                out.extend(run(q))
            except Budget:
                break
            except Exception as e:
                log.warning("radar source %s query %r failed: %s", self.name, str(q)[:80], e)
                self.errors.append(f"{type(e).__name__}: {e}"[:300])
            attempted += 1
        self.cursor = (start + attempted) % n
        return out

    @staticmethod
    def _batches(items: list, size: int) -> list[list]:
        size = max(1, size)
        return [items[i:i + size] for i in range(0, len(items), size)]


def _primary_name(sym: str, names: dict[str, list[str]]) -> str | None:
    vs = [v.lstrip("~") for v in names.get(sym, [])]
    vs = [v for v in vs if v.lower() not in AMBIGUOUS_NAMES]
    vs.sort(key=lambda v: ("." in v))  # prefer 'Amazon' over 'Amazon.com' as a query term
    return vs[0] if vs else None
    return None


def _ticker_ok(sym: str) -> bool:
    return len(sym) >= 3 and sym.isalpha() and sym not in STOPLIST


# ---------------------------------------------------------------- HN Algolia
class HackerNews(Source):
    """HN Algolia search_by_date; one query per symbol (Algolia ANDs words, so no OR batching)."""
    name = "hn"
    URL = "https://hn.algolia.com/api/v1/search_by_date"

    def _fetch(self, symbols, names, since):
        queries = []
        for s in symbols:
            q = _primary_name(s, names) or (s if _ticker_ok(s) else None)
            if q:
                queries.append(q)

        def run(q):
            r = self._get(self.URL, params={"query": q, "tags": "(story,comment)", "hitsPerPage": 50,
                                            "numericFilters": f"created_at_i>{int(since.timestamp())}"})
            posts = []
            for h in r.json().get("hits", []):
                oid = h.get("objectID")
                if not oid:
                    continue
                is_comment = "comment" in (h.get("_tags") or []) or bool(h.get("comment_text"))
                # comments are matched on their own words only (not the parent story's title)
                title = "" if is_comment else clean_text(h.get("title") or "", 300)
                body = clean_text(h.get("comment_text") if is_comment else h.get("story_text"))
                if not (title or body):
                    continue
                ts = dt.datetime.fromtimestamp(int(h.get("created_at_i") or 0), UTC)
                posts.append(Post(url=f"https://news.ycombinator.com/item?id={oid}", source=self.name,
                                  origin=f"hn:{h.get('author') or '?'}", title=title, text=body, ts=ts,
                                  is_mainstream=False))
            return posts

        return self._each(queries, run)


# ---------------------------------------------------------------- GDELT DOC 2.0
class Gdelt(Source):
    """GDELT DOC 2.0 artlist (titles only). Strictly rate limited (~1 req / 5+ s): few, batched requests."""
    name = "gdelt"
    URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def _fetch(self, symbols, names, since):
        terms = []
        for s in symbols:
            n = _primary_name(s, names)
            n = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", n or "")).strip()  # GDELT dislikes punctuation
            if len(n) >= 4:
                terms.append(f'"{n}"' if " " in n else n)
        hours = max(1, min(24 * 7, int((dt.datetime.now(UTC) - since).total_seconds() // 3600) + 1))

        def run(batch):
            q = (f"({' OR '.join(batch)})" if len(batch) > 1 else batch[0]) + " sourcelang:english"
            r = self._get(self.URL, params={"query": q, "mode": "artlist", "format": "json", "maxrecords": 250,
                                            "timespan": f"{hours}h", "sort": "datedesc"})
            try:
                data = r.json()
            except ValueError:
                raise RuntimeError(f"non-JSON reply: {r.text[:120]!r}")
            posts = []
            for a in data.get("articles", []) or []:
                url = a.get("url") or ""
                dom = (a.get("domain") or domain_of(url)).lower()
                try:
                    ts = dt.datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                except ValueError:
                    continue
                if not url or ts < since:
                    continue
                # GDELT spaces out punctuation ("Amazon . com , Inc ."): undo that before matching
                title = re.sub(r"\s+([.,:;!?%)])", r"\1", clean_text(a.get("title"), 300))
                title = re.sub(r"(\b[A-Z])\.\s(?=[A-Z]\.)", r"\1.", title)
                title = re.sub(r"\.\s+(com|net|org|io|ai)\b", r".\1", title)
                posts.append(Post(url=url, source=self.name, origin=dom, title=title,
                                  text="", ts=ts, is_mainstream=is_mainstream_domain(dom, self.mainstream_domains)))
            return posts

        return self._each(self._batches(terms, self.cfg.batch_size), run)


# ---------------------------------------------------------------- Google News RSS
class GoogleNews(Source):
    """Google News RSS search: the mainstream baseline. Small OR batches (results cap at 100 per query)."""
    name = "google_news"
    default_mainstream = True
    URL = "https://news.google.com/rss/search"

    def _fetch(self, symbols, names, since):
        import feedparser

        terms = []
        for s in symbols:
            parts = []
            n = _primary_name(s, names)
            if n:
                parts.append(f'"{n}"')
            if _ticker_ok(s):
                parts.append(f'"{s}"')
            if parts:
                terms.append(parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")")
        days = max(1, -(-int((dt.datetime.now(UTC) - since).total_seconds()) // 86400))  # ceil

        def run(batch):
            q = " OR ".join(batch) + f" when:{days}d"
            r = self._get(self.URL, params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
            feed = feedparser.parse(r.content)
            posts = []
            for e in feed.entries:
                ts = _struct_to_dt(e.get("published_parsed"))
                link = e.get("link") or ""
                if not link or ts is None or ts < since:
                    continue
                src = e.get("source") or {}
                publisher = domain_of(src.get("href", "")) or clean_text(src.get("title", ""), 80).lower() or "news.google.com"
                title = clean_text(e.get("title"), 300)
                ptitle = clean_text(src.get("title", ""), 80)
                if ptitle and title.endswith(" - " + ptitle):
                    title = title[: -len(ptitle) - 3]  # strip " - Publisher" so syndicated copies dedupe
                posts.append(Post(url=link, source=self.name, origin=publisher, title=title, text="",
                                  ts=ts, is_mainstream=True))
            return posts

        return self._each(self._batches(terms, self.cfg.batch_size), run)


# ---------------------------------------------------------------- Substack RSS
class Substack(Source):
    """Configured Substack feeds (title + subtitle + start of the body). One request per feed."""
    name = "substack"

    def _fetch(self, symbols, names, since):
        import feedparser

        def run(url):
            r = self._get(url)
            feed = feedparser.parse(r.content)
            host = domain_of(url)
            posts = []
            for e in feed.entries:
                ts = _struct_to_dt(e.get("published_parsed") or e.get("updated_parsed"))
                link = e.get("link") or ""
                if not link or ts is None or ts < since:
                    continue
                body = ""
                if e.get("content"):
                    body = e["content"][0].get("value", "")
                text = clean_text(f"{e.get('summary', '')} {body}", 3000)
                posts.append(Post(url=link, source=self.name, origin=host, title=clean_text(e.get("title"), 300),
                                  text=text, ts=ts, is_mainstream=False))
            return posts

        return self._each(list(self.cfg.feeds), run)


# ---------------------------------------------------------------- Bluesky
class Bluesky(Source):
    """app.bsky.feed.searchPosts for $CASHTAG queries. Needs BSKY_HANDLE + BSKY_APP_PASSWORD."""
    name = "bluesky"
    HOST = "https://bsky.social/xrpc"

    def _fetch(self, symbols, names, since):
        handle = os.environ.get("BSKY_HANDLE", "").strip()
        pw = os.environ.get("BSKY_APP_PASSWORD", "").strip()
        if not handle or not pw:
            log.info("radar: Bluesky skipped (BSKY_HANDLE / BSKY_APP_PASSWORD not set)")
            return []
        if self.deadline is not None and time.monotonic() > self.deadline:
            return []
        self.requests += 1
        r = self.client.post(f"{self.HOST}/com.atproto.server.createSession",
                             json={"identifier": handle, "password": pw})
        r.raise_for_status()
        auth = {"Authorization": f"Bearer {r.json()['accessJwt']}"}
        since_s = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        def run(sym):
            r = self._get(f"{self.HOST}/app.bsky.feed.searchPosts",
                          params={"q": f"${sym}", "limit": 100, "sort": "latest", "since": since_s}, headers=auth)
            posts = []
            for p in r.json().get("posts", []):
                rec = p.get("record") or {}
                author = (p.get("author") or {}).get("handle") or "?"
                uri = p.get("uri") or ""
                rkey = uri.rsplit("/", 1)[-1] if uri else ""
                t = rec.get("createdAt") or p.get("indexedAt")
                try:
                    ts = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
                    ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
                except Exception:
                    continue
                if not rkey or ts < since:
                    continue
                posts.append(Post(url=f"https://bsky.app/profile/{author}/post/{rkey}", source=self.name,
                                  origin=f"bsky:{author}", title="", text=clean_text(rec.get("text")),
                                  ts=ts, is_mainstream=False))
            return posts

        return self._each(list(symbols), run)


# ---------------------------------------------------------------- YouTube (stub)
class YouTube(Source):
    """Disabled stub. TODO: transcripts from ~5-10 finance channels via yt-dlp (only if installed),
    e.g. `yt-dlp --skip-download --write-auto-subs --sub-langs en --sub-format vtt <channel>/videos`
    for videos newer than `since`, then clean_text() the transcript and match like any other post."""
    name = "youtube"

    def _fetch(self, symbols, names, since):
        import shutil

        if shutil.which("yt-dlp") is None:
            log.info("radar: YouTube skipped (yt-dlp not installed)")
        else:
            log.info("radar: YouTube transcripts not implemented yet (TODO); skipping")
        return []


SOURCES: dict[str, type[Source]] = {c.name: c for c in (HackerNews, Gdelt, GoogleNews, Substack, Bluesky, YouTube)}
