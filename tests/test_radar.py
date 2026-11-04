"""Radar tests: no network (httpx.MockTransport), temp SQLite DBs."""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3

import httpx
import pytest

from bot.radar.collect import run_collect
from bot.radar.config import RadarConfig, SourceCfg
from bot.radar.features import compute_features
from bot.radar.matching import TickerMatcher, name_variants, normalize_company_name
from bot.radar.sources import Gdelt, GoogleNews, HackerNews, Substack, is_mainstream_domain
from bot.radar.store import Post, RadarStore, iso
from bot.radar.text import clean_text, text_hash

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 11, 3, 20, 0, tzinfo=UTC)


# ------------------------------------------------------------------ matching
@pytest.fixture
def matcher():
    raw = {
        "NVDA": "NVIDIA Corporation Common Stock", "AAPL": "Apple Inc. Common Stock",
        "UNH": "UNITEDHEALTH GROUP INCORPORATED (Delaware)", "JPM": "JPMorgan Chase & Co.",
        "AMZN": "Amazon.com, Inc. Common Stock", "HOOD": "Robinhood Markets, Inc. Class A Common Stock",
        "NOW": "ServiceNow, Inc. Common Stock", "ALL": "Allstate Corporation (The) Common Stock",
        "QQQ": "Invesco QQQ Trust, Series 1", "COST": "Costco Wholesale Corporation Common Stock",
        "AMD": "Advanced Micro Devices, Inc. Common Stock", "ON": "ON Semiconductor Corporation Common Stock",
        "FIGS": "FIGS, Inc.", "GTE": "Gran Tierra Energy Inc.", "ECO": "Okeanis Eco Tankers Corp.",
        "KIDS": "OrthoPediatrics Corp. Common Stock", "TLT": "iShares 20 Year Treasury Bond ETF",
        "MTW": "The Manitowoc Company, Inc.", "BAND": "Bandwidth Inc. Class A Common Stock",
        "XLF": "State Street Financial Select Sector SPDR ETF", "V": "VISA Inc.",
    }
    aliases = {"NVDA": ["Nvidia"], "AAPL": ["Apple"], "AMZN": ["Amazon"]}
    names = {s: name_variants(r, aliases.get(s), symbol=s) for s, r in raw.items()}
    return TickerMatcher(list(raw), names, max_symbols=6)


def test_normalize_company_name():
    assert normalize_company_name("NVIDIA Corporation Common Stock") == "NVIDIA"
    assert normalize_company_name("JPMorgan Chase & Co.") == "JPMorgan Chase"
    assert normalize_company_name("Alphabet Inc. Class A Common Stock") == "Alphabet"
    assert normalize_company_name("UNITEDHEALTH GROUP INCORPORATED (Delaware)") == "UNITEDHEALTH"
    assert normalize_company_name("The Home Depot, Inc.") == "Home Depot"
    assert normalize_company_name("ExxonMobil Holdings Corporation") == "ExxonMobil"
    assert "~Amazon" in name_variants("Amazon.com, Inc. Common Stock")
    assert name_variants("Robinhood Markets, Inc. Class A Common Stock") == ["Robinhood Markets", "~Robinhood"]
    assert name_variants("NVIDIA Corporation", ["Nvidia"], symbol="NVDA") == ["Nvidia"]
    assert name_variants("Bandwidth Inc. Class A Common Stock", symbol="BAND") == ["~Bandwidth"]
    assert name_variants("CareDx, Inc. Common Stock", symbol="CDNA") == ["CareDx"]
    assert name_variants("FIGS, Inc.", symbol="FIGS") == []
    assert name_variants("Invesco QQQ Trust, Series 1") == []  # funds: no name matching


def test_cashtags_always_match(matcher):
    assert matcher.match("Loading up on $NVDA and $hood today") == {"NVDA", "HOOD"}
    assert matcher.match("$ALL and $NOW are cheap") == {"ALL", "NOW"}   # stoplisted words still match as cashtags
    assert matcher.match("$ON looks strong") == {"ON"}
    assert matcher.match("It cost $5 and $XYZ is unknown") == set()


def test_bare_tickers_and_stoplist(matcher):
    assert matcher.match("AMD beats on data center revenue") == {"AMD"}
    assert matcher.match("QQQ is down 2% premarket") == {"QQQ"}
    # common words / acronyms that are also tickers must not match bare
    assert matcher.match("We are OPEN NOW for ALL customers, ON time. COST matters. HOOD up.") == set()
    assert matcher.match("The CEO said AI and IT spending will RUN hot") == set()
    # lowercase is never a bare ticker; shouting headlines disable bare matching
    assert matcher.match("amd and nvda") == set()
    assert matcher.match("BREAKING: MARKETS CRASH AS AMD TUMBLES") == set()
    # two-letter tickers never match bare
    assert matcher.match("ON the other hand") == set()


def test_real_false_positives_seen_in_live_run(matcher):
    """Headlines that matched wrongly in the first live collection (2026-11-03)."""
    assert matcher.match("Lake Elsinore Approves FIGS Volleyball Facility Permit") == set()
    assert matcher.match("Win a Volkswagen Golf GTE worth over 40,000 for just 40p!") == set()
    assert matcher.match("Little Perillo ECO Leads a New Dauphin Americas Portfolio") == set()
    assert matcher.match("EcoSynthetix Inc. (TSX:ECO): What Could Drive the Next Market Move as Investors Assess") == set()
    assert matcher.match("TLT strengthens Pensions team with Scottish Partner appointment") == set()
    assert matcher.match("Nov. 5 - Manitowoc County Beekeepers meeting 5:30pm at Manitowoc County Building") == set()
    assert matcher.match("Qualcomm Validates First HBC Silicon With an 18x Memory Bandwidth Gain Over LPDDR5X") == set()
    assert matcher.match("CMO and COO with 20+ years. Visa sponsorship: no. Remote: yes") == set()
    # ...while the real ones still match
    assert matcher.match("FIGS, Inc. (NYSE:FIGS) Stock Has Consensus Price Target of $19.57") == {"FIGS"}
    assert matcher.match("How Orthopediatrics Corp. (KIDS) Affects Timing") == {"KIDS"}
    assert matcher.match("Dealers see U.S. Treasury revamping issuance guidance (TLT:NASDAQ)") == {"TLT"}
    assert matcher.match("Financial sector rises in October: AI shift lifts CBOE (XLF:NYSEARCA)") == {"XLF"}
    assert matcher.match("Manitowoc shares jump after Q3 results") == {"MTW"}
    assert matcher.match("Gran Tierra Energy reports production") == {"GTE"}
    assert matcher.match("Artificial intelligence (AI) and the economy") == set()


def test_names(matcher):
    assert matcher.match("Nvidia unveils a new chip") == {"NVDA"}
    assert matcher.match("NVIDIA's results") == {"NVDA"}
    assert matcher.match("UnitedHealth cuts guidance") == {"UNH"}
    assert matcher.match("JPMorgan Chase hires") == {"JPM"}
    assert matcher.match("Robinhood Markets launches futures") == {"HOOD"}
    assert matcher.match("Robinhood launches futures") == set()          # derived short name: needs context
    assert matcher.match("Robinhood shares rise") == {"HOOD"}
    assert matcher.match("ServiceNow raises outlook") == {"NOW"}
    assert matcher.match("Costco Wholesale opens stores") == {"COST"}
    assert matcher.match("Costco stock hits a record") == {"COST"}
    # ambiguous everyday words need finance context and a capital letter
    assert matcher.match("I ate an apple pie") == set()
    assert matcher.match("Apple unveils a new phone in a keynote") == set()
    assert matcher.match("Apple shares rise after earnings") == {"AAPL"}
    assert matcher.match("Amazon rainforest fires spread") == set()
    assert matcher.match("Amazon stock jumps on AWS growth") == {"AMZN"}


def test_roundups_dropped():
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    m = TickerMatcher(syms, {}, max_symbols=6)
    assert m.match(" ".join(f"${s}" for s in syms)) == set()
    assert m.match("$AAA $BBB") == {"AAA", "BBB"}


# ------------------------------------------------------------------ text
def test_clean_text_strips_and_caps():
    raw = "<p>Hello <b>world</b></p><script>alert('x')</script> &lt;img src=x onerror=1&gt; \n\n  ok‮" + "a" * 3000
    out = clean_text(raw)
    assert "<" not in out and "alert" not in out and "‮" not in out
    assert out.startswith("Hello world ok")
    assert len(out) == 1000


# ------------------------------------------------------------------ store / dedupe
def _post(url, title, sym="NVDA", ts=NOW, origin="example.com", mainstream=False, text=""):
    return Post(url=url, source="test", origin=origin, title=title, text=text, ts=ts,
                is_mainstream=mainstream, symbols={sym})


def test_dedupe_by_url_and_text_hash(tmp_path):
    st = RadarStore(tmp_path / "r.db")
    new, d = st.insert_posts([_post("u1", "Nvidia rallies")], NOW, text_hash)
    assert (new, d) == (1, 0)
    new, d = st.insert_posts([_post("u1", "Different title same url")], NOW, text_hash)
    assert (new, d) == (0, 1)
    # same text, different URL (syndicated copy; punctuation/case differences ignored)
    new, d = st.insert_posts([_post("u2", "NVIDIA rallies!")], NOW, text_hash)
    assert (new, d) == (0, 1)
    # same URL for a different symbol is a separate row
    p = _post("u1", "Nvidia rallies")
    p.symbols = {"NVDA", "AMD"}
    assert st.insert_posts([p], NOW, text_hash) == (1, 1)
    # future-dated posts are clamped to fetched_at
    st.insert_posts([_post("u3", "x future", ts=NOW + dt.timedelta(days=2))], NOW, text_hash)
    with sqlite3.connect(tmp_path / "r.db") as c:
        assert c.execute("SELECT ts FROM posts WHERE url='u3'").fetchone()[0] == iso(NOW)


# ------------------------------------------------------------------ features
def _seed(tmp_path, daily_counts, freeze, runs_every_day=True, extra=()):
    """daily_counts[i] = posts for NVDA in baseline bucket i (i=1..len), bucket 0 = window."""
    st = RadarStore(tmp_path / "r.db")
    posts = []
    for i, n in enumerate(daily_counts):
        for k in range(n):
            t = freeze - dt.timedelta(hours=24 * i + 1 + k * 0.01)
            posts.append(_post(f"u{i}-{k}", f"post {i} {k}", ts=t, origin=f"o{k}"))
    posts.extend(extra)
    st.insert_posts(posts, freeze - dt.timedelta(minutes=5), text_hash)
    if runs_every_day:
        for i in range(len(daily_counts)):
            st.start_run(freeze - dt.timedelta(hours=24 * i + 2))
    return tmp_path / "r.db"


def test_mention_z_math(tmp_path):
    freeze = NOW
    base = [2, 4] * 15                     # 30 baseline days: mean 3, sample std ~1.017
    db = _seed(tmp_path, [9] + base, freeze)
    f = compute_features(db, ["NVDA", "AMD"], freeze)["NVDA"]
    mean = 3.0
    std = math.sqrt(sum((x - mean) ** 2 for x in base) / (len(base) - 1))
    assert f["mention_count"] == 9
    assert f["baseline_days"] == 30
    assert f["baseline_mean"] == pytest.approx(mean)
    assert f["baseline_std"] == pytest.approx(std, abs=1e-4)
    assert f["mention_z"] == pytest.approx((9 - mean) / max(std, math.sqrt(mean), 1.0), abs=1e-4)
    amd = compute_features(db, ["AMD"], freeze)["AMD"]
    assert amd["mention_count"] == 0 and amd["mention_z"] == pytest.approx(0.0)  # floor: 0/1


def test_std_floor_and_insufficient_baseline(tmp_path):
    db = _seed(tmp_path, [1] + [0] * 30, NOW)
    f = compute_features(db, ["NVDA"], NOW)["NVDA"]
    assert f["mention_z"] == pytest.approx(1.0)   # flat-zero baseline: std floored at 1.0, not inf
    (tmp_path / "b").mkdir()
    db2 = _seed(tmp_path / "b", [3] + [1] * 10, NOW)
    f2 = compute_features(db2, ["NVDA"], NOW)["NVDA"]
    assert f2["mention_z"] is None and f2["baseline_days"] == 10 and f2["mention_count"] == 3


def test_live_coverage_rule(tmp_path):
    # history exists (backfilled) but the collector never ran on those days -> no baseline
    db = _seed(tmp_path, [5] + [2] * 30, NOW, runs_every_day=False)
    RadarStore(db).start_run(NOW - dt.timedelta(hours=1))
    f = compute_features(db, ["NVDA"], NOW)["NVDA"]
    assert f["mention_count"] == 5 and f["mention_z"] is None and f["baseline_days"] == 0
    g = compute_features(db, ["NVDA"], NOW, require_live_coverage=False)["NVDA"]
    assert g["mention_z"] is not None and g["baseline_days"] == 30


def test_no_lookahead_freeze(tmp_path):
    freeze = NOW
    st = RadarStore(tmp_path / "r.db")
    st.start_run(freeze - dt.timedelta(hours=3))
    st.insert_posts([_post("before", "a", ts=freeze - dt.timedelta(hours=2))], freeze - dt.timedelta(hours=1), text_hash)
    st.insert_posts([_post("after", "b", ts=freeze + dt.timedelta(minutes=1))], freeze + dt.timedelta(hours=1), text_hash)
    # posted before the freeze but only fetched after it: point-in-time excludes it
    st.insert_posts([_post("late", "c", ts=freeze - dt.timedelta(hours=5))], freeze + dt.timedelta(hours=2), text_hash)
    # exactly at the freeze is included, exactly 24h earlier is not (window is (freeze-24h, freeze])
    st.insert_posts([_post("edge1", "d", ts=freeze), _post("edge2", "e", ts=freeze - dt.timedelta(hours=24))],
                    freeze, text_hash)
    f = compute_features(tmp_path / "r.db", ["NVDA"], freeze)["NVDA"]
    assert f["mention_count"] == 2  # 'before' + 'edge1'
    g = compute_features(tmp_path / "r.db", ["NVDA"], freeze, point_in_time=False)["NVDA"]
    assert g["mention_count"] == 3  # + 'late'
    # moving the freeze later picks up the later post
    h = compute_features(tmp_path / "r.db", ["NVDA"], freeze + dt.timedelta(hours=3))["NVDA"]
    assert h["mention_count"] == 4


def test_fringe_only_logic(tmp_path):
    st = RadarStore(tmp_path / "r.db")
    t = NOW - dt.timedelta(hours=2)
    fringe = [_post(f"f{i}", f"chatter {i}", sym="AAA", ts=t, origin=f"blog{i}.substack.com") for i in range(3)]
    same_origin = [_post(f"s{i}", f"same {i}", sym="BBB", ts=t, origin="hn:alice") for i in range(5)]
    mixed = [_post(f"m{i}", f"mixed {i}", sym="CCC", ts=t, origin=f"o{i}") for i in range(4)]
    mixed.append(_post("m-main", "reuters story", sym="CCC", ts=t, origin="reuters.com", mainstream=True))
    st.insert_posts(fringe + same_origin + mixed, NOW - dt.timedelta(hours=1), text_hash)
    f = compute_features(tmp_path / "r.db", ["AAA", "BBB", "CCC", "DDD"], NOW)
    assert f["AAA"]["fringe_only"] is True and f["AAA"]["n_sources"] == 3
    assert f["BBB"]["fringe_only"] is False and f["BBB"]["n_sources"] == 1   # one voice, 5 posts
    assert f["CCC"]["fringe_only"] is False and f["CCC"]["n_mainstream"] == 1 and f["CCC"]["n_sources"] == 5
    assert f["DDD"]["mention_count"] == 0 and f["DDD"]["summary"] is None


def test_missing_db_returns_empty(tmp_path):
    f = compute_features(tmp_path / "nope.db", ["NVDA"], NOW)
    assert f["NVDA"]["mention_count"] == 0 and f["NVDA"]["mention_z"] is None
    assert not (tmp_path / "nope.db").exists()


# ------------------------------------------------------------------ sources (mocked HTTP)
def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "trading-bot-radar/0.1"})


NAMES = {"NVDA": ["NVIDIA"], "AMD": ["Advanced Micro Devices"]}
SINCE = NOW - dt.timedelta(hours=48)


def test_source_fails_soft_on_http_error():
    src = HackerNews(SourceCfg(max_requests=5, pause_seconds=0), _client(lambda r: httpx.Response(503)))
    assert src.fetch(["NVDA", "AMD"], NAMES, SINCE) == []
    assert src.errors and src.requests == 2


def test_source_fails_soft_on_exception_and_garbage():
    def boom(request):
        raise httpx.ConnectTimeout("timed out")
    assert Gdelt(SourceCfg(pause_seconds=0), _client(boom)).fetch(["NVDA"], NAMES, SINCE) == []
    g = Gdelt(SourceCfg(pause_seconds=0), _client(lambda r: httpx.Response(200, text="Please limit requests")))
    assert g.fetch(["NVDA"], NAMES, SINCE) == [] and "non-JSON" in g.errors[0]
    s = Substack(SourceCfg(pause_seconds=0, feeds=["https://x.example/feed"]), _client(lambda r: httpx.Response(404)))
    assert s.fetch(["NVDA"], NAMES, SINCE) == []


def test_hn_parsing_and_round_robin():
    seen = []

    def handler(request):
        seen.append(request.url.params["query"])
        assert request.headers["user-agent"] == "trading-bot-radar/0.1"
        return httpx.Response(200, json={"hits": [
            {"objectID": "1", "title": "Nvidia <b>earnings</b>", "created_at_i": int(NOW.timestamp()) - 60,
             "author": "bob", "_tags": ["story"]},
            {"objectID": "2", "comment_text": "<p>I think $AMD &amp; co.</p>", "story_title": "Nvidia thing",
             "created_at_i": int(NOW.timestamp()) - 30, "author": "eve", "_tags": ["comment"]},
        ]})

    src = HackerNews(SourceCfg(max_requests=1, pause_seconds=0), _client(handler))
    posts = src.fetch(["NVDA", "AMD"], NAMES, SINCE)
    assert seen == ["NVIDIA"] and src.cursor == 1
    assert posts[0].title == "Nvidia earnings" and posts[0].origin == "hn:bob"
    assert posts[1].title == "" and posts[1].text == "I think $AMD & co."  # comment: own words only
    src2 = HackerNews(SourceCfg(max_requests=1, pause_seconds=0), _client(handler))
    src2.cursor = 1
    src2.fetch(["NVDA", "AMD"], NAMES, SINCE)
    assert seen[-1] == "Advanced Micro Devices" and src2.cursor == 0


RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Nvidia stock jumps - Reuters</title><link>https://news.google.com/a1</link>
<pubDate>Tue, 03 Nov 2026 18:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
<item><title>Old news - Reuters</title><link>https://news.google.com/a0</link>
<pubDate>Tue, 20 Oct 2026 18:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
</channel></rss>"""


def test_google_news_parsing():
    qs = []

    def handler(request):
        qs.append(request.url.params["q"])
        return httpx.Response(200, text=RSS)

    src = GoogleNews(SourceCfg(max_requests=5, batch_size=3, pause_seconds=0), _client(handler))
    posts = src.fetch(["NVDA", "AMD"], NAMES, SINCE)
    assert len(qs) == 1 and '"NVIDIA"' in qs[0] and '"AMD"' in qs[0] and "when:" in qs[0]
    assert len(posts) == 1  # old item filtered by `since`
    p = posts[0]
    assert p.title == "Nvidia stock jumps" and p.origin == "reuters.com" and p.is_mainstream


def test_mainstream_domains():
    assert is_mainstream_domain("finance.yahoo.com", {"yahoo.com"})
    assert not is_mainstream_domain("notreuters.com", {"reuters.com"})


# ------------------------------------------------------------------ collector end to end (mocked)
def test_run_collect_end_to_end(tmp_path):
    (tmp_path / "settings.yaml").write_text("universe:\n  symbols: [NVDA, AMD]\n")
    udir = tmp_path / "universe"
    udir.mkdir()
    (udir / "2026-11-01.json").write_text(json.dumps({"symbols": ["ACME"]}))
    (udir / "2026-10-01.json").write_text(json.dumps({"symbols": ["OLD"]}))
    (tmp_path / "names.json").write_text(json.dumps({
        "fetched_at": NOW.isoformat(),
        "names": {"NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices, Inc.", "ACME": "Acme Rockets Inc."}}))

    def handler(request):
        host = request.url.host
        if host == "news.google.com":
            return httpx.Response(200, text=RSS)
        if host == "hn.algolia.com":
            return httpx.Response(200, json={"hits": [
                {"objectID": "9", "title": "Acme Rockets files 8-K", "created_at_i": int(NOW.timestamp()) - 60,
                 "author": "z", "_tags": ["story"]}]})
        return httpx.Response(500)

    cfg = RadarConfig(
        db_path=str(tmp_path / "radar.db"), names_cache=str(tmp_path / "names.json"),
        settings_path=str(tmp_path / "settings.yaml"), universe_dir=str(udir),
        sources={"google_news": SourceCfg(max_requests=1, batch_size=3, pause_seconds=0),
                 "hn": SourceCfg(max_requests=2, pause_seconds=0),
                 "gdelt": SourceCfg(max_requests=1, pause_seconds=0, batch_size=5),
                 "youtube": SourceCfg(enabled=False)},
    )
    stats = run_collect(cfg, client=_client(handler), now=NOW, allow_network_names=False, sleep=lambda s: None)
    assert stats["symbols"] == 3 and stats["smallcap"] == 1
    assert stats["sources"]["google_news"]["new_rows"] == 1
    assert stats["sources"]["hn"]["new_rows"] == 1      # same story returned twice -> deduped
    assert stats["sources"]["gdelt"]["errors"]           # 500 -> soft failure, run continues
    assert stats["sources"]["bluesky"]["skipped"] == "disabled"
    with sqlite3.connect(tmp_path / "radar.db") as c:
        rows = dict(c.execute("SELECT symbol, COUNT(*) FROM posts GROUP BY symbol").fetchall())
        assert rows == {"NVDA": 1, "ACME": 1}
        assert c.execute("SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL").fetchone()[0] == 1


def test_junk_titles():
    import re as _re

    from bot.radar.collect import JUNK_TITLES, is_junk
    pats = [_re.compile(p, _re.I) for p in JUNK_TITLES]

    def junk(title, origin="x.com"):
        return is_junk(_post("u", title, origin=origin), pats, ["stocktradersdaily.com"])

    assert junk("GLD Nov 2026 395.000 put (GLD261104P00395000) Seasonal Returns Chart")
    assert junk("IOVA261127C00020000 Interactive Stock Chart | IOVA Nov 2026 20.000 call Stock")
    assert junk("iShares 20+ Year Treasury Bond ETF (TLT) Stock Price, News, Quote & History")
    assert junk("3,782 Amazon.com, Inc. $AMZN Shares Purchased by Petredis Investment Advisors LLC")
    assert junk("Procter & Gamble Company (The) $PG Stock Acquired by Welch Group LLC")
    assert junk("Nippon Life Global Investors Americas Inc. Sells 241,550 Shares of Procter & Gamble")
    assert junk("Understanding the Setup: (LIND) and Scalable Risk", origin="news.stocktradersdaily.com")
    assert not junk("Lindblad Expeditions (NASDAQ:LIND) exceeds Q3 2026 expectations")
    assert not junk("Nvidia orders data center clients to reduce China exposure")


def test_gdelt_title_normalization():
    payload = {"articles": [{"url": "https://a.example/x", "domain": "reuters.com", "seendate": "20261103T180000Z",
                             "title": "Premarket : U . S . stock futures mixed as Amazon . com , Inc . slides"}]}
    src = Gdelt(SourceCfg(pause_seconds=0), _client(lambda r: httpx.Response(200, json=payload)), ["reuters.com"])
    (p,) = src.fetch(["AMZN"], {"AMZN": ["Amazon.com"]}, SINCE)
    assert p.title == "Premarket: U.S. stock futures mixed as Amazon.com, Inc. slides"
    assert p.is_mainstream and p.origin == "reuters.com"
