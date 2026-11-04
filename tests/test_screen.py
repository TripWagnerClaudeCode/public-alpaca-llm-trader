"""Screener tests with synthetic data (no network)."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from bot.journal.store import Journal
from bot.screen.screener import (WEIGHTS, ScreenParams, compute_features, exclusion_reason, run_screen, score,
                                 zscore_winsor)
from bot.screen.universe import load_universe, write_universe

AS_OF = dt.date(2026, 11, 3)


def asset(sym, name="Acme Corp. Common Stock", exchange="NASDAQ", **kw):
    a = {"class": "us_equity", "status": "active", "tradable": True, "exchange": exchange, "symbol": sym,
         "name": name, "fractionable": True, "shortable": True}
    a.update(kw)
    return a


# ---------------------------------------------------------------- exclusions
@pytest.mark.parametrize("a, reason", [
    (asset("ABCDW", "Abcd Holdings Inc. Warrant"), "warrant"),
    (asset("ABCDW", "Abcd Holdings Inc. Warrants expiring 2030"), "warrant"),
    (asset("ABCU", "Abc Acquisition Corp. Units, each consisting of one share"), "unit"),
    (asset("ABCR", "Abc Holdings Ltd. Rights"), "right"),
    (asset("ABCP", "Abc Bancorp 7.00% Series A Non-Cumulative Preferred Stock"), "preferred"),
    (asset("XYZ", "SPDR Something ETF", exchange="ARCA"), "etf"),
    (asset("XYZN", "iPath Bloomberg Commodity ETN", exchange="ARCA"), "etf"),
    (asset("BIGF", "BlackRock Municipal Income Fund, Inc.", exchange="NYSE"), "fund"),
    (asset("TRST", "Something Royalty Trust", exchange="NYSE"), "trust"),
    (asset("ABCL", "Abc Corp. 6.25% Notes due 2031"), "debt"),
    (asset("SPAK", "Blue Sky Acquisition Corp. Class A Ordinary Shares"), "spac"),
    (asset("CCXI", "Churchill Capital Corp XI Class A Ordinary Shares"), "spac"),
    (asset("ADRX", "Foo Group Ltd. American Depositary Shares"), "adr"),
    (asset("BRK.B", "Berkshire Hathaway Inc. Class B"), "special_symbol"),
    (asset("ABC/WS", "Abc Corp"), "special_symbol"),
    (asset("OTCX", exchange="OTC"), "exchange:OTC"),
    (asset("NOPE", tradable=False), "not_tradable"),
    (asset("OLD", status="inactive"), "inactive"),
    (asset("BTC", **{"class": "crypto"}), "not_us_equity"),
])
def test_exclusions(a, reason):
    assert exclusion_reason(a) == reason


@pytest.mark.parametrize("name", [
    "CareDx, Inc. Common Stock", "Trustmark Corporation Common Stock", "United Fire Group, Inc.",
    "Unity Bancorp", "Bright Horizons Family Solutions", "Fundamental Global Inc.", "The Manitowoc Company, Inc.",
])
def test_ordinary_names_not_excluded(name):
    assert exclusion_reason(asset("OK", name)) is None


def test_adr_spac_exclusion_toggles():
    p = ScreenParams(exclude_adrs=False, exclude_spacs=False)
    assert exclusion_reason(asset("ADRX", "Foo Ltd American Depositary Shares"), p) is None
    assert exclusion_reason(asset("SPAK", "Blue Sky Acquisition Corp."), p) is None


# ---------------------------------------------------------------- z-scores and scoring
def test_zscore_winsor_clips_and_fills():
    x = pd.Series([0.0] * 50 + [1000.0, np.nan], index=[f"S{i}" for i in range(52)])
    z = zscore_winsor(x, 3.0)
    assert z.max() == 3.0 and z.min() >= -3.0
    assert z["S51"] == 0.0                     # missing -> neutral
    assert zscore_winsor(pd.Series([1.0, 1.0, 1.0])).eq(0).all()   # zero dispersion -> 0
    assert zscore_winsor(pd.Series([5.0, np.nan])).eq(0).all()     # < 2 values -> 0


def test_score_uses_spec_weights_and_radar_zero():
    assert WEIGHTS == {"mom": 0.40, "ret_3m": 0.20, "vol_surge": 0.20, "prox_high": 0.20, "mention_z": 0.0}
    f = pd.DataFrame({"mom": [1.0, -1.0], "ret_3m": [0.0, 0.0], "vol_surge": [0.0, 0.0],
                      "prox_high": [0.5, 0.5], "mention_z": [100.0, -100.0]}, index=["A", "B"])
    s, z = score(f)
    # mom z = +/-1; others constant -> 0; radar weight 0 so mention_z changes nothing
    assert s["A"] == pytest.approx(0.40) and s["B"] == pytest.approx(-0.40)
    assert z.loc["A", "mention_z"] == pytest.approx(1.0)


def _series(n, start=10.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))


def test_features_definitions():
    idx = pd.bdate_range(end="2026-11-02", periods=300)
    c = pd.Series(np.linspace(10, 40, 300), index=idx)
    v = pd.Series(1e6, index=idx)
    v.iloc[-5:] = 3e6
    close, vol = pd.DataFrame({"X": c}), pd.DataFrame({"X": v})
    f = compute_features(close, vol).loc["X"]
    assert f["mom_window"] == "12-1"
    assert f["mom"] == pytest.approx(c.iloc[-22] / c.iloc[-253] - 1)
    assert f["ret_3m"] == pytest.approx(c.iloc[-1] / c.iloc[-64] - 1)
    dv = c * v
    assert f["vol_surge"] == pytest.approx(np.log(dv.iloc[-5:].mean() / dv.iloc[-50:].mean()))
    assert f["prox_high"] == pytest.approx(1.0)
    # short history -> 6-1 momentum
    f2 = compute_features(close.iloc[-150:], vol.iloc[-150:]).loc["X"]
    cs = c.iloc[-150:]
    assert f2["mom_window"] == "6-1" and f2["mom"] == pytest.approx(cs.iloc[-22] / cs.iloc[-127] - 1)


# ---------------------------------------------------------------- pipeline
def _panel(spec, n=300):
    """spec: {sym: dict(price=, vol=, n=, drift=, stale=)} -> wide close/volume/high."""
    idx = pd.bdate_range(end="2026-11-02", periods=n)
    close, volume = {}, {}
    for i, (s, d) in enumerate(spec.items()):
        px = _series(n, 1.0, d.get("drift", 0.0), seed=i)
        px = px / px[-1] * d.get("price", 20.0)
        c = pd.Series(px, index=idx)
        if d.get("n"):
            c.iloc[: n - d["n"]] = np.nan
        if d.get("stale"):
            c.iloc[-d["stale"]:] = np.nan
        close[s] = c
        volume[s] = pd.Series(d.get("vol", 1e6), index=idx)
    close = pd.DataFrame(close)
    return {"close": close, "volume": pd.DataFrame(volume), "high": close * 1.01}


def _screen(spec, caps, extra_assets=(), top_n=20, **pkw):
    assets = [asset(s) for s in spec] + list(extra_assets)
    panel = _panel(spec)
    seen = {}

    def fetch_prices(syms):
        seen["prices"] = list(syms)
        return {k: v[[s for s in syms if s in v.columns]] for k, v in panel.items()}

    def fetch_mcaps(syms):
        seen["mcaps"] = list(syms)
        return {s: caps.get(s, {"market_cap": None, "reason": "boom"}) for s in syms}

    res = run_screen(assets, fetch_prices, fetch_mcaps, AS_OF, ScreenParams(top_n=top_n, **pkw))
    return res, {r["symbol"]: r for r in res.rows}, seen


def test_pipeline_filters_and_reasons():
    ok = {"market_cap": 1e9, "quote_type": "EQUITY", "currency": "USD"}
    spec = {
        "GOOD1": dict(price=20, vol=1e6, drift=0.002),  # $20M ADV
        "GOOD2": dict(price=30, vol=1e6, drift=-0.001),
        "CHEAP": dict(price=2.5, vol=1e7),              # price < 3
        "THIN": dict(price=20, vol=1e4),                # ADV $200k
        "YOUNG": dict(price=20, vol=1e6, n=100),        # < 126 bars
        "STALE": dict(price=20, vol=1e6, stale=5),
        "TINY": dict(price=20, vol=1e6), "HUGE": dict(price=20, vol=1e6),
        "NOCAP": dict(price=20, vol=1e6), "FUNDY": dict(price=20, vol=1e6),
    }
    caps = {"GOOD1": ok, "GOOD2": ok, "TINY": {**ok, "market_cap": 1e8}, "HUGE": {**ok, "market_cap": 9e9},
            "FUNDY": {**ok, "quote_type": "ETF"}}
    res, rows, seen = _screen(spec, caps, extra_assets=[asset("WARR", "Foo Corp Warrant"), asset("NODATA")])
    assert rows["CHEAP"]["reason"] == "price<3"
    assert rows["THIN"]["reason"] == "adv20<5M"
    assert rows["YOUNG"]["reason"] == "history<126"
    assert rows["STALE"]["reason"] == "stale_price"
    assert rows["TINY"]["reason"] == "mcap<300M"
    assert rows["HUGE"]["reason"] == "mcap>5B"
    assert rows["NOCAP"]["reason"] == "mcap_unavailable:boom"
    assert rows["FUNDY"]["reason"] == "not_equity:ETF"
    assert rows["WARR"]["reason"] == "excluded:warrant"
    assert rows["NODATA"]["reason"] == "no_price_data"
    # market caps fetched only for price/volume/history survivors
    assert set(seen["mcaps"]) == {"GOOD1", "GOOD2", "TINY", "HUGE", "NOCAP", "FUNDY"}
    assert "WARR" not in seen["prices"]
    assert res.symbols == sorted(["GOOD1", "GOOD2"], key=lambda s: rows[s]["rank"])
    assert {rows[s]["passed"] for s in ("GOOD1", "GOOD2")} == {True}
    assert all(r["passed"] is False and r["reason"] for s, r in rows.items() if s not in ("GOOD1", "GOOD2"))
    c = res.counts
    assert (c["assets"], c["eligible"], c["price_data"], c["fresh"], c["price"], c["adv"], c["history"],
            c["mcap"], c["passed"], c["top"]) == (12, 11, 10, 9, 8, 7, 6, 2, 2, 2)
    top = res.top[0]
    assert top["fractionable"] is True and top["shortable"] is True and top["market_cap"] == 1e9


def test_pipeline_ranks_momentum_first_and_top_n():
    ok = {"market_cap": 1e9, "quote_type": "EQUITY", "currency": "USD"}
    spec = {f"S{i}": dict(price=20, vol=1e6, drift=0.0005 * (i - 10)) for i in range(25)}
    res, rows, _ = _screen(spec, {s: ok for s in spec}, top_n=5)
    assert len(res.symbols) == 5 and res.counts["passed"] == 25
    ranks = sorted(rows.values(), key=lambda r: r["rank"])
    assert [r["rank"] for r in ranks] == list(range(1, 26))
    assert all(a["score"] >= b["score"] for a, b in zip(ranks, ranks[1:]))
    assert res.symbols[0] in {"S24", "S23", "S22"}  # strongest drift wins
    p = res.payload("2026-11-03T20:00:00")
    assert p["as_of"] == "2026-11-03" and p["symbols"] == res.symbols
    assert p["candidates"][0]["rank"] == 1 and "mom" in p["candidates"][0]
    assert p["filters"]["min_price"] == 3.0 and p["filters"]["weights"]["mention_z"] == 0.0
    json.dumps(p)  # serializable


# ---------------------------------------------------------------- load_universe
def _write(d, name, symbols=("AAA",)):
    (d / name).write_text(json.dumps({"as_of": name[:10], "symbols": list(symbols)}))


def test_load_universe_newest_within_age(tmp_path):
    _write(tmp_path, "2026-10-25.json", ["OLD"])
    _write(tmp_path, "2026-11-01.json", ["NEW"])
    _write(tmp_path, "2026-11-08.json", ["FUTURE"])
    assert load_universe(tmp_path, dt.date(2026, 11, 3), 8)["symbols"] == ["NEW"]
    assert load_universe(tmp_path, dt.date(2026, 10, 30), 8)["symbols"] == ["OLD"]
    assert load_universe(tmp_path, dt.date(2026, 11, 9), 8)["symbols"] == ["FUTURE"]
    assert load_universe(tmp_path, dt.date(2026, 11, 7), 5) is None   # newest <= as_of is 6 days old
    assert load_universe(tmp_path, dt.date(2026, 11, 7), 6)["symbols"] == ["NEW"]
    assert load_universe(tmp_path, dt.date(2026, 11, 10), 1) is None  # stale


def test_load_universe_age_boundary(tmp_path):
    _write(tmp_path, "2026-11-01.json", ["X"])
    assert load_universe(tmp_path, dt.date(2026, 11, 9), 8)["symbols"] == ["X"]   # exactly 8 days
    assert load_universe(tmp_path, dt.date(2026, 11, 10), 8) is None


def test_load_universe_ignores_junk(tmp_path):
    _write(tmp_path, "2026-10-30.json", ["GOOD"])
    for junk in ("2026-11-02.json.tmp", "2026-11-02-dry.json", "latest.json", "x2026-11-02.json",
                 "2026-11-02.JSON", "2026-13-40.json", "2026-11-2.json"):
        _write(tmp_path, junk, ["JUNK"])
    (tmp_path / "2026-11-01.json").write_text("{not json")
    (tmp_path / "2026-11-02.json").write_text(json.dumps({"symbols": []}))
    assert load_universe(tmp_path, dt.date(2026, 11, 3), 8)["symbols"] == ["GOOD"]


def test_load_universe_missing_dir(tmp_path):
    assert load_universe(tmp_path / "nope", dt.date(2026, 11, 3), 8) is None


def test_write_universe_roundtrip(tmp_path):
    p = write_universe(tmp_path / "u", AS_OF, {"as_of": "2026-11-03", "symbols": ["A", "B"]})
    assert p.name == "2026-11-03.json"
    assert load_universe(tmp_path / "u", AS_OF, 8)["symbols"] == ["A", "B"]
    assert not list((tmp_path / "u").glob("*.tmp"))


# ---------------------------------------------------------------- journal
def test_journal_logs_every_candidate(tmp_path):
    ok = {"market_cap": 1e9, "quote_type": "EQUITY", "currency": "USD"}
    spec = {"GOOD1": dict(price=20, vol=1e6), "GOOD2": dict(price=25, vol=1e6), "CHEAP": dict(price=1, vol=1e8)}
    res, _, _ = _screen(spec, {"GOOD1": ok, "GOOD2": ok}, extra_assets=[asset("WARR", "Foo Warrants")])
    j = Journal(tmp_path / "j.db")
    assert j.log_screen(AS_OF, "2026-11-03T20:00:00", res.rows) == 4
    df = j.screen_candidates(AS_OF)
    assert len(df) == 4 and set(df["symbol"]) == {"GOOD1", "GOOD2", "CHEAP", "WARR"}
    failed = df[df["passed"] == 0].set_index("symbol")
    assert failed.loc["CHEAP", "reason"] == "price<3" and failed.loc["WARR", "reason"] == "excluded:warrant"
    good = df[df["passed"] == 1]
    assert sorted(good["rank"]) == [1, 2] and good["score"].notna().all()
    feats = json.loads(good.iloc[0]["features"])
    assert {"mom", "ret_3m", "vol_surge", "prox_high", "fractionable", "shortable", "market_cap"} <= set(feats)
    # a re-run on the same date supersedes the old one in the latest-run view; history is kept
    j.log_screen(AS_OF, "2026-11-03T21:00:00", res.rows[:2])
    assert len(j.screen_candidates(AS_OF)) == 2
    assert len(j.query("SELECT * FROM screen_candidates")) == 6
    assert len(j.screen_candidates(passed_only=True)) == sum(1 for r in res.rows[:2] if r["passed"])


def test_partial_session_detection():
    from bot.screen.sources import session_is_open_or_pending

    tue_noon = pd.Timestamp("2026-11-03 12:00", tz="America/New_York")
    tue_eve = pd.Timestamp("2026-11-03 17:00", tz="America/New_York")
    sun = pd.Timestamp("2026-11-08 20:00", tz="America/New_York")
    assert session_is_open_or_pending(dt.date(2026, 11, 3), tue_noon)
    assert not session_is_open_or_pending(dt.date(2026, 11, 3), tue_eve)
    assert not session_is_open_or_pending(dt.date(2026, 11, 2), tue_noon)
    assert not session_is_open_or_pending(dt.date(2026, 11, 8), sun)


def test_alpaca_keys_prefer_profile_b(monkeypatch):
    from bot.screen.sources import alpaca_keys

    monkeypatch.setenv("ALPACA_API_KEY", "a")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "as")
    monkeypatch.delenv("ALPACA_B_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_B_SECRET_KEY", raising=False)
    assert alpaca_keys() == ("a", "as")
    monkeypatch.setenv("ALPACA_B_API_KEY", "b")
    monkeypatch.setenv("ALPACA_B_SECRET_KEY", "bs")
    assert alpaca_keys() == ("b", "bs")


def test_profile_paths_fallbacks(tmp_path):
    from bot.screen.paths import profile_paths

    p = profile_paths("smallcap", root=tmp_path)  # no settings file yet
    assert p["screener_dir"] == tmp_path / "data/universe/smallcap"
    assert p["journal_db"] == tmp_path / "data/journal_smallcap.db"
    (tmp_path / "config").mkdir()
    (tmp_path / "config/settings_smallcap.yaml").write_text(
        "universe:\n  screener_dir: x/uni\npaths:\n  journal_db: x/j.db\n")
    p = profile_paths("smallcap", root=tmp_path)
    assert p["screener_dir"] == tmp_path / "x/uni" and p["journal_db"] == tmp_path / "x/j.db"
    assert profile_paths("default", root=tmp_path)["journal_db"] == tmp_path / "data/journal.db"
