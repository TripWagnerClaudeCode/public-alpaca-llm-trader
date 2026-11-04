"""Network data for the screener: Alpaca asset list, bulk yfinance prices, market caps.

Everything fails soft per symbol: a failed ticker is simply missing from the result and
the screener logs it with a reason.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ET = "America/New_York"


# ---------------------------------------------------------------- Alpaca assets
def alpaca_keys() -> tuple[str, str]:
    """Profile B keys first, then A's. The asset list is the same for every account."""
    for k, s in (("ALPACA_B_API_KEY", "ALPACA_B_SECRET_KEY"), ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")):
        key, sec = os.environ.get(k, "").strip(), os.environ.get(s, "").strip()
        if key and sec:
            return key, sec
    raise RuntimeError("No Alpaca keys: set ALPACA_B_API_KEY/ALPACA_B_SECRET_KEY or ALPACA_API_KEY/ALPACA_SECRET_KEY in .env")


def alpaca_assets() -> list[dict]:
    """Raw /v2/assets dicts (active US equities). Raw JSON on purpose: alpaca-py's Asset model
    rejects exchanges it does not know yet (e.g. TXSE), which would crash the whole call."""
    from alpaca.trading.client import TradingClient

    key, sec = alpaca_keys()
    client = TradingClient(key, sec, paper=True)
    data = client.get("/assets", {"status": "active", "asset_class": "us_equity"})
    return list(data or [])


# ---------------------------------------------------------------- prices
def session_is_open_or_pending(as_of: dt.date, now: pd.Timestamp | None = None) -> bool:
    """True when `as_of` is today (ET) and the regular session has not closed yet, so the
    day's bar would be partial."""
    now = now or pd.Timestamp.now(tz=ET)
    return as_of == now.date() and now.weekday() < 5 and (now.hour, now.minute) < (16, 30)


def _quiet_yf():
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _download_chunk(symbols: list[str], start: dt.date, end: dt.date, retries: int = 2) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries + 1):
        try:
            df = yf.download(symbols, start=str(start), end=str(end), auto_adjust=True, threads=True,
                             progress=False, group_by="column", repair=False)
            if df is None or df.empty:
                raise RuntimeError("empty download")
            if not isinstance(df.columns, pd.MultiIndex):  # single-ticker shape
                df.columns = pd.MultiIndex.from_product([df.columns, symbols[:1]])
            df = df[[c for c in ("Close", "High", "Volume") if c in df.columns.get_level_values(0)]]
            long = df.stack(level=1, future_stack=True).reset_index()
            long.columns = ["date", "symbol", *[c.lower() for c in long.columns[2:]]]
            return long.dropna(subset=["close"])
        except Exception as e:  # noqa: BLE001 - fail soft, retry
            log.warning("price chunk (%d syms, attempt %d) failed: %s", len(symbols), attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame(columns=["date", "symbol", "close", "high", "volume"])


def download_prices(symbols: list[str], as_of: dt.date, cache_dir: Path, history_days: int = 300,
                    chunk: int = 200, drop_partial: bool | None = None) -> dict[str, pd.DataFrame]:
    """Bulk daily bars (adjusted) for about `history_days` trading days ending at `as_of`.

    Cached as long-format parquet keyed by as_of (plus `_intraday` when today's partial bar is
    dropped); symbols already attempted for that key are not re-downloaded."""
    _quiet_yf()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = session_is_open_or_pending(as_of) if drop_partial is None else drop_partial
    key = f"screen_prices_{as_of.isoformat()}{'_intraday' if partial else ''}"
    pq, attempted_f = cache_dir / f"{key}.parquet", cache_dir / f"{key}.attempted.json"
    cached = pd.DataFrame(columns=["date", "symbol", "close", "high", "volume"])
    attempted: set[str] = set()
    if pq.exists() and attempted_f.exists():
        try:
            cached = pd.read_parquet(pq)
            attempted = set(json.loads(attempted_f.read_text()))
        except Exception as e:  # noqa: BLE001
            log.warning("ignoring unreadable price cache %s: %s", pq, e)
            cached, attempted = cached.iloc[0:0], set()
    todo = [s for s in symbols if s not in attempted]
    if todo:
        start = as_of - dt.timedelta(days=int(history_days * 365 / 252) + 20)
        end = as_of + dt.timedelta(days=1)  # yfinance end is exclusive
        parts = [cached]
        t0 = time.time()
        for i in range(0, len(todo), chunk):
            part = _download_chunk(todo[i:i + chunk], start, end)
            parts.append(part)
            log.info("prices %d/%d (%.0fs)", min(i + chunk, len(todo)), len(todo), time.time() - t0)
        cached = pd.concat([p for p in parts if len(p)], ignore_index=True)
        cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None).dt.normalize()
        if partial:
            cached = cached[cached["date"].dt.date < as_of]
        cached = cached.drop_duplicates(["date", "symbol"], keep="last")
        cached.to_parquet(pq, index=False)
        attempted_f.write_text(json.dumps(sorted(attempted | set(todo))))
    want = cached[cached["symbol"].isin(set(symbols)) & (cached["date"].dt.date <= as_of)]
    out = {}
    for col in ("close", "high", "volume"):
        w = want.pivot(index="date", columns="symbol", values=col).sort_index()
        out[col] = w.iloc[-history_days:] if len(w) > history_days else w
    # drop calendar rows nobody traded on (holidays leaking from odd tickers)
    if len(out["close"]):
        counts = out["close"].notna().sum(axis=1)
        good = counts >= max(1, 0.2 * counts.max())
        out = {k: v.loc[good] for k, v in out.items()}
    return out


# ---------------------------------------------------------------- market caps
_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"


def _batch_quotes(symbols: list[str], chunk: int = 150) -> dict[str, dict]:
    """Yahoo's batch quote endpoint via yfinance's authenticated session (crumb handled by yfinance).
    Fast path: about 0.3 s per 150 symbols. Any failure just leaves symbols for the slow path."""
    out: dict[str, dict] = {}
    try:
        from yfinance.data import YfData

        yd = YfData()
    except Exception as e:  # noqa: BLE001
        log.warning("batch quote unavailable: %s", e)
        return out
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        try:
            r = yd.get_raw_json(_QUOTE_URL, params={"symbols": ",".join(part)})
            for q in (r.get("quoteResponse") or {}).get("result") or []:
                out[q["symbol"]] = {
                    "market_cap": q.get("marketCap"),
                    "quote_type": q.get("quoteType"),
                    "currency": q.get("currency"),
                    "long_name": q.get("longName"),
                    "shares_outstanding": q.get("sharesOutstanding"),
                    "source": "quote",
                }
        except Exception as e:  # noqa: BLE001
            log.warning("batch quote chunk failed (%d syms): %s", len(part), e)
    return out


def _one_info(sym: str) -> dict:
    import yfinance as yf

    t = yf.Ticker(sym)
    try:
        info = t.info or {}
        m = info.get("marketCap")
        if m:
            return {"market_cap": m, "quote_type": info.get("quoteType"), "currency": info.get("currency"),
                    "country": info.get("country"), "sector": info.get("sector"),
                    "industry": info.get("industry"), "source": "info"}
    except Exception as e:  # noqa: BLE001
        info_err = str(e)[:80]
    else:
        info_err = "no marketCap in info"
    try:
        m = t.fast_info["marketCap"]
        if m:
            return {"market_cap": float(m), "source": "fast_info"}
    except Exception as e:  # noqa: BLE001
        return {"market_cap": None, "reason": f"info: {info_err}; fast_info: {str(e)[:80]}"}
    return {"market_cap": None, "reason": info_err}


def fetch_market_caps(symbols: list[str], as_of: dt.date, cache_dir: Path, workers: int = 8) -> dict[str, dict]:
    """Market cap per symbol: batch quote first, then per-symbol `info`/`fast_info` in a thread pool
    for misses. Cached per as_of date. Missing caps come back with a `reason`, never raise."""
    _quiet_yf()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq = cache_dir / f"screen_mcap_{as_of.isoformat()}.parquet"
    cached: dict[str, dict] = {}
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
            cached = {r["symbol"]: {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "symbol"}
                      for r in df.to_dict("records")}
        except Exception as e:  # noqa: BLE001
            log.warning("ignoring unreadable mcap cache %s: %s", pq, e)
    todo = [s for s in symbols if not (cached.get(s) or {}).get("market_cap")]
    if todo:
        got = _batch_quotes(todo)
        miss = [s for s in todo if not (got.get(s) or {}).get("market_cap")]
        if miss:
            log.info("market cap: %d via batch quote, %d via per-symbol info", len(todo) - len(miss), len(miss))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for s, r in zip(miss, ex.map(_safe_info, miss)):
                    if r.get("market_cap") or s not in got:
                        got[s] = {**got.get(s, {}), **r}
                    else:
                        got[s]["reason"] = r.get("reason")
        cached.update(got)
        rows = [{"symbol": s, **{k: v for k, v in d.items() if isinstance(v, (str, int, float)) or v is None}}
                for s, d in cached.items()]
        pd.DataFrame(rows).to_parquet(pq, index=False)
    return {s: cached.get(s, {"market_cap": None, "reason": "no_response"}) for s in symbols}


def _safe_info(sym: str) -> dict:
    try:
        return _one_info(sym)
    except Exception as e:  # noqa: BLE001
        return {"market_cap": None, "reason": str(e)[:120]}


def fetch_details(symbols: list[str], workers: int = 8) -> dict[str, dict]:
    """country/sector/industry via `info`, for a short list (the top N). Fails soft to {}."""
    _quiet_yf()

    def one(s):
        try:
            import yfinance as yf

            i = yf.Ticker(s).info or {}
            return {k: i.get(src) for k, src in (("country", "country"), ("sector", "sector"),
                                                  ("industry", "industry"), ("long_name", "longName"))}
        except Exception:  # noqa: BLE001
            return {}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(symbols, ex.map(one, symbols)))
