"""Market data via OpenBB (yfinance for prices/fundamentals, nasdaq for news).

OpenBB ODP 5.x groups commands by provider (obb.yfinance.*), and the first import
builds its static assets (~1 min). Results are cached to Parquet per day.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _obb():
    from openbb import obb  # heavy import, deferred

    return obb


class MarketData:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- prices ----------
    def daily_bars(self, symbols: list[str], days: int, end: dt.date | None = None) -> dict[str, pd.DataFrame]:
        """Return {symbol: DataFrame[open, high, low, close, volume]} indexed by date."""
        end = end or dt.date.today()
        start = end - dt.timedelta(days=days)
        key = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:10]
        cache = self.cache_dir / f"bars_{start}_{end}_{key}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = (
                _obb()
                .yfinance.equity.price.historical(symbol=",".join(symbols), start_date=str(start), end_date=str(end))
                .to_dataframe()
            )
            if "symbol" not in df.columns:  # single-symbol responses omit the column
                df["symbol"] = symbols[0]
            df.index = pd.to_datetime(df.index)
            df.to_parquet(cache)
        out = {}
        for sym, g in df.groupby("symbol"):
            g = g[["open", "high", "low", "close", "volume"]].sort_index()
            out[str(sym)] = g[~g.index.duplicated(keep="last")]
        missing = set(symbols) - set(out)
        if missing:
            log.warning("No price data for %s", sorted(missing))
        return out

    @staticmethod
    def closes(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame({s: b["close"] for s, b in bars.items()}).sort_index()

    # ---------- fundamentals ----------
    def fundamentals(self, symbol: str) -> dict:
        keep = [
            "market_cap", "pe_ratio", "forward_pe", "peg_ratio", "earnings_growth", "revenue_growth",
            "gross_margin", "operating_margin", "profit_margin", "return_on_equity", "debt_to_equity",
            "current_ratio", "dividend_yield", "beta", "price_return_1y",
        ]
        try:
            row = _obb().yfinance.equity.fundamental.metrics(symbol=symbol).to_dataframe().iloc[0]
            return {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in keep if k in row.index}
        except Exception as e:  # ETFs and some tickers have no metrics
            log.info("fundamentals unavailable for %s: %s", symbol, e)
            return {}

    # ---------- news ----------
    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        try:
            df = _obb().news.company(symbol=symbol, limit=limit, provider="nasdaq").to_dataframe()
        except Exception as e:
            log.info("news unavailable for %s: %s", symbol, e)
            return []
        items = []
        for ts, r in df.head(limit).iterrows():
            items.append({
                "date": str(ts)[:10],
                "title": str(r.get("title", ""))[:200],
                "excerpt": str(r.get("excerpt", "") or "")[:300],
            })
        return items


def latest_bar_age_days(bars: dict[str, pd.DataFrame], today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    last = max(b.index.max() for b in bars.values()).date()
    return (today - last).days
