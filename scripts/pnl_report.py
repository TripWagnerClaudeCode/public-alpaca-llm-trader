"""Daily paper vs realistic P&L for a profile's journal.

  uv run python scripts/pnl_report.py --profile smallcap
  uv run python scripts/pnl_report.py --profile smallcap --csv reports/smallcap/pnl.csv

realistic P&L = paper P&L (equity vs the first equity row) - cumulative modeled trading cost
(half spread + sqrt-law impact, from the fill_costs table filled at reconcile). Benchmark = the
profile's universe.benchmark (IWM for smallcap) close-to-close return over the same period.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from bot.config import PROFILES, load_settings
from bot.journal.store import Journal

log = logging.getLogger("pnl_report")


def benchmark_returns(s, dates: list[str]) -> pd.Series:
    """Benchmark return from the first date to each date (as-of the last close on or before it). NaN on failure."""
    out = pd.Series(float("nan"), index=dates, dtype=float)
    if not dates:
        return out
    try:
        from bot.data.market import MarketData

        sym = s.universe.benchmark
        start = pd.Timestamp(dates[0])
        days = max(30, (pd.Timestamp.today() - start).days + 30)
        bars = MarketData(s.paths.resolve(s.paths.cache_dir)).daily_bars([sym], days)
        c = bars[sym]["close"].sort_index()
        c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
        base = c.asof(start)
        for d in dates:
            px = c.asof(pd.Timestamp(d))
            if pd.notna(base) and pd.notna(px) and base:
                out[d] = px / base - 1
    except Exception as e:  # noqa: BLE001 - the benchmark column is informational only
        log.warning("benchmark %s unavailable: %r", s.universe.benchmark, e)
    return out


def build_table(s) -> pd.DataFrame:
    j = Journal(s.paths.resolve(s.paths.journal_db))
    t = j.pnl_table()
    if t.empty:
        return t
    t["benchmark_ret"] = benchmark_returns(s, list(t["date"])).values
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smallcap")
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="also write the table to this CSV path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "openbb", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    s = load_settings(args.config, args.profile)
    t = build_table(s)
    print(f"Profile {s.profile}: journal {s.paths.journal_db}, benchmark {s.universe.benchmark}")
    if t.empty:
        print("No equity history yet (equity rows are recorded by execute and reconcile).")
        return 0
    hdr = f"{'date':<10}  {'equity':>12}  {'paper P&L':>18}  {'cum cost':>10}  {'realistic P&L':>18}  " \
          f"{s.universe.benchmark + ' ret':>9}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in t.iterrows():
        bm = f"{r['benchmark_ret']:+.2%}" if pd.notna(r["benchmark_ret"]) else "n/a"
        print(f"{r['date']:<10}  {r['equity']:>12,.2f}  {r['paper_pnl']:>+10,.2f} ({r['paper_pnl_pct']:+.2%})  "
              f"{r['cum_cost']:>10,.2f}  {r['realistic_pnl']:>+10,.2f} ({r['realistic_pnl_pct']:+.2%})  {bm:>9}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
