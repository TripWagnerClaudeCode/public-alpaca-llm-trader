"""Evaluate the weekly screener (hypothesis H1: rank IC > 0 with t > 2 over 10 trading days).

For every past screen date in the profile journal's screen_candidates table (latest run per date),
compute the Spearman rank IC of score vs the forward `--horizon`-trading-day return across all
passing candidates, the IC t-stat across dates, and the top-minus-bottom quintile spread.
Forward returns start at the last close <= screen date (the close the screen was computed on).

Usage: uv run python scripts/eval_screener.py [--profile smallcap] [--horizon 10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bot.journal.store import Journal  # noqa: E402
from bot.screen.evaluate import evaluate  # noqa: E402
from bot.screen.paths import profile_paths  # noqa: E402
from bot.screen.sources import _download_chunk, _quiet_yf  # noqa: E402


def load_closes(symbols: list[str], start: dt.date, cache_dir: Path) -> pd.DataFrame:
    _quiet_yf()
    today = dt.date.today()
    cache = cache_dir / f"screen_eval_{start}_{today}.parquet"
    if cache.exists():
        long = pd.read_parquet(cache)
    else:
        parts = [_download_chunk(symbols[i:i + 200], start, today + dt.timedelta(days=1))
                 for i in range(0, len(symbols), 200)]
        long = pd.concat([p for p in parts if len(p)], ignore_index=True) if parts else pd.DataFrame()
        if len(long):
            long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
            cache_dir.mkdir(parents=True, exist_ok=True)
            long.to_parquet(cache, index=False)
    if long.empty:
        return pd.DataFrame()
    w = long.pivot_table(index="date", columns="symbol", values="close").sort_index()
    # never use a still-forming bar of today
    if len(w) and w.index[-1].date() == today and pd.Timestamp.now(tz="America/New_York").hour < 17:
        w = w.iloc[:-1]
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smallcap")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--min-names", type=int, default=10)
    args = ap.parse_args(argv)

    paths = profile_paths(args.profile)
    if not paths["journal_db"].exists():
        print(f"No journal at {paths['journal_db']}: not enough data yet.")
        return 0
    cands = Journal(paths["journal_db"]).screen_candidates(passed_only=True)
    if cands.empty:
        print("No screen_candidates logged yet: not enough data yet.")
        return 0
    dates = sorted(cands["screen_date"].unique())
    first = dt.date.fromisoformat(dates[0]) - dt.timedelta(days=10)
    if (dt.date.today() - dt.date.fromisoformat(dates[0])).days < args.horizon * 7 / 5:
        print(f"{len(dates)} screen date(s) ({dates[0]}..{dates[-1]}); none has {args.horizon} trading days "
              f"of forward returns yet: not enough data yet.")
        return 0
    syms = sorted(cands["symbol"].unique())
    print(f"{len(dates)} screen dates, {len(syms)} symbols; downloading closes from {first}...", flush=True)
    closes = load_closes(syms, first, paths["cache_dir"])
    res = evaluate(cands, closes, args.horizon, args.min_names)

    for r in res["rows"]:
        print(f"{r['date']}  n={r['n']:4d}  IC={r['ic']:+.3f}  Q5-Q1={r['q5_minus_q1']:+.2%}  "
              f"top20-all={r['top20_minus_all']:+.2%}  ({r['from']} -> {r['to']})")
    if res["pending"]:
        print(f"pending (horizon not elapsed / too few names): {', '.join(res['pending'])}")
    out = ROOT / "data" / f"screen_eval_{args.profile}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== Screener eval (horizon {args.horizon} trading days) ===")
    if res["n_dates"] == 0:
        print("Not enough data yet.")
    else:
        t = res["ic_tstat"]
        st = res["spread_tstat"]
        print(f"dates={res['n_dates']}  meanIC={res['mean_ic']:+.4f}  t={'n/a' if t is None else f'{t:+.2f}'}  "
              f"IC>0={res['hit_rate_ic_pos']:.0%}  Q5-Q1={res['mean_q5_minus_q1']:+.2%} "
              f"(t={'n/a' if st is None else f'{st:+.2f}'})")
        if res["n_dates"] < 8:
            print("Fewer than 8 dates: treat as anecdotal.")
    print(f"Saved {out}")
    print("H1 passes if mean IC > 0 with t > 2. Overlapping 10-day windows on weekly dates inflate t a bit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
