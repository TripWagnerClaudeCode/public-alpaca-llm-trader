"""Evaluate Kronos as a cross-sectional ranking signal on US daily data.

For each weekly evaluation date we forecast the next `pred_len` trading days for every
symbol using only data up to that date, then compare against realized returns.

Metric: Spearman rank IC between forecast and realized horizon return, per date,
averaged. Also reports a long-short spread (top third minus bottom third).

Evaluate on dates AFTER Kronos's pretraining data (default start 2025-09-01),
otherwise the model may simply remember the answers.

Usage: uv run python scripts/eval_kronos.py [--start 2025-09-01] [--lookbacks 120 250 400]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from bot.config import ROOT, load_settings
from bot.data.market import MarketData
from bot.signals.kronos_signal import KronosSignal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[120, 250, 400])
    ap.add_argument("--pred-len", type=int, default=10)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--every", type=int, default=5, help="trading days between eval dates")
    args = ap.parse_args()

    s = load_settings()
    md = MarketData(s.paths.resolve(s.paths.cache_dir))
    bars = md.daily_bars(s.all_symbols, days=(dt.date.today() - dt.date(2023, 6, 1)).days)
    closes = md.closes(bars)
    dates = closes.index[(closes.index >= args.start)]
    dates = dates[: len(dates) - args.pred_len : args.every]
    print(f"{len(dates)} eval dates from {dates[0].date()} to {dates[-1].date()}, {len(bars)} symbols")

    results = {}
    for lb in args.lookbacks:
        sig = KronosSignal(s.kronos.model_copy(update={"lookback": lb, "pred_len": args.pred_len, "sample_count": args.samples}))
        ics, spreads, rows = [], [], []
        for d in dates:
            hist = {sym: b.loc[:d] for sym, b in bars.items()}
            f = sig.forecast(hist)
            pos = closes.index.get_loc(d)
            fwd = closes.iloc[pos + args.pred_len] / closes.iloc[pos] - 1
            df = pd.DataFrame({"pred": {k: v.exp_return for k, v in f.items()}, "real": fwd}).dropna()
            if len(df) < 8:
                continue
            ic = df["pred"].rank().corr(df["real"].rank())
            q = df["pred"].rank(pct=True)
            spread = df.loc[q > 2 / 3, "real"].mean() - df.loc[q <= 1 / 3, "real"].mean()
            ics.append(ic)
            spreads.append(spread)
            rows.append({"date": str(d.date()), "ic": round(ic, 4), "spread": round(spread, 4),
                         "mean_pred": round(df["pred"].mean(), 4), "mean_real": round(df["real"].mean(), 4)})
            print(f"lb={lb} {d.date()} IC={ic:+.3f} spread={spread:+.4f}", flush=True)
        ics, spreads = np.array(ics), np.array(spreads)
        results[lb] = {
            "n_dates": int(len(ics)),
            "mean_ic": float(ics.mean()),
            "ic_tstat": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 2 else None,
            "hit_rate_ic_pos": float((ics > 0).mean()),
            "mean_ls_spread": float(spreads.mean()),
            "rows": rows,
        }
    out = ROOT / "data" / "kronos_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n=== Summary (horizon %d trading days) ===" % args.pred_len)
    for lb, r in results.items():
        print(f"lookback={lb:4d}  dates={r['n_dates']:3d}  meanIC={r['mean_ic']:+.4f}  t={r['ic_tstat']:+.2f}  "
              f"IC>0={r['hit_rate_ic_pos']:.0%}  LS spread={r['mean_ls_spread']:+.4%}")
    print(f"Saved {out}")
    print("Rule of thumb: keep Kronos only if mean IC > ~0.03 with t-stat > 2 for some lookback.")


if __name__ == "__main__":
    main()
