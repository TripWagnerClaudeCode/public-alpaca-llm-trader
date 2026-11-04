"""Screener evaluation (hypothesis H1): does the score rank forward 10-day returns?

Per screen date: Spearman rank IC of score vs forward return across all passing candidates,
and the top-quintile minus bottom-quintile mean return. Across dates: mean IC, t-stat, hit rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(closes: pd.DataFrame, date, horizon: int) -> tuple[pd.Series, str | None, str | None] | None:
    """Return from the last close <= date to the close `horizon` trading days later, or None if
    that horizon has not elapsed yet."""
    idx = closes.index
    pos = idx.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0 or pos + horizon >= len(idx):
        return None
    start, end = closes.iloc[pos], closes.iloc[pos + horizon]
    return end / start - 1, str(idx[pos].date()), str(idx[pos + horizon].date())


def evaluate(cands: pd.DataFrame, closes: pd.DataFrame, horizon: int = 10, min_names: int = 10) -> dict:
    """cands: columns screen_date, symbol, score, rank (passing candidates only)."""
    rows, pending = [], []
    for d, g in cands.groupby("screen_date"):
        fr = forward_returns(closes, d, horizon)
        if fr is None:
            pending.append(str(d))
            continue
        fwd, start, end = fr
        df = pd.DataFrame({"score": g.set_index("symbol")["score"], "rank": g.set_index("symbol")["rank"]})
        df["fwd"] = fwd.reindex(df.index)
        df = df.dropna(subset=["score", "fwd"])
        if len(df) < min_names:
            pending.append(f"{d} (only {len(df)} names with returns)")
            continue
        ic = df["score"].rank().corr(df["fwd"].rank())
        q = df["score"].rank(pct=True)
        spread = df.loc[q > 0.8, "fwd"].mean() - df.loc[q <= 0.2, "fwd"].mean()
        top20 = df.loc[df["rank"] <= 20, "fwd"].mean()
        rows.append({"date": str(d), "from": start, "to": end, "n": int(len(df)), "ic": float(ic),
                     "q5_minus_q1": float(spread), "top20_minus_all": float(top20 - df["fwd"].mean()),
                     "mean_fwd": float(df["fwd"].mean())})
    ics = np.array([r["ic"] for r in rows])
    spreads = np.array([r["q5_minus_q1"] for r in rows])
    n = len(ics)
    out = {"horizon": horizon, "n_dates": n, "pending": pending, "rows": rows,
           "mean_ic": float(ics.mean()) if n else None,
           "ic_tstat": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n))) if n > 2 and ics.std(ddof=1) > 0 else None,
           "hit_rate_ic_pos": float((ics > 0).mean()) if n else None,
           "mean_q5_minus_q1": float(spreads.mean()) if n else None,
           "spread_tstat": (float(spreads.mean() / (spreads.std(ddof=1) / np.sqrt(n)))
                            if n > 2 and spreads.std(ddof=1) > 0 else None)}
    return out
