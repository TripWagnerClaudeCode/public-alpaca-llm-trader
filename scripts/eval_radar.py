"""Evaluate radar features against forward returns (spec hypotheses H2/H3), modeled on eval_kronos.py.

For every as_of date in the radar features table, compare each symbol's frozen features with its realized
forward return from the close on as_of to the close `horizon` trading days later.

- H2: Spearman rank IC of mention_z vs forward return, per date, averaged. Split into profile A symbols
  (config/settings.yaml) and small-cap symbols (any screener universe file). Expect IC > 0 for small caps only.
- H3: fringe_only effect: mean forward return and mean |forward return| for fringe_only vs not (pooled).

Usage: uv run python scripts/eval_radar.py [--horizon 10] [--min-symbols 5]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3

import numpy as np
import pandas as pd

from bot.config import ROOT
from bot.radar.config import load_radar_config
from bot.radar.universe import all_smallcap_symbols_ever, profile_a_symbols


def tstat(x: np.ndarray) -> float | None:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else None


def welch(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--min-symbols", type=int, default=5, help="min symbols with mention_z per date and group")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_radar_config(args.config)
    db = cfg.resolve(cfg.db_path)
    if not db.exists():
        print("Not enough data yet: no radar database (run scripts/radar_collect.py on a schedule first).")
        return
    with sqlite3.connect(db) as c:
        feats = pd.read_sql("SELECT as_of, symbol, mention_count, mention_z, n_sources, fringe_only FROM features", c)
    if feats.empty:
        print("Not enough data yet: features table is empty (run scripts/radar_features.py daily).")
        return

    a_syms = set(profile_a_symbols(cfg.resolve(cfg.settings_path)))
    b_syms = all_smallcap_symbols_ever(cfg.resolve(cfg.universe_dir)) - a_syms
    feats["group"] = np.where(feats["symbol"].isin(a_syms), "A", np.where(feats["symbol"].isin(b_syms), "smallcap", "other"))
    feats["as_of"] = pd.to_datetime(feats["as_of"])
    # Forward returns need horizon trading days (~1.5x calendar days) after as_of.
    latest_possible = pd.Timestamp(dt.date.today()) - pd.Timedelta(days=int(args.horizon * 1.4) + 1)
    usable = feats[feats["as_of"] <= latest_possible]
    n_dates_all = feats["as_of"].nunique()
    n_z = int(feats["mention_z"].notna().sum())
    if usable.empty or usable["mention_z"].notna().sum() == 0:
        print(f"Not enough data yet: {n_dates_all} feature dates, {n_z} rows with mention_z, "
              f"0 with a full {args.horizon}-day forward window. Keep the collector running ~6-8 weeks.")
        return

    from bot.config import load_settings
    from bot.data.market import MarketData

    s = load_settings()
    md = MarketData(s.paths.resolve(s.paths.cache_dir))
    syms = sorted(usable["symbol"].unique())
    start = usable["as_of"].min().date() - dt.timedelta(days=10)
    bars = md.daily_bars(syms, days=(dt.date.today() - start).days)
    closes = md.closes(bars)

    def fwd_return(row) -> float:
        if row["symbol"] not in closes:
            return np.nan
        col = closes[row["symbol"]].dropna()
        pos = col.index.searchsorted(row["as_of"], side="right") - 1  # last close on/before as_of
        if pos < 0 or pos + args.horizon >= len(col):
            return np.nan
        return float(col.iloc[pos + args.horizon] / col.iloc[pos] - 1)

    usable = usable.copy()
    usable["fwd"] = usable.apply(fwd_return, axis=1)
    usable = usable.dropna(subset=["fwd"])

    results: dict = {"horizon": args.horizon, "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "groups": {}}
    for grp in ["A", "smallcap"]:
        g = usable[usable["group"] == grp]
        ics, rows = [], []
        for d, gd in g.dropna(subset=["mention_z"]).groupby("as_of"):
            if len(gd) < args.min_symbols or gd["mention_z"].nunique() < 2:
                continue
            ic = gd["mention_z"].rank().corr(gd["fwd"].rank())
            if np.isfinite(ic):
                ics.append(ic)
                rows.append({"date": str(d.date()), "ic": round(float(ic), 4), "n": int(len(gd))})
        ics = np.array(ics)
        fr, nf = g[g["fringe_only"] == 1]["fwd"].to_numpy(), g[g["fringe_only"] == 0]["fwd"].to_numpy()
        results["groups"][grp] = {
            "n_dates": int(len(ics)),
            "mean_ic": float(ics.mean()) if len(ics) else None,
            "ic_tstat": tstat(ics) if len(ics) else None,
            "hit_rate_ic_pos": float((ics > 0).mean()) if len(ics) else None,
            "fringe_n": int(len(fr)), "nonfringe_n": int(len(nf)),
            "fringe_mean_fwd": float(fr.mean()) if len(fr) else None,
            "nonfringe_mean_fwd": float(nf.mean()) if len(nf) else None,
            "fringe_mean_abs_fwd": float(np.abs(fr).mean()) if len(fr) else None,
            "nonfringe_mean_abs_fwd": float(np.abs(nf).mean()) if len(nf) else None,
            "fringe_abs_welch_t": welch(np.abs(fr), np.abs(nf)),
            "rows": rows,
        }
    out = ROOT / "data" / "radar_eval.json"
    out.write_text(json.dumps(results, indent=2))

    def f(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    print(f"\n=== Radar eval (horizon {args.horizon} trading days) ===")
    for grp, r in results["groups"].items():
        print(f"{grp:9s} dates={r['n_dates']:3d}  meanIC={f(r['mean_ic'], '+.4f')}  t={f(r['ic_tstat'], '+.2f')}  "
              f"IC>0={f(r['hit_rate_ic_pos'], '.0%')}")
        print(f"{'':9s} fringe_only n={r['fringe_n']} mean fwd={f(r['fringe_mean_fwd'], '+.2%')} "
              f"|fwd|={f(r['fringe_mean_abs_fwd'], '.2%')}  vs others n={r['nonfringe_n']} "
              f"mean fwd={f(r['nonfringe_mean_fwd'], '+.2%')} |fwd|={f(r['nonfringe_mean_abs_fwd'], '.2%')}  "
              f"t(|fwd|)={f(r['fringe_abs_welch_t'], '+.2f')}")
    if all(r["n_dates"] < 20 for r in results["groups"].values()):
        print("Not enough data yet for a verdict (< 20 dates per group); treat these numbers as noise.")
    print(f"Saved {out}")
    print("Rule of thumb (H2): wire mention_z in only if small-cap mean IC > ~0.03 with t > 2, and A is ~0.")


if __name__ == "__main__":
    main()
