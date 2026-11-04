"""Compute and store daily radar features, frozen at analyze time (no lookahead).

Only posts with ts <= freeze_ts (and fetched by then) are used. The default freeze is --as-of at 17:15 ET
(profile A analyze time; B analyzes at 17:30, so 17:15 is safe for both). See bot/radar/features.py for
the mention_z / baseline rules.

Usage: uv run python scripts/radar_features.py --as-of 2026-11-03 [--freeze-ts 2026-11-03T22:15:00Z] [--no-write]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from zoneinfo import ZoneInfo

from bot.radar.config import load_radar_config
from bot.radar.features import compute_features
from bot.radar.store import RadarStore, iso
from bot.radar.universe import load_universe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    ap.add_argument("--freeze-ts", default=None, help="ISO timestamp; naive = UTC. Default: as-of 17:15 ET")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_radar_config(args.config)
    as_of = dt.date.fromisoformat(args.as_of)
    if args.freeze_ts:
        freeze = dt.datetime.fromisoformat(args.freeze_ts.replace("Z", "+00:00"))
        freeze = freeze if freeze.tzinfo else freeze.replace(tzinfo=dt.timezone.utc)
    else:
        h, m = (int(x) for x in cfg.features.default_freeze_et.split(":"))
        freeze = dt.datetime(as_of.year, as_of.month, as_of.day, h, m, tzinfo=ZoneInfo("America/New_York"))
    freeze = freeze.astimezone(dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    if freeze > now:
        print(f"NOTE: freeze {iso(freeze)} is in the future; using posts collected so far (provisional). "
              f"Re-run after the freeze for the final values.")

    a, b = load_universe(cfg, as_of)
    symbols = a + b
    fc = cfg.features
    feats = compute_features(cfg.resolve(cfg.db_path), symbols, freeze, window_hours=fc.window_hours,
                             baseline_days=fc.baseline_days, min_baseline_days=fc.min_baseline_days,
                             std_floor=fc.std_floor, require_live_coverage=fc.require_live_coverage,
                             fringe_min_sources=fc.fringe_min_sources)
    if not args.no_write:
        RadarStore(cfg.resolve(cfg.db_path)).write_features(as_of, freeze, feats)

    print(f"\n=== radar features as_of={as_of} freeze={iso(freeze)}  {len(symbols)} symbols ===")
    print(f"{'symbol':7s} {'grp':3s} {'count':>5s} {'z':>7s} {'srcs':>4s} {'main':>4s} {'fringe':>6s} {'base_mean':>9s} {'base_days':>9s}")
    for s in sorted(symbols, key=lambda x: -feats[x]["mention_count"]):
        f = feats[s]
        z = "n/a" if f["mention_z"] is None else f"{f['mention_z']:+.2f}"
        bm = "n/a" if f["baseline_mean"] is None else f"{f['baseline_mean']:.2f}"
        print(f"{s:7s} {'A' if s in a else 'B':3s} {f['mention_count']:5d} {z:>7s} {f['n_sources']:4d} "
              f"{f['n_mainstream']:4d} {str(f['fringe_only']):>6s} {bm:>9s} {f['baseline_days']:9d}")
    if all(f["mention_z"] is None for f in feats.values()):
        print(f"mention_z is n/a until {fc.min_baseline_days} baseline days with live collector coverage exist.")
    print("" if args.no_write else f"Saved to features table in {cfg.db_path}")


if __name__ == "__main__":
    main()
