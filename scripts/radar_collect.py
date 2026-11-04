"""Radar collector: one pass over every enabled source -> data/radar.db (shadow mode, no LLM).

Scheduled every 30 min by scripts/register_radar_task.ps1. Every source fails soft; the run always exits 0
unless the config itself is broken.

Usage: uv run python scripts/radar_collect.py [--config config/radar.yaml] [--json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging

from bot.radar.collect import run_collect
from bot.radar.config import load_radar_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true", help="print the run stats as JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_radar_config(args.config)
    stats = run_collect(cfg)
    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return
    print(f"\n=== radar collect {dt.datetime.now().isoformat(timespec='seconds')}  "
          f"{stats['symbols']} symbols (A={stats['profile_a']}, small-cap={stats['smallcap']})  {stats['seconds']}s ===")
    for name, st in stats["sources"].items():
        if st["skipped"]:
            print(f"{name:12s} skipped: {st['skipped']}")
            continue
        print(f"{name:12s} req={st['requests']:2d} fetched={st['fetched']:4d} junk={st.get('junk', 0):3d} matched={st['matched_posts']:4d} "
              f"new_rows={st['new_rows']:4d} dupes={st['dupes']:4d} errors={len(st['errors'])} ({st['seconds']}s)")
        for e in st["errors"]:
            print(f"{'':12s}   ! {e[:160]}")


if __name__ == "__main__":
    main()
