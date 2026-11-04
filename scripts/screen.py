"""Weekly small-cap screener (profile B). Writes data/universe/smallcap/YYYY-MM-DD.json and logs
every candidate to the profile journal's screen_candidates table.

Usage: uv run python scripts/screen.py --profile smallcap [--as-of YYYY-MM-DD] [--dry] [--top 20]

--dry prints the result without writing the universe file or the journal.
Note for --as-of in the past: the Alpaca asset list and market caps are as of today (survivorship
and cap look-ahead); prices are truncated to as_of.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bot.journal.store import Journal  # noqa: E402
from bot.screen import sources  # noqa: E402
from bot.screen.paths import profile_paths  # noqa: E402
from bot.screen.screener import ScreenParams, run_screen  # noqa: E402
from bot.screen.universe import write_universe  # noqa: E402


def _fmt_money(x):
    if x is None:
        return "-"
    return f"{x / 1e9:.2f}B" if x >= 1e9 else f"{x / 1e6:.0f}M"


def _pct(x):
    return "-" if x is None else f"{x:+.0%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="smallcap")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--dry", action="store_true", help="do not write the universe JSON or the journal")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--history-days", type=int, default=300)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(ROOT / ".env")
    t0 = time.time()
    as_of = args.as_of or dt.date.today()
    paths = profile_paths(args.profile)
    params = ScreenParams(top_n=args.top)
    cache_dir = paths["cache_dir"]

    print(f"Screen {args.profile} as of {as_of} (top {args.top}){' [dry]' if args.dry else ''}", flush=True)
    assets = sources.alpaca_assets()
    print(f"  {len(assets)} Alpaca assets ({time.time() - t0:.0f}s)", flush=True)
    if sources.session_is_open_or_pending(as_of):
        print("  market session not closed yet: today's partial bar is dropped", flush=True)

    def fetch_prices(syms):
        t = time.time()
        out = sources.download_prices(syms, as_of, cache_dir, history_days=args.history_days)
        print(f"  prices: {out['close'].shape[1]} of {len(syms)} symbols, {out['close'].shape[0]} bars "
              f"({time.time() - t:.0f}s)", flush=True)
        return out

    def fetch_mcaps(syms):
        t = time.time()
        out = sources.fetch_market_caps(syms, as_of, cache_dir)
        ok = sum(1 for v in out.values() if v.get("market_cap"))
        print(f"  market caps: {ok} of {len(syms)} ({time.time() - t:.0f}s)", flush=True)
        return out

    res = run_screen(assets, fetch_prices, fetch_mcaps, as_of, params)

    # country/sector for the shortlist only (cheap, informative for the agents and sanity checks)
    details = sources.fetch_details(res.symbols) if res.symbols else {}
    for c in res.top:
        for k, v in (details.get(c["symbol"]) or {}).items():
            if v is not None and c.get(k) is None:
                c[k] = v

    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print("\nStage counts (survivors after each filter):")
    for k, v in res.counts.items():
        print(f"  {k:<12}{v:>7}")
    reasons: dict[str, int] = {}
    for r in res.rows:
        if r["passed"] is False:
            key = "mcap_unavailable" if r["reason"].startswith("mcap_unavailable") else r["reason"]
            reasons[key] = reasons.get(key, 0) + 1
    print("Fail reasons:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28}{v:>7}")

    print(f"\nTop {len(res.top)}:")
    print(f"{'#':>3} {'symbol':<7}{'score':>7} {'price':>8} {'mcap':>8} {'adv20':>8} {'mom':>6}{'win':>5} "
          f"{'3m':>6} {'vsurge':>7} {'%hi':>5}  frac short  name")
    for c in res.top:
        print(f"{c['rank']:>3} {c['symbol']:<7}{c['score']:>7.2f} {c['price']:>8.2f} {_fmt_money(c.get('market_cap')):>8} "
              f"{_fmt_money(c.get('adv20')):>8} {_pct(c.get('mom')):>6}{(c.get('mom_window') or ''):>5} "
              f"{_pct(c.get('ret_3m')):>6} {c.get('vol_surge') or 0:>+7.2f} {c.get('prox_high') or 0:>5.0%}  "
              f"{'Y' if c['fractionable'] else 'n':^4} {'Y' if c['shortable'] else 'n':^5} "
              f"{(c.get('name') or '')[:40]} {('[' + c['country'] + ']') if c.get('country') else ''}")

    if res.counts.get("passed", 0) < args.top:
        print(f"\nWARNING: only {res.counts.get('passed', 0)} candidates passed (wanted {args.top}).")
    if not args.dry:
        if res.symbols:
            out = write_universe(paths["screener_dir"], as_of, res.payload(generated_at))
            print(f"\nWrote {out}")
        else:
            print("\nNo candidates passed; universe file NOT written (the daily run will fail closed).")
        j = Journal(paths["journal_db"])
        n = j.log_screen(as_of, generated_at, res.rows)
        print(f"Logged {n} candidates to {paths['journal_db']} (screen_candidates)")
    print(f"Elapsed {time.time() - t0:.0f}s")
    return 0 if res.symbols else 1


if __name__ == "__main__":
    sys.exit(main())
