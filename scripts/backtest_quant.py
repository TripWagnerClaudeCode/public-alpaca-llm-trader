"""Walk-forward backtest of the quantitative baselines. This is the bar the agent team must beat.

Strategies (rebalanced every --every trading days, long-only, using only past data):
  SPY          buy and hold the benchmark
  EqualWeight  1/N across the universe (minus the cash buffer)
  HRP          skfolio Hierarchical Risk Parity (what the bot holds when it has no views)
  BL+Kronos    Black-Litterman with Kronos relative forecasts as views (opt-in: --kronos, slow on CPU)

LLM agents are not backtestable without look-ahead bias (the models know what happened), so
they are evaluated by forward paper trading instead. See the README.

Usage: uv run python scripts/backtest_quant.py [--start 2021-01-01] [--every 10] [--cost-bps 5] [--kronos]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from bot.config import load_settings
from bot.data.market import MarketData
from bot.models import View
from bot.portfolio.optimizer import target_weights


def metrics(equity: pd.Series, turnover: float, years: float) -> dict:
    r = equity.pct_change().dropna()
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": (r.mean() * 252) / vol if vol else np.nan,
            "MaxDD": dd, "Calmar": cagr / abs(dd) if dd else np.nan, "AnnTurnover": turnover / years}


def run(closes: pd.DataFrame, weights_fn, start: str, every: int, cost_bps: float) -> tuple[pd.Series, float]:
    rets = closes.pct_change().fillna(0.0)
    dates = closes.index[closes.index >= start]
    w = pd.Series(0.0, index=closes.columns)
    equity, val, turnover = [], 1.0, 0.0
    for i, d in enumerate(dates):
        if i % every == 0:
            hist = closes.loc[:d].iloc[:-1]  # only data strictly before today's close is known
            new_w = weights_fn(hist, d).reindex(closes.columns, fill_value=0.0)
            tr = float((new_w - w).abs().sum())
            val *= 1 - tr * cost_bps / 1e4
            turnover += tr
            w = new_w
        day_r = float((w * rets.loc[d]).sum())
        val *= 1 + day_r
        # drift weights with prices
        grown = w * (1 + rets.loc[d])
        w = grown / (1 + day_r) if (1 + day_r) else grown
        equity.append(val)
    return pd.Series(equity, index=dates), turnover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--kronos", action="store_true")
    args = ap.parse_args()

    s = load_settings()
    md = MarketData(s.paths.resolve(s.paths.cache_dir))
    lookback_cal_days = int(s.portfolio.lookback_days * 1.5) + 30
    first = dt.date.fromisoformat(args.start) - dt.timedelta(days=lookback_cal_days)
    bars = md.daily_bars(s.all_symbols, days=(dt.date.today() - first).days)
    closes = md.closes(bars).ffill()
    uni = [x for x in s.universe.symbols if x in closes.columns]
    bench = s.universe.benchmark
    pc = s.portfolio
    cash = pc.min_cash_weight

    strategies = {
        "SPY": lambda h, d: pd.Series({bench: 1.0}),
        "EqualWeight": lambda h, d: pd.Series(1.0 / len(uni) * (1 - cash), index=uni),
        "HRP": lambda h, d: target_weights(h[uni], [], pc.model_copy(update={"method": "hrp"}), s.agents.horizon_days),
    }

    if args.kronos:
        from bot.run_daily import kronos_relative
        from bot.signals.kronos_signal import KronosSignal

        ks = KronosSignal(s.kronos.model_copy(update={"sample_count": 2}))

        def bl_kronos(h, d):
            f = kronos_relative(ks.forecast({x: bars[x].loc[: h.index[-1]] for x in uni}))
            ranked = sorted(f.items(), key=lambda kv: kv[1]["relative"])
            picks = ranked[:4] + ranked[-4:]  # strongest relative calls only
            views = [View(symbol=k, expected_return=float(np.clip(v["relative"], -0.02, 0.02)),
                          confidence=0.3, rationale="kronos") for k, v in picks]
            return target_weights(h[uni], views, pc, s.agents.horizon_days)

        strategies["BL+Kronos"] = bl_kronos

    years = (closes.index[-1] - pd.Timestamp(args.start)).days / 365.25
    rows, curves = {}, {}
    for name, fn in strategies.items():
        print(f"running {name}...", flush=True)
        eq, to = run(closes, fn, args.start, args.every, args.cost_bps)
        rows[name] = metrics(eq, to, years)
        curves[name] = eq
    df = pd.DataFrame(rows).T
    fmt = {"CAGR": "{:+.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}", "Calmar": "{:.2f}",
           "AnnTurnover": "{:.1f}x"}
    print(f"\nWalk-forward {args.start} -> {closes.index[-1].date()}  rebalance every {args.every}d, "
          f"cost {args.cost_bps}bps\n")
    print(df.apply(lambda c: c.map(fmt[c.name].format)).to_string())
    out = s.paths.resolve("data/backtest_curves.csv")
    pd.DataFrame(curves).to_csv(out)
    print(f"\nEquity curves saved to {out}")


if __name__ == "__main__":
    main()
