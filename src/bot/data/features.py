"""Deterministic technical features computed from daily bars (no LLM involved)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50.0


def technical_snapshot(bars: pd.DataFrame) -> dict:
    c = bars["close"]
    last = float(c.iloc[-1])

    def ret(n: int):
        return round(float(c.iloc[-1] / c.iloc[-1 - n] - 1), 4) if len(c) > n else None

    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1]
    vol20 = c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    hi52 = c.tail(252).max()
    return {
        "last_close": round(last, 2),
        "ret_5d": ret(5),
        "ret_20d": ret(20),
        "ret_60d": ret(60),
        "ret_252d": ret(252),
        "rsi_14": round(rsi(c), 1),
        "pct_vs_sma50": round(last / sma50 - 1, 4) if pd.notna(sma50) else None,
        "pct_vs_sma200": round(last / sma200 - 1, 4) if pd.notna(sma200) else None,
        "ann_vol_20d": round(float(vol20), 4) if pd.notna(vol20) else None,
        "pct_from_52w_high": round(last / hi52 - 1, 4),
        "avg_dollar_volume_20d": round(float((c * bars["volume"]).tail(20).mean()), 0),
    }


def market_regime_snapshot(closes: pd.DataFrame, benchmark: str) -> dict:
    """Breadth and trend summary used by the regime analyst."""
    spy = closes[benchmark].dropna()
    above200 = {
        s: bool(closes[s].iloc[-1] > closes[s].rolling(200).mean().iloc[-1])
        for s in closes.columns
        if closes[s].notna().sum() > 200
    }
    rets = closes.pct_change()
    return {
        "benchmark": technical_snapshot(pd.DataFrame({"close": spy, "volume": 0.0})) | {"symbol": benchmark},
        "breadth_pct_above_200dma": round(sum(above200.values()) / max(len(above200), 1), 3),
        "avg_pairwise_corr_60d": round(float(rets.tail(60).corr().values[np.triu_indices(len(closes.columns), 1)].mean()), 3),
        "tlt_ret_20d": technical_snapshot(pd.DataFrame({"close": closes["TLT"], "volume": 0.0}))["ret_20d"] if "TLT" in closes else None,
        "gld_ret_20d": technical_snapshot(pd.DataFrame({"close": closes["GLD"], "volume": 0.0}))["ret_20d"] if "GLD" in closes else None,
    }
