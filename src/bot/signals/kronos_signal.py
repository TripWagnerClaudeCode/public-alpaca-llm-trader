"""Kronos forecast signal.

Kronos is vendored at vendor/Kronos (not a pip package; imported as `model`).
For each symbol we forecast `pred_len` daily bars, `sample_count` times, and report:
  exp_return  - mean predicted close at horizon / last close - 1
  dispersion  - std of per-sample horizon returns (higher = less confident)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import ROOT, KronosCfg

log = logging.getLogger(__name__)
_KRONOS_PATH = str(ROOT / "vendor" / "Kronos")


@dataclass
class KronosForecast:
    symbol: str
    exp_return: float
    dispersion: float
    horizon_days: int


def future_business_days(last: pd.Timestamp, n: int) -> pd.Series:
    return pd.Series(pd.bdate_range(last + pd.Timedelta(days=1), periods=n))


class KronosSignal:
    def __init__(self, cfg: KronosCfg):
        self.cfg = cfg
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            if _KRONOS_PATH not in sys.path:
                sys.path.insert(0, _KRONOS_PATH)
            from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

            tok = KronosTokenizer.from_pretrained(self.cfg.tokenizer)
            mdl = Kronos.from_pretrained(self.cfg.model)
            self._predictor = KronosPredictor(mdl, tok, device=self.cfg.device, max_context=512)
        return self._predictor

    def _prep(self, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = bars.tail(self.cfg.lookback).copy()
        df["amount"] = df["close"] * df["volume"]
        x_ts = pd.Series(df.index)
        y_ts = future_business_days(df.index[-1], self.cfg.pred_len)
        return df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True), x_ts, y_ts

    def forecast(self, bars: dict[str, pd.DataFrame], batch_size: int = 32) -> dict[str, KronosForecast]:
        """Batched forecast. Each symbol is replicated `sample_count` times in the batch so a
        single predict_batch call yields independent samples (for dispersion)."""
        eligible = {s: b for s, b in bars.items() if len(b) >= self.cfg.lookback}
        skipped = set(bars) - set(eligible)
        if skipped:
            log.info("Kronos skipping (insufficient history): %s", sorted(skipped))
        if not eligible:
            return {}
        pred = self._load()
        jobs = []  # (symbol, df, x_ts, y_ts, last_close)
        for sym, b in eligible.items():
            x, x_ts, y_ts = self._prep(b)
            for _ in range(self.cfg.sample_count):
                jobs.append((sym, x, x_ts, y_ts, float(x["close"].iloc[-1])))
        samples: dict[str, list[float]] = {s: [] for s in eligible}
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i : i + batch_size]
            try:
                outs = pred.predict_batch(
                    [j[1] for j in chunk], [j[2] for j in chunk], [j[3] for j in chunk],
                    pred_len=self.cfg.pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
                )
            except Exception as e:
                log.warning("Kronos batch %d failed: %s", i // batch_size, e)
                continue
            for (sym, _, _, _, last), out in zip(chunk, outs):
                samples[sym].append(float(out["close"].iloc[-1]) / last - 1)
        result = {}
        for sym, r in samples.items():
            if r:
                arr = np.asarray(r)
                result[sym] = KronosForecast(sym, float(arr.mean()), float(arr.std()), self.cfg.pred_len)
        return result
