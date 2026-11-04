import numpy as np
import pandas as pd
import pytest

from bot.screen.evaluate import evaluate, forward_returns


def _closes(n_days=60, n_syms=30, seed=0):
    idx = pd.bdate_range("2026-09-01", periods=n_days)
    rng = np.random.default_rng(seed)
    # symbol i drifts by i bp/day, so a score equal to i is perfectly predictive
    data = {f"S{i}": 10 * np.exp(np.cumsum(0.0005 * i + rng.normal(0, 0.0005, n_days))) for i in range(n_syms)}
    return pd.DataFrame(data, index=idx)


def test_forward_returns_and_pending():
    c = _closes()
    fwd, start, end = forward_returns(c, "2026-09-06", 10)   # a Sunday -> uses Friday 09-04 close
    assert start == "2026-09-04" and end == "2026-09-18"
    assert fwd["S3"] == pytest.approx(c.loc["2026-09-18", "S3"] / c.loc["2026-09-04", "S3"] - 1)
    assert forward_returns(c, str(c.index[-5].date()), 10) is None


def test_evaluate_perfect_signal():
    c = _closes()
    dates = ["2026-09-06", "2026-09-13", "2026-09-20", "2026-11-22"]   # last one is pending
    cands = pd.DataFrame([{"screen_date": d, "symbol": f"S{i}", "score": float(i), "rank": 30 - i}
                          for d in dates for i in range(30)])
    r = evaluate(cands, c, horizon=10)
    assert r["n_dates"] == 3 and r["pending"] == ["2026-11-22"]
    assert r["mean_ic"] > 0.95 and r["mean_q5_minus_q1"] > 0
    assert r["rows"][0]["top20_minus_all"] > 0


def test_evaluate_inverted_and_too_few():
    c = _closes()
    cands = pd.DataFrame([{"screen_date": "2026-09-06", "symbol": f"S{i}", "score": -float(i), "rank": i + 1}
                          for i in range(30)])
    r = evaluate(cands, c)
    assert r["mean_ic"] < -0.95 and r["ic_tstat"] is None   # one date: no t-stat
    few = cands.head(5)
    r2 = evaluate(few, c, min_names=10)
    assert r2["n_dates"] == 0 and r2["mean_ic"] is None and "only 5" in r2["pending"][0]
