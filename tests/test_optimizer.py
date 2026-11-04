import pytest

from bot.models import View
from bot.portfolio.optimizer import horizon_to_daily, target_weights, views_to_skfolio


def test_horizon_to_daily_roundtrip():
    d = horizon_to_daily(0.02, 10)
    assert (1 + d) ** 10 - 1 == pytest.approx(0.02)


def test_views_conversion_drops_unknown_duplicate_and_zero_conf():
    views = [View(symbol="AAA", expected_return=0.01, confidence=0.5, rationale=""),
             View(symbol="AAA", expected_return=0.03, confidence=0.5, rationale=""),
             View(symbol="ZZZ", expected_return=0.01, confidence=0.5, rationale=""),
             View(symbol="BBB", expected_return=-0.01, confidence=0.0, rationale="")]
    strings, confs = views_to_skfolio(views, ["AAA", "BBB"], 10)
    assert len(strings) == 1 and strings[0].startswith("AAA == ")
    assert 0 < confs[0] < 1


def test_hrp_respects_cash_and_caps(closes, pcfg):
    w = target_weights(closes, [], pcfg.model_copy(update={"method": "hrp", "max_weight": 0.3}), 10)
    assert w.sum() == pytest.approx(1 - pcfg.min_cash_weight, abs=1e-4)
    assert (w >= 0).all()


def test_bl_no_usable_views_falls_back_to_hrp(closes, pcfg):
    w1 = target_weights(closes, [], pcfg, 10)
    w2 = target_weights(closes, [], pcfg.model_copy(update={"method": "hrp"}), 10)
    assert w1.round(4).equals(w2.round(4))


def test_bl_positive_view_increases_weight(closes, pcfg):
    cfg = pcfg.model_copy(update={"max_weight": 0.5})
    base = target_weights(closes, [View(symbol="AAA", expected_return=0.0, confidence=0.5, rationale="")], cfg, 10)
    up = target_weights(closes, [View(symbol="AAA", expected_return=0.02, confidence=0.6, rationale="")], cfg, 10)
    assert up["AAA"] > base["AAA"]
    assert up.max() <= 0.5 + 1e-6
    assert up.sum() <= 1 - cfg.min_cash_weight + 1e-6


def test_exclude_benchmark(closes, pcfg):
    w = target_weights(closes, [], pcfg.model_copy(update={"method": "hrp"}), 10, exclude=("AAA",))
    assert "AAA" not in w.index
