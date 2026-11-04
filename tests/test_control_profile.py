"""Frozen v1 control profiles: same strategy as live, v1 prompts, simulated book, every new feature off."""
from __future__ import annotations

import pytest

from bot import run_daily
from bot.agents.llm import prompt_version
from bot.config import FeaturesCfg, PROFILES, Settings, load_settings

# prompt_version() of src/bot/agents/prompts and prompts_smallcap at commit 8dfaa91 (Task 0).
V1_HASH = {"prompts_v1": "a9bddb8d11", "prompts_smallcap_v1": "2f8303eb06"}
PAIRS = [("default", "control"), ("smallcap", "smallcap_control")]


@pytest.mark.parametrize("live,ctl", PAIRS)
def test_control_profiles_load(live, ctl):
    a, c = load_settings(profile=live), load_settings(profile=ctl)
    assert c.profile == ctl and c.execution.mode == "control"
    assert not any(c.features.model_dump().values())
    assert c.agents.prompt_dir.endswith("_v1")
    assert c.paths.journal_db != a.paths.journal_db
    assert c.risk.kill_switch_file != a.risk.kill_switch_file
    assert c.paths.reports_dir != a.paths.reports_dir
    assert c.notify.email is False
    # strategy-freeze guard: the control trades the identical strategy
    assert c.risk == a.risk.model_copy(update={"kill_switch_file": c.risk.kill_switch_file})
    assert c.portfolio == a.portfolio
    assert c.universe == a.universe
    assert c.agents.model_dump(exclude={"prompt_dir"}) == a.agents.model_dump(exclude={"prompt_dir"})
    assert c.broker == a.broker   # same paper keys, used for market data and the clock only
    # execution (order shaping, limits) is not reasoning: the control executes exactly like live
    assert c.execution.model_dump(exclude={"mode"}) == a.execution.model_dump(exclude={"mode"})


@pytest.mark.parametrize("d", sorted(V1_HASH))
def test_prompts_v1_frozen(d):
    assert prompt_version(d) == V1_HASH[d]


def test_live_features_default_off():
    assert not any(FeaturesCfg().model_dump().values())
    s = Settings.model_validate({"universe": {"symbols": ["SPY"]}})
    assert not any(s.features.model_dump().values())


def test_control_profiles_registered():
    assert PROFILES["control"] == "config/settings_control.yaml"
    assert PROFILES["smallcap_control"] == "config/settings_smallcap_control.yaml"


class _FakeAlpaca:
    def __init__(self, **kw):
        self.kw = kw

    def latest_quote(self, sym):
        return {"bid": 9.9, "ask": 10.0}

    def is_market_open(self):
        return True

    def submit(self, order):  # pragma: no cover - must never be reached in control mode
        raise AssertionError("control mode reached Alpaca submit")


def test_make_broker_control_never_alpaca_submit(tmp_path, monkeypatch):
    import bot.broker.alpaca_broker as ab
    from bot.broker.book import BookBroker
    from bot.models import OrderPlan

    monkeypatch.setattr(ab, "AlpacaBroker", _FakeAlpaca)
    s = load_settings(profile="control")
    s.control.book_path = str(tmp_path / "book.json")
    b = run_daily.make_broker("control", s, init_book=True)
    assert isinstance(b, BookBroker)
    sub = b.submit(OrderPlan(symbol="AAPL", side="buy", notional=100.0, current_weight=0, target_weight=0.001))
    assert sub.status == "filled"


def test_make_broker_control_missing_book_fails(tmp_path, monkeypatch):
    import bot.broker.alpaca_broker as ab

    monkeypatch.setattr(ab, "AlpacaBroker", _FakeAlpaca)
    s = load_settings(profile="control")
    s.control.book_path = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        run_daily.make_broker("control", s)


def test_control_reconcile_marks_book_and_records_equity(tmp_path, monkeypatch):
    import bot.broker.alpaca_broker as ab
    from bot.journal.store import Journal
    from bot.models import OrderPlan

    class Px(_FakeAlpaca):
        price = {"bid": 9.9, "ask": 10.0}

        def latest_quote(self, sym):
            return self.price

    monkeypatch.setattr(ab, "AlpacaBroker", Px)
    s = load_settings(profile="control")
    s.control.book_path = str(tmp_path / "book.json")
    s.paths.journal_db = str(tmp_path / "j.db")
    b = run_daily.make_broker("control", s, init_book=True)
    b.submit(OrderPlan(symbol="AAPL", side="buy", notional=1000.0, current_weight=0, target_weight=0.01))
    Px.price = {"bid": 11.9, "ask": 12.1}
    run_daily.reconcile(s, "control")
    eq = Journal(tmp_path / "j.db").query("SELECT equity FROM equity")["equity"].tolist()
    assert eq == [pytest.approx(99_000.0 + 100 * 12.0)]
