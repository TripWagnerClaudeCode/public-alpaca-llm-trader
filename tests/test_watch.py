"""Intraday watch: refresh orders, record equity, and run B's existing stop every 30 minutes."""
from __future__ import annotations

import pytest

from bot import watch
from bot.broker.sim import DryRunBroker
from bot.config import load_settings
from bot.journal.store import Journal
from bot.models import PositionDetail
from bot.risk import gate


def _settings(tmp_path, profile="smallcap", on=True):
    s = load_settings(profile=profile)
    s.paths.journal_db = str(tmp_path / "j.db")
    s.risk.kill_switch_file = str(tmp_path / "KILL_SWITCH")
    s.features.watch = on
    return s


def _broker(pnl_price=84.0, symbol="XYZ", qty=10.0, open_=True):
    det = {symbol: PositionDetail(qty=qty, avg_entry_price=100.0, current_price=pnl_price)}
    b = DryRunBroker(equity=10_000, positions={symbol: qty * pnl_price} if qty else {}, details=det)
    b.is_market_open = lambda: open_
    return b


def _runs(s):
    return Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM runs WHERE phase='watch'")


def test_watch_disabled_noop(tmp_path):
    s = _settings(tmp_path, on=False)
    b = _broker()
    assert watch.run(s, "dry-run", broker=b) == 0
    assert b.submitted == [] and _runs(s).empty


def test_watch_market_closed_skips(tmp_path):
    s = _settings(tmp_path)
    b = _broker(open_=False)
    assert watch.run(s, "dry-run", broker=b) == 0
    assert b.submitted == [] and _runs(s)["status"].tolist() == ["skipped"]


def test_watch_b_stop_sells_once(tmp_path):
    s = _settings(tmp_path)
    b = _broker(pnl_price=84.0)                     # -16% vs entry, B's stop is -15%
    assert watch.run(s, "dry-run", broker=b) == 0
    assert [(o.symbol, o.side, o.close_position) for o in b.submitted] == [("XYZ", "sell", True)]
    j = Journal(s.paths.resolve(s.paths.journal_db))
    assert "XYZ" in j.active_cooldowns(__import__("datetime").date.today(), "dry-run")
    assert watch.run(s, "dry-run", broker=b) == 0   # the sell is still open: never a second sell
    assert len(b.submitted) == 1


def test_watch_no_stop_above_level(tmp_path):
    s = _settings(tmp_path)
    b = _broker(pnl_price=90.0)                     # -10%: inside B's stop
    assert watch.run(s, "dry-run", broker=b) == 0
    assert b.submitted == []


def test_watch_skips_symbol_with_no_position(tmp_path):
    s = _settings(tmp_path)
    b = _broker(qty=0.0)
    assert watch.run(s, "dry-run", broker=b) == 0
    assert b.submitted == []


def test_watch_a_never_sells(tmp_path):
    s = _settings(tmp_path, profile="default")
    assert s.risk.stop_loss_pct is None              # strategy freeze: A has no stop and gets none
    b = _broker(pnl_price=70.0)                      # -30%
    assert watch.run(s, "dry-run", broker=b) == 0
    assert b.submitted == []
    assert _runs(s)["status"].tolist() == ["ok"]


def test_watch_records_intraday_equity(tmp_path):
    s = _settings(tmp_path, profile="default")
    assert watch.run(s, "dry-run", broker=_broker(pnl_price=95.0)) == 0
    eq = Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM intraday_equity")
    assert len(eq) == 1 and eq["equity"].iloc[0] == pytest.approx(10_000)


def test_watch_auto_kill_switch_still_exits(tmp_path):
    s = _settings(tmp_path)
    gate.trip_kill_switch(s.risk, "daily loss")
    b = _broker(pnl_price=84.0)
    assert watch.run(s, "dry-run", broker=b) == 0
    assert len(b.submitted) == 1


def test_watch_manual_kill_switch_blocks_exits_but_records_equity(tmp_path):
    s = _settings(tmp_path)
    gate.kill_switch_path(s.risk).write_text("halt")
    b = _broker(pnl_price=84.0)
    assert watch.run(s, "dry-run", broker=b) == 2
    assert b.submitted == []
    assert _runs(s)["status"].tolist() == ["blocked"]
    eq = Journal(s.paths.resolve(s.paths.journal_db)).query("SELECT * FROM intraday_equity")
    assert len(eq) == 1


def test_cli_phase_watch_dispatches(monkeypatch, tmp_path):
    from bot import run_daily

    seen = {}
    monkeypatch.setattr(watch, "run", lambda s, mode: seen.update(profile=s.profile, mode=mode) or 0)
    orig = Journal.__init__
    monkeypatch.setattr(run_daily.Journal, "__init__", lambda self, p: orig(self, tmp_path / "j.db"))
    assert run_daily.main(["--profile", "smallcap", "--phase", "watch", "--mode", "dry-run"]) == 0
    assert seen == {"profile": "smallcap", "mode": "dry-run"}
