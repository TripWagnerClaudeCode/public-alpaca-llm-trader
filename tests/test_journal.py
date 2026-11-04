import datetime as dt

import pandas as pd

from bot.journal.store import Journal


def test_latest_targets_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.db")
    r = j.start_run("analyze", "dry-run")
    j.target_weights(r, pd.Series({"A": 0.1, "B": 0.2}))
    assert j.latest_targets("dry-run") is None   # still 'running'
    j.finish_run(r, "ok")
    run_id, w = j.latest_targets("dry-run")
    assert run_id == r and w["B"] == 0.2
    assert j.latest_targets("paper") is None     # dry-run never feeds paper/live


def test_peak_equity(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.record_equity(dt.date(2026, 1, 1), 100, 0, None)
    j.record_equity(dt.date(2026, 1, 2), 120, 0, None)
    j.record_equity(dt.date(2026, 1, 3), 110, 0, None)
    assert j.peak_equity() == 120
