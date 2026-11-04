"""Phase 2: the weekly Curator turns evidenced observations into lessons, under code-enforced rules."""
from __future__ import annotations

from bot import run_daily
from bot.agents.schemas import LessonEdit, LessonRetire, LessonsUpdate, NewLesson
from bot.memory.curator import curate
from bot.memory.files import Lesson, MemoryFiles
from test_memory_outcomes import _settings

CARDS = {"n_views": 40, "agents": {}, "calibration": {}, "morning_check": {"cancels": 0,
                                                                          "mean_excess_of_cancelled": None}, "symbols": {}}


class _LLM:
    def __init__(self, upd=None, fail=None):
        self.upd, self.fail, self.calls = upd, fail, []

    def structured(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return self.upd


def _env(tmp_path, lessons=(), staging=("PM long calls on payments beat the universe 6 of 8 times",)):
    s = _settings(tmp_path)
    mf = MemoryFiles(tmp_path / "mem", today=lambda: "2026-11-22")
    if lessons:
        mf.write_lessons(list(lessons))
    for line in staging:
        mf.append_staging("2026-11-18", [line])
    return s, mf


def _l(i, text="Old lesson.", tags=("pm",)):
    return Lesson(f"L-{i:04d}", text, "2026-11-10", 25, "e", list(tags))


def _upd(add=(), edit=(), retire=()):
    return LessonsUpdate(add=list(add), edit=list(edit), retire=list(retire), summary="s")


def test_small_sample_lesson_rejected_and_carried(tmp_path):
    s, mf = _env(tmp_path)
    out = curate(_LLM(_upd(add=[NewLesson(text="Tiny pattern.", evidence="3 of 4", n=4, tags=["pm"])])),
                 s, mf, CARDS, "2026-11-22")
    assert out["added"] == 0 and out["rejected"] == 1 and mf.read_lessons()[0] == []
    staged = mf.read_staging(None)
    assert any("Tiny pattern." in x and "carried" in x for x in staged)
    assert not any("6 of 8" in x for x in staged)                      # reviewed observations cleared


def test_add_edit_retire_applied_with_changelog(tmp_path):
    s, mf = _env(tmp_path, lessons=[_l(1), _l(2, "Second.")])
    upd = _upd(add=[NewLesson(text="Payments longs work.", evidence="6 of 8", n=24, tags=["pm"])],
               edit=[LessonEdit(id="L-0001", text="Old lesson, sharper.", evidence="e2", n=30)],
               retire=[LessonRetire(id="L-0002", reason="no longer holds")])
    out = curate(_LLM(upd), s, mf, CARDS, "2026-11-22")
    lessons = {x.id: x for x in mf.read_lessons()[0]}
    assert out == {"added": 1, "edited": 1, "retired": 1, "rejected": 0}
    assert lessons["L-0001"].text == "Old lesson, sharper." and lessons["L-0001"].n == 30
    assert "L-0002" not in lessons and lessons["L-0003"].text == "Payments longs work."
    log = (tmp_path / "mem" / "changelog.md").read_text(encoding="utf-8")
    assert "L-0003" in log and "L-0002" in log and "no longer holds" in log


def test_never_readds_owner_retired(tmp_path):
    s, mf = _env(tmp_path, lessons=[_l(1, "Chase gaps.")])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text("\n".join(x for x in p.read_text(encoding="utf-8").splitlines()
                           if "L-0001" not in x and "evidence: e" not in x) + "\n", encoding="utf-8")
    mf.read_lessons()                                                  # detects the owner's deletion
    out = curate(_LLM(_upd(add=[NewLesson(text="chase gaps.", evidence="x", n=40, tags=[])])), s, mf, CARDS,
                 "2026-11-22")
    assert out["added"] == 0 and out["rejected"] == 1 and mf.read_lessons()[0] == []


def test_max_lessons_cap(tmp_path):
    s, mf = _env(tmp_path, lessons=[_l(i, f"L{i}") for i in range(1, 15)])
    s.memory.max_lessons = 15
    adds = [NewLesson(text=f"new {i}", evidence="e", n=30, tags=[]) for i in range(3)]
    out = curate(_LLM(_upd(add=adds)), s, mf, CARDS, "2026-11-22")
    assert out["added"] == 1 and out["rejected"] == 2 and len(mf.read_lessons()[0]) == 15


def test_pinned_and_owner_lessons_untouched(tmp_path):
    s, mf = _env(tmp_path, lessons=[_l(1), _l(2, "the owner's own.", tags=["owner"])])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text(p.read_text(encoding="utf-8").replace("## Pinned by you\n", "## Pinned by you\nNo biotech.\n"),
                 encoding="utf-8")
    upd = _upd(edit=[LessonEdit(id="L-0002", text="rewritten", evidence="e", n=40)],
               retire=[LessonRetire(id="L-0002", reason="r")])
    out = curate(_LLM(upd), s, mf, CARDS, "2026-11-22")
    lessons, pinned = mf.read_lessons()
    assert out["edited"] == 0 and out["retired"] == 0
    assert {x.id: x.text for x in lessons}["L-0002"] == "the owner's own." and "No biotech." in pinned


def test_curate_failure_leaves_files_unchanged(tmp_path):
    s, mf = _env(tmp_path, lessons=[_l(1)])
    before = {f.name: f.read_text(encoding="utf-8") for f in (tmp_path / "mem").iterdir() if f.is_file()}
    try:
        curate(_LLM(fail=RuntimeError("usage limit reached")), s, mf, CARDS, "2026-11-22")
    except RuntimeError:
        pass
    after = {f.name: f.read_text(encoding="utf-8") for f in (tmp_path / "mem").iterdir() if f.is_file()}
    assert before == after


def test_cli_phase_curate_dispatches(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.memory.dir = str(tmp_path / "mem")
    MemoryFiles(tmp_path / "mem").append_staging("2026-11-18", ["an observation"])
    llm = _LLM(_upd())
    monkeypatch.setattr(run_daily, "load_settings", lambda *a, **k: s)
    monkeypatch.setattr(run_daily, "make_llm", lambda s: llm)
    assert run_daily.main(["--phase", "curate", "--mode", "paper"]) == 0
    assert len(llm.calls) == 1 and llm.calls[0]["model"] == "claude-opus-5-5"
    from bot.journal.store import Journal
    assert Journal(tmp_path / "j.db").query("SELECT status FROM runs WHERE phase='curate'")["status"].tolist() == ["ok"]


def test_curate_nothing_to_do_skips_llm(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    llm = _LLM(_upd())
    monkeypatch.setattr(run_daily, "make_llm", lambda s: llm)
    assert run_daily.curate_phase(s, "paper") == 0 and llm.calls == []
