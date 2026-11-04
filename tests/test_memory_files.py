"""Phase 2 memory files: lessons.md (the owner's edits win), staging.md, changelog.md, scorecard.md."""
from __future__ import annotations

import pytest

from bot.config import FeaturesCfg, load_settings
from bot.memory.files import Lesson, MemoryFiles


def _mf(tmp_path, today="2026-11-20"):
    return MemoryFiles(tmp_path / "mem", title="Lessons (large-cap)", today=lambda: today)


def _l(i, text="PM overrates momentum after earnings gaps.", n=30, tags=("pm",)):
    return Lesson(id=f"L-{i:04d}", text=text, added="2026-11-16", n=n, evidence="22 of 30 misses", tags=list(tags))


def test_roundowner_lessons_and_pinned_preserved(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1), _l(2, "Bear case on NFLX was right 4 of 5 times.", tags=["bear", "NFLX"])])
    p = tmp_path / "mem" / "lessons.md"
    txt = p.read_text(encoding="utf-8")
    assert "## Pinned by you" in txt and "[L-0001]" in txt
    p.write_text(txt.replace("## Pinned by you\n", "## Pinned by you\nNever chase biotech gaps.\n"), encoding="utf-8")
    lessons, pinned = mf.read_lessons()
    assert [x.id for x in lessons] == ["L-0001", "L-0002"]
    assert lessons[1].tags == ["bear", "NFLX"] and lessons[0].n == 30 and lessons[0].evidence == "22 of 30 misses"
    assert "Never chase biotech gaps." in pinned
    mf.write_lessons(lessons)
    assert "Never chase biotech gaps." in p.read_text(encoding="utf-8")


def test_owner_deleted_lesson_is_retired_and_never_returns(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1), _l(2, "Second lesson.")])
    p = tmp_path / "mem" / "lessons.md"
    lines = p.read_text(encoding="utf-8").splitlines()
    keep = [ln for i, ln in enumerate(lines) if "[L-0002]" not in ln and not (i and "[L-0002]" in lines[i - 1])]
    p.write_text("\n".join(keep) + "\n", encoding="utf-8")
    lessons, _ = mf.read_lessons()
    assert [x.id for x in lessons] == ["L-0001"]
    assert "L-0002" in mf.retired_ids()
    mf.write_lessons([*lessons, _l(2, "Second lesson.")])        # the bot tries to put it back
    assert [x.id for x in mf.read_lessons()[0]] == ["L-0001"]
    assert "L-0002" in (tmp_path / "mem" / "changelog.md").read_text(encoding="utf-8")


def test_owner_edited_text_is_kept(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text(p.read_text(encoding="utf-8").replace("PM overrates momentum", "PM badly overrates momentum"),
                 encoding="utf-8")
    assert mf.read_lessons()[0][0].text.startswith("PM badly overrates momentum")


def test_owner_added_line_gets_id(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    p = tmp_path / "mem" / "lessons.md"
    p.write_text(p.read_text(encoding="utf-8") + "- Small caps under $5 rarely work for us.\n", encoding="utf-8")
    lessons, _ = mf.read_lessons()
    new = lessons[-1]
    assert new.id == "L-0002" and new.tags == ["owner"] and new.n == 0 and new.added == "2026-11-20"
    assert new.text == "Small caps under $5 rarely work for us."


def test_sync_conflict_files_ignored(tmp_path):
    mf = _mf(tmp_path)
    mf.write_lessons([_l(1)])
    (tmp_path / "mem" / "lessons.sync-conflict-20261120-101010-ABCDEF.md").write_text(
        "## Active lessons\n- [L-0009] (2026-11-20, n=99) rogue\n  evidence: x\n", encoding="utf-8")
    assert [x.id for x in mf.read_lessons()[0]] == ["L-0001"]


def test_staging_append_read_clear(tmp_path):
    mf = _mf(tmp_path)
    mf.append_staging("2026-11-17", ["obs one"])
    mf.append_staging("2026-11-18", ["obs two", "obs three"])
    assert len(mf.read_staging(None)) == 3
    assert [x.endswith("obs two") or x.endswith("obs three") for x in mf.read_staging("2026-11-18")] == [True, True]
    mf.clear_staging("2026-11-17")
    assert len(mf.read_staging(None)) == 2


def test_missing_dir_created_on_first_write_and_reads_empty(tmp_path):
    mf = _mf(tmp_path)
    assert mf.read_lessons() == ([], "") and mf.read_staging(None) == []
    mf.write_scorecard("# Scorecard\n")
    assert (tmp_path / "mem" / "scorecard.md").read_text(encoding="utf-8") == "# Scorecard\n"


def test_memory_flag_defaults_off_and_controls_off():
    assert FeaturesCfg().memory is False
    for p in ("control", "smallcap_control"):
        assert load_settings(profile=p).features.memory is False


def test_tests_cannot_touch_real_memory_dir(tmp_path):
    from bot.memory import memory_root

    s = load_settings()                                  # the live settings point at the repo's memory folder
    assert s.memory.dir.startswith("memory/")
    root = memory_root(s)
    assert str(tmp_path.parent.parent) in str(root)      # redirected under pytest's temp base
    s.memory.dir = str(tmp_path / "mine")
    assert memory_root(s) == tmp_path / "mine"           # test-owned folders are used as given
