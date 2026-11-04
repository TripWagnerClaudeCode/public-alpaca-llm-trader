"""Read/write the screener's weekly universe files (`<dir>/YYYY-MM-DD.json`).

`load_universe` is the interface consumed by the daily pipeline (fail closed: it returns None
when there is no usable file, and the caller blocks the run).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _dated_files(dir: Path) -> list[tuple[dt.date, Path]]:
    out = []
    d = Path(dir)
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = _NAME.fullmatch(p.name)
        if not m or not p.is_file():
            continue
        try:
            out.append((dt.date.fromisoformat(m.group(1)), p))
        except ValueError:  # e.g. 2026-13-45.json
            continue
    return sorted(out, reverse=True)


def load_universe(dir: Path, as_of: dt.date, max_age_days: int) -> dict | None:
    """Newest `YYYY-MM-DD.json` in `dir` dated <= as_of and at most `max_age_days` old.

    Returns the parsed dict (guaranteed to have a non-empty "symbols" list), else None.
    A file that is unreadable or has no symbols is skipped with a warning and the next
    (older, still in-window) file is tried. Files not named exactly YYYY-MM-DD.json are ignored.
    """
    for d, p in _dated_files(Path(dir)):
        if d > as_of:
            continue
        if (as_of - d).days > max_age_days:
            return None  # sorted newest first, so everything after is older still
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("unreadable universe file %s: %s", p, e)
            continue
        syms = data.get("symbols") if isinstance(data, dict) else None
        if not isinstance(syms, list) or not syms:
            log.warning("universe file %s has no symbols", p)
            continue
        return data
    return None


def write_universe(dir: Path, as_of: dt.date, payload: dict) -> Path:
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{as_of.isoformat()}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic, so a reader never sees a half-written file
    return path
