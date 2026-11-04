"""Profile paths for the screener scripts, read straight from the profile YAML.

Deliberately independent of `bot.config` so the screener runs even before the profile's
settings file (or its new fields) exists; missing keys fall back to the profile-B defaults.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PROFILE_FILES = {"default": "config/settings.yaml", "smallcap": "config/settings_smallcap.yaml"}
DEFAULTS = {"screener_dir": "data/universe/smallcap", "journal_db": "data/journal_smallcap.db",
            "cache_dir": "data/cache"}

log = logging.getLogger(__name__)


def profile_paths(profile: str, root: Path = ROOT) -> dict[str, Path]:
    cfg: dict = {}
    p = root / PROFILE_FILES.get(profile, f"config/settings_{profile}.yaml")
    if p.exists():
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            log.warning("could not parse %s (%s); using defaults", p, e)
    uni, paths = cfg.get("universe") or {}, cfg.get("paths") or {}
    journal = paths.get("journal_db") or ("data/journal.db" if profile == "default" else DEFAULTS["journal_db"])
    vals = {"screener_dir": uni.get("screener_dir") or DEFAULTS["screener_dir"],
            "journal_db": journal,
            "cache_dir": paths.get("cache_dir") or DEFAULTS["cache_dir"]}
    return {k: (Path(v) if Path(v).is_absolute() else root / v) for k, v in vals.items()}
