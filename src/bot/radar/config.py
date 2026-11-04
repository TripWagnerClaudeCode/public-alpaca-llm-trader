"""Radar configuration: config/radar.yaml (independent of config/settings.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from bot.config import ROOT


class SourceCfg(BaseModel):
    enabled: bool = True
    max_requests: int = 10           # hard cap on HTTP requests per collector run
    batch_size: int = 1              # symbols OR-ed into one query (where the API supports OR)
    pause_seconds: float = 1.0       # politeness delay between requests to the same API
    min_interval_minutes: int = 0    # skip the source if it ran more recently than this
    feeds: list[str] = []            # substack only


class FeaturesCfg(BaseModel):
    window_hours: int = 24
    baseline_days: int = 30
    min_baseline_days: int = 14
    std_floor: float = 1.0
    require_live_coverage: bool = True
    fringe_min_sources: int = 3
    default_freeze_et: str = "17:15"   # analyze time (A 17:15 ET, B 17:30 ET); earliest of the two


class RadarConfig(BaseModel):
    db_path: str = "data/radar.db"
    names_cache: str = "data/cache/radar_names.json"
    names_max_age_days: int = 7
    settings_path: str = "config/settings.yaml"
    universe_dir: str = "data/universe/smallcap"
    user_agent: str = "trading-bot-radar/0.1"
    http_timeout_seconds: float = 12.0
    max_runtime_seconds: float = 150.0
    lookback_hours: int = 48
    max_symbols_per_post: int = 6
    features: FeaturesCfg = FeaturesCfg()
    sources: dict[str, SourceCfg] = {}
    mainstream_domains: list[str] = []
    aliases: dict[str, list[str]] = {}
    ambiguous_names: list[str] = []
    stoplist_extra: list[str] = []
    junk_title_patterns: list[str] = []
    junk_origins: list[str] = []

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    def source(self, name: str) -> SourceCfg:
        return self.sources.get(name, SourceCfg(enabled=False))


def load_radar_config(path: str | Path | None = None) -> RadarConfig:
    path = Path(path) if path else ROOT / "config" / "radar.yaml"
    if not path.exists():
        return RadarConfig()
    with open(path, encoding="utf-8") as f:
        return RadarConfig.model_validate(yaml.safe_load(f) or {})
