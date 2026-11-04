"""Typed configuration: config/settings.yaml + secrets from .env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]

# --profile name -> settings file (relative to ROOT). --config overrides.
PROFILES = {"default": "config/settings.yaml", "smallcap": "config/settings_smallcap.yaml",
            # frozen v1 controls (simulated book, v1 prompts, no new features): spec 2026-11-04 Phase 0
            "control": "config/settings_control.yaml",
            "smallcap_control": "config/settings_smallcap_control.yaml"}


class Universe(BaseModel):
    symbols: list[str] = []
    benchmark: str = "SPY"
    source: Literal["static", "screener"] = "static"   # screener: symbols come from the weekly screener file
    screener_dir: str = "data/universe/smallcap"
    max_age_days: int = 8                                # older screener output blocks the run (fail closed)

    @model_validator(mode="after")
    def _symbols_required_for_static(self):
        if self.source == "static" and not self.symbols:
            raise ValueError("universe.symbols is required when universe.source is 'static'")
        return self


class BrokerCfg(BaseModel):
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"


class DataCfg(BaseModel):
    history_days: int = 800
    max_staleness_days: int = 4
    news_per_symbol: int = 5


class KronosCfg(BaseModel):
    enabled: bool = True
    model: str = "NeoQuasar/Kronos-small"
    tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    lookback: int = Field(400, le=512)
    pred_len: int = 10
    sample_count: int = 8
    device: str = "cpu"


class AgentsCfg(BaseModel):
    analyst_model: str = "claude-opus-5-5"
    analyst_effort: str = "low"
    debate_model: str = "claude-opus-5-5"
    debate_effort: str = "medium"
    pm_model: str = "claude-opus-5-5"
    pm_effort: str = "high"
    debate_rounds: int = 1
    max_daily_output_tokens: int = 150_000
    horizon_days: int = 10
    max_abs_view_return: float = 0.08
    prompt_dir: str = "prompts"   # under src/bot/agents/ unless absolute
    backend: str = "claude-cli"   # claude-cli (Claude subscription, no API key) | api (ANTHROPIC_API_KEY)
    cli_path: str | None = None   # claude CLI path; default: PATH, then ~/.local/bin


class PortfolioCfg(BaseModel):
    method: str = "mean_risk"
    risk_measure: str = "variance"
    tau: float = 0.05
    max_weight: float = 0.15
    min_cash_weight: float = 0.05
    l2_coef: float = 0.0005
    delta: float = 2.5
    lookback_days: int = 504


class RiskCfg(BaseModel):
    max_position_weight: float = 0.15
    max_gross_exposure: float = 0.95
    max_orders_per_day: int = 20
    max_daily_turnover: float = 0.40
    min_order_notional: float = 5.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    kill_switch_file: str = "KILL_SWITCH"
    stop_loss_pct: float | None = None          # unrealized P&L <= -x vs avg entry -> forced exit
    stop_cooldown_days: int = 0                 # trading days re-entry is blocked after a stop
    max_adv_participation: float | None = None  # order notional <= x * 20-day avg dollar volume
    max_spread_pct: float | None = None         # stage 2: skip buys when quoted spread > x of mid
    # Phase 3 intraday buys (locked: the bot may never tune these)
    intraday_max_weight_frac: float = 0.5       # an intraday buy is at most this x max_position_weight
    intraday_max_buys_per_day: int = 2


class ExecutionCfg(BaseModel):
    mode: str = "paper"
    require_approval: bool = False
    live_auto_execute: bool = False
    order_type: str = "market"
    order_style: Literal["notional", "auto"] = "notional"  # auto: whole-share limit orders when not fractionable
    limit_buffer_pct: float = 0.005                         # marketable limit = ask*(1+x) / bid*(1-x)
    limit_fractional_buys: bool = False                     # auto only: fractionable buys as fractional limits too
    record_quotes: bool = False                             # stage 2: journal quotes + modeled fill costs
    default_spread_pct: float = 0.003                       # cost model spread for fills without a quote
                                                            # (also the dry-run sim quote spread)


class PathsCfg(BaseModel):
    cache_dir: str = "data/cache"
    journal_db: str = "data/journal.db"
    reports_dir: str = "reports"

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path


class NotifyCfg(BaseModel):
    email: bool = True
    modes: list[str] = ["paper", "live"]                       # dry-run never emails
    phases: list[str] = ["analyze", "execute", "reconcile"]    # errors/blocked always email


class RadarCfg(BaseModel):
    enabled: bool = False   # stage 2, shadow mode only
    db_path: str = "data/radar.db"


class FeaturesCfg(BaseModel):
    """Upgrade features (spec 2026-11-04). Every flag defaults to False: the frozen controls set none."""
    watch: bool = False            # 30-minute intraday watch phase
    morning_check: bool = False    # gap guard + Morning Check agent at execute
    memory: bool = False           # Phase 2: outcome scoring, scorecards, lessons, memory block in prompts
    research: bool = False         # Phase 3: researcher deep dives (web + SEC) before the PM finalizes
    reviewers: bool = False        # Phase 3: second PM + Risk Reviewer after the PM (cut/drop only)
    event_review: bool = False     # Phase 3: intraday Event Review on held names (two agents must agree)
    intraday_buys: bool = False    # Phase 3: intraday buys from the existing universe, two approvals


class ResearchCfg(BaseModel):
    """Phase 3 deep dives (on/off: features.research). The researcher gets WebSearch/WebFetch only."""
    max_dives: int = 3                  # per analyze run (usage budget: 3 per profile per day)
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    planner_effort: str = "medium"      # the PM model choosing which symbols to dive into
    max_turns: int = 12                 # tool-call budget per dive
    timeout_s: int = 300                # wall clock per dive
    filings_days: int = 30
    max_filing_chars: int = 6000


class ReviewersCfg(BaseModel):
    """Phase 3 analyze reviewers (on/off: features.reviewers): an independent second PM and a Risk Reviewer.
    Both can only cut confidence or drop views."""
    second_pm_model: str = "claude-opus-5-5"
    second_pm_effort: str = "medium"    # the first PM runs at high: a different effort mix, same inputs
    disagree_factor: float = 0.5        # opposite-direction second-PM view -> confidence x this
    risk_model: str = "claude-opus-5-5"
    risk_effort: str = "high"


class EventsCfg(BaseModel):
    """Phase 3 intraday Event Review (on/off: features.event_review). Trigger thresholds may be tuned by the
    Phase 4 reviewer; the caps and window are usage/timing rules, not risk limits."""
    k_sigma: float = 2.5                         # move trigger: |mid / prior close - 1| >= k x daily sigma
    max_reviews_per_day: int = 6                 # per profile (usage budget)
    max_buy_pipelines_per_day: int = 2           # intraday buy pipelines per profile (usage budget)
    max_reviews_per_watch: int = 2               # one 30-minute watch must finish before the next stop check
    max_buy_pipelines_per_watch: int = 1
    agent_budget_s: float = 1200                 # wall clock for all agent work in one watch
    window: tuple[str, str] = ("09:45", "15:45")  # ET: no LLM-initiated orders in the first/last 15 minutes
    backoff_hours: float = 5                     # agents off this long after a usage-limit refusal
    analyst_model: str = "claude-sonnet-5-5"
    analyst_effort: str = "medium"
    reviewer_model: str = "claude-opus-5-5"
    reviewer_effort: str = "high"
    max_turns: int = 8
    timeout_s: int = 240


class MorningCheckCfg(BaseModel):
    """Morning Check at execute (on/off: features.morning_check). It may only cancel or shrink buys."""
    gap_sigma_k: float = 3.0            # deterministic guard: cancel a buy gapping down > k x daily sigma
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_items_per_symbol: int = 8       # overnight headlines / radar posts per symbol


class MemoryCfg(BaseModel):
    """Phase 2 memory (on/off: features.memory). Empty dir disables memory even with the flag on."""
    dir: str = ""                       # synced folder for lessons.md / staging.md / changelog.md / scorecard.md
    max_block_chars: int = 4000         # memory block per agent call
    min_lesson_n: int = 20              # scored outcomes a lesson needs as evidence
    max_lessons: int = 15
    reviewer_model: str = "claude-sonnet-5-5"
    reviewer_effort: str = "medium"
    curator_model: str = "claude-opus-5-5"
    curator_effort: str = "high"


class ControlCfg(BaseModel):
    """Frozen control profiles only (execution.mode: control)."""
    book_path: str = "data/book_control.json"
    starting_cash: float = 100_000.0


class Settings(BaseModel):
    profile: str = "default"
    universe: Universe
    broker: BrokerCfg = BrokerCfg()
    data: DataCfg = DataCfg()
    kronos: KronosCfg = KronosCfg()
    agents: AgentsCfg = AgentsCfg()
    portfolio: PortfolioCfg = PortfolioCfg()
    risk: RiskCfg = RiskCfg()
    execution: ExecutionCfg = ExecutionCfg()
    paths: PathsCfg = PathsCfg()
    notify: NotifyCfg = NotifyCfg()
    radar: RadarCfg = RadarCfg()
    features: FeaturesCfg = FeaturesCfg()
    morning_check: MorningCheckCfg = MorningCheckCfg()
    control: ControlCfg = ControlCfg()
    memory: MemoryCfg = MemoryCfg()
    research: ResearchCfg = ResearchCfg()
    reviewers: ReviewersCfg = ReviewersCfg()
    events: EventsCfg = EventsCfg()

    @property
    def all_symbols(self) -> list[str]:
        return list(dict.fromkeys([*self.universe.symbols, self.universe.benchmark]))


def settings_path(profile: str = "default", config: str | Path | None = None) -> Path:
    if config:
        return Path(config)
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    return ROOT / PROFILES[profile]


def load_settings(path: str | Path | None = None, profile: str = "default") -> Settings:
    load_dotenv(ROOT / ".env")
    path = settings_path(profile, path)
    with open(path, encoding="utf-8") as f:
        return Settings.model_validate(yaml.safe_load(f))


def secret(name: str, required: bool = True) -> str:
    load_dotenv(ROOT / ".env")
    val = os.environ.get(name, "").strip()
    if required and not val:
        raise RuntimeError(f"Missing required secret {name}; set it in .env (see .env.example)")
    return val
