import pytest
from pydantic import ValidationError

from bot.agents.llm import load_prompt, prompt_version
from bot.config import ROOT, Settings, Universe, load_settings, settings_path

# Snapshot of load_settings("config/settings.yaml").model_dump() taken before profile B existed.
BASELINE_A = {
    "universe": {"symbols": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "JPM", "V", "UNH", "XOM",
                             "LLY", "COST", "HD", "PG", "JNJ", "WMT", "MA", "NFLX", "AMD", "QQQ", "XLE", "XLF",
                             "XLV", "TLT", "GLD"], "benchmark": "SPY"},
    "data": {"history_days": 800, "max_staleness_days": 4, "news_per_symbol": 5},
    "kronos": {"enabled": False, "model": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
               "lookback": 400, "pred_len": 10, "sample_count": 8, "device": "cpu"},
    "agents": {"analyst_model": "claude-opus-5-5", "analyst_effort": "low", "debate_model": "claude-opus-5-5",
               "debate_effort": "medium", "pm_model": "claude-opus-5-5", "pm_effort": "high", "debate_rounds": 1,
               "max_daily_output_tokens": 150000, "horizon_days": 10, "max_abs_view_return": 0.08},
    "portfolio": {"method": "mean_risk", "risk_measure": "variance", "tau": 0.05, "max_weight": 0.15,
                  "min_cash_weight": 0.05, "l2_coef": 0.0005, "delta": 2.5, "lookback_days": 504},
    "risk": {"max_position_weight": 0.15, "max_gross_exposure": 0.95, "max_orders_per_day": 20,
             "max_daily_turnover": 0.4, "min_order_notional": 5.0, "max_daily_loss_pct": 0.03,
             "max_drawdown_pct": 0.15, "kill_switch_file": "KILL_SWITCH"},
    "execution": {"mode": "paper", "require_approval": False, "live_auto_execute": False, "order_type": "market"},
    "paths": {"cache_dir": "data/cache", "journal_db": "data/journal.db", "reports_dir": "reports"},
    "notify": {"email": True, "modes": ["paper", "live"], "phases": ["analyze", "execute", "reconcile"]},
}


def test_default_profile_reproduces_bot_a():
    s = load_settings()
    dump = s.model_dump()
    for section, values in BASELINE_A.items():
        for k, v in values.items():
            assert dump[section][k] == v, (section, k)
    # new fields sit at their bot-A defaults
    assert s.profile == "default"
    assert (s.broker.api_key_env, s.broker.secret_key_env) == ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    assert s.universe.source == "static"
    assert s.agents.prompt_dir == "prompts"
    assert s.execution.order_style == "auto" and s.execution.limit_fractional_buys is True   # buys as limits (2026-11-04)
    assert s.execution.record_quotes is False
    assert s.risk.stop_loss_pct is None and s.risk.stop_cooldown_days == 0
    assert s.risk.max_adv_participation is None and s.risk.max_spread_pct is None
    assert s.radar.enabled is False
    assert load_settings(profile="default").model_dump() == dump


def test_smallcap_profile_values():
    s = load_settings(profile="smallcap")
    assert s.profile == "smallcap"
    assert s.broker.api_key_env == "ALPACA_B_API_KEY" and s.broker.secret_key_env == "ALPACA_B_SECRET_KEY"
    assert s.universe.source == "screener" and s.universe.max_age_days == 8 and s.universe.symbols == []
    assert s.agents.prompt_dir == "prompts_smallcap" and s.agents.max_abs_view_return == 0.20
    a = load_settings().agents
    assert (s.agents.analyst_model, s.agents.debate_model, s.agents.pm_model) == \
        (a.analyst_model, a.debate_model, a.pm_model)
    p = s.portfolio
    assert (p.method, p.lookback_days, p.tau, p.delta, p.l2_coef, p.max_weight, p.min_cash_weight) == \
        ("mean_risk", 252, 0.10, 1.0, 0.0001, 0.12, 0.05)
    r = s.risk
    assert (r.max_position_weight, r.max_gross_exposure, r.max_daily_turnover, r.max_orders_per_day,
            r.min_order_notional, r.max_daily_loss_pct, r.max_drawdown_pct) == (0.12, 0.95, 0.50, 20, 25.0, 0.06, 0.25)
    assert (r.stop_loss_pct, r.stop_cooldown_days, r.max_adv_participation, r.max_spread_pct) == (0.15, 10, 0.02, 0.015)
    assert r.kill_switch_file == "KILL_SWITCH_SMALLCAP"
    assert s.execution.order_style == "auto" and s.execution.record_quotes is True
    assert s.paths.journal_db == "data/journal_smallcap.db" and s.paths.reports_dir == "reports/smallcap"
    assert s.kronos.enabled is False and s.radar.enabled is True
    assert s.notify == load_settings().notify


def test_profile_paths_and_config_override(tmp_path):
    assert settings_path("default") == ROOT / "config" / "settings.yaml"
    assert settings_path("smallcap") == ROOT / "config" / "settings_smallcap.yaml"
    other = tmp_path / "x.yaml"
    assert settings_path("smallcap", other) == other
    with pytest.raises(ValueError):
        settings_path("nope")


def test_symbols_optional_only_for_screener():
    with pytest.raises(ValidationError):
        Universe(source="static")
    assert Universe(source="screener").symbols == []
    assert Settings(universe=Universe(source="screener")).all_symbols == ["SPY"]


def test_prompt_dir(tmp_path):
    for n in ("regime", "portfolio_manager"):
        (tmp_path / f"{n}.md").write_text(f"custom {n} {{horizon}}", encoding="utf-8")
    assert load_prompt("regime", str(tmp_path)) == "custom regime {horizon}"
    assert prompt_version(str(tmp_path)) != prompt_version()
    assert load_prompt("regime") != load_prompt("regime", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        prompt_version(str(tmp_path / "missing"))


def test_team_uses_prompt_dir(tmp_path, acfg, pcfg):
    from bot.agents.schemas import AnalystReport, DebateCase, PMDecision, RegimeReport
    from bot.agents.team import AgentTeam, TeamInputs

    names = ["regime", "technical", "fundamentals", "news", "bull", "bear", "portfolio_manager"]
    for n in names:
        (tmp_path / f"{n}.md").write_text(f"B-{n} {{horizon}} {{max_view}} {{min_cash}} {{universe}}",
                                          encoding="utf-8")
    seen = []

    class FakeLLM:
        def structured(self, *, model, effort, system, user, schema):
            seen.append(system)
            return {
                RegimeReport: lambda: RegimeReport(regime="neutral", confidence=0.5, suggested_invested_fraction=0.9,
                                                   summary="", key_points=[]),
                AnalystReport: lambda: AnalystReport(summary="", notes=[]),
                DebateCase: lambda: DebateCase(side="bull", overview="", arguments=[]),
                PMDecision: lambda: PMDecision(market_view="", cash_weight=0.1, views=[]),
            }[schema]()

    cfg = acfg.model_copy(update={"prompt_dir": str(tmp_path)})
    AgentTeam(FakeLLM(), cfg, pcfg).run(TeamInputs(as_of="2026-11-03", universe=["AAA"], regime={},
                                                    technical={}, fundamentals={}, news={}))
    assert len(seen) == 7 and all(x.startswith("B-") for x in seen)
    assert "B-portfolio_manager 10 0.08 0.05 AAA" in seen


def test_smallcap_prompts_have_pm_placeholders():
    d = ROOT / "src" / "bot" / "agents" / "prompts_smallcap"
    if not (d / "portfolio_manager.md").exists():
        pytest.skip("prompts_smallcap not written yet")
    import re

    text = (d / "portfolio_manager.md").read_text(encoding="utf-8")
    a_text = (ROOT / "src" / "bot" / "agents" / "prompts" / "portfolio_manager.md").read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{(\w+)\}", text))
    assert set(re.findall(r"\{(\w+)\}", a_text)) <= placeholders <= {"horizon", "max_view", "min_cash", "universe"}
    text.format(horizon=10, max_view=0.2, min_cash=0.05, universe="AAA")  # no stray braces
    assert prompt_version("prompts_smallcap") != prompt_version()


def test_both_profiles_limit_fractionable_buys():
    # B since security finding 003; A since the 2026-11-04 upgrade (spec: marketable limit orders)
    assert load_settings(profile="smallcap").execution.limit_fractional_buys is True
    assert load_settings().execution.limit_fractional_buys is True
    assert load_settings().execution.order_style == "auto"
