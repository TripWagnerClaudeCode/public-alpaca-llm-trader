import numpy as np
import pandas as pd
import pytest

from bot.config import AgentsCfg, PortfolioCfg, RiskCfg
from bot.models import AccountState


@pytest.fixture
def risk(tmp_path):
    return RiskCfg(kill_switch_file=str(tmp_path / "KILL_SWITCH"))


@pytest.fixture
def pcfg():
    return PortfolioCfg()


@pytest.fixture
def acfg():
    return AgentsCfg()


@pytest.fixture
def closes():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=600)
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    common = rng.normal(0.0004, 0.01, len(idx))
    data = {s: 100 * np.exp(np.cumsum(common * (0.5 + i * 0.2) + rng.normal(0.0002, 0.012, len(idx))))
            for i, s in enumerate(syms)}
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def cash_account():
    return AccountState(equity=10_000, last_equity=10_000, cash=10_000, buying_power=10_000, positions={})


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch, tmp_path_factory):
    """Tests must never reach the real claude CLI (it spends your Claude subscription). The one function that
    launches it raises instead; agent steps treat that as unavailable. Tests that fake the LLM class are unaffected."""
    from bot.agents import llm

    def refuse(self, **k):
        raise RuntimeError("real claude CLI disabled in tests")

    monkeypatch.setattr(llm.LLM, "_structured_cli", refuse)

    # research() may only launch a fake CLI script that lives under pytest's temp base
    real_research = llm.LLM.research
    base = tmp_path_factory.getbasetemp().resolve()

    def guarded_research(self, **k):
        from pathlib import Path

        target = Path(self._cli_cmd[-1]).resolve() if self._cli_cmd else None
        try:
            target.relative_to(base)
        except (AttributeError, ValueError):
            raise RuntimeError("real claude CLI disabled in tests") from None
        return real_research(self, **k)

    monkeypatch.setattr(llm.LLM, "research", guarded_research)

    def no_api(*a, **k):
        raise RuntimeError("Anthropic API client disabled in tests (it would bill ANTHROPIC_API_KEY)")

    monkeypatch.setattr(llm.anthropic, "Anthropic", no_api)
    monkeypatch.setattr(llm, "find_claude_cli", lambda explicit=None: "claude-disabled-in-tests")


@pytest.fixture(autouse=True)
def _no_real_sec(monkeypatch):
    """Tests never reach SEC EDGAR: a request without an injected (mock) session raises, which every
    caller treats as 'EDGAR unavailable'."""
    monkeypatch.setenv("SEC_USER_AGENT", "test-bot test@example.com")
    from bot.research import edgar

    real = edgar._get

    def guarded(url, session):
        if session is None:
            raise RuntimeError("SEC network disabled in tests")
        return real(url, session)

    monkeypatch.setattr(edgar, "_get", guarded)


@pytest.fixture(autouse=True)
def _no_real_memory_dir(monkeypatch, tmp_path_factory):
    """The live settings point memory at a real memory folder; tests must never write there.
    Any memory folder outside pytest's temp base is redirected into it."""
    import bot.memory as memory

    base = tmp_path_factory.getbasetemp()
    real = memory.memory_root

    def guarded(s):
        root = real(s)
        try:
            root.resolve().relative_to(base.resolve())
            return root
        except ValueError:
            return base / "redirected-memory" / root.name

    monkeypatch.setattr(memory, "memory_root", guarded)
