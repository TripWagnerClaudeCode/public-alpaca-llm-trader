"""Daily Post-Trade Reviewer (Sonnet): turns newly scored outcomes into dated observations in
staging.md. Observations are context only; the weekly Curator decides what becomes a lesson."""
from __future__ import annotations

import json

from bot.agents.llm import load_prompt
from bot.agents.schemas import PostTradeNotes
from bot.agents.team import wrap

MAX_OBSERVATIONS = 5


def run_post_trade(llm, s, journal, files, newly_scored: list[dict], cards: dict, today: str) -> int:
    data = json.dumps({"newly_scored": newly_scored, "scorecards": cards}, default=str, sort_keys=True)
    user = (f"Today is {today}. These views just reached their 10-trading-day horizon. Each row has the "
            f"realized excess return vs the universe and every agent's call on that symbol.\n{wrap(data)}")
    out = llm.structured(model=s.memory.reviewer_model, effort=s.memory.reviewer_effort,
                         system=load_prompt("post_trade", s.agents.prompt_dir), user=user, schema=PostTradeNotes)
    obs = out.observations[:MAX_OBSERVATIONS]
    files.append_staging(today, [f"{o.text} [agents: {', '.join(o.agents)}] [symbols: {', '.join(o.symbols)}] "
                                 f"evidence: {o.evidence}" for o in obs])
    return len(obs)
