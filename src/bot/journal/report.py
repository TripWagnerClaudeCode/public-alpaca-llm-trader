"""Human-readable markdown report for each analyze run."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def write_report(path_dir: Path, run_id: int, mode: str, team_out, targets: pd.Series,
                 kronos: dict, gate_notes: list[str], cost_usd: float, regime=None,
                 pnl: dict | None = None) -> Path:
    """`pnl` (Journal.pnl_summary, only for profiles with execution.record_quotes) adds a
    paper-vs-realistic P&L section; None leaves the report exactly as before."""
    path_dir.mkdir(parents=True, exist_ok=True)
    path = path_dir / f"{dt.date.today()}_run{run_id}.md"
    L = [f"# Trading bot report: {dt.date.today()} (run {run_id}, {mode})", ""]
    if regime is not None:
        L += [f"**Regime:** {regime.regime} (confidence {regime.confidence:.0%}). {regime.summary}", ""]
    if team_out is not None:
        pm = team_out.reports.get("portfolio_manager")
        if pm is not None:
            L += ["## Portfolio manager", "", pm.market_view, "", f"Cash weight: {team_out.cash_weight:.0%}", ""]
        L += ["## Views", "", "| Symbol | Exp. return | Confidence | Rationale |", "|---|---|---|---|"]
        for v in sorted(team_out.views, key=lambda v: -v.expected_return):
            L.append(f"| {v.symbol} | {v.expected_return:+.2%} | {v.confidence:.0%} | {v.rationale} |")
        if team_out.rejected:
            L += ["", "Rejected views: " + "; ".join(team_out.rejected)]
        L.append("")
    L += ["## Target weights", "", "| Symbol | Weight | Kronos rel. fcst |", "|---|---|---|"]
    for s, w in targets.sort_values(ascending=False).items():
        if w > 0:
            k = kronos.get(s)
            L.append(f"| {s} | {w:.1%} | {k['relative']:+.2%} |" if k else f"| {s} | {w:.1%} | n/a |")
    L += ["", f"Invested: {targets.sum():.1%}  ·  LLM cost this run: ${cost_usd:.2f}", ""]
    if pnl is not None:
        from bot.journal.store import format_pnl

        L += ["## P&L: paper vs realistic", ""] + [f"- {x}" for x in format_pnl(pnl)] + [""]
    if gate_notes:
        L += ["## Risk gate", ""] + [f"- {n}" for n in gate_notes]
    path.write_text("\n".join(L), encoding="utf-8")
    return path
