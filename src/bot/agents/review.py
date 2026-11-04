"""Phase 3 reviewers (spec 2026-11-04 section 4). Deterministic application of their outputs:

- second_pm_cut: an independent second PM's views only CUT confidence where it disagrees on direction.
  Its views are never added and never raise anything.
- apply_risk_review: the Risk Reviewer may keep, lower or drop a view, never raise one.
"""
from __future__ import annotations

from bot.agents.schemas import RiskReview
from bot.models import View


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def second_pm_cut(views: list[View], pm2: list[View], factor: float) -> tuple[list[View], list[str]]:
    """Opposite-sign view from PM2 -> confidence x factor. Same sign or no PM2 view -> unchanged."""
    other = {v.symbol: v for v in pm2}
    out, notes = [], []
    for v in views:
        o = other.get(v.symbol)
        if o is not None and _sign(o.expected_return) * _sign(v.expected_return) < 0:
            new = round(v.confidence * factor, 6)
            notes.append(f"{v.symbol}: second PM disagrees ({o.expected_return:+.2%} vs {v.expected_return:+.2%}), "
                         f"confidence {v.confidence:.2f} -> {new:.2f}")
            v = v.model_copy(update={"confidence": new})
        out.append(v)
    return out, notes


def apply_risk_review(views: list[View], review: RiskReview) -> tuple[list[View], list[str]]:
    """keep: unchanged; lower: confidence = min(old, given); drop: removed. Unknown symbols ignored."""
    items = {i.symbol.upper().strip(): i for i in review.items}
    out, notes = [], []
    for v in views:
        it = items.get(v.symbol)
        if it is None or it.action == "keep":
            out.append(v)
        elif it.action == "drop":
            notes.append(f"{v.symbol}: dropped by Risk Reviewer ({it.objection})")
        else:
            new = min(v.confidence, max(0.0, it.confidence))
            if new < v.confidence:
                notes.append(f"{v.symbol}: Risk Reviewer lowered confidence {v.confidence:.2f} -> {new:.2f} "
                             f"({it.objection})")
            out.append(v.model_copy(update={"confidence": new}))
    return out, notes
