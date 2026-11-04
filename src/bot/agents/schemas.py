"""Structured outputs for each agent. Validated by Pydantic after the API returns."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SymbolNote(BaseModel):
    symbol: str
    stance: Literal["bullish", "neutral", "bearish"]
    strength: float = Field(ge=0, le=1)
    key_points: list[str]


class AnalystReport(BaseModel):
    summary: str
    notes: list[SymbolNote]


class RegimeReport(BaseModel):
    regime: Literal["risk_on", "neutral", "risk_off"]
    confidence: float = Field(ge=0, le=1)
    suggested_invested_fraction: float = Field(ge=0.5, le=0.95)
    summary: str
    key_points: list[str]


class SymbolArgument(BaseModel):
    symbol: str
    argument: str


class DebateCase(BaseModel):
    side: Literal["bull", "bear"]
    overview: str
    arguments: list[SymbolArgument]


class PMView(BaseModel):
    symbol: str
    expected_return: float
    confidence: float = Field(ge=0, le=1)
    rationale: str


class PMDecision(BaseModel):
    market_view: str
    cash_weight: float = Field(ge=0, le=0.5)
    views: list[PMView]


class MorningItem(BaseModel):
    symbol: str
    action: Literal["keep", "reduce", "cancel"]
    fraction: float = Field(1.0, description="for reduce: share of the planned buy to keep, 0 to 1")
    reason: str


class MorningCheck(BaseModel):
    items: list[MorningItem]
    summary: str


class Observation(BaseModel):
    text: str
    agents: list[str]
    symbols: list[str]
    evidence: str


class PostTradeNotes(BaseModel):
    observations: list[Observation]
    summary: str


class NewLesson(BaseModel):
    text: str
    evidence: str
    n: int = Field(description="scored outcomes supporting this lesson")
    tags: list[str]


class LessonEdit(BaseModel):
    id: str
    text: str
    evidence: str
    n: int


class LessonRetire(BaseModel):
    id: str
    reason: str


class LessonsUpdate(BaseModel):
    add: list[NewLesson]
    edit: list[LessonEdit]
    retire: list[LessonRetire]
    summary: str


class Finding(BaseModel):
    claim: str
    source_url: str
    date: str
    support: Literal["strong", "moderate", "weak"]


class DiveFindings(BaseModel):
    symbol: str
    question: str
    answer: str
    findings: list[Finding]


class ResearchRequest(BaseModel):
    symbol: str
    question: str


class ResearchPlan(BaseModel):
    requests: list[ResearchRequest]


class RiskItem(BaseModel):
    symbol: str
    action: Literal["keep", "lower", "drop"]
    confidence: float = Field(ge=0, le=1, description="for lower: the new confidence (never higher than the view's)")
    objection: str


class RiskReview(BaseModel):
    items: list[RiskItem]
    summary: str


class EventCall(BaseModel):
    action: Literal["hold", "trim", "exit"]
    fraction: float = Field(1.0, description="for trim: share of the position to sell, 0 to 1")
    reason: str


class BuyApproval(BaseModel):
    approve: bool
    reason: str
