"""API I/O modellari (Pydantic) — OpenAPI auto-doc uchun.

Case Result aniq §7.1 shaklida `serializers.serialize_case` orqali dict sifatida
qaytariladi (detections/factors ichida "class" kaliti saqlanishi uchun).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.core.enums import CaseStatus, DecisionType, RiskLevel


class CaseCreatedOut(BaseModel):
    case_id: str
    status: CaseStatus


class DecisionIn(BaseModel):
    decision: DecisionType
    notes: str | None = None
    operator_id: str | None = None


class DecisionOut(BaseModel):
    audit_id: str


class AuditEntryOut(BaseModel):
    id: str
    actor: str
    action: str
    payload: dict | None = None
    ts: datetime


class AuditListOut(BaseModel):
    entries: list[AuditEntryOut]


class CaseListItem(BaseModel):
    case_id: str
    status: CaseStatus
    risk_level: RiskLevel | None = None
    degraded: bool
    created_at: datetime


class CaseListOut(BaseModel):
    items: list[CaseListItem]
    total: int


# ---- Case Result (§7.1) — hujjat/doc uchun; runtime dict qaytaradi ----
class RiskOut(BaseModel):
    level: RiskLevel
    score: float
    computed_by: str
    factors: list[dict]


class TranscriptOut(BaseModel):
    text: str | None = None
    language: str | None = None
    confidence: float | None = None
    available: bool


class ExplanationOut(BaseModel):
    text: str | None = None
    generated_by: str | None = None
    available: bool


class CaseResultOut(BaseModel):
    case_id: str
    status: CaseStatus
    risk: RiskOut | None = None
    detections: list[dict] = Field(default_factory=list)
    transcript: TranscriptOut | None = None
    explanation: ExplanationOut | None = None
    operator_notes: str | None = None
    timings_ms: dict | None = None
    degraded: bool = False


class HealthOut(BaseModel):
    status: str
    models: dict
    gpu: dict
