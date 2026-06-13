"""Markaziy enum'lar — DB'da string sifatida saqlanadi (ADR-006: Postgres-uyumli)."""
from __future__ import annotations

from enum import Enum


class CaseStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AuditAction(str, Enum):
    CASE_CREATED = "CASE_CREATED"
    STT_DONE = "STT_DONE"
    DETECTION_DONE = "DETECTION_DONE"
    RISK_COMPUTED = "RISK_COMPUTED"
    EXPLANATION_DONE = "EXPLANATION_DONE"
    TTS_DONE = "TTS_DONE"
    OPERATOR_CONFIRMED = "OPERATOR_CONFIRMED"
    OPERATOR_REJECTED = "OPERATOR_REJECTED"
    OPERATOR_OVERRIDE = "OPERATOR_OVERRIDE"
    MODEL_FAILED = "MODEL_FAILED"


class DecisionType(str, Enum):
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    OVERRIDE = "OVERRIDE"


# Operator qarori -> audit action (POST /cases/{id}/decision)
DECISION_TO_ACTION = {
    DecisionType.CONFIRM: AuditAction.OPERATOR_CONFIRMED,
    DecisionType.REJECT: AuditAction.OPERATOR_REJECTED,
    DecisionType.OVERRIDE: AuditAction.OPERATOR_OVERRIDE,
}


class AttachmentKind(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    REPORT = "report"
