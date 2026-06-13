"""ORM -> §7.1 Case Result dict.

`available` / `computed_by` / `generated_by` / `degraded` maydonlari har doim
mavjud (operator qaysi qism AI'dan, qaysi qism deterministik ekanini ko'radi).
"""
from __future__ import annotations

from app.models.entities import AuditLog, Case


def serialize_case(case: Case) -> dict:
    risk = None
    if case.risk_result is not None:
        risk = {
            "level": case.risk_result.level,
            "score": case.risk_result.score,
            "computed_by": case.risk_result.computed_by,
            "factors": case.risk_result.factors or [],
        }

    detections = [
        {"class": d.cls, "confidence": d.confidence, "bbox": d.bbox} for d in case.detections
    ]

    transcript = None
    if case.transcript is not None:
        transcript = {
            "text": case.transcript.content,
            "language": case.transcript.language,
            "confidence": case.transcript.confidence,
            "available": case.transcript.available,
        }

    explanation = None
    if case.explanation is not None:
        explanation = {
            "text": case.explanation.content,
            "generated_by": case.explanation.model_version,
            "available": case.explanation.available,
        }

    return {
        "case_id": case.id,
        "status": case.status,
        "risk": risk,
        "detections": detections,
        "transcript": transcript,
        "explanation": explanation,
        "operator_notes": case.operator_notes,
        "timings_ms": case.timings,
        "degraded": case.degraded,
    }


def serialize_audit(entry: AuditLog) -> dict:
    return {
        "id": entry.id,
        "actor": entry.actor,
        "action": entry.action,
        "payload": entry.payload,
        "ts": entry.ts,
    }
