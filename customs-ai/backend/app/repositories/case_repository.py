"""Case repository — ORM CRUD (ADR-006: faqat ORM, DB-agnostik).

Har metod o'z sessiyasini boshqaradi va commit qiladi. Bu har bosqichni
darhol durable qiladi (audit/startup recovery uchun muhim).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.core.enums import CaseStatus
from app.models.entities import (
    Attachment,
    Case,
    Detection,
    Explanation,
    RiskResult,
    Transcript,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nid() -> str:
    return str(uuid.uuid4())


class CaseRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    # ---- yaratish ----
    def create(self, *, operator_id: str | None, operator_notes: str | None) -> str:
        cid = _nid()
        with self._sf() as s:
            s.add(
                Case(
                    id=cid,
                    status=CaseStatus.PENDING.value,
                    operator_id=operator_id,
                    operator_notes=operator_notes,
                    degraded=False,
                )
            )
            s.commit()
        return cid

    def add_attachment(self, case_id: str, att: dict) -> None:
        with self._sf() as s:
            s.add(
                Attachment(
                    id=_nid(),
                    case_id=case_id,
                    kind=att["kind"],
                    path=att["path"],
                    sha256=att["sha256"],
                )
            )
            s.commit()

    def exists_attachment_sha256(self, sha256: str) -> bool:
        """Shu sha256'li attachment bormi? (scanner restart'lararo dedup uchun)."""
        with self._sf() as s:
            return s.execute(
                select(Attachment.id).where(Attachment.sha256 == sha256).limit(1)
            ).first() is not None

    # ---- o'qish ----
    def get(self, case_id: str) -> Case | None:
        with self._sf() as s:
            stmt = (
                select(Case)
                .options(
                    selectinload(Case.detections),
                    selectinload(Case.risk_result),
                    selectinload(Case.transcript),
                    selectinload(Case.explanation),
                    selectinload(Case.attachments),
                )
                .where(Case.id == case_id)
            )
            return s.execute(stmt).scalar_one_or_none()

    def get_status(self, case_id: str) -> str | None:
        with self._sf() as s:
            return s.execute(
                select(Case.status).where(Case.id == case_id)
            ).scalar_one_or_none()

    def list_audit(self, case_id: str) -> list:
        from app.models.entities import AuditLog

        with self._sf() as s:
            stmt = (
                select(AuditLog)
                .where(AuditLog.case_id == case_id)
                .order_by(AuditLog.ts.asc())
            )
            return list(s.execute(stmt).scalars().all())

    def list(
        self,
        *,
        status: str | None = None,
        risk_level: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        with self._sf() as s:
            base = select(Case).options(selectinload(Case.risk_result))
            count_stmt = select(func.count()).select_from(Case)
            if status:
                base = base.where(Case.status == status)
                count_stmt = count_stmt.where(Case.status == status)
            if risk_level:
                base = base.join(RiskResult).where(RiskResult.level == risk_level)
                count_stmt = count_stmt.join(RiskResult).where(RiskResult.level == risk_level)

            base = base.order_by(Case.created_at.desc()).limit(limit).offset(offset)
            rows = list(s.execute(base).scalars().all())
            total = s.execute(count_stmt).scalar_one()
            items = [
                {
                    "case_id": c.id,
                    "status": c.status,
                    "risk_level": c.risk_result.level if c.risk_result else None,
                    "degraded": c.degraded,
                    "created_at": c.created_at,
                }
                for c in rows
            ]
            return items, total

    # ---- yangilash (case mutable; audit emas) ----
    def set_status(self, case_id: str, status: str, *, completed: bool = False) -> None:
        with self._sf() as s:
            case = s.get(Case, case_id)
            if case is None:
                return
            case.status = status
            if completed:
                case.completed_at = _now()
            s.commit()

    def set_degraded(self, case_id: str, value: bool = True) -> None:
        with self._sf() as s:
            case = s.get(Case, case_id)
            if case:
                case.degraded = value
                s.commit()

    def set_timings(self, case_id: str, timings: dict) -> None:
        with self._sf() as s:
            case = s.get(Case, case_id)
            if case:
                case.timings = timings
                s.commit()

    def save_detections(self, case_id: str, detections: list[dict]) -> None:
        with self._sf() as s:
            for d in detections:
                s.add(
                    Detection(
                        id=_nid(),
                        case_id=case_id,
                        cls=d["class"],
                        confidence=float(d["confidence"]),
                        bbox=d["bbox"],
                    )
                )
            s.commit()

    def save_risk(self, case_id: str, risk: dict) -> None:
        with self._sf() as s:
            s.add(
                RiskResult(
                    id=_nid(),
                    case_id=case_id,
                    level=risk["level"],
                    score=float(risk["score"]),
                    computed_by=risk["computed_by"],
                    factors=risk.get("factors", []),
                )
            )
            s.commit()

    def save_transcript(self, case_id: str, transcript: dict) -> None:
        with self._sf() as s:
            s.add(
                Transcript(
                    id=_nid(),
                    case_id=case_id,
                    content=transcript.get("text"),
                    language=transcript.get("language"),
                    confidence=transcript.get("confidence"),
                    available=bool(transcript.get("available", False)),
                )
            )
            s.commit()

    def save_explanation(self, case_id: str, explanation: dict) -> None:
        with self._sf() as s:
            s.add(
                Explanation(
                    id=_nid(),
                    case_id=case_id,
                    content=explanation.get("text"),
                    model_version=explanation.get("generated_by"),
                    available=bool(explanation.get("available", False)),
                )
            )
            s.commit()

    # ---- startup recovery (§10) ----
    def recover_stuck(self) -> int:
        """PROCESSING'da osilib qolgan case'larni FAILED ga o'tkazadi."""
        with self._sf() as s:
            stuck = list(
                s.execute(
                    select(Case).where(Case.status == CaseStatus.PROCESSING.value)
                ).scalars().all()
            )
            for c in stuck:
                c.status = CaseStatus.FAILED.value
                c.completed_at = _now()
            s.commit()
            return len(stuck)
