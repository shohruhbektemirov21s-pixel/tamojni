"""ORM modellar — §8 DB schema.

ADR-006: faqat ORM, SQLite-spetsifik SQL yo'q. ID = String(36) UUID
(cross-DB), JSON ustunlar SQLite va Postgres'da ham ishlaydi.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    operator_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    risk_result: Mapped["RiskResult | None"] = relationship(
        back_populates="case", uselist=False, cascade="all, delete-orphan"
    )
    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="case", uselist=False, cascade="all, delete-orphan"
    )
    explanation: Mapped["Explanation | None"] = relationship(
        back_populates="case", uselist=False, cascade="all, delete-orphan"
    )
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    # "class" reserved bo'lgani uchun atribut cls, ustun nomi "class".
    cls: Mapped[str] = mapped_column("class", String(64))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list] = mapped_column(JSON)

    case: Mapped["Case"] = relationship(back_populates="detections")


class RiskResult(Base):
    __tablename__ = "risk_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    level: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    computed_by: Mapped[str] = mapped_column(String(64))
    factors: Mapped[list] = mapped_column(JSON)

    case: Mapped["Case"] = relationship(back_populates="risk_result")


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    case: Mapped["Case"] = relationship(back_populates="transcript")


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    case: Mapped["Case"] = relationship(back_populates="explanation")


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # image | audio | report
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64))  # chain-of-custody

    case: Mapped["Case"] = relationship(back_populates="attachments")


class AuditLog(Base):
    """Append-only (Tamoyil 3). UPDATE/DELETE app-level guard bilan bloklanadi."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    actor: Mapped[str] = mapped_column(String(64))  # "system" | operator_id
    action: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True, nullable=False
    )

    case: Mapped["Case"] = relationship(back_populates="audit_logs")
