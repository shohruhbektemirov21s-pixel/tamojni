"""Audit append-only guard (Tamoyil 3).

App-level — DB-agnostik (SQLite va Postgres'da bir xil ishlaydi, ADR-006).
Har qanday sessiyada AuditLog UPDATE/DELETE urinishi AuditImmutable beradi.
"""
from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.core.errors import AuditImmutable
from app.models.entities import AuditLog

_INSTALLED = False


def install_audit_guard() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    @event.listens_for(Session, "before_flush")
    def _block_audit_mutation(session: Session, flush_context, instances) -> None:  # noqa: ANN001
        for obj in session.dirty:
            if isinstance(obj, AuditLog) and session.is_modified(obj):
                raise AuditImmutable(
                    "Audit yozuvini o'zgartirib bo'lmaydi (append-only)",
                    detail={"id": obj.id},
                )
        for obj in session.deleted:
            if isinstance(obj, AuditLog):
                raise AuditImmutable(
                    "Audit yozuvini o'chirib bo'lmaydi (append-only)",
                    detail={"id": obj.id},
                )
