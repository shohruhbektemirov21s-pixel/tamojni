"""Audit Service — APPEND-ONLY (Tamoyil 3).

Har input, har model chiqishi, har operator qarori shu yerdan yoziladi.
O'zgartirish/o'chirish models/guard.py tomonidan bloklanadi.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session, sessionmaker

from app.models.entities import AuditLog


class AuditService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def log(
        self,
        case_id: str,
        actor: str,
        action: str,
        payload: dict | None = None,
        *,
        session: Session | None = None,
    ) -> str:
        """Audit yozuvini qo'shadi va uning id'sini qaytaradi.

        `session` berilsa, mavjud tranzaksiya ichida flush qiladi (case bilan
        atomar). Aks holda o'z sessiyasini ochib commit qiladi.
        """
        entry = AuditLog(
            id=str(uuid.uuid4()),
            case_id=case_id,
            actor=actor,
            action=action,
            payload=payload,
        )
        if session is not None:
            session.add(entry)
            session.flush()
            return entry.id

        with self._sf() as s:
            s.add(entry)
            s.commit()
            return entry.id
