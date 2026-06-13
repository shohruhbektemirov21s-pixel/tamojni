"""Audit append-only guard (Tamoyil 3): UPDATE/DELETE bloklanadi."""
from __future__ import annotations

import pytest

from app.core.errors import AuditImmutable
from app.db import init_db, make_engine, make_session_factory
from app.models.entities import AuditLog, Case
from app.models.guard import install_audit_guard


def _setup(tmp_path):
    install_audit_guard()
    engine = make_engine(f"sqlite:///{tmp_path}/audit.db")
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as s:
        s.add(Case(id="c1", status="PENDING"))
        s.add(AuditLog(id="a1", case_id="c1", actor="system", action="CASE_CREATED"))
        s.commit()
    return sf


def test_audit_update_blocked(tmp_path):
    sf = _setup(tmp_path)
    with sf() as s:
        entry = s.get(AuditLog, "a1")
        entry.action = "TAMPERED"
        with pytest.raises(AuditImmutable):
            s.commit()


def test_audit_delete_blocked(tmp_path):
    sf = _setup(tmp_path)
    with sf() as s:
        entry = s.get(AuditLog, "a1")
        s.delete(entry)
        with pytest.raises(AuditImmutable):
            s.commit()
