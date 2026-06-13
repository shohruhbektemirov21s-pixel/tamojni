"""DB engine va sessiya fabrikasi.

ADR-006: SQLite (WAL) Faza 1; sozlamalar Postgres'ga ko'chishga tayyor.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.base import Base


def make_engine(db_url: str) -> Engine:
    connect_args = {}
    if db_url.startswith("sqlite"):
        # FastAPI/worker turli thread'lardan kiradi.
        connect_args = {"check_same_thread": False}
    engine = create_engine(db_url, connect_args=connect_args, future=True)

    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")      # bir vaqtda o'qish+yozish
            cur.execute("PRAGMA foreign_keys=ON")       # FK majburlash
            cur.execute("PRAGMA busy_timeout=5000")     # lock kutish
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: commit'dan keyin ham yuklangan atributlar ochiq qoladi
    # (detached obyektni serializatsiya qilish uchun).
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
