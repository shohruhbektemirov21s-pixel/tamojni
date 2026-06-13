"""FastAPI entrypoint — app factory, lifespan, exception handlers.

Bind: 127.0.0.1 (Faza 1 auth = faqat loopback, tashqaridan kirib bo'lmaydi).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import cases, decisions, health, ws
from app.core.config import Settings, get_settings_singleton, load_risk_config
from app.core.errors import AppError, ValidationFailed
from app.core.events import EventBus
from app.core.orchestrator import GpuOrchestrator
from app.core.providers import build_providers
from app.core.worker import CaseWorker
from app.db import init_db, make_engine, make_session_factory
from app.models.guard import install_audit_guard
from app.repositories.case_repository import CaseRepository
from app.services.audit_service import AuditService
from app.services.file_store import FileStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("customs.main")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings_singleton()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    install_audit_guard()

    repo = CaseRepository(session_factory)
    audit = AuditService(session_factory)
    file_store = FileStore(settings.files_dir, settings.min_free_disk_mb)
    orchestrator = GpuOrchestrator(settings)
    providers = build_providers(settings, orchestrator)
    risk_config = load_risk_config(settings)
    event_bus = EventBus(subscriber_buffer=settings.event_subscriber_buffer)
    worker = CaseWorker(
        settings=settings,
        repo=repo,
        audit=audit,
        orchestrator=orchestrator,
        providers=providers,
        risk_config=risk_config,
        event_bus=event_bus,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # threadsafe publish (watchdog) uchun ishlayotgan loop'ni bog'laymiz
        event_bus.bind_loop(asyncio.get_running_loop())
        # startup recovery (§10): PROCESSING'da osilib qolganlar -> FAILED
        recovered = repo.recover_stuck()
        if recovered:
            log.warning("Startup recovery: %d osilib qolgan case FAILED ga o'tkazildi", recovered)
        await orchestrator.start()
        await worker.start()
        log.info("Backend tayyor: http://%s:%s", settings.host, settings.port)
        yield
        await worker.stop()
        await orchestrator.stop()

    app = FastAPI(
        title="Offline Bojxona AI — Backend",
        version="0.1.0",
        description="Air-gapped, audit-ready backend (human-in-the-loop, deterministik risk).",
        lifespan=lifespan,
    )

    # DI uchun state
    app.state.settings = settings
    app.state.repo = repo
    app.state.audit = audit
    app.state.file_store = file_store
    app.state.orchestrator = orchestrator
    app.state.worker = worker
    app.state.event_bus = event_bus

    _register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(cases.router)
    app.include_router(decisions.router)
    app.include_router(ws.router)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_request: Request, exc: AppError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        err = ValidationFailed("So'rov validatsiyadan o'tmadi", detail=exc.errors())
        return JSONResponse(status_code=err.http_status, content=err.to_dict())

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        log.exception("Kutilmagan xato")
        err = AppError("Ichki xatolik", detail=str(exc))
        return JSONResponse(status_code=err.http_status, content=err.to_dict())


# Uvicorn uchun modul-level app (production: `uvicorn app.main:app`)
app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings_singleton()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
