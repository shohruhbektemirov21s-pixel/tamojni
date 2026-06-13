"""GET /health — backend + modellar holati."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    st = request.app.state
    settings = st.settings
    orch = st.orchestrator

    mode = "mock" if settings.use_mocks else "real"
    if settings.use_mocks:
        llm_status = "mock"
    else:
        llm_status = "up" if await orch.is_healthy() else "down"

    return HealthOut(
        status="ok",
        models={"yolo": mode, "stt": mode, "llm": llm_status},
        gpu={"managed": settings.manage_ollama, "daemon_alive": orch.process_alive()},
    )
