"""Case endpointlari (§7): create, get, list, audit, SSE stream."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.enums import AuditAction, CaseStatus
from app.core.errors import NotFound, ValidationFailed
from app.schemas import (
    AuditEntryOut,
    AuditListOut,
    CaseCreatedOut,
    CaseListItem,
    CaseListOut,
    CaseResultOut,
)
from app.services.serializers import serialize_audit, serialize_case

router = APIRouter(tags=["cases"])

_TERMINAL = {CaseStatus.DONE.value, CaseStatus.FAILED.value}


@router.post("/cases", status_code=201, response_model=CaseCreatedOut)
async def create_case(
    request: Request,
    image: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    notes: str | None = Form(None),
    operator_id: str | None = Form(None),
) -> CaseCreatedOut:
    st = request.app.state
    st.file_store.check_capacity()  # disk to'la -> DiskFull(503)

    img_bytes = await image.read()
    if not img_bytes:
        raise ValidationFailed("Rasm bo'sh yoki yuborilmadi", detail={"field": "image"})

    case_id = await asyncio.to_thread(
        st.repo.create, operator_id=operator_id, operator_notes=notes
    )

    img_att = st.file_store.save(case_id, "image", image.filename or "scan.png", img_bytes)
    await asyncio.to_thread(st.repo.add_attachment, case_id, img_att)

    has_audio = False
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes:
            has_audio = True
            au_att = st.file_store.save(case_id, "audio", audio.filename or "audio.wav", audio_bytes)
            await asyncio.to_thread(st.repo.add_attachment, case_id, au_att)

    await asyncio.to_thread(
        st.audit.log, case_id, operator_id or "operator",
        AuditAction.CASE_CREATED.value, {"has_audio": has_audio},
    )
    await st.worker.enqueue(case_id)
    return CaseCreatedOut(case_id=case_id, status=CaseStatus.PENDING)


@router.get("/cases", response_model=CaseListOut)
async def list_cases(
    request: Request,
    status: str | None = Query(None),
    risk_level: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> CaseListOut:
    st = request.app.state
    items, total = await asyncio.to_thread(
        st.repo.list, status=status, risk_level=risk_level, limit=limit, offset=offset
    )
    return CaseListOut(items=[CaseListItem(**it) for it in items], total=total)


@router.get("/cases/{case_id}", response_model=CaseResultOut)
async def get_case(request: Request, case_id: str):
    st = request.app.state
    case = await asyncio.to_thread(st.repo.get, case_id)
    if case is None:
        raise NotFound("Case topilmadi", detail={"case_id": case_id})
    # §7.1 aniq shaklini saqlash uchun dict qaytaramiz (detections ichida "class").
    return JSONResponse(content=jsonable(serialize_case(case)))


@router.get("/cases/{case_id}/audit", response_model=AuditListOut)
async def get_audit(request: Request, case_id: str) -> AuditListOut:
    st = request.app.state
    if await asyncio.to_thread(st.repo.get_status, case_id) is None:
        raise NotFound("Case topilmadi", detail={"case_id": case_id})
    entries = await asyncio.to_thread(st.repo.list_audit, case_id)
    return AuditListOut(entries=[AuditEntryOut(**serialize_audit(e)) for e in entries])


@router.get("/cases/{case_id}/stream")
async def stream_case(request: Request, case_id: str):
    """SSE: PENDING -> PROCESSING -> DONE/FAILED."""
    st = request.app.state
    if await asyncio.to_thread(st.repo.get_status, case_id) is None:
        raise NotFound("Case topilmadi", detail={"case_id": case_id})

    async def event_gen():
        last = None
        while True:
            if await request.is_disconnected():
                break
            status = await asyncio.to_thread(st.repo.get_status, case_id)
            if status != last:
                last = status
                payload = json.dumps({"case_id": case_id, "status": status})
                yield f"event: status\ndata: {payload}\n\n"
            if status in _TERMINAL:
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---- datetime'larni JSON uchun tayyorlash ----
def jsonable(obj):
    from datetime import datetime

    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
