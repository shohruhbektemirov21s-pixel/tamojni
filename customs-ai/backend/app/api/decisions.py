"""POST /cases/{id}/decision — operator qarori (human-in-the-loop, Tamoyil 1)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from app.core.enums import DECISION_TO_ACTION
from app.core.errors import NotFound
from app.schemas import DecisionIn, DecisionOut

router = APIRouter(tags=["decisions"])


@router.post("/cases/{case_id}/decision", response_model=DecisionOut)
async def submit_decision(request: Request, case_id: str, body: DecisionIn) -> DecisionOut:
    st = request.app.state
    if await asyncio.to_thread(st.repo.get_status, case_id) is None:
        raise NotFound("Case topilmadi", detail={"case_id": case_id})

    action = DECISION_TO_ACTION[body.decision]
    audit_id = await asyncio.to_thread(
        st.audit.log,
        case_id,
        body.operator_id or "operator",
        action.value,
        {"decision": body.decision.value, "notes": body.notes},
    )
    return DecisionOut(audit_id=audit_id)
