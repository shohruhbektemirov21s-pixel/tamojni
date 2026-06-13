"""WebSocket `/ws` — real-time event push (§7).

Multi-client: bir nechta operator UI bir vaqtda subscribe qila oladi. Ixtiyoriy
`?case_id=...` filtri bilan faqat bitta case'ni tinglash mumkin; aks holda hamma
event (tizim eventlari + alertlar bilan birga) oqib keladi.

Slow-consumer izolyatsiyasi event bus darajasida (per-subscriber bounded queue,
drop-oldest) — bu yerda biz faqat ikki yo'nalishni boshqaramiz:
  - SEND loop: bus'dan event olib, JSON push qiladi.
  - RECV loop: client xabarlarini (ping/keep-alive) o'qiydi va UZILISHNI aniqlaydi.
Biri tugashi bilan ikkinchisi bekor qilinadi va subscription yopiladi.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.events import EventBus

router = APIRouter()
log = logging.getLogger("customs.ws")


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, case_id: str | None = Query(None)) -> None:
    bus: EventBus = websocket.app.state.event_bus
    await websocket.accept()
    sub = bus.subscribe(case_id=case_id)

    # ulanish tasdig'i — client qaysi filtr bilan ulanganini biladi
    await websocket.send_json(
        {"type": "ready", "case_id": case_id, "seq": 0, "ts": time.time(), "data": {}}
    )
    log.info("WS ulandi (case_id=%s, jami subscriber=%d)", case_id, bus.subscriber_count)

    async def _send() -> None:
        while True:
            event = await sub.get()
            await websocket.send_json(event.to_wire())

    async def _recv() -> None:
        # Clientdan kelgan xabarlarni o'qiymiz (ping/keep-alive yoki yopilish).
        # Maqsad — uzilishni aniqlash; mazmunni hozircha e'tiborsiz qoldiramiz.
        while True:
            await websocket.receive_text()

    send_task = asyncio.create_task(_send(), name="ws-send")
    recv_task = asyncio.create_task(_recv(), name="ws-recv")
    try:
        done, pending = await asyncio.wait(
            {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        # birinchi tugagan task istisnoni yutib yubormasligi uchun natijani so'raymiz
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        for task in (send_task, recv_task):
            task.cancel()
        sub.close()
        if sub.dropped:
            log.warning(
                "WS sekin client (case_id=%s): %d event tashlandi", case_id, sub.dropped
            )
        log.info("WS uzildi (case_id=%s, qolgan subscriber=%d)", case_id, bus.subscriber_count)
