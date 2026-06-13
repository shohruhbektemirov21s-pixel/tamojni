"""Event bus — real-time push'ning markazi (Tamoyil: event-driven, polling YO'Q).

Pipeline (worker / scanner / on-demand endpointlar) bu yerga `publish()` qiladi;
WebSocket `/ws` clientlari `subscribe()` orqali tinglaydi. Fan-out NON-BLOCKING:
sekin client butun real-time oqimni TO'XTATA OLMAYDI — har subscriber'da alohida
bounded navbat bor, to'lib qolsa eng eskisi tashlanadi + `dropped` hisoblanadi
(slow-consumer izolyatsiyasi). Bu ingestion navbatidagi backpressure'dan
boshqacha: u yerda burst case'lar navbatga turadi (drop emas), bu yerda esa
sekin UI client pipeline'ni ushlab turmasligi uchun ataylab drop-oldest.

Event konverti (§7 WebSocket schema):
    {"type": str, "case_id": str|None, "seq": int, "ts": float, "data": {...}}

`seq` — bus bo'yicha monotonik; client tartiblash/uzilishdan keyin qayta
ulanganda nimani o'tkazib yuborganini bilishi uchun. `case_id=None` — tizim
darajasidagi event (masalan backpressure ogohlantirishi), HAMMA subscriber'ga
boradi.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any

log = logging.getLogger("customs.events")


class EventType(str, Enum):
    # --- scanner ingestion ---
    SCAN_INGESTED = "scan_ingested"      # papkaga fayl tushdi -> avto case
    CASE_CREATED = "case_created"        # case yozildi (qo'lda yoki avto)
    # --- Tier 1 (DARHOL, <1s, har skanga) ---
    TIER1_DONE = "tier1_done"            # detect + DETERMINISTIK risk tayyor
    ALERT = "alert"                      # HIGH risk -> operatorga signal
    # --- STT streaming (jonli) ---
    STT_PARTIAL = "stt_partial"
    STT_DONE = "stt_done"
    # --- LLM streaming (jonli, on-demand / flagged) ---
    EXPLANATION_TOKEN = "explanation_token"
    EXPLANATION_DONE = "explanation_done"
    # --- TTS ---
    TTS_READY = "tts_ready"
    # --- lifecycle / sog'liq ---
    CASE_DONE = "case_done"
    CASE_FAILED = "case_failed"
    MODEL_FAILED = "model_failed"        # degradatsiya (Tamoyil 6)
    BACKPRESSURE = "backpressure"        # ingestion navbati to'lib boryapti


@dataclass(slots=True)
class Event:
    type: EventType
    case_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        """WebSocket/JSON uchun konvert (§7 schema)."""
        return {
            "type": self.type.value,
            "case_id": self.case_id,
            "seq": self.seq,
            "ts": self.ts,
            "data": self.data,
        }


class Subscription:
    """Bitta tinglovchi (odatda bitta WebSocket client).

    `case_id=None` -> hamma event; aks holda faqat shu case + tizim eventlari.
    Navbat bounded; to'lib qolsa eng eski event tashlanadi (slow-consumer
    pipeline'ni ushlab turmasligi uchun). `dropped` — tashlangan event soni.
    """

    __slots__ = ("bus", "case_id", "_queue", "dropped")

    def __init__(self, bus: EventBus, case_id: str | None, maxsize: int) -> None:
        self.bus = bus
        self.case_id = case_id
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _matches(self, event: Event) -> bool:
        # tizim eventi (case_id=None) hammaga; aks holda subscriber filtri.
        if self.case_id is None or event.case_id is None:
            return True
        return event.case_id == self.case_id

    def _offer(self, event: Event) -> None:
        """NON-BLOCKING: navbat to'lsa eng eskisini tashlab, yangisini qo'yadi."""
        try:
            self._queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        # drop-oldest
        try:
            self._queue.get_nowait()
            self.dropped += 1
        except asyncio.QueueEmpty:
            pass
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:  # poyga holatida — bu eventni ham tashlaymiz
            self.dropped += 1

    async def get(self) -> Event:
        return await self._queue.get()

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> Event:
        return await self._queue.get()

    def close(self) -> None:
        self.bus._unsubscribe(self)


class EventBus:
    """In-process pub/sub. `publish()` SYNC va NON-BLOCKING — event loop ichidan
    chaqiriladi (worker, endpointlar). Boshqa thread'dan (watchdog) push qilish
    uchun `publish_threadsafe()` ishlating."""

    def __init__(self, *, subscriber_buffer: int = 1000) -> None:
        self._subs: set[Subscription] = set()
        self._seq = count(1)
        self._buffer = subscriber_buffer
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Lifespan startup'da chaqiriladi — threadsafe publish uchun loop havolasi."""
        self._loop = loop

    # ---- subscribe ----
    def subscribe(self, case_id: str | None = None) -> Subscription:
        sub = Subscription(self, case_id, self._buffer)
        self._subs.add(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    # ---- publish ----
    def publish(
        self, type: EventType, case_id: str | None = None, data: dict[str, Any] | None = None
    ) -> Event:
        """Event yaratib, mos subscriber'larga fan-out qiladi. Event loop ichidan."""
        event = Event(
            type=type,
            case_id=case_id,
            data=data or {},
            seq=next(self._seq),
            ts=time.time(),
        )
        for sub in self._subs:
            if sub._matches(event):
                sub._offer(event)
        return event

    def publish_threadsafe(
        self, type: EventType, case_id: str | None = None, data: dict[str, Any] | None = None
    ) -> None:
        """Boshqa thread'dan (masalan watchdog observer) xavfsiz publish.

        Loop bog'lanmagan bo'lsa (test/startupdan oldin) — bevosita publish.
        """
        if self._loop is None:
            self.publish(type, case_id, data)
            return
        self._loop.call_soon_threadsafe(self.publish, type, case_id, data)
