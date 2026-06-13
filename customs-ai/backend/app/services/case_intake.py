"""CaseIntake — case yaratishning YAGONA yo'li (API + scanner uchun umumiy).

API endpointi (qo'lda) ham, WatchedFolderSource (avtomatik) ham shu servis orqali
case ochadi — logika takrorlanmaydi. Oqim:
    case yarat -> rasm (+audio) saqla -> CASE_CREATED audit -> worker.enqueue
    -> event push (scan_meta bo'lsa SCAN_INGESTED, so'ng CASE_CREATED).

`submit` event loop ichida chaqiriladi (await). Scanner (watchdog) boshqa
thread'da bo'lgani uchun `submit_threadsafe` orqali loop'ga ko'chiradi.
"""
from __future__ import annotations

import asyncio
import logging

from app.core.enums import AuditAction, CaseStatus
from app.core.errors import ValidationFailed
from app.core.events import EventBus, EventType

log = logging.getLogger("customs.intake")


class CaseIntake:
    def __init__(self, *, repo, audit, file_store, worker, event_bus: EventBus) -> None:
        self.repo = repo
        self.audit = audit
        self.file_store = file_store
        self.worker = worker
        self.bus = event_bus
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Lifespan startup'da — threadsafe submit (scanner thread) uchun."""
        self._loop = loop

    async def submit(
        self,
        *,
        image_bytes: bytes,
        image_filename: str | None = None,
        audio_bytes: bytes | None = None,
        audio_filename: str | None = None,
        notes: str | None = None,
        operator_id: str | None = None,
        source: str = "manual",
        scan_meta: dict | None = None,
    ) -> str:
        """Case yaratadi va case_id qaytaradi. Event loop ichida."""
        self.file_store.check_capacity()  # disk to'la -> DiskFull(503)
        if not image_bytes:
            raise ValidationFailed("Rasm bo'sh yoki yuborilmadi", detail={"field": "image"})

        case_id = await asyncio.to_thread(
            self.repo.create, operator_id=operator_id, operator_notes=notes
        )
        img_att = self.file_store.save(case_id, "image", image_filename or "scan.png", image_bytes)
        await asyncio.to_thread(self.repo.add_attachment, case_id, img_att)

        has_audio = False
        if audio_bytes:
            has_audio = True
            au_att = self.file_store.save(
                case_id, "audio", audio_filename or "audio.wav", audio_bytes
            )
            await asyncio.to_thread(self.repo.add_attachment, case_id, au_att)

        await asyncio.to_thread(
            self.audit.log, case_id, operator_id or "operator",
            AuditAction.CASE_CREATED.value, {"has_audio": has_audio, "source": source},
        )
        await self.worker.enqueue(case_id)

        # Scanner case'i bo'lsa avval SCAN_INGESTED (fayl tizimga kirdi) push qilamiz.
        if scan_meta is not None:
            self.bus.publish(EventType.SCAN_INGESTED, case_id, {**scan_meta, "source": source})
        self.bus.publish(
            EventType.CASE_CREATED, case_id, {"has_audio": has_audio, "source": source}
        )
        log.info("Case %s ochildi (source=%s, audio=%s)", case_id, source, has_audio)
        return case_id

    def submit_threadsafe(self, **kwargs) -> "asyncio.Future":
        """Boshqa thread'dan (watchdog) submit'ni loop'ga ko'chiradi.

        concurrent.futures.Future qaytaradi — chaqiruvchi xohlasa `.result()` bilan
        case_id'ni deterministik kutishi mumkin. Xato bo'lsa log qilinadi.
        """
        if self._loop is None:
            raise RuntimeError("CaseIntake loop bog'lanmagan (bind_loop chaqirilmagan)")
        fut = asyncio.run_coroutine_threadsafe(self.submit(**kwargs), self._loop)
        fut.add_done_callback(self._on_done)
        return fut

    @staticmethod
    def _on_done(fut: "asyncio.Future") -> None:
        exc = fut.exception()
        if exc is not None:
            log.error("Avtomatik case yaratishda xato: %s", exc)
