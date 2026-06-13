"""POST /tts — matnni ovozga aylantiradi (Dev 3 synthesizer).

Audio baytlari to'g'ridan-to'g'ri HTTP javobida qaytadi (event'da EMAS — katta).
`tts_ready` eventi metadata bilan push qilinadi (audio'siz). Ixtiyoriy `case_id`
berilsa, TTS_DONE audit shu case ostida yoziladi (AuditLog case_id majburiy).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.core.enums import AuditAction
from app.core.errors import NotFound, ServiceUnavailable, ValidationFailed
from app.core.events import EventType
from app.schemas import TtsIn

router = APIRouter(tags=["tts"])

_MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}


@router.post("/tts")
async def synthesize(request: Request, body: TtsIn) -> Response:
    st = request.app.state
    text = (body.text or "").strip()
    if not text:
        raise ValidationFailed("Matn bo'sh", detail={"field": "text"})
    if len(text) > st.settings.tts_max_chars:
        raise ValidationFailed(
            "Matn juda uzun", detail={"max_chars": st.settings.tts_max_chars, "got": len(text)}
        )
    language = body.language or st.settings.tts_default_language

    # case_id berilgan bo'lsa — mavjudligini tekshiramiz (audit FK uchun)
    if body.case_id is not None:
        if await asyncio.to_thread(st.repo.get_status, body.case_id) is None:
            raise NotFound("Case topilmadi", detail={"case_id": body.case_id})

    synth = st.providers["synthesizer"]
    try:
        res = await asyncio.to_thread(synth.synthesize_speech, text, language)
    except Exception as exc:  # noqa: BLE001 - degradatsiya (Tamoyil 6)
        raise ServiceUnavailable("TTS hozircha mavjud emas", detail={"error": str(exc)}) from exc

    if not res.get("available") or not res.get("audio_bytes"):
        raise ServiceUnavailable("TTS audio qaytarmadi")

    audio: bytes = res["audio_bytes"]
    fmt = res.get("format", "wav")
    meta = {"format": fmt, "sample_rate": res.get("sample_rate"),
            "bytes": len(audio), "language": language}

    if body.case_id is not None:
        await asyncio.to_thread(
            st.audit.log, body.case_id, body.operator_id or "operator",
            AuditAction.TTS_DONE.value, meta,
        )
    st.event_bus.publish(EventType.TTS_READY, body.case_id, meta)

    return Response(content=audio, media_type=_MEDIA.get(fmt, "application/octet-stream"))
