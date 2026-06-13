"""STT — EGASI: Dev 3.

Backend Core `Transcriber` kontraktini chaqiradi. faster-whisper (CTranslate2),
device="cpu", compute_type="int8" — GPU'ni vision/LLM uchun bo'shatadi (ADR-002)
va detection bilan PARALLEL ketadi.

Degradatsiya (Tamoyil 2 / §10): worker `transcribe` ni 30s timeout bilan o'raydi
(`core/worker.py::_run_stt`). Qattiq xatolikda (model yuklanmadi, audio buzuq,
inference crash) bu yerda EXCEPTION ko'tariladi — worker uni tutib MODEL_FAILED
audit yozadi va aniq xato dict'ini qaytaradi:
    {"text": "", "language": None, "confidence": 0.0, "available": False}
Shu tarzda butun case TO'XTAMAYDI. "Nutq topilmadi" (sukunat) — xato EMAS:
available=True, text="" qaytadi.

⚠️ HAQIQAT (o'zbek tili): Whisper rus tilida a'lo, lekin O'ZBEKDA sifat past
(lotin/kiril aralash, dialekt, bojxona terminlari). MVP base "medium" bilan
boshlanadi; production'da o'zbek uchun LoRA fine-tune kerak (stt/ track).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from app.pipelines.whisper_utils import (
    Chunk,
    Seg,
    aggregate_confidence,
    join_text,
    merge_overlapping_segments,
    normalize_language,
    plan_chunks,
)

log = logging.getLogger("customs.stt")


def _empty_result() -> dict:
    return {"text": "", "language": None, "confidence": 0.0, "available": False}


class WhisperTranscriber:
    """faster-whisper (CPU/int8) transkriber.

    Parametrlar:
        model_size: "medium" (MVP). Aniqlik/tezlik bo'yicha "small"/"large-v3" ham.
        model_path: lokal model katalogi (100% offline — oldindan yuklab olingan).
                    Berilsa model_size o'rniga shu ishlatiladi (HF download YO'Q).
        device/compute_type: ADR-002 -> "cpu" / "int8".
        default_language: None -> auto-detect. "ru"/"uz" majburlash mumkin.
        vad_filter: sukunatni kesish (Silero VAD, faster-whisper ichida).
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        *,
        model_path: str | None = None,
        default_language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        cpu_threads: int = 0,
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.default_language = default_language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self._model = None  # lazy

    # ---- model (lazy, offline) ----
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # lazy: core bularsiz ishlaydi
        except ImportError as exc:  # pragma: no cover - muhitga bog'liq
            raise RuntimeError(
                "faster-whisper o'rnatilmagan. offline-wheels orqali o'rnating "
                "(requirements-stt.txt)."
            ) from exc

        # model_path berilsa lokal katalog (offline); aks holda model_size nomi.
        # download_root oldindan yuklangan keshga ishora qiladi -> tarmoq YO'Q.
        source = self.model_path or self.model_size
        if self.model_path and not Path(self.model_path).exists():
            raise FileNotFoundError(f"Whisper model katalogi topilmadi: {self.model_path}")

        log.info(
            "Whisper yuklanmoqda: source=%s device=%s compute=%s",
            source, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            source,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=self.download_root,
            local_files_only=bool(self.download_root or self.model_path),  # offline
        )
        return self._model

    # ---- ichki: bitta oraliqni transkripsiya qilish ----
    def _run(
        self, audio_path: str, language: str | None, clip: tuple[float, float] | None = None
    ) -> tuple[list[Seg], str | None]:
        model = self._ensure_model()
        kwargs: dict = {
            "language": language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        if clip is not None:
            kwargs["clip_timestamps"] = [clip[0], clip[1]]
        fw_segments, info = model.transcribe(audio_path, **kwargs)
        segs = [
            Seg(
                start=float(s.start),
                end=float(s.end),
                text=s.text,
                avg_logprob=float(getattr(s, "avg_logprob", -0.3)),
                no_speech_prob=float(getattr(s, "no_speech_prob", 0.0)),
            )
            for s in fw_segments  # generator — bu yerda materializatsiya bo'ladi
        ]
        detected = getattr(info, "language", None)
        return segs, detected

    # ---- KONTRAKT: fayl rejimi ----
    def transcribe(self, audio_path: str, language: str | None) -> dict:
        """-> {"text", "language": "ru|uz|...", "confidence": float, "available": bool}

        Qattiq xatoda EXCEPTION ko'taradi (worker degradatsiya qiladi). Sukunat
        bo'lsa available=True, text="" qaytadi (xato emas).
        """
        lang_req = language if language is not None else self.default_language
        segs, detected = self._run(audio_path, lang_req)

        text = join_text(segs)
        confidence = aggregate_confidence(segs)
        lang_code, supported = normalize_language(detected, lang_req)
        if not supported and lang_code is not None:
            log.warning("STT: qo'llab-quvvatlanmaydigan til aniqlandi: %s", lang_code)

        result = {
            "text": text,
            "language": lang_code,
            "confidence": confidence,
            "available": True,
        }
        log.info(
            "transcribe(%s): lang=%s conf=%.3f len=%d",
            audio_path, lang_code, confidence, len(text),
        )
        return result

    # ---- NEAR-REAL-TIME: VAD chunking (generator) ----
    def transcribe_realtime(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        total_s: float | None = None,
        chunk_s: float = 25.0,
        overlap_s: float = 2.0,
    ) -> Iterator[dict]:
        """Chunk-ma-chunk qisman natijalarni yield qiladi (near-real-time).

        To'liq streaming Whisper murakkab; chunked yondashuv (§ scope). Har chunk
        tugagach to'plangan transcript yangilanadi va yield bo'ladi — UI/operator
        oraliq natijani ko'ra oladi. Yakuniy yield = to'liq, merge qilingan matn.

        total_s berilmasa audio davomiyligidan o'qiladi (soundfile).
        """
        lang_req = language if language is not None else self.default_language
        if total_s is None:
            total_s = _probe_duration(audio_path)

        chunks: list[Chunk] = plan_chunks(total_s, chunk_s=chunk_s, overlap_s=overlap_s)
        if not chunks:
            yield {**_empty_result(), "available": True, "partial": False}
            return

        collected: list[Seg] = []
        detected_lang: str | None = None
        for ch in chunks:
            segs, detected = self._run(audio_path, lang_req, clip=(ch.start, ch.end))
            # chunk vaqtini global vaqtga siljitish (clip_timestamps lokal beradi)
            for s in segs:
                s.start += ch.start
                s.end += ch.start
            collected.extend(segs)
            if detected_lang is None:
                detected_lang = detected

            merged = merge_overlapping_segments(collected)
            lang_code, _ = normalize_language(detected_lang, lang_req)
            yield {
                "text": join_text(merged),
                "language": lang_code,
                "confidence": aggregate_confidence(merged),
                "available": True,
                "partial": ch.index < len(chunks) - 1,
                "progress": round((ch.end / total_s), 3) if total_s else 1.0,
            }


def _probe_duration(audio_path: str) -> float:
    """Audio davomiyligi (soniya). soundfile lazy — STT'siz ham core ishlaydi."""
    try:
        import soundfile as sf  # lazy

        info = sf.info(audio_path)
        return float(info.frames) / float(info.samplerate)
    except Exception as exc:  # noqa: BLE001
        log.warning("Audio davomiyligini o'qib bo'lmadi (%s) -> 0", exc)
        return 0.0
