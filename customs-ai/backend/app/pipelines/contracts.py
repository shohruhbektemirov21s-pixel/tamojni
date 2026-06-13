"""Integratsiya kontraktlari (rasmiy interfeyslar).

Bu interfeyslarni Backend Core (men) CHAQIRADI; Dev 2/3/4 IMPLEMENTATSIYA qiladi.
Mock'lar bilan parallel ishlash mumkin. Implementatsiya bu Protocol'larga
struktural mos kelishi shart (runtime_checkable — isinstance bilan tekshiriladi).

Qaytariladigan dict'lar §7.1 (Case Result) maydonlariga aynan mos kelishi kerak.
`available` / `computed_by` / `generated_by` maydonlari MAJBURIY — operator
qaysi qism AI'dan, qaysi qism deterministik ekanini ko'rishi shart.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    """Dev 2 — Vision (YOLO/ONNX). ADR-002: GPU'da emas, CPU/ONNX'da."""

    def detect(self, image_path: str) -> list[dict]:
        """-> [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]"""
        ...


@runtime_checkable
class RiskEngine(Protocol):
    """Dev 2 — DETERMINISTIK risk dvigateli (Tamoyil 2). LLM EMAS."""

    def compute_risk(self, detections: list[dict], config: dict) -> dict:
        """-> {"level": "LOW|MEDIUM|HIGH", "score": float,
               "computed_by": str, "factors": [ ... ]}"""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Dev 3 — STT (faster-whisper, CPU)."""

    def transcribe(self, audio_path: str, language: str | None) -> dict:
        """-> {"text": str|None, "language": str|None,
               "confidence": float|None, "available": bool}"""
        ...


@runtime_checkable
class StreamingTranscriber(Protocol):
    """Dev 3 — JONLI STT. Backend partiallarni `stt_partial` event qilib push qiladi.

    Sinxron generator (blocking thread'da aylanadi); har bir partial darhol
    relay bo'ladi (batch EMAS). Oxirgi element `is_final=True` bo'lishi SHART —
    u `stt_done` event'iga aylanadi. Generator tugashi = transkripsiya yakuni.
    """

    def transcribe_stream(
        self, audio_path: str, language: str | None
    ) -> Iterator[dict]:
        """yield -> {"text": str, "is_final": bool, "language": str|None,
                     "confidence": float|None}
        Oxirgi yield (is_final=True) to'liq natijani beradi."""
        ...


@runtime_checkable
class Explainer(Protocol):
    """Dev 4 — LLM (Qwen3) sintez. FAQAT tushuntirish matni yozadi;
    risk score'ni O'ZGARTIRMAYDI (Tamoyil 1/2)."""

    def generate_explanation(
        self,
        detections: list[dict],
        transcript: dict,
        operator_notes: str | None,
        risk: dict,
    ) -> dict:
        """-> {"text": str|None, "generated_by": str|None, "available": bool}"""
        ...


@runtime_checkable
class StreamingExplainer(Protocol):
    """Dev 4 — JONLI LLM sintez. Backend tokenlarni `explanation_token` qilib
    push qiladi (token-streaming, batch EMAS). On-demand (operator case ochganda)
    yoki HIGH-flagged case uchun — har skanga EMAS (yagona GPU bottleneck).

    Sinxron token generator (blocking thread'da, GPU lock ostida aylanadi).
    Generator tugashi = to'liq matn tayyor (`explanation_done`)."""

    def generate_explanation_stream(
        self,
        detections: list[dict],
        transcript: dict,
        operator_notes: str | None,
        risk: dict,
    ) -> Iterator[str]:
        """yield -> token (str). Birlashtirilgani to'liq tushuntirish matni."""
        ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Dev 3 — TTS (matn -> ovoz). POST /tts va `tts_ready` event uchun."""

    def synthesize_speech(self, text: str, language: str | None) -> dict:
        """-> {"audio_bytes": bytes|None, "format": str, "sample_rate": int|None,
               "available": bool}"""
        ...
