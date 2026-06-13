"""Mock provayderlar + reference risk engine.

Backend Core'ni Dev 2/3/4 implementatsiyasidan MUSTAQIL ishlatish uchun.
Degradatsiya testlari uchun `fail`/`hang` bayroqli variantlar ham bor.

`reference_compute_risk` — DETERMINISTIK noisy-OR (arxitektura §9). Bu Dev 2 ning
haqiqiy logikasi uchun reference; u config/risk_rules.yaml dan og'irlik oladi.
"""
from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# Reference deterministik risk engine (Tamoyil 2)
# ---------------------------------------------------------------------------
def reference_compute_risk(detections: list[dict], config: dict) -> dict:
    config = config or {}
    class_weights: dict = config.get("class_weights", {})
    thresholds: dict = config.get("thresholds", {"HIGH": 0.7, "MEDIUM": 0.4})
    default_weight: float = config.get("default_weight", 0.2)

    factors: list[dict] = []
    for d in detections:
        weight = class_weights.get(d["class"], default_weight)
        contribution = round(weight * float(d["confidence"]), 3)
        factors.append(
            {
                "rule": "prohibited_class",
                "class": d["class"],
                "confidence": float(d["confidence"]),
                "weight": weight,
                "contribution": contribution,
            }
        )

    # noisy-OR: bir nechta signal kuchayadi, lekin hech qachon 1.0 dan oshmaydi.
    prod = 1.0
    for f in factors:
        prod *= 1 - f["contribution"]
    score = round(1 - prod, 3)

    if score >= thresholds.get("HIGH", 0.7):
        level = "HIGH"
    elif score >= thresholds.get("MEDIUM", 0.4):
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": score,
        "computed_by": "rule_engine_ref_v1",
        "factors": factors,
    }


class ReferenceRiskEngine:
    """RiskEngine kontraktiga mos reference implementatsiya (Dev 2 almashtiradi)."""

    def compute_risk(self, detections: list[dict], config: dict) -> dict:
        return reference_compute_risk(detections, config)


# ---------------------------------------------------------------------------
# Mock model provayderlari
# ---------------------------------------------------------------------------
class MockDetector:
    def __init__(self, result: list[dict] | None = None, fail: bool = False) -> None:
        self._result = (
            result
            if result is not None
            else [{"class": "pichoq", "confidence": 0.82, "bbox": [10, 10, 120, 140]}]
        )
        self._fail = fail

    def detect(self, image_path: str) -> list[dict]:
        if self._fail:
            raise RuntimeError("mock detector failure")
        return [dict(d) for d in self._result]


class MockTranscriber:
    def __init__(
        self,
        text: str = "namuna transkript matni",
        fail: bool = False,
        hang_s: float | None = None,
    ) -> None:
        self._text = text
        self._fail = fail
        self._hang_s = hang_s

    def transcribe(self, audio_path: str, language: str | None) -> dict:
        if self._fail:
            raise RuntimeError("mock stt failure")
        if self._hang_s:
            time.sleep(self._hang_s)  # timeout degradatsiyasini sinash uchun
        return {
            "text": self._text,
            "language": language or "ru",
            "confidence": 0.9,
            "available": True,
        }


class MockExplainer:
    def __init__(self, fail: bool = False, hang_s: float | None = None) -> None:
        self._fail = fail
        self._hang_s = hang_s

    def generate_explanation(
        self,
        detections: list[dict],
        transcript: dict,
        operator_notes: str | None,
        risk: dict,
    ) -> dict:
        if self._fail:
            raise RuntimeError("mock llm failure")
        if self._hang_s:
            time.sleep(self._hang_s)
        n = len(detections)
        return {
            "text": (
                f"Aniqlangan {n} ta obyekt asosida risk darajasi {risk['level']} "
                f"({risk['score']}). Operator ko'rib chiqishi tavsiya etiladi."
            ),
            "generated_by": "mock-qwen3",
            "available": True,
        }


# ---------------------------------------------------------------------------
# Streaming mock'lar (4-bosqich relay'ini sinash uchun)
# ---------------------------------------------------------------------------
class MockStreamingTranscriber:
    """StreamingTranscriber kontrakti: partiallarni so'zma-so'z yield qiladi."""

    model_version = "mock-whisper-stream"

    def __init__(self, text: str = "namuna jonli transkript", fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    def transcribe_stream(self, audio_path: str, language: str | None):
        if self._fail:
            raise RuntimeError("mock stt stream failure")
        words = self._text.split()
        acc = ""
        for w in words:
            acc = (acc + " " + w).strip()
            yield {"text": acc, "is_final": False, "language": language or "ru", "confidence": None}
        yield {
            "text": self._text, "is_final": True,
            "language": language or "ru", "confidence": 0.9,
        }


class MockStreamingExplainer:
    """StreamingExplainer kontrakti: tushuntirishni token-token yield qiladi."""

    model_version = "mock-qwen3-stream"

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def generate_explanation_stream(
        self,
        detections: list[dict],
        transcript: dict,
        operator_notes: str | None,
        risk: dict,
    ):
        if self._fail:
            raise RuntimeError("mock llm stream failure")
        text = (
            f"Aniqlangan {len(detections)} ta obyekt asosida risk {risk['level']} "
            f"({risk['score']}). Operator ko'rib chiqsin."
        )
        for tok in text.split(" "):
            yield tok + " "


class MockSynthesizer:
    """SpeechSynthesizer kontrakti: matnni minimal WAV baytlariga aylantiradi (offline)."""

    def __init__(self, fail: bool = False, sample_rate: int = 16000) -> None:
        self._fail = fail
        self._sr = sample_rate

    def synthesize_speech(self, text: str, language: str | None) -> dict:
        if self._fail:
            raise RuntimeError("mock tts failure")
        # Haqiqiy audio emas — deterministik, tarmoqsiz bayt (uzunligi matnga bog'liq).
        body = ("WAVE_MOCK::" + (language or "uz") + "::" + text).encode("utf-8")
        return {
            "audio_bytes": body,
            "format": "wav",
            "sample_rate": self._sr,
            "available": True,
        }
