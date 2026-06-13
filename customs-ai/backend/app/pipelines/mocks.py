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
