"""Provayderlarni yig'ish — mock (default) yoki real (Dev 2/3/4).

use_mocks=True bo'lsa Backend Core mustaqil ishlaydi. Real provayderlar tayyor
bo'lganda use_mocks=False qilinadi; agar real modul hali stub/model yo'q bo'lsa,
mock'ga xavfsiz qaytadi (jamoa parallel ishlashi uchun).

Vision (Dev 2): RuleEngine HAR DOIM real (deterministik, modelga bog'liq emas).
Detector esa faqat ONNX model + labels mavjud bo'lsa real bo'ladi; aks holda
mock — domain-fine-tuned model kelguncha hech kim bloklanmaydi.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.core.orchestrator import GpuOrchestrator

log = logging.getLogger("customs.providers")


def _build_detector(settings: Settings):
    """ONNX model mavjud bo'lsa real OnnxYoloDetector, aks holda mock.

    Import/yuklash xatosi case'ni qulatmasin — har qanday muammoda mock'ga
    tushamiz va OGOHLANTIRAMIZ (Tamoyil 6: degradatsiya, fail emas).
    """
    from app.pipelines.mocks import MockDetector

    model_path = settings.yolo_model_path
    if not model_path or not Path(model_path).exists():
        log.warning(
            "Vision: ONNX model topilmadi (yolo_model_path=%s) -> MOCK detector. "
            "⚠️ Domain-fine-tuned X-ray model kerak (ml/ track).",
            model_path,
        )
        return MockDetector()

    try:
        from app.pipelines.detection import OnnxYoloDetector

        detector = OnnxYoloDetector(
            model_path=str(model_path),
            labels_path=str(settings.yolo_labels_path) if settings.yolo_labels_path else None,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
            imgsz=settings.yolo_imgsz,
        )
        log.info("Vision: REAL OnnxYoloDetector (%s)", model_path)
        return detector
    except Exception as exc:  # noqa: BLE001 - har qanday xatoda mock fallback
        log.warning("Vision: OnnxYoloDetector yuklab bo'lmadi (%s) -> MOCK", exc)
        return MockDetector()


def _build_transcriber(settings: Settings):
    """Real WhisperTranscriber (faster-whisper CPU/int8); xatoda mock fallback.

    Tamoyil 6: import/yuklash muammosi case'ni qulatmasin. Lokal model_path
    berilgan-u, lekin yo'q bo'lsa ham mock'ga tushamiz va ogohlantiramiz.
    """
    from app.pipelines.mocks import MockTranscriber

    if settings.stt_model_path and not Path(settings.stt_model_path).exists():
        log.warning(
            "STT: lokal model topilmadi (stt_model_path=%s) -> MOCK transcriber.",
            settings.stt_model_path,
        )
        return MockTranscriber()

    try:
        from app.pipelines.speech import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_size=settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            model_path=str(settings.stt_model_path) if settings.stt_model_path else None,
            default_language=settings.stt_language,
            beam_size=settings.stt_beam_size,
            vad_filter=settings.stt_vad,
            cpu_threads=settings.stt_cpu_threads,
            download_root=str(settings.stt_download_root) if settings.stt_download_root else None,
        )
        log.info("STT: REAL WhisperTranscriber (%s, %s)", settings.stt_model_size, settings.stt_compute_type)
        return transcriber
    except Exception as exc:  # noqa: BLE001 - har qanday xatoda mock fallback
        log.warning("STT: WhisperTranscriber yuklab bo'lmadi (%s) -> MOCK", exc)
        return MockTranscriber()


def _build_explainer(settings: Settings, orchestrator: GpuOrchestrator):
    """Real OllamaExplainer (Qwen3 4B); llm_enabled=False yoki import xatosida mock.

    Tamoyil 6: konstruktor tarmoqqa chiqMAYDI (lazy) — Ollama o'chiq bo'lsa ham
    case'lar kelaveradi, qattiq xato faqat chaqiruvda yuzaga keladi va worker uni
    degradatsiya qiladi. Shuning uchun bu yerda Ollama'ga health-check QILMAYMIZ.
    """
    from app.pipelines.mocks import MockExplainer

    if not settings.llm_enabled:
        log.warning("LLM: llm_enabled=False -> MOCK explainer.")
        return MockExplainer()

    try:
        from app.pipelines.synthesis import OllamaExplainer

        explainer = OllamaExplainer(
            base_url=orchestrator.base_url,
            model=settings.llm_model,
            keep_alive=settings.llm_keep_alive,
            request_timeout_s=settings.llm_timeout_s,
            num_ctx=settings.llm_num_ctx,
            num_predict=settings.llm_num_predict,
            temperature=settings.llm_temperature,
        )
        log.info("LLM: REAL OllamaExplainer (%s @ %s)", settings.llm_model, orchestrator.base_url)
        return explainer
    except Exception as exc:  # noqa: BLE001 - har qanday xatoda mock fallback
        log.warning("LLM: OllamaExplainer yuklab bo'lmadi (%s) -> MOCK", exc)
        return MockExplainer()


def _build_synthesizer(settings: Settings):
    """Real TTS (Dev 3) — speech.py'da bo'lsa; aks holda mock (offline)."""
    from app.pipelines.mocks import MockSynthesizer

    try:
        from app.pipelines.speech import WhisperTtsSynthesizer  # Dev 3 yetkazadi

        synth = WhisperTtsSynthesizer()
        log.info("TTS: REAL synthesizer (%s)", type(synth).__name__)
        return synth
    except Exception as exc:  # noqa: BLE001 - yo'q/xato bo'lsa mock
        log.warning("TTS: real synthesizer yo'q (%s) -> MOCK", exc)
        return MockSynthesizer()


def build_providers(settings: Settings, orchestrator: GpuOrchestrator) -> dict:
    # RuleEngine deterministik va modelga bog'liq emas — har doim real ishlatamiz.
    from app.pipelines.scoring import RuleEngine

    if settings.use_mocks:
        from app.pipelines.mocks import (
            MockDetector,
            MockExplainer,
            MockSynthesizer,
            MockTranscriber,
        )

        log.info("Provayderlar: MOCK rejimi (RuleEngine real)")
        return {
            "detector": MockDetector(),
            "risk_engine": RuleEngine(),
            "transcriber": MockTranscriber(),
            "explainer": MockExplainer(),
            "synthesizer": MockSynthesizer(),
        }

    # Real rejim: detector/STT model bo'lsa real, LLM llm_enabled bo'lsa real.
    providers: dict = {
        "detector": _build_detector(settings),
        "risk_engine": RuleEngine(),
        "transcriber": _build_transcriber(settings),
        "explainer": _build_explainer(settings, orchestrator),
        "synthesizer": _build_synthesizer(settings),
    }
    log.info(
        "Provayderlar: REAL rejim (detector=%s, transcriber=%s, explainer=%s)",
        type(providers["detector"]).__name__,
        type(providers["transcriber"]).__name__,
        type(providers["explainer"]).__name__,
    )
    return providers
