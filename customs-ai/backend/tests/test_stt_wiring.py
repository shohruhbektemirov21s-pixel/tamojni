"""STT provider wiring + degradatsiya kontrakti (modelsiz)."""
from __future__ import annotations

from pathlib import Path


def test_transcriber_falls_back_to_mock_when_model_missing(tmp_path):
    from app.core.config import Settings
    from app.core.orchestrator import GpuOrchestrator
    from app.core.providers import build_providers
    from app.pipelines.mocks import MockTranscriber

    s = Settings(
        use_mocks=False,
        stt_model_path=tmp_path / "yoq_model",  # mavjud emas
        data_dir=tmp_path,
    )
    providers = build_providers(s, GpuOrchestrator(s))
    assert isinstance(providers["transcriber"], MockTranscriber)


def test_whisper_transcriber_lazy_no_model_load_on_construct(tmp_path):
    """Konstruktor model yuklaMASLIGI kerak (lazy) — faster-whisper'siz ham."""
    from app.pipelines.speech import WhisperTranscriber

    t = WhisperTranscriber(model_size="medium")
    assert t._model is None
    assert t.device == "cpu" and t.compute_type == "int8"


def test_missing_local_model_dir_raises_on_use(tmp_path):
    from app.pipelines.speech import WhisperTranscriber

    t = WhisperTranscriber(model_path=str(tmp_path / "yoq"))
    # _ensure_model faster-whisper bo'lmasa RuntimeError, model yo'qligida
    # FileNotFoundError — ikkalasi ham degradatsiya uchun Exception (worker tutadi).
    import pytest

    with pytest.raises((FileNotFoundError, RuntimeError)):
        t._ensure_model()


def test_empty_result_contract_shape():
    from app.pipelines.speech import _empty_result

    r = _empty_result()
    assert r == {"text": "", "language": None, "confidence": 0.0, "available": False}
