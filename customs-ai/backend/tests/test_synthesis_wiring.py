"""LLM sintez provider wiring + kontrakt shakli (modelsiz/tarmoqsiz, Dev 4)."""
from __future__ import annotations


def test_explainer_uses_mock_when_llm_disabled(tmp_path):
    from app.core.config import Settings
    from app.core.orchestrator import GpuOrchestrator
    from app.core.providers import build_providers
    from app.pipelines.mocks import MockExplainer

    s = Settings(use_mocks=False, llm_enabled=False, data_dir=tmp_path)
    providers = build_providers(s, GpuOrchestrator(s))
    assert isinstance(providers["explainer"], MockExplainer)


def test_explainer_is_real_when_enabled(tmp_path):
    from app.core.config import Settings
    from app.core.orchestrator import GpuOrchestrator
    from app.core.providers import build_providers
    from app.pipelines.synthesis import OllamaExplainer

    s = Settings(use_mocks=False, llm_enabled=True, data_dir=tmp_path)
    providers = build_providers(s, GpuOrchestrator(s))
    assert isinstance(providers["explainer"], OllamaExplainer)


def test_construct_does_no_network(tmp_path):
    """Konstruktor lazy bo'lishi shart — Ollama o'chiq bo'lsa ham case'lar kelaveradi."""
    from app.pipelines.synthesis import OllamaExplainer

    # Mavjud bo'lmagan portga ko'rsatamiz; konstruktor TARMOQQA chiqmasligi kerak.
    ex = OllamaExplainer(base_url="http://127.0.0.1:1", model="qwen3:4b")
    assert ex.model == "qwen3:4b"
    assert ex.base_url == "http://127.0.0.1:1"


def test_empty_result_contract_shape():
    from app.pipelines.synthesis import GENERATED_BY, _empty_result

    r = _empty_result()
    assert r == {"text": "", "generated_by": GENERATED_BY, "available": False}
    assert GENERATED_BY == "qwen3-4b"
