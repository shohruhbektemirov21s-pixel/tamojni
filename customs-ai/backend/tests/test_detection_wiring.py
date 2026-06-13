"""Detector wiring: label yuklash, provider tartibi, mock fallback (modelsiz)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipelines.detection import OnnxYoloDetector, _select_providers


def test_provider_order_prefers_directml_then_cpu_never_cuda():
    avail = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    chosen = _select_providers(avail)
    assert chosen[0] == "DmlExecutionProvider"
    assert "CUDAExecutionProvider" not in chosen  # ADR-002
    assert chosen[-1] == "CPUExecutionProvider"


def test_provider_falls_back_to_cpu_only():
    assert _select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert _select_providers([]) == ["CPUExecutionProvider"]


def test_labels_loaded_from_explicit_list(tmp_path):
    det = OnnxYoloDetector(str(tmp_path / "m.onnx"), labels=["qurol", "pichoq"])
    assert det.labels == ["qurol", "pichoq"]


def test_labels_loaded_from_file(tmp_path):
    lp = tmp_path / "labels.txt"
    lp.write_text("# header\nqurol\no'q-dori\n\npichoq\n", encoding="utf-8")
    det = OnnxYoloDetector(str(tmp_path / "m.onnx"), labels_path=str(lp))
    assert det.labels == ["qurol", "o'q-dori", "pichoq"]


def test_labels_sibling_autoload(tmp_path):
    (tmp_path / "labels.txt").write_text("qurol\n", encoding="utf-8")
    det = OnnxYoloDetector(str(tmp_path / "model.onnx"))
    assert det.labels == ["qurol"]


def test_missing_labels_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxYoloDetector(str(tmp_path / "model.onnx"))


def test_registry_labels_match_taxonomy():
    """v0_baseline/labels.txt taxonomy.yaml class indekslari bilan mos."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    tax = yaml.safe_load((root / "config" / "taxonomy.yaml").read_text(encoding="utf-8"))
    expected = [tax["classes"][i] for i in sorted(tax["classes"])]
    labels_file = root / "model_registry" / "v0_baseline" / "labels.txt"
    got = [
        ln.strip()
        for ln in labels_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert got == expected


def test_provider_fallback_to_mock_when_no_model(tmp_path):
    """yolo_model_path yo'q -> build_providers MockDetector qaytaradi."""
    from app.core.config import Settings
    from app.core.orchestrator import GpuOrchestrator
    from app.core.providers import build_providers
    from app.pipelines.mocks import MockDetector
    from app.pipelines.scoring import RuleEngine

    s = Settings(use_mocks=False, yolo_model_path=tmp_path / "yoq.onnx", data_dir=tmp_path)
    providers = build_providers(s, GpuOrchestrator(s))
    assert isinstance(providers["detector"], MockDetector)
    assert isinstance(providers["risk_engine"], RuleEngine)  # RuleEngine doim real
