"""Reference deterministik risk engine (Tamoyil 2)."""
from __future__ import annotations

from app.pipelines.mocks import reference_compute_risk

CFG = {
    "default_weight": 0.2,
    "class_weights": {"qurol": 1.0, "pichoq": 0.8},
    "thresholds": {"HIGH": 0.7, "MEDIUM": 0.4},
}


def test_deterministic_same_input_same_output():
    dets = [{"class": "pichoq", "confidence": 0.82, "bbox": [0, 0, 1, 1]}]
    a = reference_compute_risk(dets, CFG)
    b = reference_compute_risk(dets, CFG)
    assert a == b


def test_empty_detections_is_low():
    r = reference_compute_risk([], CFG)
    assert r["level"] == "LOW"
    assert r["score"] == 0.0
    assert r["factors"] == []


def test_high_weight_class_triggers_high():
    r = reference_compute_risk(
        [{"class": "qurol", "confidence": 0.9, "bbox": [0, 0, 1, 1]}], CFG
    )
    assert r["level"] == "HIGH"
    assert r["computed_by"] == "rule_engine_ref_v1"


def test_noisy_or_never_exceeds_one():
    dets = [{"class": "qurol", "confidence": 0.99, "bbox": [0, 0, 1, 1]} for _ in range(5)]
    r = reference_compute_risk(dets, CFG)
    assert 0.0 <= r["score"] <= 1.0


def test_factors_contain_explainability_fields():
    r = reference_compute_risk(
        [{"class": "pichoq", "confidence": 0.5, "bbox": [0, 0, 1, 1]}], CFG
    )
    f = r["factors"][0]
    for key in ("rule", "class", "confidence", "weight", "contribution"):
        assert key in f
