"""Production deterministik risk engine (scoring.RuleEngine, Tamoyil 2 / ADR-004)."""
from __future__ import annotations

from app.pipelines.scoring import RuleEngine, compute_risk, decide_level

CFG = {
    "default_weight": 0.2,
    "class_weights": {"qurol": 1.0, "pichoq": 0.8, "valyuta_toplami": 0.7},
    "thresholds": {"HIGH": 0.7, "MEDIUM": 0.4},
    "min_confidence": 0.25,
    "per_class_min_confidence": {"qurol": 0.10},
    "prohibited_classes": ["qurol", "pichoq", "valyuta_toplami"],
}


def _det(cls, conf, bbox=(0, 0, 1, 1)):
    return {"class": cls, "confidence": conf, "bbox": list(bbox)}


# --- determinizm (qabul mezoni: bir xil input -> bir xil score) ---
def test_deterministic_identical_output():
    dets = [_det("pichoq", 0.82), _det("qurol", 0.6)]
    assert compute_risk(dets, CFG) == compute_risk(dets, CFG)


def test_order_invariant():
    a = [_det("qurol", 0.9), _det("pichoq", 0.5)]
    b = [_det("pichoq", 0.5), _det("qurol", 0.9)]
    assert compute_risk(a, CFG)["score"] == compute_risk(b, CFG)["score"]


def test_computed_by_is_v1():
    r = compute_risk([_det("qurol", 0.9)], CFG)
    assert r["computed_by"] == "rule_engine_v1"


# --- "topilmadi" -> LOW ---
def test_empty_is_low():
    r = compute_risk([], CFG)
    assert r["level"] == "LOW" and r["score"] == 0.0 and r["factors"] == []


# --- noisy-OR chegaralari ---
def test_noisy_or_bounded():
    dets = [_det("qurol", 0.99) for _ in range(8)]
    r = compute_risk(dets, CFG)
    assert 0.0 <= r["score"] <= 1.0


def test_high_weight_triggers_high():
    r = compute_risk([_det("qurol", 0.9)], CFG)
    assert r["level"] == "HIGH"


# --- recall-favoring per-class floor ---
def test_weapon_below_global_floor_still_counts():
    # conf 0.15: global floor 0.25 dan past, lekin qurol floor 0.10 dan baland.
    r = compute_risk([_det("qurol", 0.15)], CFG)
    weapon_factor = next(f for f in r["factors"] if f["class"] == "qurol")
    assert weapon_factor["rule"] == "prohibited_class"
    assert weapon_factor["contribution"] > 0


def test_nonweapon_below_floor_ignored_but_audited():
    # pichoq conf 0.20 < global floor 0.25 -> hisobga olinmaydi, audit ko'rinadi.
    r = compute_risk([_det("pichoq", 0.20)], CFG)
    f = r["factors"][0]
    assert f["rule"] == "below_confidence_floor"
    assert f["contribution"] == 0.0
    assert "floor" in f
    assert r["score"] == 0.0


# --- audit factor maydonlari ---
def test_factor_audit_fields():
    r = compute_risk([_det("qurol", 0.9)], CFG)
    f = r["factors"][0]
    for key in ("rule", "class", "confidence", "weight", "contribution"):
        assert key in f


def test_watch_vs_prohibited_tagging():
    cfg = dict(CFG)
    cfg["prohibited_classes"] = ["qurol"]  # pichoq endi watch
    cfg["per_class_min_confidence"] = {"pichoq": 0.0}
    r = compute_risk([_det("pichoq", 0.9)], cfg)
    assert r["factors"][0]["rule"] == "watch_class"


# --- mustahkamlik (pipeline qulamaydi) ---
def test_malformed_detections_skipped():
    dets = [
        _det("qurol", 0.9),
        {"class": "", "confidence": 0.9},          # bo'sh class
        {"class": "pichoq", "confidence": "bad"},  # yaroqsiz conf
        "not a dict",                                # umuman dict emas
    ]
    r = compute_risk(dets, CFG)
    # faqat qurol hisoblanadi
    counted = [f for f in r["factors"] if f["rule"] != "below_confidence_floor"]
    assert [f["class"] for f in counted] == ["qurol"]


def test_confidence_clamped():
    r = compute_risk([_det("qurol", 1.5)], CFG)  # >1 -> 1.0 ga qisiladi
    assert r["factors"][0]["confidence"] == 1.0


def test_unknown_class_uses_default_weight():
    r = compute_risk([_det("notebook", 0.9)], {**CFG, "min_confidence": 0.0})
    assert r["factors"][0]["weight"] == CFG["default_weight"]


# --- daraja chegaralari ---
def test_decide_level_boundaries():
    th = {"HIGH": 0.7, "MEDIUM": 0.4}
    assert decide_level(0.7, th) == "HIGH"
    assert decide_level(0.69, th) == "MEDIUM"
    assert decide_level(0.4, th) == "MEDIUM"
    assert decide_level(0.39, th) == "LOW"


def test_none_config_safe():
    r = compute_risk([_det("qurol", 0.9)], None)
    assert r["level"] in {"LOW", "MEDIUM", "HIGH"}


def test_class_via_engine():
    eng = RuleEngine()
    assert eng.compute_risk([_det("qurol", 0.9)], CFG)["computed_by"] == "rule_engine_v1"
