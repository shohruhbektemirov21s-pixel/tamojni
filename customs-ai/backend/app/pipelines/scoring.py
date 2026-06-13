"""Deterministik risk engine — EGASI: Dev 2 (Tamoyil 2, ADR-004).

Score 100% DETERMINISTIK: bir xil (detections, config) -> bir xil natija, har doim.
LLM bu yerga ARALASHMAYDI. Har score auditga tayyor `factors[]` bilan keladi
(qaysi qoida, qaysi class, confidence, og'irlik, hissa).

Agregatsiya: noisy-OR (§9). Bir nechta mustaqil signal riskni kuchaytiradi,
lekin natija hech qachon [0, 1] dan chiqmaydi.

Konfiguratsiya `config/risk_rules.yaml` dan keladi (operator sozlay oladi):
    default_weight: 0.2
    class_weights: {qurol: 1.0, pichoq: 0.8, ...}
    thresholds: {HIGH: 0.7, MEDIUM: 0.4}
    min_confidence: 0.25                 # global "shovqin" pollik (floor)
    per_class_min_confidence:            # recall foydasiga past pollik
      qurol: 0.10                        # qurolni past confidence'da ham hisobga ol
    prohibited_classes: [qurol, ...]     # (ixtiyoriy) faqat audit teglash uchun

Recall-favoring (qabul mezoni: qurol RECALL >= 0.95): xavfli class'lar uchun
`per_class_min_confidence` ni PAST qo'yamiz — false-negative bojxonada false-
positive'dan qimmatroq. Bu YOLO conf threshold'idan mustaqil ikkinchi himoya.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("customs.risk")

COMPUTED_BY = "rule_engine_v1"

# Hisob aniqligi (audit reproduktsiyasi uchun qat'iy). Float order-bog'liqligini
# yo'qotish maqsadida detections oldindan deterministik tartiblanadi.
_ROUND = 4

# Konfiguratsiya umuman bo'sh kelganda xavfsiz default (fail-safe: hech narsani
# yashirib yubormaslik uchun og'irliklar nolga emas, mo''tadil qiymatga tushadi).
_FALLBACK: dict[str, Any] = {
    "default_weight": 0.2,
    "class_weights": {},
    "thresholds": {"HIGH": 0.7, "MEDIUM": 0.4},
    "min_confidence": 0.0,
    "per_class_min_confidence": {},
    "prohibited_classes": [],
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _coerce_detection(raw: Any) -> dict | None:
    """Bitta detection'ni xavfsiz normallashtiradi. Buzuq bo'lsa -> None.

    Risk engine pipeline'ni qulatmaydi (Tamoyil 6): yaroqsiz element o'tkazib
    yuboriladi va log qilinadi, qolganlari hisoblanadi.
    """
    if not isinstance(raw, dict):
        log.warning("risk: dict bo'lmagan detection o'tkazib yuborildi: %r", raw)
        return None
    cls = raw.get("class")
    conf = raw.get("confidence")
    if not isinstance(cls, str) or not cls:
        log.warning("risk: 'class' yaroqsiz, o'tkazildi: %r", raw)
        return None
    try:
        conf_f = _clamp(float(conf))
    except (TypeError, ValueError):
        log.warning("risk: 'confidence' yaroqsiz, o'tkazildi: %r", raw)
        return None
    bbox = raw.get("bbox") or [0, 0, 0, 0]
    try:
        bbox = [float(v) for v in bbox][:4]
        if len(bbox) < 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
    except (TypeError, ValueError):
        bbox = [0.0, 0.0, 0.0, 0.0]
    return {"class": cls, "confidence": conf_f, "bbox": bbox}


def _floor_for(cls: str, cfg: dict) -> float:
    per_class: dict = cfg.get("per_class_min_confidence") or {}
    if cls in per_class:
        return _clamp(float(per_class[cls]))
    return _clamp(float(cfg.get("min_confidence", 0.0)))


def decide_level(score: float, thresholds: dict) -> str:
    """Deterministik daraja tanlash (yagona haqiqat manbasi)."""
    if score >= float(thresholds.get("HIGH", 0.7)):
        return "HIGH"
    if score >= float(thresholds.get("MEDIUM", 0.4)):
        return "MEDIUM"
    return "LOW"


def _merge_config(config: dict | None) -> dict:
    cfg = dict(_FALLBACK)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    return cfg


def compute_risk(detections: list[dict], config: dict | None) -> dict:
    """DETERMINISTIK risk hisoblash (RiskEngine kontrakti).

    Qaytaradi:
        {"level": "LOW|MEDIUM|HIGH", "score": float,
         "computed_by": "rule_engine_v1",
         "factors": [{"rule","class","confidence","weight","contribution"}, ...]}

    factors[] AUDIT uchun: skor'ga hissa qo'shgan har qoida ko'rinadi, shuningdek
    pollikdan past tushib HISOBGA OLINMAGAN detection'lar ham
    rule="below_confidence_floor", contribution=0.0 bilan yoziladi (operator
    "nega ko'rsatilmadi?" degan savolga javob topa olsin).
    """
    cfg = _merge_config(config)
    class_weights: dict = cfg.get("class_weights") or {}
    default_weight = float(cfg.get("default_weight", 0.2))
    thresholds: dict = cfg.get("thresholds") or {"HIGH": 0.7, "MEDIUM": 0.4}
    prohibited = set(cfg.get("prohibited_classes") or [])

    # 1) Normallashtirish + deterministik tartiblash (float order'ni barqaror qil).
    norm = [d for d in (_coerce_detection(r) for r in (detections or [])) if d]
    norm.sort(key=lambda d: (d["class"], -d["confidence"], d["bbox"]))

    factors: list[dict] = []
    contributions: list[float] = []
    for d in norm:
        cls, conf = d["class"], d["confidence"]
        floor = _floor_for(cls, cfg)
        weight = float(class_weights.get(cls, default_weight))

        if conf < floor:
            # Hisobga olinmadi, lekin AUDIT uchun ko'rinadi (false-negative izi).
            factors.append(
                {
                    "rule": "below_confidence_floor",
                    "class": cls,
                    "confidence": round(conf, _ROUND),
                    "weight": round(weight, _ROUND),
                    "contribution": 0.0,
                    "floor": round(floor, _ROUND),
                }
            )
            continue

        contribution = _clamp(weight * conf, 0.0, 0.999)
        rule = "prohibited_class" if (not prohibited or cls in prohibited) else "watch_class"
        factors.append(
            {
                "rule": rule,
                "class": cls,
                "confidence": round(conf, _ROUND),
                "weight": round(weight, _ROUND),
                "contribution": round(contribution, _ROUND),
            }
        )
        contributions.append(contribution)

    # 2) noisy-OR agregatsiya: 1 - prod(1 - c_i). Bo'sh -> 0.0.
    prod = 1.0
    for c in contributions:
        prod *= 1.0 - c
    score = round(_clamp(1.0 - prod), _ROUND)

    level = decide_level(score, thresholds)
    return {
        "level": level,
        "score": score,
        "computed_by": COMPUTED_BY,
        "factors": factors,
    }


class RuleEngine:
    """RiskEngine kontraktining production implementatsiyasi (Tamoyil 2).

    Holatsiz (stateless) — har chaqiruv faqat (detections, config) ga bog'liq,
    shu sabab determinizm kafolatlanadi. config worker tomonidan
    `load_risk_config(settings)` orqali bir marta yuklanib uzatiladi.
    """

    def compute_risk(self, detections: list[dict], config: dict) -> dict:
        return compute_risk(detections, config)
