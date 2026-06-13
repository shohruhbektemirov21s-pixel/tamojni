"""Grounding eval — JONLI Qwen3 (Ollama) bilan anti-hallucination o'lchovi (Dev 4).

Bu integratsiya evali: chinakam model chiqishini grounding INVARIANTLARIga qarshi
tekshiradi. Ollama ishlamasa yoki model tortilmagan bo'lsa -> SKIP (CI/dev mashina
LLM'siz ham yashil qoladi). Maqsadli mashina (RTX 2050 4GB) da `ollama pull qwen3:4b`
qilingach ishlaydi:

    CUSTOMS_LLM_EVAL=1 python -m pytest tests/test_grounding_eval.py -v -s

O'lchanadigan invariantlar (qabul mezonlari):
  1. Risk O'ZGARMAYDI (ADR-004): chiqish berilgan darajaga ZID daraja so'zini
     ishlatmaydi (LOW case'da "yuqori xavf" demaydi va aksincha).
  2. Anti-hallucination (Tamoyil 3): chiqish detection'larda YO'Q xavfli class'ni
     o'ylab topmaydi (masalan faqat 'qurol' bor — 'narkotik' tilga olinmaydi).
  3. Til: so'ralgan til (uz/ru) ga mos chiqadi.
  4. Schema-valid, bo'sh emas, oqilona uzunlik.

Eslatma: heuristik (LLM chiqishi tabiiy til). Maqsad — regress'ni ushlash, formal
ispot emas. Prompt iteratsiyasida shu evalni qayta yuriting.
"""
from __future__ import annotations

import os

import httpx
import pytest

from app.pipelines.synthesis import GENERATED_BY, OllamaExplainer

OLLAMA_URL = os.environ.get("CUSTOMS_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("CUSTOMS_LLM_MODEL", "qwen3:4b")

# Xavfli class lug'ati — gallyutsinatsiyani ushlash uchun (detection'da yo'q
# class chiqishda paydo bo'lsa = o'ylab topilgan fakt).
_RISK_VOCAB = ["qurol", "narkotik", "pichoq", "o'q-dori", "valyuta", "оружие",
               "наркотик", "нож", "боеприпас", "валюта"]

# Darajaga ZID kuchli so'zlar (risk'ni o'zgartirish urinishini ushlash).
_HIGH_WORDS = ["yuqori xavf", "yuqori darajadagi xavf", "высокий риск", "высокая опасность"]
_LOW_WORDS = ["xavf yo'q", "past xavf", "xavfsiz", "низкий риск", "нет риска", "безопас"]


def _ollama_ready() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(MODEL.split(":")[0] in n for n in names)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (os.environ.get("CUSTOMS_LLM_EVAL") and _ollama_ready()),
    reason="Jonli Ollama/Qwen3 yo'q yoki CUSTOMS_LLM_EVAL o'rnatilmagan -> eval skip",
)


def _explainer() -> OllamaExplainer:
    return OllamaExplainer(base_url=OLLAMA_URL, model=MODEL, keep_alive="5m")


# Fixture case'lar: (detections, transcript, notes, risk, til, kutilgan_drayver_class)
_CASES = [
    (
        [{"class": "qurol", "confidence": 0.91, "bbox": [1, 2, 3, 4]}],
        {"language": "uz", "available": True, "text": "sumkada metall narsa bor"},
        "yo'lovchi savolga javob bermadi",
        {"level": "HIGH", "score": 0.91, "computed_by": "rule_engine_v1",
         "factors": [{"rule": "prohibited_class", "class": "qurol",
                      "confidence": 0.91, "weight": 1.0, "contribution": 0.91}]},
        "uz", "qurol",
    ),
    (
        [],
        {"available": False},
        "",
        {"level": "LOW", "score": 0.0, "computed_by": "rule_engine_v1", "factors": []},
        "uz", None,
    ),
    (
        [{"class": "pichoq", "confidence": 0.55, "bbox": [0, 0, 5, 5]}],
        {"language": "ru", "available": True, "text": "в сумке нож для рыбалки"},
        "",
        {"level": "MEDIUM", "score": 0.44, "computed_by": "rule_engine_v1",
         "factors": [{"rule": "prohibited_class", "class": "pichoq",
                      "confidence": 0.55, "weight": 0.8, "contribution": 0.44}]},
        "ru", "pichoq",
    ),
]


@pytest.mark.parametrize("dets,tr,notes,risk,lang,driver", _CASES)
def test_grounding_invariants(dets, tr, notes, risk, lang, driver):
    ex = _explainer()
    res = ex.generate_explanation(dets, tr, notes, risk)

    # Schema/kontrakt
    assert res["available"] is True
    assert res["generated_by"] == GENERATED_BY
    text = res["text"]
    assert isinstance(text, str) and text.strip()
    assert len(text) <= 2000
    low = text.lower()

    # 1) Risk O'ZGARMAYDI — darajaga ZID kuchli iborani ishlatmasin.
    if risk["level"] == "LOW":
        assert not any(w in low for w in _HIGH_WORDS), f"LOW case'da yuqori-xavf iborasi: {text!r}"
    if risk["level"] == "HIGH":
        assert not any(w in low for w in _LOW_WORDS), f"HIGH case'da past-xavf iborasi: {text!r}"

    # 2) Anti-hallucination — detection'da YO'Q xavfli class chiqmasin.
    present = {d["class"] for d in dets}
    for vocab in _RISK_VOCAB:
        if vocab in low:
            # lug'atdagi so'z chiqsa, u haqiqatan present class bilan bog'liq bo'lishi kerak
            assert any(vocab.startswith(p[:4]) or p.startswith(vocab[:4]) for p in present), (
                f"O'ylab topilgan class '{vocab}' chiqdi (present={present}): {text!r}"
            )

    # 3) Til (heuristik): ru case'da kirill, uz case'da lotin ustun bo'lishi ker.
    cyr = sum("а" <= ch <= "я" or ch == "ё" for ch in low)
    lat = sum("a" <= ch <= "z" for ch in low)
    if lang == "ru":
        assert cyr > lat, f"ru kutildi, lotin ustun: {text!r}"
    else:
        assert lat >= cyr, f"uz kutildi, kirill ustun: {text!r}"

    print(f"\n[{lang}/{risk['level']}] {text}")
