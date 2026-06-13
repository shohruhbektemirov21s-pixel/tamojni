"""OllamaExplainer unit testlari — httpx MockTransport (jonli Ollama YO'Q).

Tekshiriladi (qabul mezonlari):
  * Happy path: schema-valid JSON -> {"text","generated_by","available":True}.
  * Grounding: risk score/level prompt'ga kiradi; LLM uni O'ZGARTIRMAYDI (ADR-004).
  * Til tanlash: transcript.language (ru/uz) -> chiqish tili.
  * Schema fail -> 1x re-prompt -> tuzalsa available=True.
  * Schema 2x fail -> RAISE (worker degradatsiya qiladi).
  * Infra xato (HTTP 5xx / connect) -> RAISE (worker daemon restart qiladi).
  * <think> va atrofdagi matn oqib chiqsa ham JSON ajratiladi.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.pipelines.synthesis import GENERATED_BY, OllamaExplainer

RISK_HIGH = {
    "level": "HIGH",
    "score": 0.91,
    "computed_by": "rule_engine_v1",
    "factors": [
        {"rule": "prohibited_class", "class": "qurol", "confidence": 0.91,
         "weight": 1.0, "contribution": 0.91},
        {"rule": "below_confidence_floor", "class": "pichoq", "confidence": 0.05,
         "weight": 0.8, "contribution": 0.0, "floor": 0.15},
    ],
}
DETECTIONS = [{"class": "qurol", "confidence": 0.91, "bbox": [1, 2, 3, 4]}]


def _ollama_response(text_payload: str) -> httpx.Response:
    """Ollama /api/chat javobini taqlid qiladi."""
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": text_payload}}
    )


def _explainer_with(handler) -> OllamaExplainer:
    return OllamaExplainer(
        base_url="http://test", model="qwen3:4b",
        transport=httpx.MockTransport(handler),
    )


# --------------------------------------------------------------------------- #
def test_happy_path_returns_contract_dict():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response(json.dumps({"text": "Yuqori ishonchli qurol aniqlandi."}))

    ex = _explainer_with(handler)
    res = ex.generate_explanation(
        DETECTIONS, {"text": "salom", "language": "uz", "available": True},
        "operator izohi", RISK_HIGH,
    )
    assert res == {
        "text": "Yuqori ishonchli qurol aniqlandi.",
        "generated_by": GENERATED_BY,
        "available": True,
    }
    # Ollama'ga to'g'ri parametrlar yuborilgan (grounded/determinizm).
    body = captured["body"]
    assert body["stream"] is False
    assert body["think"] is False
    assert body["format"]["required"] == ["text"]
    assert body["options"]["temperature"] == 0.0


def test_prompt_is_grounded_in_given_risk_not_changed():
    """ADR-004: berilgan score/level prompt'ga kiradi; LLM uni qayta baholamaydi."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response(json.dumps({"text": "ok"}))

    ex = _explainer_with(handler)
    ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)

    user_msg = captured["body"]["messages"][-1]["content"]
    system_msg = captured["body"]["messages"][0]["content"]
    assert "0.91" in user_msg          # berilgan score
    assert "qurol" in user_msg          # berilgan fakt
    assert "O'ZGARTIRMA" in user_msg or "изменять" in user_msg
    # below_floor fakt "ballga kirmagan" deb ko'rsatiladi (audit izi)
    assert "pichoq" in user_msg
    # tizim prompt'i darajani o'zgartirishni taqiqlaydi
    assert "O'ZGARTIRMA" in system_msg


def test_language_selection_ru():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response(json.dumps({"text": "Обнаружено оружие."}))

    ex = _explainer_with(handler)
    ex.generate_explanation(DETECTIONS, {"language": "ru", "available": True, "text": "x"},
                            None, RISK_HIGH)
    system_msg = captured["body"]["messages"][0]["content"]
    assert "РУССКОМ" in system_msg     # ru tizim prompt'i


def test_language_defaults_to_uz_when_unknown():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response(json.dumps({"text": "ok"}))

    ex = _explainer_with(handler)
    ex.generate_explanation(DETECTIONS, {"language": "en", "available": True}, None, RISK_HIGH)
    assert "O'ZBEK" in captured["body"]["messages"][0]["content"]


def test_schema_fail_then_retry_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _ollama_response("buzuq javob, JSON emas")     # 1-urinish: fail
        return _ollama_response(json.dumps({"text": "to'g'ri javob"}))  # 2-urinish: ok

    ex = _explainer_with(handler)
    res = ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)
    assert res["available"] is True
    assert res["text"] == "to'g'ri javob"
    assert calls["n"] == 2               # aynan 1x re-prompt


def test_schema_fail_twice_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_response("hech qachon JSON emas")

    ex = _explainer_with(handler)
    with pytest.raises(RuntimeError):    # worker buni tutib degradatsiya qiladi
        ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)


def test_empty_text_is_invalid_and_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_response(json.dumps({"text": "   "}))   # bo'sh -> yaroqsiz

    ex = _explainer_with(handler)
    with pytest.raises(RuntimeError):
        ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)


def test_http_5xx_raises_for_worker_restart():
    """Infra xato (OOM 500 kabi) -> RAISE -> worker daemon restart qiladi (ADR-001)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    ex = _explainer_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)


def test_connect_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("ulanish yo'q", request=request)

    ex = _explainer_with(handler)
    with pytest.raises(httpx.ConnectError):
        ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)


def test_think_tags_and_surrounding_text_are_stripped():
    payload = ("<think>foydalanuvchi qurol haqida...</think>\n"
               "Mana javob: " + json.dumps({"text": "Qurol aniqlandi."}))

    def handler(request: httpx.Request) -> httpx.Response:
        return _ollama_response(payload)

    ex = _explainer_with(handler)
    res = ex.generate_explanation(DETECTIONS, {"available": False}, None, RISK_HIGH)
    assert res["available"] is True
    assert res["text"] == "Qurol aniqlandi."


def test_low_risk_no_drivers_still_explains():
    risk_low = {"level": "LOW", "score": 0.0, "computed_by": "rule_engine_v1", "factors": []}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ollama_response(json.dumps({"text": "Xavfli obyekt topilmadi."}))

    ex = _explainer_with(handler)
    res = ex.generate_explanation([], {"available": False}, None, risk_low)
    assert res["available"] is True
    user_msg = captured["body"]["messages"][-1]["content"]
    assert "PAST" in user_msg or "0.0" in user_msg
