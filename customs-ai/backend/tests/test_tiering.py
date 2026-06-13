"""Tier 1/Tier 2 strategiyasi + real-time event push (2-bosqich).

- Tier 1 (detect+risk) HAR DOIM ishlaydi va push qilinadi.
- Tier 2 (LLM) sozlanadigan: default faqat HIGH; bo'lmasa on-demand'ga qoladi
  (degradatsiya EMAS). llm_auto_always -> har case.
- /ws orqali tier1_done / alert / stt_done / case_done jonli keladi.
"""
from __future__ import annotations

from app.pipelines.mocks import MockDetector
from tests.helpers import PNG, client_for, wait_done

QUROL = [{"class": "qurol", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]  # -> HIGH


def _audit_actions(client, case_id):
    return [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]


def test_low_case_skips_llm_not_degraded(tmp_path):
    """Default flag: HIGH bo'lmagan case'da LLM avtomatik ishlamaydi (degraded EMAS)."""
    with client_for(tmp_path) as client:
        client.app.state.worker.p["detector"] = MockDetector(result=[])  # bo'sh -> LOW
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["status"] == "DONE"
        assert result["risk"]["level"] == "LOW"
        assert result["degraded"] is False            # ishlamaslik != yiqilish
        assert result["explanation"]["available"] is False
        actions = _audit_actions(client, case_id)
        assert "RISK_COMPUTED" in actions
        assert "EXPLANATION_DONE" not in actions      # LLM umuman ishlamadi


def test_high_case_auto_runs_llm(tmp_path):
    """Default flag (llm_auto_on_high): HIGH risk -> LLM avtomatik ishlaydi."""
    with client_for(tmp_path) as client:
        client.app.state.worker.p["detector"] = MockDetector(result=QUROL)
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["risk"]["level"] == "HIGH"
        assert result["explanation"]["available"] is True
        assert "EXPLANATION_DONE" in _audit_actions(client, case_id)


def test_always_flag_runs_llm_on_low(tmp_path):
    """llm_auto_always=True -> LOW case'da ham LLM ishlaydi (eski xulq)."""
    with client_for(tmp_path, llm_auto_always=True) as client:
        client.app.state.worker.p["detector"] = MockDetector(result=[])
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)
        assert result["risk"]["level"] == "LOW"
        assert result["explanation"]["available"] is True


def _drain_until(wsconn, stop_type, limit=30):
    types, events = [], []
    for _ in range(limit):
        ev = wsconn.receive_json()
        types.append(ev["type"])
        events.append(ev)
        if ev["type"] == stop_type:
            break
    return types, events


def test_ws_pipeline_event_sequence(tmp_path):
    """Skaner case'i -> /ws'da case_created/tier1_done/stt_done/case_done jonli."""
    with client_for(tmp_path) as client:
        with client.websocket_connect("/ws") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
            types, _ = _drain_until(wsconn, "case_done")

    for expected in ("case_created", "tier1_done", "stt_done", "case_done"):
        assert expected in types, f"{expected} push bo'lmadi: {types}"


def test_ws_high_risk_emits_alert(tmp_path):
    """HIGH risk -> tier1_done + alert push qilinadi."""
    with client_for(tmp_path) as client:
        client.app.state.worker.p["detector"] = MockDetector(result=QUROL)
        with client.websocket_connect("/ws") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
            types, events = _drain_until(wsconn, "case_done")

    assert "alert" in types
    tier1 = next(e for e in events if e["type"] == "tier1_done")
    assert tier1["case_id"] == case_id
    assert tier1["data"]["risk"]["level"] == "HIGH"
    alert = next(e for e in events if e["type"] == "alert")
    assert alert["data"]["level"] == "HIGH"
