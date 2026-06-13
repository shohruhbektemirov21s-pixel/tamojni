"""Degradatsiya (§10, Tamoyil 6): model yiqilsa case to'liq risk bilan yakunlanadi."""
from __future__ import annotations

from app.pipelines.mocks import MockExplainer, MockTranscriber
from tests.helpers import PNG, WAV, client_for, wait_done


def test_stt_failure_degrades_not_fails(tmp_path):
    with client_for(tmp_path) as client:
        client.app.state.worker.p["transcriber"] = MockTranscriber(fail=True)

        case_id = client.post("/cases", files={"image": PNG, "audio": WAV}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["status"] == "DONE"          # FAIL emas
        assert result["degraded"] is True
        assert result["transcript"]["available"] is False
        assert result["risk"] is not None          # deterministik risk baribir bor

        actions = [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]
        assert "MODEL_FAILED" in actions
        assert "RISK_COMPUTED" in actions


def test_stt_timeout_degrades(tmp_path):
    with client_for(tmp_path, stt_timeout_s=0.2) as client:
        client.app.state.worker.p["transcriber"] = MockTranscriber(hang_s=3.0)

        case_id = client.post("/cases", files={"image": PNG, "audio": WAV}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["status"] == "DONE"
        assert result["degraded"] is True
        assert result["transcript"]["available"] is False
        assert result["risk"] is not None


def test_llm_failure_degrades_not_fails(tmp_path):
    with client_for(tmp_path) as client:
        client.app.state.worker.p["explainer"] = MockExplainer(fail=True)

        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["status"] == "DONE"
        assert result["degraded"] is True
        assert result["explanation"]["available"] is False
        # Risk LLM'dan OLDIN hisoblangani uchun mavjud va to'liq
        assert result["risk"] is not None
        assert result["risk"]["level"] in ("LOW", "MEDIUM", "HIGH")


def test_detection_failure_degrades(tmp_path):
    from app.pipelines.mocks import MockDetector

    with client_for(tmp_path) as client:
        client.app.state.worker.p["detector"] = MockDetector(fail=True)

        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)

        assert result["status"] == "DONE"
        assert result["degraded"] is True
        # bo'sh detections -> risk LOW, lekin case yashaydi
        assert result["risk"] is not None
        assert result["detections"] == []
