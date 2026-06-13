"""POST /tts — matn -> audio (4-bosqich). Offline mock synthesizer."""
from __future__ import annotations

from app.pipelines.mocks import MockSynthesizer
from tests.helpers import PNG, client_for, wait_done


def test_tts_returns_audio(tmp_path):
    with client_for(tmp_path) as client:
        r = client.post("/tts", json={"text": "Diqqat, yuqori xavf", "language": "uz"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/")
        assert len(r.content) > 0


def test_tts_empty_text_400(tmp_path):
    with client_for(tmp_path) as client:
        r = client.post("/tts", json={"text": "   "})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "validation_failed"


def test_tts_with_case_audits_and_emits(tmp_path):
    with client_for(tmp_path) as client:
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        wait_done(client, case_id)

        with client.websocket_connect(f"/ws?case_id={case_id}") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            r = client.post("/tts", json={"text": "Tekshiruv natijasi", "case_id": case_id})
            assert r.status_code == 200
            # tts_ready event push qilinadi
            ev = wsconn.receive_json()
            assert ev["type"] == "tts_ready"
            assert ev["data"]["bytes"] == len(r.content)

        actions = [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]
        assert "TTS_DONE" in actions


def test_tts_unknown_case_404(tmp_path):
    with client_for(tmp_path) as client:
        r = client.post("/tts", json={"text": "salom", "case_id": "yo-q"})
        assert r.status_code == 404


def test_tts_failure_degrades_503(tmp_path):
    with client_for(tmp_path) as client:
        client.app.state.providers["synthesizer"] = MockSynthesizer(fail=True)
        r = client.post("/tts", json={"text": "salom"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "service_unavailable"
