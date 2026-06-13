"""API kontrakti, error format, decision oqimi va startup recovery (§10)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.enums import CaseStatus
from app.main import create_app
from tests.helpers import PNG, client_for, wait_done


def test_health(tmp_path):
    with client_for(tmp_path) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["models"]["llm"] == "mock"
        assert body["gpu"]["managed"] is False


def test_error_format_404(tmp_path):
    with client_for(tmp_path) as client:
        r = client.get("/cases/does-not-exist")
        assert r.status_code == 404
        err = r.json()["error"]
        assert err["code"] == "not_found"
        assert "message" in err


def test_create_without_image_is_422(tmp_path):
    with client_for(tmp_path) as client:
        r = client.post("/cases")  # image majburiy
        assert r.status_code in (400, 422)
        assert "error" in r.json()  # bizning yagona format


def test_decision_flow(tmp_path):
    with client_for(tmp_path) as client:
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        wait_done(client, case_id)

        r = client.post(f"/cases/{case_id}/decision", json={"decision": "CONFIRM", "notes": "ok"})
        assert r.status_code == 200
        assert r.json()["audit_id"]

        actions = [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]
        assert "OPERATOR_CONFIRMED" in actions


def test_list_pagination(tmp_path):
    with client_for(tmp_path) as client:
        for _ in range(3):
            cid = client.post("/cases", files={"image": PNG}).json()["case_id"]
            wait_done(client, cid)
        body = client.get("/cases", params={"limit": 2, "offset": 0}).json()
        assert body["total"] == 3
        assert len(body["items"]) == 2


def test_stream_emits_terminal(tmp_path):
    with client_for(tmp_path) as client:
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        wait_done(client, case_id)  # allaqachon terminal
        with client.stream("GET", f"/cases/{case_id}/stream") as r:
            assert r.status_code == 200
            text = "\n".join(list(r.iter_lines()))
        assert "DONE" in text


def test_startup_recovery(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", use_mocks=True, manage_ollama=False)
    # 1-app: PROCESSING'da osilib qolgan case yaratamiz (lifespan'siz)
    app1 = create_app(settings)
    case_id = app1.state.repo.create(operator_id=None, operator_notes=None)
    app1.state.repo.set_status(case_id, CaseStatus.PROCESSING.value)

    # 2-app: bir xil DB, lifespan startup recovery'ni ishga tushiradi
    app2 = create_app(settings)
    with TestClient(app2):
        assert app2.state.repo.get_status(case_id) == CaseStatus.FAILED.value
