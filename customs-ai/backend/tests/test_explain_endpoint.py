"""POST /cases/{id}/explain — on-demand LLM streaming (4-bosqich).

Default flag (HIGH'da emas) -> auto LLM ishlamaydi; operator /explain bilan
so'raydi -> tokenlar /ws orqali oqadi, explanation saqlanadi.
"""
from __future__ import annotations

from app.pipelines.mocks import MockStreamingExplainer
from tests.helpers import PNG, client_for, wait_done


def _drain_until(wsconn, stop_type, limit=60):
    events = []
    for _ in range(limit):
        ev = wsconn.receive_json()
        events.append(ev)
        if ev["type"] == stop_type:
            break
    return events


def test_explain_unknown_case_404(tmp_path):
    with client_for(tmp_path) as client:
        r = client.post("/cases/yo-q/explain")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"


def test_on_demand_explain_streams_and_persists(tmp_path):
    with client_for(tmp_path) as client:
        client.app.state.worker.p["explainer"] = MockStreamingExplainer()

        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        result = wait_done(client, case_id)
        # default: HIGH emas -> auto LLM ishlamadi
        assert result["explanation"]["available"] is False

        with client.websocket_connect(f"/ws?case_id={case_id}") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            r = client.post(f"/cases/{case_id}/explain")
            assert r.status_code == 202
            assert r.json()["status"] == "accepted"
            events = _drain_until(wsconn, "explanation_done")

    types = [e["type"] for e in events]
    assert types.count("explanation_token") >= 2
    assert "explanation_done" in types
    assert events[-1]["data"]["on_demand"] is True

    # endi explanation saqlangan
    res = client.get(f"/cases/{case_id}").json()
    assert res["explanation"]["available"] is True
    actions = [e["action"] for e in client.get(f"/cases/{case_id}/audit").json()["entries"]]
    assert "EXPLANATION_DONE" in actions
