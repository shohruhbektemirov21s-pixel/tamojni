"""Integration: bo'sh case end-to-end (mock modellar) + audit izlari."""
from __future__ import annotations

from tests.helpers import PNG, client_for, wait_done


def test_empty_case_end_to_end(tmp_path):
    # to'liq Tier1+Tier2 yo'lini sinaymiz -> LLM'ni har case'ga yoqamiz
    with client_for(tmp_path, llm_auto_always=True) as client:
        r = client.post("/cases", files={"image": PNG})
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "PENDING"
        case_id = body["case_id"]

        result = wait_done(client, case_id)

        assert result["status"] == "DONE"
        assert result["degraded"] is False

        # Deterministik risk har doim bor (Tamoyil 2)
        assert result["risk"] is not None
        assert result["risk"]["level"] in ("LOW", "MEDIUM", "HIGH")
        assert result["risk"]["computed_by"].startswith("rule_engine")

        # detections §7.1 shaklida ("class" kaliti)
        assert len(result["detections"]) >= 1
        assert "class" in result["detections"][0]

        # audio yo'q -> transcript mavjud emas, lekin degradatsiya emas
        assert result["transcript"]["available"] is False

        # mock LLM tushuntirish beradi
        assert result["explanation"]["available"] is True

        # timings to'liq
        for key in ("stt", "detection", "synthesis", "total"):
            assert key in result["timings_ms"]
        assert result["timings_ms"]["total"] < 2000  # mock e2e tez

    # audit izlari (Tamoyil 3)
    with client_for(tmp_path) as _:  # nothing
        pass


def test_audit_trail_recorded(tmp_path):
    with client_for(tmp_path, llm_auto_always=True) as client:
        case_id = client.post("/cases", files={"image": PNG}).json()["case_id"]
        wait_done(client, case_id)

        entries = client.get(f"/cases/{case_id}/audit").json()["entries"]
        actions = [e["action"] for e in entries]
        for expected in (
            "CASE_CREATED",
            "DETECTION_DONE",
            "RISK_COMPUTED",
            "EXPLANATION_DONE",
        ):
            assert expected in actions, f"{expected} audit'da yo'q: {actions}"
