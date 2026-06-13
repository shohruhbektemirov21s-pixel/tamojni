"""WebSocket /ws — multi-client push, case filtri, threadsafe publish.

Push'ni server loop'iga `publish_threadsafe` orqali yuboramiz — bu watchdog
(boshqa thread) yo'lining aynan o'zi.
"""
from __future__ import annotations

from app.core.events import EventType
from tests.helpers import client_for


def test_ws_ready_and_push(tmp_path):
    with client_for(tmp_path) as client:
        bus = client.app.state.event_bus
        with client.websocket_connect("/ws") as wsconn:
            ready = wsconn.receive_json()
            assert ready["type"] == "ready"

            bus.publish_threadsafe(
                EventType.TIER1_DONE, case_id="c1", data={"level": "HIGH"}
            )
            ev = wsconn.receive_json()
            assert ev["type"] == "tier1_done"
            assert ev["case_id"] == "c1"
            assert ev["data"]["level"] == "HIGH"
            assert isinstance(ev["seq"], int)


def test_ws_case_filter(tmp_path):
    with client_for(tmp_path) as client:
        bus = client.app.state.event_bus
        with client.websocket_connect("/ws?case_id=c1") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            # c2 filtrlanadi, c1 keladi -> birinchi qabul qilingani c1 bo'lishi shart
            bus.publish_threadsafe(EventType.STT_PARTIAL, case_id="c2")
            bus.publish_threadsafe(EventType.STT_PARTIAL, case_id="c1")
            ev = wsconn.receive_json()
            assert ev["case_id"] == "c1"


def test_ws_multi_client(tmp_path):
    with client_for(tmp_path) as client:
        bus = client.app.state.event_bus
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            assert a.receive_json()["type"] == "ready"
            assert b.receive_json()["type"] == "ready"
            bus.publish_threadsafe(EventType.ALERT, case_id="cX", data={"level": "HIGH"})
            for conn in (a, b):
                ev = conn.receive_json()
                assert ev["type"] == "alert" and ev["case_id"] == "cX"
