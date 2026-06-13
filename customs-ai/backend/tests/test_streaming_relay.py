"""STT/LLM streaming relay — partial/token jonli push (4-bosqich).

Streaming mock'lar inject qilinadi; worker isinstance bilan streaming yo'lni
tanlaydi. /ws'da stt_partial/explanation_token JONLI (batch emas) keladi.
"""
from __future__ import annotations

from app.pipelines.mocks import MockStreamingExplainer, MockStreamingTranscriber
from tests.helpers import PNG, WAV, client_for


def _drain_until(wsconn, stop_type, limit=80):
    events = []
    for _ in range(limit):
        ev = wsconn.receive_json()
        events.append(ev)
        if ev["type"] == stop_type:
            break
    return events


def test_stt_and_llm_stream_live(tmp_path):
    # LLM har case'ga (always), streaming provayderlar inject qilingan
    with client_for(tmp_path, llm_auto_always=True) as client:
        client.app.state.worker.p["transcriber"] = MockStreamingTranscriber()
        client.app.state.worker.p["explainer"] = MockStreamingExplainer()

        with client.websocket_connect("/ws") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            client.post("/cases", files={"image": PNG, "audio": WAV})
            events = _drain_until(wsconn, "case_done")

    types = [e["type"] for e in events]
    # STT jonli: bir nechta partial + yakuniy stt_done
    assert types.count("stt_partial") >= 2
    assert "stt_done" in types
    # LLM jonli: bir nechta token + explanation_done
    assert types.count("explanation_token") >= 2
    assert "explanation_done" in types

    # partiallar stt_done'dan OLDIN kelishi shart (jonli, batch emas)
    assert types.index("stt_partial") < types.index("stt_done")
    assert types.index("explanation_token") < types.index("explanation_done")

    # token'lar birlashganda explanation matni hosil bo'ladi
    tokens = [e["data"]["token"] for e in events if e["type"] == "explanation_token"]
    assert "".join(tokens).strip()


def test_batch_provider_still_works(tmp_path):
    """Streaming bo'lmagan provayder (default mock) batch yo'l bilan ishlaydi."""
    with client_for(tmp_path, llm_auto_always=True) as client:
        with client.websocket_connect("/ws") as wsconn:
            assert wsconn.receive_json()["type"] == "ready"
            client.post("/cases", files={"image": PNG, "audio": WAV})
            events = _drain_until(wsconn, "case_done")
    types = [e["type"] for e in events]
    assert "stt_done" in types and "explanation_done" in types
    assert "stt_partial" not in types          # batch -> partial yo'q
    assert "explanation_token" not in types
