"""EventBus birliklari — fan-out, case filtri, slow-consumer drop-oldest.

pytest-asyncio'siz: async sahnani asyncio.run bilan haydaymiz (loyiha uslubi).
"""
from __future__ import annotations

import asyncio

from app.core.events import EventBus, EventType


def test_fanout_and_case_filter():
    async def scenario():
        bus = EventBus(subscriber_buffer=10)
        all_sub = bus.subscribe()              # hamma event
        c1_sub = bus.subscribe(case_id="c1")   # faqat c1 + tizim eventlari

        bus.publish(EventType.TIER1_DONE, case_id="c1")
        bus.publish(EventType.TIER1_DONE, case_id="c2")
        bus.publish(EventType.BACKPRESSURE, case_id=None, data={"depth": 5})  # tizim

        got_all = [await all_sub.get() for _ in range(3)]
        assert [e.case_id for e in got_all] == ["c1", "c2", None]

        # c1 subscriber: o'z case'i + tizim eventi (None), c2 EMAS
        got_c1 = [await c1_sub.get() for _ in range(2)]
        assert [e.case_id for e in got_c1] == ["c1", None]
        assert c1_sub._queue.empty()

        # seq monotonik o'sadi
        assert [e.seq for e in got_all] == sorted(e.seq for e in got_all)

    asyncio.run(scenario())


def test_drop_oldest_on_overflow():
    async def scenario():
        bus = EventBus(subscriber_buffer=2)
        sub = bus.subscribe()
        for i in range(5):
            bus.publish(EventType.STT_PARTIAL, case_id="c", data={"i": i})
        # maxsize=2, 5 event -> 3 ta eng eski tashlandi
        assert sub.dropped == 3
        e1, e2 = await sub.get(), await sub.get()
        assert [e1.data["i"], e2.data["i"]] == [3, 4]  # faqat oxirgi 2 saqlandi
        assert sub._queue.empty()

    asyncio.run(scenario())


def test_unsubscribe_stops_delivery():
    async def scenario():
        bus = EventBus(subscriber_buffer=10)
        sub = bus.subscribe()
        assert bus.subscriber_count == 1
        sub.close()
        assert bus.subscriber_count == 0
        bus.publish(EventType.TIER1_DONE, case_id="c1")  # hech kimga bormaydi
        assert sub._queue.empty()

    asyncio.run(scenario())


def test_to_wire_shape():
    bus = EventBus()
    ev = bus.publish(EventType.ALERT, case_id="c9", data={"level": "HIGH"})
    wire = ev.to_wire()
    assert set(wire) == {"type", "case_id", "seq", "ts", "data"}
    assert wire["type"] == "alert"
    assert wire["case_id"] == "c9"
    assert wire["data"] == {"level": "HIGH"}
    assert isinstance(wire["seq"], int) and wire["seq"] >= 1
