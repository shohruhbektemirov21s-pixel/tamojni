"""Backpressure — burst'da navbat ushlaydi + edge-triggered signal (4-bosqich).

Drop EMAS: bounded put tabiiy sekinlashtiradi; watermark'dan oshganda/tushganda
`backpressure` event push qilinadi. Worker'ni start QILMASDAN enqueue mantiqini
sinaymiz (navbat drenajlanmaydi -> to'ladi).
"""
from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.core.events import EventBus, EventType
from app.core.worker import CaseWorker


def _worker(tmp_path, bus, **over):
    return CaseWorker(
        settings=Settings(data_dir=tmp_path / "d", queue_maxsize=4, **over),
        repo=None, audit=None, orchestrator=None, providers={}, risk_config={},
        event_bus=bus,
    )


def test_edge_triggered_signals(tmp_path):
    async def scenario():
        bus = EventBus()
        w = _worker(tmp_path, bus)
        sub = bus.subscribe()
        w._signal_backpressure(3)   # high=int(4*0.8)=3 -> active
        w._signal_backpressure(4)   # allaqachon active -> takror emas
        w._clear_backpressure(2)    # low=int(4*0.5)=2 -> bo'shadi
        w._clear_backpressure(1)    # allaqachon bo'sh -> takror emas

        evs = []
        while not sub._queue.empty():
            evs.append(await sub.get())
        bp = [e.data["active"] for e in evs if e.type == EventType.BACKPRESSURE]
        assert bp == [True, False]

    asyncio.run(scenario())


def test_enqueue_triggers_backpressure_no_drop(tmp_path):
    async def scenario():
        bus = EventBus()
        w = _worker(tmp_path, bus)
        sub = bus.subscribe()
        # worker START qilinmaydi -> navbat drenajlanmaydi
        for i in range(3):
            await w.enqueue(f"c{i}")
        # hech narsa tashlanmadi: 3 ta ham navbatda
        assert w._queue.qsize() == 3
        evs = []
        while not sub._queue.empty():
            evs.append(await sub.get())
        assert any(e.type == EventType.BACKPRESSURE and e.data["active"] for e in evs)

    asyncio.run(scenario())
