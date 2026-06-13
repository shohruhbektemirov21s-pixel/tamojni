"""ScannerSource — skaner ingestion abstraksiyasi (pluggable).

Vendor SDK yoki tarmoq manbasi A1 Sprint 0'da tasdiqlanadi; o'shangacha default
WatchedFolderSource (papka kuzatuvi) ishlaydi. Yangi manba shu ABC'ni
implementatsiya qiladi va lifespan'da start/stop qilinadi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ScannerSource(ABC):
    """Skanerdan rasm oqimini qabul qilib, CaseIntake orqali case ochadi.

    Implementatsiya `start()` da kuzatuvni boshlaydi (non-blocking — o'z thread'i
    yoki observer'i bilan), `stop()` da toza yopadi.
    """

    @abstractmethod
    async def start(self) -> None:
        """Kuzatuvni boshlaydi. Event loop ichida chaqiriladi."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Kuzatuvni to'xtatadi va resurslarni bo'shatadi."""
        ...
