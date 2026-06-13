"""Pluggable ScannerSource stub'lari (vendor integratsiyasi — A1 Sprint 0).

Vendor SDK yoki tarmoq protokoli tasdiqlangach to'ldiriladi. Hozircha kontrakt
shaklini belgilab turadi; ulanса WatchedFolderSource o'rniga (yoki yonida)
lifespan'da start qilinadi. Ikkalasi ham bir xil CaseIntake'ga push qiladi.
"""
from __future__ import annotations

from app.ingestion.base import ScannerSource


class SdkSource(ScannerSource):
    """Vendor SDK callback'lari orqali rasm oladi (push-based)."""

    async def start(self) -> None:
        raise NotImplementedError("SdkSource vendor SDK tasdiqlangach implementatsiya qilinadi")

    async def stop(self) -> None:
        return None


class NetworkSource(ScannerSource):
    """Skaner tarmoq protokoli (masalan DICOM C-STORE SCP) orqali rasm oladi."""

    async def start(self) -> None:
        raise NotImplementedError("NetworkSource vendor protokoli tasdiqlangach implementatsiya qilinadi")

    async def stop(self) -> None:
        return None
