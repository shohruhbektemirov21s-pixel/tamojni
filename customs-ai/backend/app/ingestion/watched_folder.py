"""WatchedFolderSource — papkani kuzatib, yangi rasm -> avtomatik case.

watchdog Observer faqat YETKAZIB BERISH mexanizmi; butun ingestion logikasi
(`_ingest_file`) sof va watchdog'siz ham chaqirilishi mumkin (test uchun qulay).

Gotcha'lar (qabul mezoni):
  - Partial fayl: skaner yozib tugatguncha kutamiz (hajm `stable_checks` marta
    o'zgarmasligi). Timeout -> partial deb skip.
  - sha256 dedup: bir xil kontent qayta tushsa (watchdog ikki marta otsa yoki
    fayl qayta nusxalansa) — skip. In-memory + (ixtiyoriy) DB tekshiruvi.
  - Bo'sh/o'qib bo'lmaydigan fayl: skip + SCAN_REJECTED (case ochilmaydi).
    Korrupt-rasm esa case oladi — detection bosqichida degradatsiya qilinadi.
  - Hidden/temp (.tmp/.part/...) va noma'lum kengaytmalar e'tiborga olinmaydi.

observer thread bloklanmasligi uchun har fayl kichik ThreadPool'da ishlanadi;
case yaratish esa `intake.submit_threadsafe` orqali event loop'ga ko'chiriladi.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from app.core.config import Settings
from app.core.events import EventBus, EventType
from app.ingestion.base import ScannerSource
from app.services.case_intake import CaseIntake

log = logging.getLogger("customs.scanner")

_IGNORE_SUFFIXES = (".tmp", ".part", ".partial", ".crdownload", ".swp", ".filepart", ".lock")


class WatchedFolderSource(ScannerSource):
    def __init__(
        self,
        *,
        folder: Path,
        intake: CaseIntake,
        event_bus: EventBus,
        settings: Settings,
        repo=None,
    ) -> None:
        self.folder = Path(folder)
        self.intake = intake
        self.bus = event_bus
        self.s = settings
        self.repo = repo  # ixtiyoriy: restart'lararo sha256 dedup uchun DB tekshiruvi
        self._observer = None
        self._executor: ThreadPoolExecutor | None = None
        self._seen_sha: set[str] = set()
        self._lock = threading.Lock()
        self._exts = settings.scanner_extension_set

    # ---- lifecycle ----
    async def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="scan-ingest")

        # Startup'dan oldin papkada turgan fayllarni ham qabul qilamiz (yo'qotmaslik).
        if self.s.scanner_process_existing:
            for p in sorted(self.folder.iterdir()):
                if self._is_candidate(p):
                    self._dispatch(p)

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        source = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    source._dispatch(Path(event.src_path))

            def on_moved(self, event):
                # skaner ko'pincha .tmp ga yozib, tugagach final nomga ko'chiradi
                if not event.is_directory:
                    source._dispatch(Path(event.dest_path))

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self.folder), recursive=False)
        self._observer.start()
        log.info("WatchedFolderSource kuzatyapti: %s (ext=%s)", self.folder, sorted(self._exts))

    async def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # ---- dispatch ----
    def _dispatch(self, path: Path) -> None:
        """observer thread'ni bloklamaslik uchun ishlovni pool'ga beradi."""
        if self._executor is None:  # start'dan oldin/keyin himoya
            self._ingest_file(path)
            return
        self._executor.submit(self._safe_ingest, path)

    def _safe_ingest(self, path: Path) -> None:
        try:
            self._ingest_file(path)
        except Exception:  # noqa: BLE001 - bitta fayl xatosi kuzatuvni qulatmasin
            log.exception("Ingestion xatosi: %s", path)

    # ---- core (sof, watchdog'siz testlanadi) ----
    def _ingest_file(self, path: Path) -> Future | None:
        if not self._is_candidate(path):
            return None
        if not self._wait_stable(path):
            self._reject("partial_or_unstable", path)
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            self._reject("read_error", path, {"error": str(exc)})
            return None
        if not data:
            self._reject("empty", path)
            return None

        sha = hashlib.sha256(data).hexdigest()
        if not self._claim(sha):
            self._reject("duplicate", path, {"sha256": sha})
            return None

        log.info("Skaner fayli qabul qilindi: %s (%d bayt, sha=%s)", path.name, len(data), sha[:12])
        return self.intake.submit_threadsafe(
            image_bytes=data,
            image_filename=path.name,
            source=f"scanner:{self.folder.name}",
            scan_meta={"sha256": sha, "filename": path.name, "size": len(data), "path": str(path)},
        )

    # ---- helpers ----
    def _is_candidate(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name
        if name.startswith(".") or name.endswith(_IGNORE_SUFFIXES):
            return False
        return path.suffix.lower() in self._exts

    def _wait_stable(self, path: Path) -> bool:
        """Hajm `stable_checks` marta ketma-ket o'zgarmasa (va >0) -> tayyor."""
        poll = self.s.scanner_poll_interval_s
        needed = self.s.scanner_stable_checks
        deadline = time.monotonic() + self.s.scanner_stable_timeout_s
        last = -1
        stable = 0
        while time.monotonic() < deadline:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                return False  # ko'chirilgan/o'chirilgan
            if size > 0 and size == last:
                stable += 1
                if stable >= needed:
                    return True
            else:
                stable = 0
                last = size
            time.sleep(poll)
        return False  # hali o'syapti -> partial

    def _claim(self, sha: str) -> bool:
        """Atomar: sha ko'rilmagan bo'lsa belgilab True, aks holda False (dublikat)."""
        with self._lock:
            if sha in self._seen_sha:
                return False
            if self.repo is not None:
                try:
                    if self.repo.exists_attachment_sha256(sha):
                        self._seen_sha.add(sha)
                        return False
                except Exception:  # noqa: BLE001 - DB tekshiruvi best-effort
                    log.debug("sha dedup DB tekshiruvi o'tkazib yuborildi", exc_info=True)
            self._seen_sha.add(sha)
            return True

    def _reject(self, reason: str, path: Path, extra: dict | None = None) -> None:
        log.warning("Skaner fayli rad etildi (%s): %s", reason, path.name)
        self.bus.publish_threadsafe(
            EventType.SCAN_REJECTED,
            None,
            {"reason": reason, "filename": path.name, **(extra or {})},
        )
