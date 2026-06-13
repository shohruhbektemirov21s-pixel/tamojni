"""GPU Orchestrator — ADR-001: PROCESS SUPERVISOR, model loader EMAS.

Og'ir GPU inference daemon'ini (Ollama) start/stop/health qiladi va yagona GPU
lock'ni boshqaradi (Tamoyil 7: VRAM yagona egasi; ADR-002: GPU faqat Ollama'niki).

Nega process supervisor: process o'limi = OS tomonidan KAFOLATLANGAN, darhol VRAM
tozalash. In-process torch.cuda.empty_cache() ga ISHONMA (lazy GC -> OOM).

manage_ollama=False (dev/test): daemon boshqarilmaydi, gpu_session faqat lock
beradi va hech qachon tashqi tarmoqqa chiqmaydi.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess

import httpx

from app.core.config import Settings
from app.core.errors import ServiceUnavailable

log = logging.getLogger("customs.orchestrator")


class GpuOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()
        self._base = f"http://{settings.ollama_host}:{settings.ollama_port}"

    @property
    def base_url(self) -> str:
        return self._base

    # ---- lifecycle ----
    async def start(self) -> None:
        if not self.s.manage_ollama:
            return
        if self._proc is not None and self._proc.poll() is None:
            return
        log.info("Ollama daemon ishga tushirilmoqda")
        self._proc = subprocess.Popen(  # noqa: S603
            [self.s.ollama_binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(timeout=30.0)

    async def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            log.info("Ollama daemon to'xtatilmoqda")
            self._proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._proc.wait, 10)
        self._proc = None

    async def restart(self) -> None:
        """OOM/crash'dan keyin VRAM'ni kafolatli tozalash uchun."""
        log.warning("Ollama daemon qayta ishga tushirilmoqda (VRAM reclaim)")
        await self.stop()
        await self.start()

    # ---- health ----
    def process_alive(self) -> bool:
        if not self.s.manage_ollama:
            return True  # tashqi/qo'lda boshqariladigan daemon — tirik deb hisoblaymiz
        return self._proc is not None and self._proc.poll() is None

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._base}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def _wait_healthy(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await self.is_healthy():
                return
            await asyncio.sleep(0.5)
        raise ServiceUnavailable("Ollama GPU daemon ishga tushmadi", detail={"base": self._base})

    # ---- GPU lock (serial) ----
    @contextlib.asynccontextmanager
    async def gpu_session(self):
        """Bir vaqtda bitta GPU ishi (Tamoyil 7). Kerak bo'lsa daemon'ni tiklaydi."""
        async with self._lock:
            if self.s.manage_ollama and not self.process_alive():
                await self.restart()
            yield self
