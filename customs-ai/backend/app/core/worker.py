"""Case Worker — pipeline orkestratsiyasi (asyncio, GPU serial).

Oqim (§6.1):
    PROCESSING -> (parallel: STT[CPU] || detection[CPU])
               -> DETERMINISTIK risk (LLM'dan OLDIN)
               -> LLM synthesis (GPU, serial, retry+timeout)
               -> persist + DONE

Degradatsiya (§10, Tamoyil 6): bitta model yiqilsa, case TASHLANMAYDI —
qisman natija + degraded=true + available=false + MODEL_FAILED audit.
Risk score deterministik bo'lgani uchun LLM/STT yiqilsa ham case to'liq risk
bilan yakunlanadi.

Faza 1: bir vaqtda bitta case (queue). Bu GPU serial'ligini tabiiy ta'minlaydi;
gpu_session() lock esa qo'shimcha himoya.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from app.core.config import Settings
from app.core.enums import AuditAction, CaseStatus
from app.core.events import EventBus
from app.core.orchestrator import GpuOrchestrator
from app.repositories.case_repository import CaseRepository
from app.services.audit_service import AuditService

log = logging.getLogger("customs.worker")

_EMPTY_TRANSCRIPT = {"text": None, "language": None, "confidence": None, "available": False}
_EMPTY_EXPLANATION = {"text": None, "generated_by": None, "available": False}


class CaseWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: CaseRepository,
        audit: AuditService,
        orchestrator: GpuOrchestrator,
        providers: dict,
        risk_config: dict,
        event_bus: EventBus | None = None,
    ) -> None:
        self.s = settings
        self.repo = repo
        self.audit = audit
        self.orch = orchestrator
        self.p = providers
        self.risk_config = risk_config
        # 2-bosqich relay shu yerga push qiladi (tier1_done, stt_partial, ...).
        # None bo'lsa (eski testlar) push o'tkazib yuboriladi.
        self.bus = event_bus
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.queue_maxsize)
        self._task: asyncio.Task | None = None
        self._running = False

    # ---- lifecycle ----
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="case-worker")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def enqueue(self, case_id: str) -> None:
        await self._queue.put(case_id)

    async def _loop(self) -> None:
        while self._running:
            try:
                case_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self.process(case_id)
            except Exception:  # noqa: BLE001
                log.exception("Case %s ishlovida kutilmagan xato", case_id)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        self.repo.set_status, case_id, CaseStatus.FAILED.value, completed=True
                    )
            finally:
                self._queue.task_done()

    # ---- audit helperlari ----
    async def _audit(self, case_id: str, action: AuditAction, payload: dict | None = None) -> None:
        await asyncio.to_thread(self.audit.log, case_id, "system", action.value, payload)

    async def _audit_fail(self, case_id: str, stage: str, exc: Exception) -> None:
        await asyncio.to_thread(
            self.audit.log,
            case_id,
            "system",
            AuditAction.MODEL_FAILED.value,
            {"stage": stage, "error": str(exc)},
        )

    # ---- bosqichlar ----
    async def _run_stt(self, case_id: str, audio_path: str | None) -> tuple[dict, float, bool]:
        if not audio_path:
            return dict(_EMPTY_TRANSCRIPT), 0.0, False  # audio yo'q = degradatsiya emas
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(self.p["transcriber"].transcribe, audio_path, None),
                timeout=self.s.stt_timeout_s,
            )
            res.setdefault("available", True)
            return res, (time.perf_counter() - t0) * 1000, False
        except Exception as exc:  # timeout ham shu yerga tushadi
            log.warning("STT degradatsiya (case %s): %s", case_id, exc)
            await self._audit_fail(case_id, "stt", exc)
            return dict(_EMPTY_TRANSCRIPT), (time.perf_counter() - t0) * 1000, True

    async def _run_detection(self, case_id: str, image_path: str | None) -> tuple[list, float, bool]:
        t0 = time.perf_counter()
        try:
            dets = [] if image_path is None else await asyncio.to_thread(
                self.p["detector"].detect, image_path
            )
            return dets, (time.perf_counter() - t0) * 1000, False
        except Exception as exc:  # noqa: BLE001
            log.warning("Detection degradatsiya (case %s): %s", case_id, exc)
            await self._audit_fail(case_id, "detection", exc)
            return [], (time.perf_counter() - t0) * 1000, True

    async def _run_synthesis(
        self, case_id: str, detections: list, transcript: dict, notes: str | None, risk: dict
    ) -> tuple[dict, float, bool]:
        t0 = time.perf_counter()
        attempts = self.s.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                async with self.orch.gpu_session():
                    res = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.p["explainer"].generate_explanation,
                            detections,
                            transcript,
                            notes,
                            risk,
                        ),
                        timeout=self.s.llm_timeout_s,
                    )
                res.setdefault("available", True)
                return res, (time.perf_counter() - t0) * 1000, False
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "LLM synthesis urinish %d/%d yiqildi (case %s): %s",
                    attempt + 1, attempts, case_id, exc,
                )
                # Ehtimoliy OOM -> process restart (kafolatli VRAM tozalash).
                with contextlib.suppress(Exception):
                    await self.orch.restart()
                if attempt == attempts - 1:
                    await self._audit_fail(case_id, "llm", exc)
        return dict(_EMPTY_EXPLANATION), (time.perf_counter() - t0) * 1000, True

    # ---- asosiy pipeline ----
    async def process(self, case_id: str) -> None:
        t_start = time.perf_counter()
        timings: dict = {}
        degraded = False

        await asyncio.to_thread(self.repo.set_status, case_id, CaseStatus.PROCESSING.value)
        case = await asyncio.to_thread(self.repo.get, case_id)
        if case is None:
            log.error("process: case %s topilmadi", case_id)
            return

        image_path = audio_path = None
        for a in case.attachments:
            if a.kind == "image":
                image_path = a.path
            elif a.kind == "audio":
                audio_path = a.path
        operator_notes = case.operator_notes

        # 1) parallel: STT (CPU) || detection (CPU)
        (transcript, stt_ms, stt_failed), (detections, det_ms, det_failed) = await asyncio.gather(
            self._run_stt(case_id, audio_path),
            self._run_detection(case_id, image_path),
        )
        timings["stt"] = round(stt_ms)
        timings["detection"] = round(det_ms)
        degraded = degraded or stt_failed or det_failed

        await asyncio.to_thread(self.repo.save_transcript, case_id, transcript)
        if transcript.get("available"):
            await self._audit(case_id, AuditAction.STT_DONE, {"language": transcript.get("language")})
        await asyncio.to_thread(self.repo.save_detections, case_id, detections)
        if not det_failed:
            await self._audit(case_id, AuditAction.DETECTION_DONE, {"count": len(detections)})

        # 2) DETERMINISTIK risk — LLM'dan OLDIN (Tamoyil 2)
        risk = await asyncio.to_thread(
            self.p["risk_engine"].compute_risk, detections, self.risk_config
        )
        await asyncio.to_thread(self.repo.save_risk, case_id, risk)
        await self._audit(
            case_id, AuditAction.RISK_COMPUTED, {"level": risk["level"], "score": risk["score"]}
        )

        # 3) LLM synthesis (GPU, serial, retry+timeout) — faqat tushuntirish
        explanation, synth_ms, synth_failed = await self._run_synthesis(
            case_id, detections, transcript, operator_notes, risk
        )
        timings["synthesis"] = round(synth_ms)
        degraded = degraded or synth_failed
        await asyncio.to_thread(self.repo.save_explanation, case_id, explanation)
        if explanation.get("available"):
            await self._audit(
                case_id, AuditAction.EXPLANATION_DONE, {"generated_by": explanation.get("generated_by")}
            )

        # 4) yakunlash
        timings["total"] = round((time.perf_counter() - t_start) * 1000)
        await asyncio.to_thread(self.repo.set_timings, case_id, timings)
        if degraded:
            await asyncio.to_thread(self.repo.set_degraded, case_id, True)
        await asyncio.to_thread(
            self.repo.set_status, case_id, CaseStatus.DONE.value, completed=True
        )
        log.info("Case %s yakunlandi (risk=%s, degraded=%s, %dms)",
                 case_id, risk["level"], degraded, timings["total"])
