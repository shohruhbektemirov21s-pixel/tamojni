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
from app.core.events import EventBus, EventType
from app.core.orchestrator import GpuOrchestrator
from app.pipelines.contracts import StreamingExplainer, StreamingTranscriber
from app.repositories.case_repository import CaseRepository
from app.services.audit_service import AuditService
from app.services.serializers import serialize_case

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
        # Backpressure (burst himoyasi) — edge-triggered watermark signallari.
        maxsize = max(1, settings.queue_maxsize)
        self._bp_high = max(1, int(maxsize * settings.queue_high_watermark_ratio))
        self._bp_low = int(maxsize * settings.queue_low_watermark_ratio)
        self._bp_active = False
        # On-demand explain (POST /cases/{id}/explain) — bir case uchun bittadan.
        self._explaining: set[str] = set()
        self._explain_tasks: set[asyncio.Task] = set()

    # ---- lifecycle ----
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="case-worker")

    async def stop(self) -> None:
        self._running = False
        for t in list(self._explain_tasks):
            t.cancel()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def enqueue(self, case_id: str) -> None:
        # bounded put: navbat to'la bo'lsa BU YERDA tabiiy backpressure (bloklanadi,
        # drop EMAS). Watermark'dan oshganda operatorga ogohlantirish push qilamiz.
        await self._queue.put(case_id)
        self._signal_backpressure(self._queue.qsize())

    # ---- backpressure (edge-triggered) ----
    def _signal_backpressure(self, depth: int) -> None:
        if not self._bp_active and depth >= self._bp_high:
            self._bp_active = True
            self._emit(EventType.BACKPRESSURE, None,
                       {"active": True, "depth": depth, "maxsize": self._queue.maxsize})
            log.warning("Backpressure: navbat to'lib boryapti (%d/%d)", depth, self._queue.maxsize)

    def _clear_backpressure(self, depth: int) -> None:
        if self._bp_active and depth <= self._bp_low:
            self._bp_active = False
            self._emit(EventType.BACKPRESSURE, None,
                       {"active": False, "depth": depth, "maxsize": self._queue.maxsize})
            log.info("Backpressure bo'shadi (%d/%d)", depth, self._queue.maxsize)

    async def _loop(self) -> None:
        while self._running:
            try:
                case_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            self._clear_backpressure(self._queue.qsize())
            try:
                await self.process(case_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("Case %s ishlovida kutilmagan xato", case_id)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        self.repo.set_status, case_id, CaseStatus.FAILED.value, completed=True
                    )
                self._emit(EventType.CASE_FAILED, case_id, {"error": str(exc)})
            finally:
                self._queue.task_done()

    # ---- event push helperi (real-time relay) ----
    def _emit(self, type: EventType, case_id: str | None, data: dict | None = None) -> None:
        """Event bus'ga non-blocking push. Bus yo'q bo'lsa (eski test) — no-op.

        process() event loop ichida aylanadi, shuning uchun publish() to'g'ridan-to'g'ri.
        """
        if self.bus is not None:
            self.bus.publish(type, case_id, data)

    def _emit_threadsafe(self, type: EventType, case_id: str | None, data: dict | None = None) -> None:
        """Streaming consumer THREAD'idan push (partial/token) — loop'ga ko'chiradi."""
        if self.bus is not None:
            self.bus.publish_threadsafe(type, case_id, data)

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
        # degradatsiyani jonli bildiramiz (Tamoyil 6: degraded, fail emas)
        self._emit(EventType.MODEL_FAILED, case_id, {"stage": stage, "error": str(exc)})

    # ---- bosqichlar ----
    async def _run_stt(self, case_id: str, audio_path: str | None) -> tuple[dict, float, bool]:
        if not audio_path:
            return dict(_EMPTY_TRANSCRIPT), 0.0, False  # audio yo'q = degradatsiya emas
        transcriber = self.p["transcriber"]
        streaming = isinstance(transcriber, StreamingTranscriber)
        t0 = time.perf_counter()
        try:
            if streaming:
                # JONLI: partiallar thread ichidan stt_partial bo'lib push qilinadi.
                res = await asyncio.wait_for(
                    asyncio.to_thread(self._consume_stt_stream, case_id, transcriber, audio_path),
                    timeout=self.s.stt_timeout_s,
                )
            else:
                res = await asyncio.wait_for(
                    asyncio.to_thread(transcriber.transcribe, audio_path, None),
                    timeout=self.s.stt_timeout_s,
                )
            res.setdefault("available", True)
            return res, (time.perf_counter() - t0) * 1000, False
        except Exception as exc:  # timeout ham shu yerga tushadi
            log.warning("STT degradatsiya (case %s): %s", case_id, exc)
            await self._audit_fail(case_id, "stt", exc)
            return dict(_EMPTY_TRANSCRIPT), (time.perf_counter() - t0) * 1000, True

    def _consume_stt_stream(self, case_id: str, transcriber, audio_path: str) -> dict:
        """THREAD: transcribe_stream partiallarini stt_partial qilib push qiladi.
        Oxirgi (is_final=True) yield to'liq natijani beradi; bo'lmasa oxirgi partial."""
        final: dict | None = None
        last_text = ""
        for partial in transcriber.transcribe_stream(audio_path, self.s.stt_language):
            is_final = bool(partial.get("is_final"))
            last_text = partial.get("text", last_text)
            self._emit_threadsafe(
                EventType.STT_PARTIAL, case_id,
                {"text": partial.get("text"), "is_final": is_final,
                 "language": partial.get("language")},
            )
            if is_final:
                final = {
                    "text": partial.get("text"),
                    "language": partial.get("language"),
                    "confidence": partial.get("confidence"),
                    "available": True,
                }
        if final is None:  # provider final bermadi -> oxirgi partialdan tuzamiz
            final = {"text": last_text or None, "language": self.s.stt_language,
                     "confidence": None, "available": bool(last_text)}
        return final

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
        explainer = self.p["explainer"]
        streaming = isinstance(explainer, StreamingExplainer)
        t0 = time.perf_counter()
        attempts = self.s.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                async with self.orch.gpu_session():
                    if streaming:
                        # JONLI: tokenlar thread ichidan explanation_token bo'lib push.
                        res = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._consume_llm_stream, case_id, explainer,
                                detections, transcript, notes, risk,
                            ),
                            timeout=self.s.llm_timeout_s,
                        )
                    else:
                        res = await asyncio.wait_for(
                            asyncio.to_thread(
                                explainer.generate_explanation,
                                detections, transcript, notes, risk,
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

    def _consume_llm_stream(
        self, case_id: str, explainer, detections, transcript, notes, risk
    ) -> dict:
        """THREAD: generate_explanation_stream tokenlarini explanation_token qilib push."""
        tokens: list[str] = []
        for tok in explainer.generate_explanation_stream(detections, transcript, notes, risk):
            tokens.append(tok)
            self._emit_threadsafe(EventType.EXPLANATION_TOKEN, case_id, {"token": tok})
        return {
            "text": "".join(tokens),
            "generated_by": getattr(explainer, "model_version", None) or "stream-llm",
            "available": True,
        }

    # ---- Tier 2 (LLM) qaror: sozlanadigan flag (Tamoyil 7) ----
    def _should_run_llm(self, level: str) -> bool:
        """LLM (GPU) avtomatik ishlaydimi? Default: faqat HIGH. On-demand
        (POST /cases/{id}/explain) har doim mumkin (bu yerda emas)."""
        if self.s.llm_auto_always:
            return True
        return self.s.llm_auto_on_high and level == "HIGH"

    # ---- asosiy pipeline (tiered, event-driven) ----
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

        # STT (CPU, sekin bo'lishi mumkin) detection bilan PARALLEL boshlanadi,
        # lekin Tier 1'ni BLOKLAMAYDI — natijasi LLM'dan oldin kutiladi.
        stt_task = asyncio.create_task(self._run_stt(case_id, audio_path))

        # ===== Tier 1 (DARHOL, <1s): detection -> DETERMINISTIK risk -> PUSH =====
        detections, det_ms, det_failed = await self._run_detection(case_id, image_path)
        timings["detection"] = round(det_ms)
        await asyncio.to_thread(self.repo.save_detections, case_id, detections)
        if not det_failed:
            await self._audit(case_id, AuditAction.DETECTION_DONE, {"count": len(detections)})

        risk = await asyncio.to_thread(
            self.p["risk_engine"].compute_risk, detections, self.risk_config
        )
        await asyncio.to_thread(self.repo.save_risk, case_id, risk)
        await self._audit(
            case_id, AuditAction.RISK_COMPUTED, {"level": risk["level"], "score": risk["score"]}
        )
        # Tier 1 tayyor — operatorga DARHOL push (persist'dan KEYIN, izchillik uchun)
        self._emit(
            EventType.TIER1_DONE,
            case_id,
            {
                "risk": {"level": risk["level"], "score": risk["score"],
                         "computed_by": risk["computed_by"]},
                "detections": detections,
                "detection_failed": det_failed,
            },
        )
        if risk["level"] == "HIGH":
            self._emit(EventType.ALERT, case_id, {"level": "HIGH", "score": risk["score"]})

        # ===== STT natijasini kutamiz (LLM uchun kirish) =====
        transcript, stt_ms, stt_failed = await stt_task
        timings["stt"] = round(stt_ms)
        await asyncio.to_thread(self.repo.save_transcript, case_id, transcript)
        if transcript.get("available"):
            await self._audit(case_id, AuditAction.STT_DONE, {"language": transcript.get("language")})
        self._emit(
            EventType.STT_DONE,
            case_id,
            {"available": transcript.get("available", False),
             "language": transcript.get("language"),
             "text": transcript.get("text")},
        )
        degraded = degraded or det_failed or stt_failed

        # ===== Tier 2 (LLM, GPU): faqat flag bo'yicha (har skanga EMAS) =====
        if self._should_run_llm(risk["level"]):
            explanation, synth_ms, synth_failed = await self._run_synthesis(
                case_id, detections, transcript, operator_notes, risk
            )
            degraded = degraded or synth_failed
            await asyncio.to_thread(self.repo.save_explanation, case_id, explanation)
            if explanation.get("available"):
                await self._audit(
                    case_id, AuditAction.EXPLANATION_DONE,
                    {"generated_by": explanation.get("generated_by")},
                )
            self._emit(
                EventType.EXPLANATION_DONE,
                case_id,
                {"available": explanation.get("available", False),
                 "generated_by": explanation.get("generated_by"),
                 "text": explanation.get("text")},
            )
        else:
            # On-demand: hozir generatsiya YO'Q (degradatsiya emas) — operator
            # POST /cases/{id}/explain bilan so'rashi mumkin (4-bosqich).
            synth_ms = 0.0
            await asyncio.to_thread(self.repo.save_explanation, case_id, dict(_EMPTY_EXPLANATION))
            log.info("Case %s: LLM on-demand'ga qoldirildi (risk=%s)", case_id, risk["level"])
        timings["synthesis"] = round(synth_ms)

        # ===== yakunlash =====
        timings["total"] = round((time.perf_counter() - t_start) * 1000)
        await asyncio.to_thread(self.repo.set_timings, case_id, timings)
        if degraded:
            await asyncio.to_thread(self.repo.set_degraded, case_id, True)
        await asyncio.to_thread(
            self.repo.set_status, case_id, CaseStatus.DONE.value, completed=True
        )
        self._emit(
            EventType.CASE_DONE,
            case_id,
            {"status": CaseStatus.DONE.value, "risk_level": risk["level"], "degraded": degraded},
        )
        log.info("Case %s yakunlandi (risk=%s, degraded=%s, %dms)",
                 case_id, risk["level"], degraded, timings["total"])

    # ---- On-demand LLM (POST /cases/{id}/explain) ----
    def request_explain(self, case_id: str) -> bool:
        """LLM streaming'ni fon vazifa sifatida boshlaydi. Shu case allaqachon
        ishlovda bo'lsa False qaytaradi (takror so'rovni rad etish uchun)."""
        if case_id in self._explaining:
            return False
        self._explaining.add(case_id)
        task = asyncio.create_task(self._explain_task(case_id), name=f"explain-{case_id}")
        self._explain_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._explaining.discard(case_id)
            self._explain_tasks.discard(t)

        task.add_done_callback(_done)
        return True

    async def _explain_task(self, case_id: str) -> None:
        try:
            await self.explain_on_demand(case_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("On-demand explain xatosi (case %s)", case_id)
            self._emit(EventType.MODEL_FAILED, case_id, {"stage": "llm_on_demand", "error": str(exc)})

    async def explain_on_demand(self, case_id: str) -> None:
        """Mavjud case uchun LLM tushuntirishini JONLI stream qiladi (GPU, serial).

        Risk/detection allaqachon hisoblangan (Tier 1) — bu faqat tushuntirish
        matnini yozadi (Tamoyil 1/2: risk score O'ZGARMAYDI)."""
        case = await asyncio.to_thread(self.repo.get, case_id)
        if case is None:
            log.error("explain_on_demand: case %s topilmadi", case_id)
            return
        data = serialize_case(case)
        risk = data["risk"]
        if risk is None:
            log.warning("explain_on_demand: case %s'da risk yo'q (Tier 1 tugamagan)", case_id)
            self._emit(EventType.MODEL_FAILED, case_id, {"stage": "llm_on_demand", "error": "risk yo'q"})
            return
        detections = data["detections"]
        transcript = data["transcript"] or dict(_EMPTY_TRANSCRIPT)
        notes = case.operator_notes

        explanation, _ms, failed = await self._run_synthesis(
            case_id, detections, transcript, notes, risk
        )
        await asyncio.to_thread(self.repo.save_explanation, case_id, explanation)
        if explanation.get("available"):
            await self._audit(
                case_id, AuditAction.EXPLANATION_DONE,
                {"generated_by": explanation.get("generated_by"), "on_demand": True},
            )
        if failed:
            await asyncio.to_thread(self.repo.set_degraded, case_id, True)
        self._emit(
            EventType.EXPLANATION_DONE, case_id,
            {"available": explanation.get("available", False),
             "generated_by": explanation.get("generated_by"),
             "text": explanation.get("text"), "on_demand": True},
        )
