# CLAUDE.md — Offline Bojxona AI Tizimi

> Repo ildizida turadi. Bu — kontraktlar va qoidalarning **yagona manbai**. Kontraktlarni o'zgartirsang, shu faylni yangila.

## Loyiha
Windows + Linux uchun 100% **offline, air-gapped, real-time** AI yordamchi: X-ray skaner rasmidan kontrabanda ehtimolini baholaydi, ovozni (uz/ru) jonli matnga, matnni ovozga (TTS) aylantiradi, izohli hisobot beradi. Hardware (Faza 1): RTX 2050 4GB + Ryzen 5 7000.

## Buzilmas tamoyillar (HAR DOIM)
1. **AI qaror qabul qilmaydi** — signal + tushuntirish beradi, qaror operatorda.
2. **Risk score DETERMINISTIK** (rule engine) — LLM hisoblamaydi/o'zgartirmaydi.
3. **Audit-first** — har event append-only jurnalda (UPDATE/DELETE taqiq, attachment sha256).
4. **Offline by design** — kod tashqi tarmoqqa chiqmaydi (no-network test = CI gate).
5. **Client/backend ajralgan** — faqat HTTP/WebSocket.
6. **Degradatsiya, fail emas** — model yiqilsa qisman natija + `available=false`.
7. **VRAM yagona egasi** — GPU faqat Ollama LLM'niki (YOLO CPU'da).

## Asosiy qarorlar (ADR)
- Risk = noisy-OR rule engine; weights/thresholds `config/risk_rules.yaml`, `config/taxonomy.yaml` dan (kod emas).
- LLM (Qwen3 4B, Ollama) — **on-demand / HIGH-flagged, har skanga EMAS** (config flag; default shu).
- YOLO detection — ONNX Runtime **CPU/DirectML**, GPU EMAS.
- Qwen2.5-VL (VLM) — Faza 1'da **YO'Q** (kechiktirilgan).
- STT — faster-whisper CPU (uz: `rubaiSTT`→ct2); TTS — MMS-TTS uz CPU. Lotin+kiril.
- DB — SQLite (WAL) + SQLAlchemy ORM, **Postgres-uyumli** (SQLite-spetsifik SQL yo'q).
- GPU Orchestrator = **process supervisor** (model loader emas).
- Scanner — `WatchedFolderSource` (default) + pluggable adapter.
- Desktop — **Electron + React/TS** (cross-platform: Linux + Windows).

## Real-time modeli
Event-driven + streaming + push. **Polling YO'Q.**
- **Tier 1** (har skanga, <1s): detect + risk + alert → darhol push.
- **Tier 2** (async, bloklamaydi): LLM token-stream + TTS, on-demand/HIGH.

## Vertikallar (ownership — o'z chegarangda qol)
| Dev | Egasi |
|-----|-------|
| 1 | Backend event-driven core (FastAPI, event bus, /ws, orchestrator, DB, audit) |
| 2 | Vision: detection (ONNX, CPU) + deterministik risk engine |
| 3 | Speech: STT (uz to'liq) + streaming + TTS |
| 4 | LLM: streaming tushuntirish (score'ni emas, faqat matn) |
| 5 | Desktop: Electron live UI |
| 6 | QA: acceptance gate, chaos, e2e, ML-eval |

## Integratsiya kontraktlari (UMUMIY SEAM — o'zboshimchalik bilan o'zgartirma)
```python
# Komponent interfeyslari:
detect(image_path) -> list[dict]                                # Dev2
compute_risk(detections, config) -> dict                        # Dev2
transcribe(audio_path, language) -> dict                        # Dev3
transcribe_stream(audio_chunks, language) -> Iterator[dict]     # Dev3
synthesize_speech(text, language="uz", out_path=None) -> dict   # Dev3
generate_explanation_stream(detections, transcript, notes, risk) -> Iterator[str]  # Dev4
generate_explanation(detections, transcript, notes, risk) -> dict                  # Dev4 (fallback)
```
```
# Data tiplari:
Detection   = {"class","confidence","bbox":[x1,y1,x2,y2]}
RiskResult  = {"level":"LOW|MEDIUM|HIGH","score","computed_by","factors":[...]}
Transcript  = {"text","language","confidence","available"}
Explanation = {"text","generated_by","available"}
```
```
# WebSocket /ws (push, multi-client):
{"event":"scan_ingested","case_id"}
{"event":"tier1_done","case_id","result":{detections,risk}}
{"event":"stt_partial","case_id","text"}
{"event":"stt_done","case_id","transcript":{...}}
{"event":"explanation_token","case_id","token"}
{"event":"explanation_done","case_id","explanation":{...}}
{"event":"tts_ready","case_id","audio_url"}
{"event":"alert","case_id","level"}
```
**REST:** `POST /cases` (multipart fallback), `GET /cases`, `GET /cases/{id}`, `GET /cases/{id}/audit`, `POST /cases/{id}/decision`, `POST /cases/{id}/explain`, `POST /tts`.
**Xato formati:** `{"error":{"code","message","detail"}}`.
**Case Result:** `computed_by` / `generated_by` / `available` / `degraded` maydonlari **MAJBURIY** (provenance).

## Acceptance gate (RELEASE BLOCKER)
No-network (tashqi socket = fail) · risk determinizm 100% · Tier1 < 1s · YOLO recall(qurol) ≥ 0.95 · ru WER ≤ 15% · degradatsiya qamrovi.

## Konvensiyalar
- Modellar oldindan yuklab olinadi → `model_registry/` (read-only); runtime offline.
- **CPU:** detection, STT, TTS. **GPU:** faqat Ollama LLM.
- Har o'zgarishdan keyin **commit**; testlar yashil bo'lsin; katta bosqichlar orasida `/clear`.
- Repo: `backend/app/{api,core,pipelines,services,models,repositories}`, `desktop/` (Electron), `ml/`, `deploy/`, `docs/`.
- Taxmin noaniq bo'lsa — ochiq ayt, davom et. Production-grade kod, nazariy emas.
