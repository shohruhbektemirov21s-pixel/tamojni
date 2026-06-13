# Backend Core — Offline Bojxona AI

> Tizimning umurtqa pog'onasi. Dev 2/3/4 shu yerga ulanadi. Arxitektura v2.0 ga amal qiladi.

## Tez boshlash (dev, mock modellar bilan)

```bash
cd customs-ai/backend
# bog'liqliklar (offline: deploy/offline-wheels/README.md)
pip install -r requirements.txt

# ishga tushirish (mock provayderlar bilan, GPU/model shart emas)
python -m app.main
#  -> http://127.0.0.1:8000/health
#  -> http://127.0.0.1:8000/docs   (OpenAPI auto-doc)

# testlar
python -m pytest -q
```

## Arxitektura xaritasi (qaysi fayl qaysi qarorni bajaradi)

| Talab / ADR | Fayl |
|-------------|------|
| API endpointlari (§7) | `app/api/health.py`, `app/api/cases.py`, `app/api/decisions.py` |
| Yagona xatolik formati `{"error":{...}}` | `app/core/errors.py` + `app/main.py` (handlers) |
| Case Worker (asyncio, GPU serial, parallel STT‖detection) | `app/core/worker.py` |
| GPU Orchestrator = process supervisor (ADR-001) | `app/core/orchestrator.py` |
| GPU yagona ega, PyTorch GPU'da emas (ADR-002) | `orchestrator.py` (lock) + `pipelines/detection.py` (ONNX) |
| Deterministik risk, LLM'dan oldin (Tamoyil 2) | `app/core/worker.py` + `pipelines/scoring.py` |
| DB schema, ORM, SQLite WAL (§8, ADR-006) | `app/models/entities.py`, `app/db.py`, `app/repositories/` |
| Audit append-only (Tamoyil 3) | `app/services/audit_service.py` + `app/models/guard.py` |
| Degradatsiya (§10, Tamoyil 6) | `app/core/worker.py` (`_run_*`) |
| Startup recovery | `repositories/case_repository.py::recover_stuck` + `main.py` lifespan |
| Offline by design (Tamoyil 4) | `app/core/netguard.py` + `tests/test_no_network.py` |
| Config (YAML+env) | `app/core/config.py`, `config/*.yaml` |

## Integratsiya kontraktlari (Dev 2/3/4)

Rasmiy interfeyslar: **`app/pipelines/contracts.py`** (Protocol). Backend ularni
chaqiradi; siz implementatsiya qilasiz. Mock'lar `app/pipelines/mocks.py` da —
ular tayyor bo'lguncha tizim to'liq ishlaydi.

```python
# Dev 2 — Vision (app/pipelines/detection.py, scoring.py)
def detect(image_path: str) -> list[dict]:
    # [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
def compute_risk(detections: list[dict], config: dict) -> dict:
    # {"level","score","computed_by","factors":[...]}   # DETERMINISTIK

# Dev 3 — STT (app/pipelines/speech.py)
def transcribe(audio_path: str, language: str | None) -> dict:
    # {"text","language","confidence","available"}

# Dev 4 — LLM (app/pipelines/synthesis.py)
def generate_explanation(detections, transcript, operator_notes, risk) -> dict:
    # {"text","generated_by","available"}   # FAQAT tushuntirish; risk'ni o'zgartirmaydi
```

**Real provayderlarni ulash:** `app/core/providers.py::build_providers` ichida
TODO bloki bor; tayyor bo'lganda mock o'rniga real klassni qo'ying va
`CUSTOMS_USE_MOCKS=false` qiling.

### Dev 2 — Vision vertikali (TAYYOR: runtime; DAVOM ETMOQDA: ML track)

- **Risk Engine** (`pipelines/scoring.py`): production `RuleEngine` — 100%
  deterministik noisy-OR, config-driven (`config/risk_rules.yaml`), recall-favoring
  per-class floor, auditga tayyor `factors[]` (`below_confidence_floor` izi bilan).
  `computed_by="rule_engine_v1"`. Mock rejimida ham RuleEngine REAL ishlatiladi.
- **Detector** (`pipelines/detection.py`): `OnnxYoloDetector` — ONNX Runtime
  (ADR-002: DirectML→CPU, CUDA EMAS), X-ray letterbox + sof-numpy NMS
  (`pipelines/yolo_postprocess.py`). Model `model_registry/<v>/model.onnx` +
  `labels.txt` dan. Model yo'q bo'lsa → mock fallback (jamoa bloklanmaydi).
- **Yoqish:** `CUSTOMS_USE_MOCKS=false`,
  `CUSTOMS_YOLO_MODEL_PATH=model_registry/<v>/model.onnx`. Vision deps:
  `pip install -r requirements-vision.txt`.
- **ML track** (`../ml/`): bootstrap (SIXray/PIDray/OPIXray) → `train_baseline.py`
  → `evaluate.py` (qurol recall ≥ 0.95 gate) → `export_onnx.py` → versiyalangan
  `model_registry/`. ⚠️ Domain gap (xavf #1) sabab production model o'z
  on-prem datasetimizda. Litsenziya tekshiruvi: `ml/datasets/README.md` (Sprint 0).

### Dev 3 — STT vertikali (TAYYOR: runtime; DAVOM ETMOQDA: o'zbek fine-tune)

- **Transcriber** (`pipelines/speech.py`): `WhisperTranscriber` — faster-whisper
  (CTranslate2), `device=cpu`, `compute_type=int8` (ADR-002 — GPU Ollama'niki,
  detection bilan PARALLEL). ru asosiy + uz ikkilamchi (yoki auto-detect), Silero
  VAD. Confidence = davomiylik-vaznlangan `exp(avg_logprob)` (`whisper_utils.py`).
- **Ikki rejim**: (1) fayl — `transcribe(audio_path, language)`; (2) near-real-time
  — `transcribe_realtime(...)` generator (VAD chunking + overlap merge, qisman
  natija yield).
- **Degradatsiya**: qattiq xatoda Exception → worker `_run_stt` tutadi, MODEL_FAILED
  audit + `{available: false}` (case TO'XTAMAYDI). Sukunat ≠ xato (text="", available=true).
- **Yoqish**: `CUSTOMS_USE_MOCKS=false`, `CUSTOMS_STT_MODEL_PATH=<lokal ct2 model>`
  (100% offline, `local_files_only`). Deps: `pip install -r requirements-stt.txt`.
- **ML track** (`../stt/`): Common Voice(uz)+on-prem → `finetune_whisper_lora.py`
  → `convert_ct2.py` → `evaluate_wer.py` (ru WER ≤ 15% gate; uz baseline).
  ⚠️ Whisper o'zbekда past — LoRA fine-tune SHART; rus uchun shart emas.

## API qisqacha (§7)

| Method | Endpoint | Izoh |
|--------|----------|------|
| GET | `/health` | backend + model holati |
| POST | `/cases` | multipart: `image` (majburiy), `audio?`, `notes?`, `operator_id?` → 201 |
| GET | `/cases/{id}` | Case Result (§7.1) |
| GET | `/cases/{id}/stream` | SSE: PENDING→PROCESSING→DONE/FAILED |
| POST | `/cases/{id}/decision` | `{decision: CONFIRM\|REJECT\|OVERRIDE, notes?}` |
| GET | `/cases` | ro'yxat (filter: status, risk_level; pagination) |
| GET | `/cases/{id}/audit` | to'liq audit trail |

Barcha xatolik: `{"error": {"code", "message", "detail"}}`.

## Qabul mezonlari — holati

- ✅ Bo'sh case e2e (mock) oqadi, har bosqich audit'ga yoziladi — `tests/test_e2e_case.py`
- ✅ LLM/STT yiqilsa ham case to'liq risk bilan yakunlanadi (degraded/available) — `tests/test_degradation.py`
- ✅ No-network test (tashqi socket = fail) — `tests/test_no_network.py`
- ✅ Audit append-only — `tests/test_audit_append_only.py`
- ✅ Startup recovery, OpenAPI auto-doc — `tests/test_api_and_recovery.py`

```
21 passed
```
