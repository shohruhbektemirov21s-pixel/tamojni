# Offline Bojxona AI Tizimi

Windows uchun **100% offline, air-gapped, real-time** AI yordamchi. Skanerdan kelayotgan
X-ray rasmlar darhol baholanadi, ovoz jonli matnga aylanadi, tushuntirish token-streaming
bilan keladi, matn ovozga (TTS) qaytariladi.

Maqsadli hardware (Faza 1): **RTX 2050 4GB + Ryzen 5 7000**.

## Buzilmas tamoyillar

1. **AI qaror qabul qilmaydi** — signal + tushuntirish beradi, qaror operatorda (human-in-the-loop).
2. **Risk score deterministik** — rule engine hisoblaydi, LLM emas.
3. **Audit-first** — har event append-only audit jurnalida.
4. **Offline by design** — tashqi tarmoq yo'q (CI'da no-network test).
5. **Client ajralgan** — faqat HTTP/WebSocket.
6. **Degradatsiya, fail emas** — bitta model yiqilsa, case tashlanmaydi.
7. **VRAM yagona ega** — GPU faqat Ollama LLM'niki; YOLO/Whisper CPU'da.

## Repozitoriya tarkibi

| Katalog | Tavsif |
|---|---|
| `customs-ai/backend/` | FastAPI event-driven core: event bus, WebSocket `/ws`, real-time pipeline, audit, GPU orchestrator (SQLAlchemy + SQLite WAL). |
| `customs-ai/desktop/` | Operator UI (Electron + React + TypeScript). |
| `customs-ai/ml/` | Vision (YOLO/ONNX) trening va eksport. |
| `customs-ai/stt/` | Nutq-matn (Whisper) trening/baholash. |
| `customs-ai/deploy/` | Offline Windows paketlash (NSSM + offline wheels). |

## Backend ishga tushirish

```bash
cd customs-ai/backend
pip install -r requirements.txt
python -m pytest -q          # testlar
uvicorn app.main:app         # 127.0.0.1:8000 (faqat loopback)
```

Mock provayderlar bilan butun pipeline modelsiz ishlaydi (`CUSTOMS_USE_MOCKS=true`,
default). Real modellar tayyor bo'lganda `CUSTOMS_USE_MOCKS=false`.
