# Windows offline installer (skeleton)

> ⚠️ Arxitektura §16: offline packaging eng past baholangan ko'mik. Sprint 0'da
> target mashinada **boshidan** sinab ko'ring (CUDA/Ollama/wheels mosligi).

## O'rnatish bosqichlari (air-gapped Windows)

1. **Python 3.11** o'rnatish (offline installer).
2. **venv + bog'liqliklar** — `deploy/offline-wheels/README.md` ga qarang.
3. **Ollama** offline o'rnatish + GGUF modellarni joylashtirish (`qwen3:4b`).
4. **Model weights** (`backend/models_weights/`) — YOLO ONNX, Whisper CT2.
5. **Service** — `install-service.ps1` (NSSM, auto-start, loopback bind).
6. **Smoke test** — `http://127.0.0.1:8000/health` → `{"status":"ok"}`.

## Arxitektura cheklovlari

- Backend faqat `127.0.0.1` ga bind (Faza 1 auth).
- GPU yagona egasi: Ollama (ADR-002). PyTorch GPU'da ishlatilmaydi.
- Hech qanday komponent tashqi tarmoqqa chiqmaydi (Tamoyil 4).
