# Offline pip wheels (air-gapped Windows o'rnatish)

Air-gapped mashinada `pip install` internet talab qiladi — shuning uchun wheel'larni
**internetli mashinada** oldindan yuklab, target mashinaga (USB/diskda) ko'chiramiz.

## 1. Internetli mashinada (bir martalik)

Target = Windows x64 + Python 3.11. `download_wheels.sh` (yoki Windows'da `.ps1`):

```bash
pip download -r ../../backend/requirements.txt \
    -d ./wheels \
    --platform win_amd64 \
    --python-version 3.11 \
    --only-binary=:all:
```

> Eslatma: ba'zi paketlar pure-python (`--only-binary` ularga ta'sir qilmaydi).
> `--platform` mos kelmasa, target Python versiyasini aniq tekshiring.

## 2. Air-gapped Windows mashinada

`wheels/` papkasini ko'chiring va internetsiz o'rnating:

```powershell
python -m venv C:\customs-ai\venv
C:\customs-ai\venv\Scripts\pip install --no-index --find-links wheels -r requirements.txt
```

## 3. Modellar (alohida artefaktlar)

- Ollama GGUF modellari (`qwen3:4b`) — `ollama pull` internetli mashinada, keyin
  `~/.ollama/models` ni ko'chirish (yoki offline import).
- YOLO ONNX weights — `ml/` training track'dan versiyalangan artefakt sifatida.
- faster-whisper modeli — CTranslate2 formatda oldindan yuklab.

Hech qanday model runtime'da internetdan yuklanmaydi (Tamoyil 4).
