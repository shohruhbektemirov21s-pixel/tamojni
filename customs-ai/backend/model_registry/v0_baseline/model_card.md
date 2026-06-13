# Model Card — v0_baseline (BOOTSTRAP / POC)

> ⚠️ Bu **POC validatsiya** modeli, PRODUCTION EMAS. Ochiq X-ray datasetlarda
> o'qitilgan; O'zbekiston bojxona real skaneridagi domain'ga to'liq mos kelmaydi
> (xavf #1: X-ray domain gap). Operator qarorida ishonch manbasi sifatida
> foydalanilMAYDI — faqat pipeline'ni uchidan-uchiga sinash uchun.

## Asosiy ma'lumot
- **Arxitektura**: YOLOv11 (n/s) -> ONNX (opset 12), imgsz 640
- **Runtime**: ONNX Runtime, CPU yoki DirectML (ADR-002 — CUDA EMAS)
- **Class'lar**: `labels.txt` ga qarang (taxonomy.yaml bilan sinxron)
- **Train datasetlari**: SIXray + PIDray + OPIXray (bootstrap, litsenziya Sprint 0)
- **Eksport**: `python ml/export_onnx.py --weights runs/.../best.pt`

## Metrikalar (eval set; ml/evaluate.py — deterministik)
| class              | precision | recall | qayd |
|--------------------|-----------|--------|------|
| qurol              | TBD       | TBD    | ⚠️ RECALL >= 0.95 TALAB (qabul mezoni) |
| o'q-dori           | TBD       | TBD    | |
| narkotik_paket     | TBD       | TBD    | datasetda kam — o'z datasetda kuchaytiriladi |
| pichoq             | TBD       | TBD    | OPIXray boy |
| valyuta_toplami    | n/a       | n/a    | ochiq datasetda yo'q — o'z datasetda |
| organik_anomaliya  | n/a       | n/a    | o'z datasetda |
| yashirilgan_boshliq| n/a       | n/a    | o'z datasetda |

## Ma'lum cheklovlar
- valyuta/organik/yashirin-bo'shliq class'lari ochiq datasetda YO'Q —
  bularni faqat o'z (on-prem) datasetimizdan o'rgatamiz.
- Dual-energy psevdo-rang taqsimoti dataset bo'yicha farq qiladi (kalibrlash kerak).
- Recall qabul gate'idan o'tmaguncha production'ga CHIQARILMAYDI.

## Yaxlitlik
- `sha256.txt`: model.onnx ning SHA-256 (offline tekshiruv).
