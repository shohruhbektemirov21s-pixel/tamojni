# ML Track — X-ray vision (EGASI: Dev 2)

> Bu **uzunroq parallel track** (ROADMAP §16). Runtime MVP (detection.py +
> scoring.py) Dev 1'ni allaqachon unblock qildi; bu track real domain modelni
> yetishtiradi. Bu skriptlar **train mashinasida** ishlaydi (offline target'da emas).

## Eng katta xavf: X-ray domain gap (xavf #1)
Bojxona skaneri **dual-energy X-ray** (psevdo-rang: to'q sariq=organik, ko'k=metall)
chiqaradi — RGB foto EMAS. COCO/tabiiy rasmda o'qitilgan modellar bu yerda
ishonchsiz. Shuning uchun **domain fine-tuning SHART**. Generic modelга ishonmaymiz.

## Bosqichlar

### 1. Bootstrap (POC validatsiya) — ochiq datasetlar
- **SIXray, PIDray, OPIXray** da baseline YOLOv11 o'qitamiz.
- ⚠️ **Litsenziya Sprint 0'da tekshiriladi** (`datasets/README.md`). Faqat
  bootstrap/POC — PRODUCTION'da ishlatilMAYDI.
- Maqsad: pipeline uchidan-uchiga ishlashini, recall metodikasini tekshirish.

### 2. Taksonomiya (`config/taxonomy.yaml`)
- O'zbekiston bojxona qoidalariga ko'ra taqiqlangan class'lar.
- **Ekspert bilan tasdiqlanadi** (huquqiy baza). Hozircha POC reference.

### 3. O'z datasetimiz (production manbasi)
- **Real skanerdan eksport** — faqat **offline/on-prem** (maxfiy ma'lumot).
- Labellash: **CVAT yoki Label Studio (self-hosted)** — cloud labeling YO'Q.
- **Class imbalance** -> **hard-negative mining**: noto'g'ri ijobiy/salbiylarni
  yig'ib, keyingi round'ga qo'shamiz (ayniqsa qurol false-negative'larini).
- valyuta/organik/yashirin-bo'shliq class'lari faqat shu yerdan keladi
  (ochiq datasetda yo'q).

## Skriptlar (ketma-ketlik)
| skript | vazifa |
|--------|--------|
| `sync_labels.py`    | taxonomy.yaml -> model_registry/<v>/labels.txt (yagona manba) |
| `prepare_dataset.py`| ochiq/o'z datasetni YOLO formatiga + label map qilish |
| `train_baseline.py` | Ultralytics YOLOv11 fine-tune (recall-favoring) |
| `evaluate.py`       | DETERMINISTIK per-class recall; qurol recall >= 0.95 gate |
| `export_onnx.py`    | best.pt -> model.onnx (opset 12) + sha256 + labels nusxasi |
| `benchmark_latency.py` | REAL-TIME latency gate (qabul mezoni #1: detect() < 300ms); p50/p95, bosqichlarga ajratadi |

## Latency gate (qabul mezoni #1: real-time kritik yo'l)
`detect()` < 300ms/rasm bo'lishi SHART (operator har skanga darhol natija ko'radi).
`benchmark_latency.py` buni p95 bilan tekshiradi va bosqichlarga ajratadi
(preprocess / inference / postprocess) — bottleneck ko'rinsin:
```
# To'liq (real model + onnxruntime):
python ml/benchmark_latency.py --model backend/model_registry/v0_baseline/model.onnx \
    --image sample_xray.png --iters 50 --warmup 5 --budget-ms 300
# Modelsiz (CI smoke): sof-numpy pre/post-process regressiyasini ushlaydi.
python ml/benchmark_latency.py
```
Inference budjeti model kelganda to'liq o'lchanadi; pre/post-process (men egalik
qiladigan numpy qism) ~11ms p95 — budjetning kichik ulushi.

## Qabul gate'i (release oldidan)
- **qurol RECALL >= 0.95** (`evaluate.py` qizil/yashil qaytaradi).
- Model `model_registry/<versiya>/` ga model_card bilan joylashadi.
- model.onnx git'da emas — offline yetkaziladi + sha256 tekshiriladi.

## O'rnatish (train mashinasi)
```
pip install -r ml/requirements-train.txt
```
