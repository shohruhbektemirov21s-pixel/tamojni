# Model Registry — versiyalangan vision artefaktlari (EGASI: Dev 2)

Bu papka YOLOv11 -> ONNX runtime artefaktlarini **versiyalangan** holda saqlaydi.
Backend `Settings.yolo_model_path` orqali aniq versiyaga ishora qiladi —
"latest" YO'Q (auditda qaysi model qaror berganini aniq bilish shart).

## Tuzilish

```
model_registry/
  README.md
  v0_baseline/            # bootstrap POC (SIXray/PIDray/OPIXray fine-tune)
    labels.txt            # class INDEKSI -> nom (taxonomy.yaml bilan sinxron)
    model_card.md         # qaysi dataset, metrika, recall, cheklovlar
    model.onnx            # ⚠️ git'da YO'Q (katta binar; offline yetkaziladi)
    sha256.txt            # artefakt yaxlitligi (offline tekshiruv)
```

## Qoidalar

1. **Immutable**: chiqarilgan versiya o'zgartirilmaydi; yangi model = yangi papka.
2. **labels.txt** `config/taxonomy.yaml` dan generatsiya qilinadi
   (`python ml/sync_labels.py`). Qo'lda tahrirlanmaydi.
3. **model.onnx** git'ga commit qilinMAYDI (`.gitignore`). Offline-installer
   yoki `model_registry/<v>/` ga qo'lda nusxalanadi; `sha256.txt` bilan tekshiriladi.
4. Har model **model_card.md** bilan keladi: train dataset, sana, per-class
   recall/precision, ayniqsa **qurol recall (>= 0.95 talab)**, ma'lum cheklovlar.
5. **Litsenziya**: ochiq datasetlar (SIXray/PIDray/OPIXray) FAQAT bootstrap/POC
   uchun. Production model O'ZIMIZ yig'gan/labellangan datasetda o'qitiladi.

## Ishlatish

```
# backend/.env
CUSTOMS_USE_MOCKS=false
CUSTOMS_YOLO_MODEL_PATH=model_registry/v0_baseline/model.onnx
```

model.onnx mavjud bo'lmasa, provayder xavfsiz tarzda MOCK detector'ga tushadi
(jamoa bloklanmaydi).
