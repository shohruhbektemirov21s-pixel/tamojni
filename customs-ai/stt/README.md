# STT Track — Speech-to-Text (EGASI: Dev 3)

Runtime (`backend/app/pipelines/speech.py`) Dev 1'ni allaqachon unblock qildi.
Bu papka **o'zbek tili sifatini** ko'taradigan ML track (ROADMAP §16, parallel).
Skriptlar **train mashinasida** ishlaydi (offline target'da emas).

## Eng katta haqiqat: o'zbek tili sifati
Whisper **rus tilida a'lo** (WER ≤ 15% real), lekin **o'zbekда past** —
lotin/kiril aralash, dialektlar, bojxona terminlari. Shuning uchun:
- **Rus**: base "medium" yetarli. Fine-tune SHART EMAS (qayta o'qitish buzishi mumkin).
- **O'zbek**: **LoRA fine-tune SHART** (kam-data, kam-resurs uchun barqaror).

## Pipeline (ketma-ketlik)
| skript | vazifa |
|--------|--------|
| `prepare_uz_data.py`       | Common Voice (uz) + on-prem audio -> JSONL manifest + split |
| `finetune_whisper_lora.py` | Whisper LoRA/PEFT fine-tune (FAQAT o'zbek) |
| `convert_ct2.py`           | base+LoRA merge -> CTranslate2 int8 (CPU runtime) |
| `evaluate_wer.py`          | WER (ru<=15% gate; uz baseline) — yadro pure-python |

## Ma'lumot manbalari
- **Common Voice (uz)** — CC0, ⚠️ Sprint 0'da litsenziya tasdig'i (`data/README.md`).
- **Real operator audio** — FAQAT on-prem/offline (maxfiy; repo'ga/tarmoqqa chiqmaydi).
- Bojxona domeni terminlari uchun real audio kritik (Common Voice umumiy nutq).

## Qabul mezoni (§15)
- **rus WER ≤ 0.15** — `evaluate_wer.py --lang ru` gate (qizil/yashil).
- **o'zbek** — baseline o'lchanadi; fine-tune'dan keyin yaxshilanish ko'rsatiladi.

## Real ru WER smoke (`smoke_wer_ru.py`)
`test_wer.py` WER **yadrosini** tekshiradi (modelsiz). `smoke_wer_ru.py` esa BUTUN
production yo'lini o'lchaydi: real audio → `WhisperTranscriber.transcribe()` (Dev 1
chaqiradigan aynan shu kontrakt) → `wer`/`check_gate`.
```
python stt/smoke_wer_ru.py --generate --model small   # gTTS fixture + o'lchov
python stt/smoke_wer_ru.py --manifest <real_audio.jsonl> --model medium  # real audio
```
**Natija (gTTS toza audio, 6 utterance, ё-fold bilan):** small WER **2.5%**,
medium WER ≈ 2–5% — gate ✅ PASS.

⚠️ **gTTS toza audio = OPTIMISTIK chegara.** Bu measurement harness'ni va kod yo'lini
tasdiqlaydi, lekin dala/operator audio shovqinli — production gate `--manifest` orqali
**real operator audio** bilan o'lchanishi shart.

### Smoke kuzatuvlari (WER adolatliligi)
1. **ё↔е** — `normalize_text` endi ё→е birlashtiradi (rus yozuvida almashinadi). ✅ qilindi.
2. **raqam↔so'z** — Whisper `«две тысячи»` ni `«2000»` deb yozadi → WER nohaq oshadi.
   Bojxonada summalar/og'irliklar tez-tez uchraydi. TAVSIYA: WER normalizatsiyasiga
   raqam↔so'z kanonizatsiyasi (masalan `num2words`, opsional dep) qo'shilsin — bu
   yadroni "deps-siz" saqlash uchun graceful-fallback bilan. Hali QILINMAGAN (jamoa
   dep qarori). Real ASR aniqligi toza audioda amalda mukammal — "xatolar" matn-form.

## Runtime'ga ulash (fine-tune tugagach)
```
python stt/convert_ct2.py --adapter stt/runs/uz_lora --out stt/models/uz_medium_ct2
# backend/.env
CUSTOMS_USE_MOCKS=false
CUSTOMS_STT_MODEL_PATH=stt/models/uz_medium_ct2   # 100% offline (local_files_only)
```

## O'rnatish (train mashinasi)
```
pip install -r stt/requirements-train.txt
```
