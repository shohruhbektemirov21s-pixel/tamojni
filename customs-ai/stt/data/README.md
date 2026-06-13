# STT Datasets — litsenziya va manba (EGASI: Dev 3)

> ⚠️ **SPRINT 0 BLOCKER**: har manbani ishlatishdan OLDIN litsenziyani tekshiring.
> Bu papka git'da kuzatilmaydi (`.gitignore`). Audio/transkript hech qachon
> repo'ga yoki tashqi tarmoqqa chiqmaydi (Tamoyil 1: offline).

## Manbalar

| manba | til | litsenziya | tijorat/davlat? | holat |
|-------|-----|------------|------------------|-------|
| Mozilla Common Voice (uz) | uz | CC0-1.0 | ✅ ruxsat (CC0) | ⚠️ versiyani tasdiqlang |
| Real operator audio | ru/uz | ichki (maxfiy) | on-prem only | yig'iladi |

> Common Voice CC0 — erkin, lekin **bojxona domeni** terminlari yo'q (umumiy nutq).
> Domen sifati uchun real operator audiosi SHART (on-prem labellash).

## Manifest formati (JSONL)
```
{"audio": "/abs/yo'l/clip.wav", "text": "referens transkript", "source": "common_voice_uz"}
```

## Workflow
```
python stt/prepare_uz_data.py --cv-tsv cv/uz/validated.tsv --cv-clips cv/uz/clips --out stt/data
python stt/prepare_uz_data.py --onprem-dir /secure/operator_audio --out stt/data
# -> uz_train.jsonl / uz_val.jsonl / uz_test.jsonl
```
