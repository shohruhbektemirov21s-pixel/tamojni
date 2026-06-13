#!/usr/bin/env python3
"""Real ru WER smoke — end-to-end gate validatsiyasi (§15: rus WER <= 0.15).

Sintetik unit-test (test_wer.py) WER YADROSINI tekshiradi. Bu skript esa BUTUN
production yo'lini o'lchaydi:
    real audio  ->  app.pipelines.speech.WhisperTranscriber.transcribe()  (KONTRAKT)
                ->  evaluate_wer.wer / check_gate  (mavjud yadro)
Ya'ni model inference + audio dekodlash + kontrakt dict'i + WER — reimplementatsiya
EMAS, aynan Dev 1 chaqiradigan funksiya.

⚠️ HALOL CHEKLOV: --generate gTTS bilan TOZA (studio) audio yasaydi. Bu measurement
harness'ni va kod yo'lini tasdiqlaydi va real WER raqamini beradi, lekin ENG YAXSHI
HOLAT (optimistik) chegara — dala/operator audio shovqinli, WER yuqoriroq bo'ladi.
Production gate'i real operator audio manifesti bilan o'lchanishi shart.

Foydalanish:
    # 1) fixture yaratish (internet kerak — gTTS; faqat dev mashinada)
    python stt/smoke_wer_ru.py --generate --fixtures-dir stt/data/smoke_ru
    # 2) real model bilan o'lchash (offline; model oldindan yuklangan/yuklab olinadi)
    python stt/smoke_wer_ru.py --manifest stt/data/smoke_ru/manifest.jsonl \
        --model small --lang ru
    # ikkalasi birga:
    python stt/smoke_wer_ru.py --generate --model small
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# --- repo modullarini import qilish uchun yo'llarni sozlash ---
_HERE = Path(__file__).resolve().parent          # .../customs-ai/stt
_ROOT = _HERE.parent                              # .../customs-ai
sys.path.insert(0, str(_ROOT / "backend"))        # -> app.pipelines.speech
sys.path.insert(0, str(_HERE))                    # -> evaluate_wer

from evaluate_wer import check_gate, corpus_wer, normalize_text, wer  # noqa: E402

# Bojxona/operator domeni ru jumlalar (ma'lum reference). Domenga yaqin terminlar.
_RU_SENTENCES: list[str] = [
    "Откройте багажное отделение для досмотра.",
    "В посылке обнаружены предметы, запрещённые к перевозке.",
    "Пассажир декларирует две тысячи долларов наличными.",
    "На рентгеновском снимке виден предмет, похожий на нож.",
    "Груз следует из Ташкента в Москву транзитом.",
    "Предъявите таможенную декларацию и документы на товар.",
]


def _gen_fixtures(fixtures_dir: Path) -> Path:
    """gTTS -> mp3 -> ffmpeg -> wav 16k mono. Manifest JSONL qaytaradi."""
    try:
        from gtts import gTTS
    except ImportError:
        raise SystemExit("gTTS yo'q: pip install gTTS (faqat fixture yaratish uchun)")

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    manifest = fixtures_dir / "manifest.jsonl"
    rows: list[dict] = []
    for i, text in enumerate(_RU_SENTENCES):
        mp3 = fixtures_dir / f"ru_{i:02d}.mp3"
        wav = fixtures_dir / f"ru_{i:02d}.wav"
        print(f"[gen {i+1}/{len(_RU_SENTENCES)}] {text!r}")
        gTTS(text=text, lang="ru").save(str(mp3))
        # 16 kHz mono PCM — Whisper kutadigan format (faster-whisper o'zi resample
        # qiladi, lekin normallashtirilgan fixture deterministik).
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", "16000", "-ac", "1", str(wav)],
            check=True,
        )
        mp3.unlink(missing_ok=True)
        rows.append({"audio": str(wav), "text": text})
    with manifest.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> manifest: {manifest} ({len(rows)} ta)")
    return manifest


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_transcriber(model: str, model_path: str | None, download_root: str | None):
    """REAL production transkriber (Dev 1 aynan shuni chaqiradi)."""
    from app.pipelines.speech import WhisperTranscriber

    return WhisperTranscriber(
        model_size=model,
        device="cpu",
        compute_type="int8",
        model_path=model_path,
        download_root=download_root,
        vad_filter=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Real ru WER smoke (§15 gate)")
    ap.add_argument("--generate", action="store_true", help="gTTS fixture yaratish")
    ap.add_argument("--fixtures-dir", default=str(_HERE / "data" / "smoke_ru"))
    ap.add_argument("--manifest", default=None, help="JSONL {audio, text}")
    ap.add_argument("--model", default="small", help="whisper o'lcham (small/medium)")
    ap.add_argument("--model-path", default=None, help="lokal ct2 model (offline)")
    ap.add_argument("--download-root", default=None, help="oldindan yuklangan kesh")
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    if args.generate:
        manifest_path = _gen_fixtures(fixtures_dir)
    elif args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = fixtures_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest topilmadi: {manifest_path} (--generate qiling)")

    rows = _load_manifest(manifest_path)
    print(f"\nManifest: {manifest_path}  ({len(rows)} ta utterance)")
    print(f"Model: {args.model}  device=cpu compute=int8  lang={args.lang}\n")

    tr = _build_transcriber(args.model, args.model_path, args.download_root)

    refs: list[str] = []
    hyps: list[str] = []
    n_unavailable = 0
    print(f"{'#':>2}  {'WER':>6}  {'conf':>5}  {'avail':>5}  hyp")
    print("-" * 72)
    for i, row in enumerate(rows):
        res = tr.transcribe(row["audio"], args.lang)   # <-- KONTRAKT chaqiruvi
        if not res.get("available"):
            n_unavailable += 1
        ref, hyp = row["text"], res.get("text", "")
        refs.append(ref)
        hyps.append(hyp)
        u_wer = wer(ref, hyp)
        print(f"{i:>2}  {u_wer:>6.3f}  {res.get('confidence', 0.0):>5.2f}  "
              f"{str(res.get('available')):>5}  {normalize_text(hyp)[:50]}")

    c_wer = corpus_wer(refs, hyps)
    ok, msg = check_gate(args.lang, c_wer)
    print("-" * 72)
    print(f"\nKORPUS WER ({args.lang}): {c_wer:.4f}  ({c_wer*100:.2f}%)")
    if n_unavailable:
        print(f"⚠️  {n_unavailable} utterance available=False qaytdi (degradatsiya)")
    print(f"GATE: {'✅ PASS' if ok else '❌ FAIL'} — {msg}")
    print("\n⚠️  Eslatma: agar fixture gTTS (toza) bo'lsa — bu OPTIMISTIK chegara. "
          "Production gate real operator audio bilan o'lchanadi.")
    # ru gate buzilsa noldan farqli chiqish kodi (CI uchun)
    return 0 if (ok or args.lang != "ru") else 1


if __name__ == "__main__":
    raise SystemExit(main())
