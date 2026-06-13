#!/usr/bin/env python3
"""O'zbek STT datasetini tayyorlash — Common Voice (uz) + on-prem operator audio.

Chiqish: JSONL manifest ({audio, text, source}) + train/val/test split. Fine-tune
(finetune_whisper_lora.py) va eval (evaluate_wer.py) shu manifestni o'qiydi.

⚠️ Tamoyil 1 (offline): real operator audiolari FAQAT on-prem qoladi — bu skript
ularni tashqariga chiqarmaydi, faqat lokal manifest yasaydi.
⚠️ Litsenziya: Common Voice (CC0) — tijoriy ishlatishga ruxsat, lekin Sprint 0'da
tasdiqlang (stt/data/README.md).

Foydalanish:
    # Common Voice uz tsv -> manifest
    python stt/prepare_uz_data.py --cv-tsv cv/uz/validated.tsv --cv-clips cv/uz/clips \
        --out stt/data
    # on-prem audio papkasi (yonida <name>.txt transkript) -> manifest
    python stt/prepare_uz_data.py --onprem-dir /secure/operator_audio --out stt/data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def from_common_voice(tsv: Path, clips: Path) -> list[dict]:
    rows: list[dict] = []
    with tsv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            sentence = (r.get("sentence") or "").strip()
            path = (r.get("path") or "").strip()
            if not sentence or not path:
                continue
            audio = clips / path
            rows.append({"audio": str(audio), "text": sentence, "source": "common_voice_uz"})
    return rows


def from_onprem(directory: Path) -> list[dict]:
    """Har audio yonida bir xil nomli .txt transkript bo'lishi kutiladi."""
    rows: list[dict] = []
    for audio in sorted(directory.rglob("*")):
        if audio.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            continue
        txt = audio.with_suffix(".txt")
        if not txt.exists():
            print(f"[skip] transkript yo'q: {audio.name}")
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if text:
            rows.append({"audio": str(audio), "text": text, "source": "onprem"})
    return rows


def split_and_write(rows: list[dict], out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)  # determinizm
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * 0.1))
    n_val = max(1, int(n * 0.1))
    test, val, train = rows[:n_test], rows[n_test : n_test + n_val], rows[n_test + n_val :]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("uz_train", train), ("uz_val", val), ("uz_test", test)]:
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] {p} ({len(part)} namuna)")


def main() -> int:
    ap = argparse.ArgumentParser(description="O'zbek STT dataset tayyorlash")
    ap.add_argument("--cv-tsv", type=Path, help="Common Voice validated.tsv")
    ap.add_argument("--cv-clips", type=Path, help="Common Voice clips/ papkasi")
    ap.add_argument("--onprem-dir", type=Path, help="on-prem audio papkasi (+ .txt)")
    ap.add_argument("--out", type=Path, default=Path("stt/data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict] = []
    if args.cv_tsv and args.cv_clips:
        cv = from_common_voice(args.cv_tsv, args.cv_clips)
        print(f"[OK] Common Voice: {len(cv)} namuna")
        rows += cv
    if args.onprem_dir:
        op = from_onprem(args.onprem_dir)
        print(f"[OK] on-prem: {len(op)} namuna")
        rows += op

    if not rows:
        print("Manba berilmadi (--cv-tsv/--cv-clips yoki --onprem-dir).")
        return 2
    split_and_write(rows, args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
