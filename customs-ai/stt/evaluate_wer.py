#!/usr/bin/env python3
"""WER baholash — ru va uz uchun ALOHIDA (qabul mezoni §15).

Yadro (`wer`, `normalize_text`) sof Python — jiwer/torch kerak EMAS, shuning uchun
deterministik birlik testi yoziladi. jiwer mavjud bo'lsa CLI undan ham foydalanadi.

Qabul gate'i:
    rus WER <= 0.15 (15%) — MAJBURIY.
    o'zbek WER — baseline o'lchanadi (fine-tune target keyin belgilanadi; ⚠️ Whisper
    o'zbekда past — bu kutilgan, fine-tune track buni yaxshilaydi).

Foydalanish:
    python stt/evaluate_wer.py --manifest stt/data/uz_test.jsonl --model medium --lang uz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

RU_TARGET_WER = 0.15

# O'zbek lotin/kiril aralash — normallashtirish WER'ni adolatli qiladi.
# (apostrof variantlari o'/g' uchun yagona shaklga keltiriladi.)
_APOS = {"`": "'", "ʻ": "'", "ʼ": "'", "’": "'", "‘": "'", "´": "'"}
_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Determinik normallashtirish: lower, apostrof birlashtirish, punktuatsiya olib
    tashlash, probel siqish. ru va uz uchun bir xil (adolatli taqqoslash)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _APOS.items():
        text = text.replace(k, v)
    text = text.lower()
    # Rus ё↔е: yozuvда almashtiriladi (Whisper "ё", reference ko'pincha "е").
    # Birlashtirmaslik ru WER'ni nohaq oshiradi (smoke: "запрещённые" vs "запрещенные").
    text = text.replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _levenshtein_words(ref: list[str], hyp: list[str]) -> int:
    """So'z darajasidagi tahrir masofasi (S+D+I)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """Word Error Rate [0, +inf). Bo'sh reference -> 0.0 agar hyp ham bo'sh, aks
    holda 1.0 (to'liq xato)."""
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein_words(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


def corpus_wer(references: list[str], hypotheses: list[str], *, normalize: bool = True) -> float:
    """Korpus darajasidagi WER: jami tahrir / jami so'z (namuna-o'rtacha emas)."""
    if len(references) != len(hypotheses):
        raise ValueError("references va hypotheses uzunligi teng bo'lishi kerak")
    total_dist = total_words = 0
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref, hyp = normalize_text(ref), normalize_text(hyp)
        r, h = ref.split(), hyp.split()
        total_dist += _levenshtein_words(r, h)
        total_words += len(r)
    if total_words == 0:
        return 0.0
    return round(total_dist / total_words, 4)


def check_gate(lang: str, value: float) -> tuple[bool, str]:
    """ru uchun gate (<=15%); uz uchun faqat baseline reportlanadi."""
    if lang == "ru":
        ok = value <= RU_TARGET_WER
        return ok, f"ru WER {value:.3f} {'<=' if ok else '>'} {RU_TARGET_WER} target"
    return True, f"uz WER {value:.3f} (baseline — fine-tune track buni yaxshilaydi)"


def _load_manifest(path: str) -> list[dict]:
    """JSONL: har qator {'audio': yo'l, 'text': referens}."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="WER baholash (ru/uz alohida)")
    ap.add_argument("--manifest", required=True, help="JSONL: {audio, text}")
    ap.add_argument("--model", default="medium", help="faster-whisper model / lokal path")
    ap.add_argument("--lang", default="uz", choices=["ru", "uz"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    rows = _load_manifest(args.manifest)
    if not rows:
        print("Manifest bo'sh.", file=sys.stderr)
        return 2

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper yo'q: pip install -r backend/requirements-stt.txt", file=sys.stderr)
        return 2

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    refs, hyps = [], []
    for row in rows:
        segments, _ = model.transcribe(row["audio"], language=args.lang)
        hyp = " ".join(s.text for s in segments)
        refs.append(row["text"])
        hyps.append(hyp)

    value = corpus_wer(refs, hyps)
    passed, msg = check_gate(args.lang, value)
    print(f"[{args.lang}] korpus WER = {value:.4f}  (n={len(rows)})")
    print(("[GATE PASS] " if passed else "[GATE FAIL] ") + msg)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
