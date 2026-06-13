"""STT sof-yordamchilar (faster-whisper'siz testlanadi) — EGASI: Dev 3.

Bu yerda model/ct2 kerak EMAS: confidence agregatsiyasi, near-real-time chunk
rejasi, overlap merge va til normallashtirish — deterministik, sof Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Qo'llab-quvvatlanadigan tillar (Tamoyil: ru asosiy, uz ikkilamchi).
# Boshqa til auto-detect bo'lsa ham qaytariladi, lekin past_confidence belgisi
# bilan (operator chalg'imasligi uchun).
PRIMARY = "ru"
SECONDARY = "uz"
SUPPORTED = (PRIMARY, SECONDARY)


@dataclass
class Seg:
    """faster-whisper Segment'ning minimal, test qilinadigan ko'rinishi."""

    start: float
    end: float
    text: str
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def segment_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """Bitta segment ishonchi [0,1].

    avg_logprob (Whisper, ~[-1,0]) -> exp() bilan ehtimollikka; nutq yo'qligi
    ehtimoli bilan jazolanadi. Determinizm: faqat kirishlarga bog'liq.
    """
    p = math.exp(min(0.0, float(avg_logprob)))  # logprob>0 bo'lmaydi, himoya
    p *= 1.0 - max(0.0, min(1.0, float(no_speech_prob)))
    return max(0.0, min(1.0, p))


def aggregate_confidence(segments: list[Seg]) -> float:
    """Davomiylik bo'yicha vaznlangan o'rtacha confidence [0,1].

    Bo'sh -> 0.0. Davomiylik 0 bo'lsa (degenerate) -> oddiy o'rtacha.
    """
    if not segments:
        return 0.0
    total = sum(s.duration for s in segments)
    if total <= 0:
        vals = [segment_confidence(s.avg_logprob, s.no_speech_prob) for s in segments]
        return round(sum(vals) / len(vals), 4)
    acc = sum(
        segment_confidence(s.avg_logprob, s.no_speech_prob) * s.duration for s in segments
    )
    return round(acc / total, 4)


def join_text(segments: list[Seg]) -> str:
    """Segment matnlarini start bo'yicha tartiblab birlashtiradi (barqaror)."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    parts = [s.text.strip() for s in ordered if s.text and s.text.strip()]
    return " ".join(parts).strip()


def normalize_language(detected: str | None, requested: str | None) -> tuple[str | None, bool]:
    """Yakuniy til kodi + 'supported?' bayrog'i.

    requested berilgan bo'lsa o'sha hal qiluvchi. Aks holda aniqlangan til.
    Qaytadi: (lang_code, is_supported).
    """
    lang = (requested or detected or None)
    if lang is None:
        return None, False
    lang = lang.lower()
    return lang, lang in SUPPORTED


@dataclass
class Chunk:
    index: int
    start: float
    end: float


def plan_chunks(
    total_s: float, chunk_s: float = 25.0, overlap_s: float = 2.0
) -> list[Chunk]:
    """Near-real-time uchun chunk rejasi (overlap bilan).

    Whisper konteksti ~30s; chunk_s=25 + overlap=2 kontekst uzilishini kamaytiradi.
    Determinizm: faqat kirishlarга bog'liq. total_s<=chunk_s -> bitta chunk.
    """
    if total_s <= 0:
        return []
    if chunk_s <= 0:
        raise ValueError("chunk_s > 0 bo'lishi kerak")
    overlap_s = max(0.0, min(overlap_s, chunk_s - 0.1))
    step = chunk_s - overlap_s
    chunks: list[Chunk] = []
    start = 0.0
    idx = 0
    while start < total_s:
        end = min(start + chunk_s, total_s)
        chunks.append(Chunk(index=idx, start=round(start, 3), end=round(end, 3)))
        if end >= total_s:
            break
        start += step
        idx += 1
    return chunks


def merge_overlapping_segments(segments: list[Seg], overlap_tol: float = 0.5) -> list[Seg]:
    """Chunk overlap'idan kelib chiqqan takroriy segmentlarni tozalaydi.

    Global vaqtga keltirilgan segmentlar bo'yicha: start o'sishi bo'yicha tartibla,
    oldingisi qoplagan hududga (tol bilan) tushgan keyingi takrorni tashla.
    Determinizm: barqaror tartib.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    kept: list[Seg] = [ordered[0]]
    for s in ordered[1:]:
        prev = kept[-1]
        same_text = s.text.strip() == prev.text.strip()
        starts_inside = s.start < prev.end - overlap_tol
        if same_text and starts_inside:
            continue  # to'liq takror
        kept.append(s)
    return kept
