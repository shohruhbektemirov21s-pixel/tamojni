"""STT sof-yordamchilar (whisper_utils) — modelsiz, deterministik."""
from __future__ import annotations

import math

import pytest

from app.pipelines.whisper_utils import (
    Seg,
    aggregate_confidence,
    join_text,
    merge_overlapping_segments,
    normalize_language,
    plan_chunks,
    segment_confidence,
)


# --- confidence ---
def test_segment_confidence_range():
    c = segment_confidence(avg_logprob=-0.2, no_speech_prob=0.0)
    assert 0.0 <= c <= 1.0
    assert abs(c - math.exp(-0.2)) < 1e-6


def test_segment_confidence_penalized_by_no_speech():
    high = segment_confidence(-0.1, 0.0)
    low = segment_confidence(-0.1, 0.9)
    assert low < high


def test_aggregate_empty_is_zero():
    assert aggregate_confidence([]) == 0.0


def test_aggregate_duration_weighted():
    # uzun, ishonchli segment qisqa, ishonchsizdan ko'ra og'irroq.
    segs = [
        Seg(0, 10, "a", avg_logprob=-0.05, no_speech_prob=0.0),   # yuqori conf, uzun
        Seg(10, 11, "b", avg_logprob=-2.0, no_speech_prob=0.0),    # past conf, qisqa
    ]
    conf = aggregate_confidence(segs)
    # natija yuqori segmentga yaqinroq bo'lishi kerak
    assert conf > segment_confidence(-2.0, 0.0)


def test_aggregate_deterministic():
    segs = [Seg(0, 2, "x", -0.3, 0.1), Seg(2, 4, "y", -0.5, 0.0)]
    assert aggregate_confidence(segs) == aggregate_confidence(segs)


# --- join text ---
def test_join_text_ordered_and_stripped():
    segs = [Seg(2, 3, "  dunyo "), Seg(0, 1, "salom")]
    assert join_text(segs) == "salom dunyo"


def test_join_text_skips_empty():
    assert join_text([Seg(0, 1, "  "), Seg(1, 2, "matn")]) == "matn"


# --- language normalize ---
def test_language_requested_wins():
    lang, sup = normalize_language(detected="en", requested="ru")
    assert lang == "ru" and sup is True


def test_language_uses_detected_when_no_request():
    lang, sup = normalize_language(detected="uz", requested=None)
    assert lang == "uz" and sup is True


def test_language_unsupported_flagged():
    lang, sup = normalize_language(detected="en", requested=None)
    assert lang == "en" and sup is False


def test_language_none():
    assert normalize_language(None, None) == (None, False)


# --- chunk planning (near-real-time) ---
def test_plan_single_chunk_when_short():
    chunks = plan_chunks(total_s=10.0, chunk_s=25.0, overlap_s=2.0)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0 and chunks[0].end == 10.0


def test_plan_multiple_chunks_with_overlap():
    chunks = plan_chunks(total_s=60.0, chunk_s=25.0, overlap_s=5.0)
    assert chunks[0].start == 0.0 and chunks[0].end == 25.0
    # step = 20 -> ikkinchi 20..45 (overlap 5)
    assert chunks[1].start == 20.0
    assert chunks[-1].end == 60.0  # oxirgi total'ga clamp


def test_plan_covers_full_audio():
    chunks = plan_chunks(total_s=100.0, chunk_s=25.0, overlap_s=2.0)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 100.0


def test_plan_empty_for_zero_duration():
    assert plan_chunks(0.0) == []


def test_plan_rejects_bad_chunk():
    with pytest.raises(ValueError):
        plan_chunks(10.0, chunk_s=0)


def test_plan_deterministic():
    assert plan_chunks(73.0, 25, 3) == plan_chunks(73.0, 25, 3)


# --- overlap merge ---
def test_merge_drops_duplicate_in_overlap():
    segs = [
        Seg(0, 5, "salom dunyo"),
        Seg(4, 5, "salom dunyo"),   # overlap'dagi takror (start < prev.end - tol)
        Seg(5, 10, "yangi gap"),
    ]
    merged = merge_overlapping_segments(segs, overlap_tol=0.5)
    texts = [s.text for s in merged]
    assert texts == ["salom dunyo", "yangi gap"]


def test_merge_keeps_distinct_text():
    segs = [Seg(0, 5, "bir"), Seg(4, 9, "ikki")]
    assert len(merge_overlapping_segments(segs)) == 2


def test_merge_empty():
    assert merge_overlapping_segments([]) == []
