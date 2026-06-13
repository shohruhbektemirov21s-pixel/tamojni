"""WER yadrosi (sof-python) — deterministik, jiwer/torch'siz."""
from __future__ import annotations

from evaluate_wer import (
    RU_TARGET_WER,
    check_gate,
    corpus_wer,
    normalize_text,
    wer,
)


# --- normalizatsiya ---
def test_normalize_lower_punct_ws():
    assert normalize_text("  Salom,  DUNYO! ") == "salom dunyo"


def test_normalize_unifies_apostrophes():
    # o' / g' uchun turli apostroflar yagona shaklga keladi
    a = normalize_text("o`zbek")
    b = normalize_text("oʻzbek")
    c = normalize_text("o’zbek")
    assert a == b == c == "o'zbek"


def test_normalize_empty():
    assert normalize_text("") == ""


def test_normalize_folds_ru_yo_to_ye():
    # ё↔е yozuvда almashtiriladi -> WER adolatli (smoke kuzatuvi)
    assert normalize_text("запрещённые") == normalize_text("запрещенные") == "запрещенные"
    assert wer("предметы запрещённые", "предметы запрещенные") == 0.0


# --- WER ---
def test_wer_perfect():
    assert wer("salom dunyo", "salom dunyo") == 0.0


def test_wer_one_substitution():
    # 1 ta xato / 2 so'z = 0.5
    assert wer("salom dunyo", "salom olam") == 0.5


def test_wer_deletion_and_insertion():
    assert wer("bir ikki uch", "bir uch") == round(1 / 3, 4)       # deletion
    assert wer("bir uch", "bir ikki uch") == round(1 / 2, 4)        # insertion


def test_wer_empty_ref_empty_hyp():
    assert wer("", "") == 0.0


def test_wer_empty_ref_nonempty_hyp():
    assert wer("", "shovqin") == 1.0


def test_wer_normalization_makes_fair():
    # punktuatsiya/registr WER'ni oshirmasligi kerak
    assert wer("Salom, dunyo!", "salom dunyo") == 0.0


def test_wer_deterministic():
    assert wer("bir ikki uch", "bir uch uch") == wer("bir ikki uch", "bir uch uch")


# --- corpus WER ---
def test_corpus_wer_aggregates_by_words():
    refs = ["bir ikki", "uch tort besh"]
    hyps = ["bir ikki", "uch tort olti"]   # 1 xato / jami 5 so'z = 0.2
    assert corpus_wer(refs, hyps) == 0.2


def test_corpus_wer_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        corpus_wer(["a"], ["a", "b"])


# --- gate ---
def test_ru_gate_pass():
    ok, msg = check_gate("ru", 0.12)
    assert ok and "ru" in msg


def test_ru_gate_fail():
    ok, _ = check_gate("ru", 0.20)
    assert not ok


def test_ru_gate_boundary():
    ok, _ = check_gate("ru", RU_TARGET_WER)  # <= -> pass
    assert ok


def test_uz_gate_is_baseline_only():
    ok, msg = check_gate("uz", 0.45)   # yuqori bo'lsa ham gate uz uchun pass (baseline)
    assert ok and "baseline" in msg
