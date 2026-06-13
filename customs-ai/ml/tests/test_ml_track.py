"""ML track: deterministik recall, qabul gate'i, labels sinxron."""
from __future__ import annotations

from evaluate import GATED_CLASSES, check_gate, iou, per_class_recall
from sync_labels import ordered_labels


def _gt(img, cls, bbox):
    return {"image": img, "class": cls, "bbox": bbox}


def _pred(img, cls, score, bbox):
    return {"image": img, "class": cls, "score": score, "bbox": bbox}


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_recall_perfect_detection():
    gts = [_gt(1, "qurol", [0, 0, 10, 10])]
    preds = [_pred(1, "qurol", 0.9, [0, 0, 10, 10])]
    r = per_class_recall(preds, gts)
    assert r["qurol"]["recall"] == 1.0 and r["qurol"]["fn"] == 0


def test_recall_missed_is_false_negative():
    gts = [_gt(1, "qurol", [0, 0, 10, 10]), _gt(1, "qurol", [50, 50, 60, 60])]
    preds = [_pred(1, "qurol", 0.9, [0, 0, 10, 10])]  # ikkinchisini topmadi
    r = per_class_recall(preds, gts)
    assert r["qurol"]["recall"] == 0.5 and r["qurol"]["fn"] == 1


def test_recall_deterministic():
    gts = [_gt(1, "qurol", [0, 0, 10, 10])]
    preds = [_pred(1, "qurol", 0.9, [0, 0, 10, 10]), _pred(1, "qurol", 0.8, [0, 0, 9, 9])]
    assert per_class_recall(preds, gts) == per_class_recall(preds, gts)


def test_each_gt_matched_once():
    # ikkita prediction bitta GT'ga — faqat bittasi TP.
    gts = [_gt(1, "qurol", [0, 0, 10, 10])]
    preds = [_pred(1, "qurol", 0.9, [0, 0, 10, 10]), _pred(1, "qurol", 0.8, [0, 0, 10, 10])]
    r = per_class_recall(preds, gts)
    assert r["qurol"]["tp"] == 1


def test_gate_passes_when_recall_high():
    recalls = {c: {"recall": 0.96, "tp": 24, "fn": 1, "total": 25} for c in GATED_CLASSES}
    passed, failures = check_gate(recalls, target=0.95)
    assert passed and failures == []


def test_gate_fails_on_low_weapon_recall():
    recalls = {"qurol": {"recall": 0.90, "tp": 18, "fn": 2, "total": 20},
               "o'q-dori": {"recall": 0.99, "tp": 99, "fn": 1, "total": 100}}
    passed, failures = check_gate(recalls, target=0.95)
    assert not passed
    assert any("qurol" in f for f in failures)


def test_gate_fails_when_class_absent():
    passed, failures = check_gate({}, target=0.95)
    assert not passed and len(failures) == len(GATED_CLASSES)


def test_ordered_labels_from_taxonomy():
    labels = ordered_labels()
    assert labels[0] == "qurol"  # indeks 0
    assert "pichoq" in labels
    assert len(labels) == len(set(labels))  # dublikatsiz
