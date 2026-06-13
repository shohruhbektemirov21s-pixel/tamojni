#!/usr/bin/env python3
"""DETERMINISTIK baholash — per-class recall + qabul gate'i (qurol recall >= 0.95).

Yadro (`per_class_recall`) sof Python — onnxruntime/ultralytics/torch kerak EMAS,
shuning uchun deterministik birlik testi yoziladi. Greedy IoU matching:
score kamayishi bo'yicha har prediction eng mos (IoU >= thres, bir xil class,
bir xil rasm) GT ga biriktiriladi; har GT bir marta ishlatiladi.

CLI rejimi (`--weights`, `--data`) Ultralytics val'iga delegatsiya qiladi
(train mashinasida). Gate ikkala rejimda ham bir xil mantiq.

Foydalanish:
    python ml/evaluate.py --weights runs/detect/train/weights/best.pt \
                          --data ml/datasets/xray.yaml --recall-target 0.95
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

# Qabul mezoni: bojxonada false-negative xavfli. Qurol class uchun qattiq gate.
DEFAULT_RECALL_TARGET = 0.95
GATED_CLASSES = ("qurol", "o'q-dori")


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def per_class_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_thres: float = 0.5,
) -> dict[str, dict]:
    """Deterministik per-class recall (greedy IoU matching).

    predictions: [{"image": id, "class": str, "score": float, "bbox": [x1,y1,x2,y2]}]
    ground_truths: [{"image": id, "class": str, "bbox": [...]}]
    Qaytaradi: {class: {"tp": int, "fn": int, "total": int, "recall": float}}
    """
    gts_by_key: dict[tuple, list[dict]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for gt in ground_truths:
        gts_by_key[(gt["image"], gt["class"])].append({"bbox": gt["bbox"], "used": False})
        totals[gt["class"]] += 1

    tp: dict[str, int] = defaultdict(int)
    # Determinizm: score kamayishi, keyin (image, class, bbox) bo'yicha barqaror.
    preds = sorted(
        predictions,
        key=lambda p: (-float(p["score"]), str(p["image"]), p["class"], tuple(p["bbox"])),
    )
    for p in preds:
        cands = gts_by_key.get((p["image"], p["class"]), [])
        best_i, best_iou = -1, iou_thres
        for i, g in enumerate(cands):
            if g["used"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v >= best_iou:
                best_iou, best_i = v, i
        if best_i >= 0:
            cands[best_i]["used"] = True
            tp[p["class"]] += 1

    out: dict[str, dict] = {}
    for cls, total in sorted(totals.items()):
        t = tp.get(cls, 0)
        out[cls] = {
            "tp": t,
            "fn": total - t,
            "total": total,
            "recall": round(t / total, 4) if total else 0.0,
        }
    return out


def check_gate(
    recalls: dict[str, dict],
    target: float = DEFAULT_RECALL_TARGET,
    gated: tuple[str, ...] = GATED_CLASSES,
) -> tuple[bool, list[str]]:
    """Gate: gated class'lar recall >= target bo'lishi shart. (passed, sabablar)."""
    failures: list[str] = []
    for cls in gated:
        r = recalls.get(cls)
        if r is None:
            failures.append(f"{cls}: eval setda namuna yo'q (recall noma'lum)")
        elif r["recall"] < target:
            failures.append(f"{cls}: recall {r['recall']:.3f} < {target} ({r['fn']} false-negative)")
    return (not failures, failures)


def _print_table(recalls: dict[str, dict]) -> None:
    print(f"{'class':<22}{'recall':>8}{'tp':>6}{'fn':>6}{'total':>7}")
    for cls, r in recalls.items():
        print(f"{cls:<22}{r['recall']:>8.3f}{r['tp']:>6}{r['fn']:>6}{r['total']:>7}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministik per-class recall + gate")
    ap.add_argument("--weights", help=".pt yoki .onnx (Ultralytics val)")
    ap.add_argument("--data", help="dataset yaml (Ultralytics)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    args = ap.parse_args()

    if not args.weights or not args.data:
        print("CLI rejimi uchun --weights va --data kerak.", file=sys.stderr)
        print("(Yadro `per_class_recall` testlardan to'g'ridan-to'g'ri chaqiriladi.)")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics o'rnatilmagan: pip install -r ml/requirements-train.txt", file=sys.stderr)
        return 2

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, iou=args.iou)
    names = model.names
    recalls: dict[str, dict] = {}
    # Ultralytics per-class recall (mp/ mr indeks bo'yicha) -> bizning formatga.
    for i, cls in names.items():
        try:
            r = float(metrics.box.r[i])  # per-class recall
        except (IndexError, TypeError):
            r = 0.0
        recalls[cls] = {"tp": -1, "fn": -1, "total": -1, "recall": round(r, 4)}

    _print_table(recalls)
    passed, failures = check_gate(recalls, args.recall_target)
    if not passed:
        print("\n[GATE FAIL] qabul mezoni bajarilmadi:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\n[GATE PASS] gated class'lar recall >= {args.recall_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
