#!/usr/bin/env python3
"""Baseline YOLOv11 fine-tune (Ultralytics) — X-ray domain.

Train mashinasida ishlaydi (torch/CUDA shu yerda; target runtime ONNX/CPU).
Recall-favoring: kichik conf, hard-negative round'lariga moslangan augmentatsiya.

Foydalanish:
    python ml/train_baseline.py --data ml/datasets/xray.yaml --model yolo11s.pt \
        --epochs 100 --imgsz 640 --name xray_v0
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv11 X-ray fine-tune")
    ap.add_argument("--data", required=True, help="dataset yaml (YOLO format)")
    ap.add_argument("--model", default="yolo11s.pt", help="boshlang'ich weights (n/s tavsiya)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="xray_baseline")
    ap.add_argument("--device", default="0", help="GPU id yoki 'cpu'")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics yo'q: pip install -r ml/requirements-train.txt")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        project="ml/runs",
        # --- recall-favoring va X-ray domain'ga moslash ---
        cls=0.7,            # class loss og'irligini biroz oshiramiz (yo'qotmaslik)
        hsv_h=0.0,          # X-ray psevdo-rang — rang aralashtirishni o'chiramiz
        hsv_s=0.3,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        # kichik/yashirin obyektlar uchun:
        scale=0.5,
        degrees=5.0,
        seed=0,             # determinizm (qayta ishlab chiqarish uchun)
        deterministic=True,
        plots=True,
    )
    print("Train tugadi. Eval + gate uchun: python ml/evaluate.py --weights "
          f"ml/runs/{args.name}/weights/best.pt --data {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
