#!/usr/bin/env python3
"""train (.pt) -> runtime artefakt (.onnx) + sha256 + labels nusxasi.

ADR-002: target runtime ONNX/CPU/DirectML. Eksport STATIK input (imgsz),
opset 12, soddalashtirilgan graf — ORT CPU'da barqaror va tez.

Artefaktlar model_registry/<versiya>/ ga joylashadi:
    model.onnx, labels.txt, sha256.txt, model_card.md (qo'lda to'ldiriladi).

Foydalanish:
    python ml/export_onnx.py --weights ml/runs/xray_v0/weights/best.pt \
        --version v1 --imgsz 640
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "backend" / "model_registry"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="best.pt -> model_registry/<v>/model.onnx")
    ap.add_argument("--weights", required=True, help="train natijasi best.pt")
    ap.add_argument("--version", required=True, help="registry versiya papkasi, masalan v1")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics yo'q: pip install -r ml/requirements-train.txt")

    out_dir = REGISTRY / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    # CUDA EMAS — ADR-002. simplify=True ORT grafini ixchamlaydi. dynamic=False:
    # statik shakl runtime'da barqaror latency beradi.
    exported = model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False
    )
    exported = Path(exported)
    target = out_dir / "model.onnx"
    shutil.copy2(exported, target)

    # labels.txt — taxonomy'dan generatsiya (model nomlari bilan mos kelishini ham
    # tekshirib qo'ying: model.names tartibi labels.txt bilan bir xil bo'lsin).
    from sync_labels import ordered_labels, render  # type: ignore

    labels = ordered_labels()
    model_names = [model.names[i] for i in sorted(model.names)]
    if model_names != labels:
        print("[WARN] model.names taxonomy labels'idan farq qiladi:")
        print(f"  model:    {model_names}")
        print(f"  taxonomy: {labels}")
        print("  -> dataset class tartibini taxonomy.yaml bilan moslang!")
    (out_dir / "labels.txt").write_text(render(labels), encoding="utf-8")

    digest = sha256_of(target)
    (out_dir / "sha256.txt").write_text(f"{digest}  model.onnx\n", encoding="utf-8")

    print(f"[OK] {target}")
    print(f"[OK] sha256: {digest}")
    print(f"[OK] labels: {out_dir / 'labels.txt'} ({len(labels)} class)")
    print(f"\nRuntime'da yoqish: CUSTOMS_YOLO_MODEL_PATH=model_registry/{args.version}/model.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
