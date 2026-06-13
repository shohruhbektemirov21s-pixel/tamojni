#!/usr/bin/env python3
"""Datasetni YOLO formatiga + bizning taksonomiyaga moslash.

Ochiq datasetlar (SIXray/PIDray/OPIXray) har xil class nomlari ishlatadi. Biz
ularni `taxonomy.yaml > dataset_label_map` orqali bizning class indekslarimizga
o'tkazamiz. Xaritada `null` bo'lgan class'lar TASHLANADI (bizda yo'q).

Bu skript YOLO-format label fayllarini (class cx cy w h, normalized) qayta yozadi
va dataset yaml hosil qiladi. Rasm konvertatsiyasi kerak bo'lsa alohida qadamda.

⚠️ Ochiq datasetlar FAQAT bootstrap/POC. Litsenziya datasets/README.md da.
⚠️ O'z (real skaner) dataseti ON-PREM labellanadi — bu skript uni ham qabul qiladi
   (allaqachon bizning taksonomiyada bo'lsa, --identity bilan o'tkazib yuboriladi).

Foydalanish:
    python ml/prepare_dataset.py --src ml/datasets/raw/sixray \
        --dst ml/datasets/yolo --source-classes ml/datasets/raw/sixray/classes.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "backend" / "config" / "taxonomy.yaml"


def load_taxonomy(path: Path = TAXONOMY) -> tuple[list[str], dict[str, str | None]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = data["classes"]
    ordered = [classes[i] for i in sorted(classes)]
    label_map = data.get("dataset_label_map") or {}
    return ordered, label_map


def build_index_remap(
    source_classes: list[str], target_classes: list[str], label_map: dict[str, str | None]
) -> dict[int, int | None]:
    """source class indeksi -> target class indeksi (yoki None = tashlash)."""
    target_idx = {name: i for i, name in enumerate(target_classes)}
    remap: dict[int, int | None] = {}
    for si, sname in enumerate(source_classes):
        # to'g'ridan-to'g'ri bizning class bo'lsa (o'z datasetimiz) — identity.
        if sname in target_idx:
            remap[si] = target_idx[sname]
            continue
        mapped = label_map.get(sname, "__MISSING__")
        if mapped == "__MISSING__":
            print(f"[WARN] '{sname}' uchun moslik yo'q (taxonomy.dataset_label_map) -> tashlanadi")
            remap[si] = None
        elif mapped is None:
            remap[si] = None  # ataylab tashlanadi
        else:
            remap[si] = target_idx.get(mapped)
            if remap[si] is None:
                print(f"[WARN] '{sname}' -> '{mapped}' taxonomiyada yo'q -> tashlanadi")
    return remap


def remap_label_file(src: Path, dst: Path, remap: dict[int, int | None]) -> int:
    """Bitta YOLO .txt faylni qayta yozadi. Qaytaradi: saqlangan qatorlar soni."""
    kept = 0
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        ci = int(float(parts[0]))
        new_ci = remap.get(ci)
        if new_ci is None:
            continue
        lines_out.append(f"{new_ci} {' '.join(parts[1:])}")
        kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="Datasetni taxonomiyaga moslash")
    ap.add_argument("--src", required=True, help="manba labels papkasi (YOLO .txt)")
    ap.add_argument("--dst", required=True, help="chiqish labels papkasi")
    ap.add_argument("--source-classes", required=True, help="manba class nomlari fayli (qatorma-qator)")
    args = ap.parse_args()

    target_classes, label_map = load_taxonomy()
    source_classes = [
        ln.strip()
        for ln in Path(args.source_classes).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    remap = build_index_remap(source_classes, target_classes, label_map)

    src_dir, dst_dir = Path(args.src), Path(args.dst)
    total_files = total_kept = 0
    for txt in sorted(src_dir.rglob("*.txt")):
        rel = txt.relative_to(src_dir)
        total_kept += remap_label_file(txt, dst_dir / rel, remap)
        total_files += 1

    # dataset yaml (Ultralytics formati)
    yaml_path = dst_dir.parent / "xray.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dst_dir.parent.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: n for i, n in enumerate(target_classes)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] {total_files} fayl, {total_kept} bbox saqlandi -> {dst_dir}")
    print(f"[OK] dataset yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
