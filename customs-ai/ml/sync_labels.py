#!/usr/bin/env python3
"""taxonomy.yaml -> labels.txt (yagona haqiqat manbasi).

YOLO class indeksi tartibi taxonomy.yaml `classes:` xaritasidan olinadi va
model_registry/<versiya>/labels.txt ga yoziladi. labels.txt qo'lda
tahrirlanMAYDI — har doim shu skript bilan generatsiya qilinadi.

Foydalanish:
    python ml/sync_labels.py --version v0_baseline
    python ml/sync_labels.py --check    # CI: labels.txt taxonomy bilan sinxronmi?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "backend" / "config" / "taxonomy.yaml"
REGISTRY = ROOT / "backend" / "model_registry"

_HEADER = (
    "# Avtomatik generatsiya: python ml/sync_labels.py (config/taxonomy.yaml dan).\n"
    "# Qo'lda tahrirlaMANG. Qator tartibi = YOLO class indeksi (0 dan).\n"
)


def ordered_labels(taxonomy_path: Path = TAXONOMY) -> list[str]:
    """taxonomy.yaml `classes:` {indeks: nom} -> indeks bo'yicha tartiblangan nomlar."""
    data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
    classes = data.get("classes")
    if not classes:
        raise ValueError(f"{taxonomy_path}: 'classes:' xaritasi yo'q yoki bo'sh")
    try:
        items = sorted(((int(k), str(v)) for k, v in classes.items()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'classes:' kalitlari butun indeks bo'lishi kerak: {exc}") from exc
    idxs = [i for i, _ in items]
    if idxs != list(range(len(idxs))):
        raise ValueError(f"class indekslari 0..N uzluksiz bo'lishi kerak, topildi: {idxs}")
    return [name for _, name in items]


def render(labels: list[str]) -> str:
    return _HEADER + "\n".join(labels) + "\n"


def read_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="taxonomy.yaml -> labels.txt")
    ap.add_argument("--version", default="v0_baseline", help="model_registry papka nomi")
    ap.add_argument("--check", action="store_true", help="sinxronlikni tekshir (yozmaydi)")
    args = ap.parse_args()

    labels = ordered_labels()
    out = REGISTRY / args.version / "labels.txt"

    if args.check:
        existing = read_existing(out)
        if existing != labels:
            print(f"[FAIL] {out} taxonomy bilan sinxron emas.", file=sys.stderr)
            print(f"  kutilgan: {labels}", file=sys.stderr)
            print(f"  topilgan: {existing}", file=sys.stderr)
            return 1
        print(f"[OK] {out} sinxron ({len(labels)} class).")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(labels), encoding="utf-8")
    print(f"[OK] yozildi: {out} ({len(labels)} class)")
    for i, name in enumerate(labels):
        print(f"  {i}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
