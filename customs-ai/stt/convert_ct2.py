#!/usr/bin/env python3
"""LoRA adapter -> merge -> CTranslate2 (faster-whisper CPU/int8) konvertatsiya.

Runtime CPU/int8 talab qiladi (ADR-002). Bu skript:
  1. base + LoRA adapter'ni birlashtiradi (merge_and_unload)
  2. HF formatga saqlaydi
  3. ct2-transformers-converter bilan int8 ct2 modelga aylantiradi

Natija lokal katalog -> backend CUSTOMS_STT_MODEL_PATH orqali yoqadi (offline).

Foydalanish:
    python stt/convert_ct2.py --base openai/whisper-medium \
        --adapter stt/runs/uz_lora --out stt/models/uz_medium_ct2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA -> ct2 (int8) konvertatsiya")
    ap.add_argument("--base", default="openai/whisper-medium")
    ap.add_argument("--adapter", required=True, help="LoRA adapter papkasi")
    ap.add_argument("--out", required=True, help="ct2 model chiqish papkasi")
    ap.add_argument("--quantization", default="int8")
    args = ap.parse_args()

    try:
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        raise SystemExit("Train deps yo'q: pip install -r stt/requirements-train.txt")

    print("[1/3] base + LoRA merge...")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    processor = WhisperProcessor.from_pretrained(args.adapter)

    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / "merged_hf"
        model.save_pretrained(merged)
        processor.save_pretrained(merged)
        print(f"[2/3] merged HF saqlandi -> {merged}")

        print("[3/3] ct2 konvertatsiya (int8)...")
        cmd = [
            "ct2-transformers-converter",
            "--model", str(merged),
            "--output_dir", args.out,
            "--quantization", args.quantization,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
        ]
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc != 0:
            print("ct2 konvertatsiya muvaffaqiyatsiz (ctranslate2 o'rnatilganmi?)", file=sys.stderr)
            return rc

    print(f"[OK] ct2 model: {args.out}")
    print(f"Runtime'da yoqish: CUSTOMS_USE_MOCKS=false CUSTOMS_STT_MODEL_PATH={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
