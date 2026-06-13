#!/usr/bin/env python3
"""Whisper LoRA/PEFT fine-tune — o'zbek bojxona domeni (Dev 3).

⚠️ FAQAT o'zbek uchun. Rus tili allaqachon yaxshi — fine-tune SHART EMAS
(qayta o'qitish rus sifatini buzishi mumkin).

Train mashinasida ishlaydi (GPU bu yerda — bu ADR-002'ni buzmaydi: ADR-002
faqat TARGET runtime'ga taalluqli; train alohida). Natija HF Whisper -> keyin
convert_ct2.py bilan faster-whisper (CPU/int8) formatiga o'tkaziladi.

LoRA sabab: to'liq fine-tune 4GB/oddiy GPU'da og'ir; LoRA kam-resursli, kam-data
(o'zbek) uchun barqaror va katastrofik unutishni kamaytiradi.

Foydalanish:
    python stt/finetune_whisper_lora.py --train stt/data/uz_train.jsonl \
        --val stt/data/uz_val.jsonl --base openai/whisper-medium --out stt/runs/uz_lora
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper LoRA fine-tune (o'zbek)")
    ap.add_argument("--train", required=True, help="JSONL {audio, text}")
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", default="openai/whisper-medium")
    ap.add_argument("--out", default="stt/runs/uz_lora")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lang", default="uz")
    args = ap.parse_args()

    try:
        import torch
        from datasets import Audio, load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError:
        raise SystemExit("Train deps yo'q: pip install -r stt/requirements-train.txt")

    processor = WhisperProcessor.from_pretrained(args.base, language=args.lang, task="transcribe")
    ds = load_dataset(
        "json", data_files={"train": args.train, "val": args.val}
    ).cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names)

    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.config.forced_decoder_ids = None
    # LoRA: faqat attention proyeksiyalari -> kam parametr, kam unutish.
    lora = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        report_to="none",
        seed=0,  # determinizm
    )

    from dataclasses import dataclass

    @dataclass
    class Collator:
        processor: object

        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=Collator(processor),
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"[OK] LoRA adapter saqlandi: {args.out}")
    print("Keyingi: merge + ct2 konvertatsiya -> python stt/convert_ct2.py --adapter " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
