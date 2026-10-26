"""Fine-tunes the base model into Decider² with LoRA (optionally QLoRA).

    locket-train --config configs/decider-2-7b.json --out runs/decider-2-7b [--qlora]

Loss is computed on the answer only (TRL prompt/completion format), so the model learns to
decide, not to reproduce prompts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from . import data
from .decider import DeciderConfig
from .schema import to_example


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_examples(cfg: dict, split: str) -> list[dict]:
    return [to_example(d) for d in data.for_split(cfg, split)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/decider-2-7b.json")
    ap.add_argument("--out", default="runs/decider-2-7b")
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (fits a 24 GB GPU)")
    ap.add_argument("--limit", type=int, default=0, help="cap train examples, for smoke tests")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = read_config(args.config)
    train = split_examples(cfg, "train")
    val = split_examples(cfg, "validation")
    if args.limit:
        train = train[: args.limit]
        val = val[: max(1, args.limit // 10)]
    print(f"train examples: {len(train)}  validation examples: {len(val)}")

    t = cfg["train"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model_kwargs = {"dtype": getattr(torch, t.get("dtype", "bfloat16"))}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], task_type="CAUSAL_LM"
    )
    # Older transformers spell a warmup fraction `warmup_ratio`; newer ones take it in `warmup_steps`.
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    warmup = {"warmup_ratio" if "warmup_ratio" in names else "warmup_steps": t["warmup_ratio"]}
    sft = SFTConfig(
        output_dir=args.out,
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_length"],
        **warmup,
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        bf16=t.get("dtype", "bfloat16") == "bfloat16",
        gradient_checkpointing=True,
        completion_only_loss=True,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=Dataset.from_list(train).remove_columns(["source"]),
        eval_dataset=Dataset.from_list(val).remove_columns(["source"]),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    DeciderConfig(base_model=cfg["base_model"], extra={"name": cfg["name"]}).write(args.out)
    print(f"adapter saved to {args.out}; next: locket-calibrate --model {cfg['base_model']} --adapter {args.out}")


if __name__ == "__main__":
    main()
