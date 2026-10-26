"""Merges a trained LoRA adapter into the base weights, giving a standalone Decider² model.

    locket-merge --adapter runs/decider-2-7b --out models/locket-decider-2-7b

The merged folder loads with plain transformers or vLLM, and keeps decider_config.json.
"""

from __future__ import annotations

import argparse
import os
import shutil

from .decider import CONFIG_FILE, DeciderConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", help="base model (default: the one recorded at training time)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or DeciderConfig.read(args.adapter).base_model
    if not base:
        raise SystemExit("no base model recorded in the adapter; pass --base")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    src = os.path.join(args.adapter, CONFIG_FILE)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(args.out, CONFIG_FILE))
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
