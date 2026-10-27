"""Writes the Hugging Face model card from real evaluation results and uploads the model.

    HF_TOKEN=... locket-release --model models/locket-decider-2-7b --repo <user-or-org>/locket-decider-2-7b \
        --eval runs/eval.json --base-eval runs/eval-base.json [--private] [--dry-run]

The card only reports numbers found in the eval JSON files written by `locket-eval`.
"""

from __future__ import annotations

import argparse
import json
import os

from .data import SOURCES
from .decider import DeciderConfig

DATASET_IDS = {
    "banking77": "PolyAI/banking77",
    "clinc_oos": "clinc_oos",
    "massive_en": "AmazonScience/massive",
    "massive_tr": "AmazonScience/massive",
    "multi_nli": "nyu-mll/multi_nli",
    "boolq": "google/boolq",
}


def results_table(trained: dict, base: dict | None) -> str:
    rows = ["| Eval set | n | Accuracy | Macro F1 | ECE ↓ |" + (" Base accuracy |" if base else ""), "|---|---:|---:|---:|---:|" + ("---:|" if base else "")]
    names = list(trained["by_source"]) + ["overall"]
    for name in names:
        m = trained["overall"] if name == "overall" else trained["by_source"][name]
        line = f"| {'**overall**' if name == 'overall' else name} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['ece']:.3f} |"
        if base:
            b = base["overall"] if name == "overall" else base["by_source"].get(name)
            line += f" {b['accuracy']:.3f} |" if b else " – |"
        rows.append(line)
    return "\n".join(rows)


def model_card(repo: str, config: DeciderConfig, trained: dict | None, base: dict | None, sources: list[str]) -> str:
    datasets = sorted({DATASET_IDS[s] for s in sources if s in DATASET_IDS})
    front = [
        "---",
        "license: apache-2.0",
        f"base_model: {config.base_model}" if config.base_model else None,
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "language:\n- en\n- tr",
        "tags:\n- decision-making\n- classification\n- routing\n- calibrated\n- locket-research",
        ("datasets:\n" + "\n".join(f"- {d}" for d in datasets)) if datasets else None,
        "---",
    ]
    parts = ["\n".join(x for x in front if x)]
    parts.append(
        f"""
# Locket Decider² (7B)

An open-weights model for enterprise decisions: given an input and a list of options, it chooses
one, reports a calibrated confidence, and explains the choice in one sentence. Fine-tuned from
[{config.base_model}](https://huggingface.co/{config.base_model}) by Locket Research.

## Use

Option scoring, calibrated probabilities and `needs_review` come from the
[`locket-decider`](https://github.com/burkiuze/LocketDecider-) package:

```python
from locket_decider import Decision, Option
from locket_decider.decider import Decider

decider = Decider.load("{repo}")
decider.decide(Decision(
    input="I was charged twice for my March invoice and need a refund before Friday.",
    options=[Option("billing"), Option("technical"), Option("sales")],
))
# {{"choice": "A", "label": "billing", "confidence": ..., "needs_review": ..., "reason": "..."}}
```

It is also a standard chat model, so it runs on any OpenAI-compatible server
(`vllm serve {repo}`) with the prompt format in
[`schema.py`](https://github.com/burkiuze/LocketDecider-/blob/main/locket_decider/schema.py); it then
answers with JSON text instead of scored probabilities.

## Calibration

| | |
|---|---|
| Softmax temperature | {config.temperature:.3f} |
| Review threshold | {config.min_confidence:.3f} |
"""
    )
    cal = config.extra.get("calibration")
    if cal:
        parts.append(
            f"Fitted on {cal['validation_n']} validation decisions so that automated decisions reach "
            f"{cal['target_accuracy']:.0%} accuracy; expected calibration error went from "
            f"{cal['ece_before']:.3f} to {cal['ece_after']:.3f}.\n"
        )
    parts.append("## Evaluation\n")
    if trained:
        parts.append(
            "Held-out test splits, rewritten as decisions with shuffled options (see the repository). "
            + ("The last column is the untouched base model scored the same way.\n\n" if base else "\n\n")
            + results_table(trained, base)
            + "\n"
        )
        sel = trained["overall"].get("selective")
        if sel:
            parts.append(
                f"\nAt the review threshold, {sel['coverage']:.1%} of decisions are automated at "
                f"{sel['accuracy']:.1%} accuracy; the rest are flagged `needs_review`.\n"
            )
    else:
        parts.append("Not evaluated yet.\n")
    parts.append(
        """
## Training data

"""
        + "\n".join(f"- `{s}` ({SOURCES[s][2]})" for s in sources if s in SOURCES)
        + """

## Limitations

- It decides between the options it is given; it does not know facts outside the input.
- Confidence is calibrated on the evaluation distribution above; recalibrate (`locket-calibrate`)
  on your own data before automating decisions with it.
- Keep a person in the loop for decisions with legal, medical, financial or safety consequences.
"""
    )
    return "\n".join(parts)


def _read(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="merged model folder from locket-merge")
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. my-org/locket-decider-2-7b")
    ap.add_argument("--eval", help="locket-eval --out JSON for the trained model")
    ap.add_argument("--base-eval", help="locket-eval --out JSON for the base model")
    ap.add_argument("--train-config", default="configs/decider-2-7b.json")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write README.md into the model folder but do not upload")
    args = ap.parse_args(argv)

    with open(args.train_config, encoding="utf-8") as f:
        sources = list(json.load(f)["data"].get("train", {}))
    card = model_card(args.repo, DeciderConfig.read(args.model), _read(args.eval), _read(args.base_eval), sources)
    with open(os.path.join(args.model, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"model card written to {args.model}/README.md")
    if args.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=args.model, commit_message="Release Locket Decider² 7B")
    print(f"uploaded: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
