"""Scores a model on held-out decisions, and calibrates its confidence.

    locket-eval      --model PATH [--adapter DIR] [--split eval]       -> metrics JSON + table
    locket-calibrate --model PATH  --adapter DIR  [--target-accuracy 0.95]

Calibration fits a softmax temperature on the validation split and picks the lowest confidence
threshold whose automated decisions still reach the target accuracy; both are written to the
model's decider_config.json and used by `Decider.decide`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from . import data
from .calibration import ece, fit_temperature, macro_f1, nll, selective, softmax, threshold_for_accuracy
from .decider import Decider, DeciderConfig
from .train import read_config


def collect(decider: Decider, decisions, progress: bool = True) -> list[dict]:
    rows = []
    for i, d in enumerate(decisions):
        rows.append(
            {
                "source": d.source,
                "scores": decider.scores(d).tolist(),
                "answer": d.answer,
                "labels": [o.label for o in d.options],
            }
        )
        if progress and (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(decisions)}", file=sys.stderr)
    return rows


def metrics(rows: list[dict], temperature: float, threshold: float | None = None) -> dict:
    probs = [softmax(r["scores"], temperature) for r in rows]
    pred = [int(p.argmax()) for p in probs]
    conf = [float(p.max()) for p in probs]
    correct = [p == r["answer"] for p, r in zip(pred, rows)]
    out = {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if rows else 0.0,
        "macro_f1": macro_f1([r["labels"][p] for p, r in zip(pred, rows)], [r["labels"][r["answer"]] for r in rows]),
        "ece": ece(conf, correct),
        "nll": nll([r["scores"] for r in rows], [r["answer"] for r in rows], temperature),
    }
    if threshold is not None:
        out["selective"] = selective(conf, correct, threshold)
    return out


def report(rows: list[dict], temperature: float, threshold: float | None) -> dict:
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    return {
        "temperature": temperature,
        "overall": metrics(rows, temperature, threshold),
        "by_source": {s: metrics(rs, temperature, threshold) for s, rs in sorted(by_source.items())},
    }


def table(rep: dict) -> str:
    lines = ["| source | n | accuracy | macro F1 | ECE |", "|---|---:|---:|---:|---:|"]
    for name, m in [*rep["by_source"].items(), ("**overall**", rep["overall"])]:
        lines.append(f"| {name} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['ece']:.3f} |")
    sel = rep["overall"].get("selective")
    if sel:
        lines.append(
            f"\nAt confidence ≥ {sel['threshold']:.3f}: {sel['coverage']:.1%} of decisions automated "
            f"at {sel['accuracy']:.1%} accuracy."
        )
    return "\n".join(lines)


def _common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--model", required=True, help="base model id/path, or a merged Decider² model")
    ap.add_argument("--adapter", help="LoRA adapter directory from locket-train")
    ap.add_argument("--config", default="configs/decider-2-7b.json")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of decisions scored")
    ap.add_argument("--qlora", action="store_true", help="load the model in 4-bit")


def _decisions(cfg: dict, split: str, limit: int):
    ds = data.for_split(cfg, split)
    return ds[:limit] if limit else ds


def main_eval(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(ap)
    ap.add_argument("--split", default="eval", choices=["eval", "validation"])
    ap.add_argument("--temperature", type=float, help="override the calibrated temperature (1.0 = uncalibrated)")
    ap.add_argument("--out", help="write the full report JSON here")
    args = ap.parse_args(argv)

    cfg = read_config(args.config)
    decider = Decider.load(args.model, args.adapter, load_in_4bit=args.qlora)
    rows = collect(decider, _decisions(cfg, args.split, args.limit))
    t = args.temperature if args.temperature is not None else decider.config.temperature
    rep = report(rows, t, decider.config.min_confidence or None)
    print(table(rep))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)


def main_calibrate(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(ap)
    ap.add_argument("--target-accuracy", type=float, default=0.95, help="accuracy required of automated decisions")
    args = ap.parse_args(argv)

    cfg = read_config(args.config)
    decider = Decider.load(args.model, args.adapter, load_in_4bit=args.qlora)
    rows = collect(decider, _decisions(cfg, "validation", args.limit))
    scores, answers = [r["scores"] for r in rows], [r["answer"] for r in rows]

    t = fit_temperature(scores, answers)
    probs = [softmax(s, t) for s in scores]
    conf = [float(p.max()) for p in probs]
    correct = [int(p.argmax()) == a for p, a in zip(probs, answers)]
    threshold = threshold_for_accuracy(conf, correct, args.target_accuracy)

    out_dir = args.adapter or args.model
    config = DeciderConfig.read(out_dir)
    config.temperature = t
    config.min_confidence = threshold if threshold is not None else 1.0
    config.extra["calibration"] = {
        "target_accuracy": args.target_accuracy,
        "validation_n": len(rows),
        "ece_before": ece([float(softmax(s).max()) for s in scores], [int(np.argmax(s)) == a for s, a in zip(scores, answers)]),
        "ece_after": ece(conf, correct),
    }
    config.write(out_dir)
    print(f"temperature {t:.3f}, min_confidence {config.min_confidence:.3f} -> {out_dir}/decider_config.json")
    print(table(report(rows, t, config.min_confidence)))


if __name__ == "__main__":
    main_eval()
