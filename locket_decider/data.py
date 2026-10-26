"""Training data: public, permissively licensed datasets turned into decisions.

Every source is rewritten as "input + a shuffled subset of options", so the model learns to
read the options it is given rather than memorise a fixed label set. A share of examples drop
the right answer and add "none of the above", which teaches the model to abstain.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from .schema import NONE_DESCRIPTION, NONE_LABEL, MAX_OPTIONS, Decision, Option


@dataclass
class Record:
    """One labelled item before option sampling."""

    input: str
    answer: str  # label name
    labels: list[str]  # the full label set this item is drawn from
    task: str
    descriptions: dict[str, str] | None = None
    abstain: bool = False  # true when no label fits (e.g. out-of-scope queries)


def make_decision(
    rec: Record,
    rng: random.Random,
    min_options: int = 2,
    max_options: int = 12,
    p_none: float = 0.1,
    source: str = "",
) -> Decision:
    """Samples distractors and shuffles them; sometimes the answer is dropped for "none of the above".

    "None of the above" is always the last option when present, as people usually write it. It also
    appears as a distractor when the answer is listed, so its presence is never a giveaway.
    """
    desc = rec.descriptions or {}
    others = [lab for lab in rec.labels if lab != rec.answer]
    most = min(max_options, MAX_OPTIONS - 1)
    drop = rec.abstain or (len(others) >= 2 and rng.random() < p_none)

    if drop:
        hi = min(most, len(others))
        picked = rng.sample(others, rng.randint(min(min_options, hi), hi))
    else:
        hi = min(most, len(others) + 1)
        picked = rng.sample(others, rng.randint(min(min_options, hi), hi) - 1) + [rec.answer]
    rng.shuffle(picked)
    opts = [Option(lab, desc.get(lab, "")) for lab in picked]

    if drop or rng.random() < p_none:
        opts.append(Option(NONE_LABEL, NONE_DESCRIPTION, abstain=True))
    answer = len(opts) - 1 if drop else picked.index(rec.answer)
    return Decision(input=rec.input, options=opts, task=rec.task, answer=answer, source=source)


def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip()


# --- sources ---------------------------------------------------------------------------------
# Each loader yields Records from one split. `datasets` is imported lazily so the rest of the
# package (and its tests) work without it.


def _load(name: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split)


def banking77(split: str) -> Iterator[Record]:
    ds = _load("PolyAI/banking77", None, split)
    names = [humanize(n) for n in ds.features["label"].names]
    for row in ds:
        yield Record(row["text"], names[row["label"]], names, "Route this banking customer message to the right intent.")


def clinc_oos(split: str) -> Iterator[Record]:
    ds = _load("clinc_oos", "plus", split)
    names = [humanize(n) for n in ds.features["intent"].names]
    in_scope = [n for n in names if n != "oos"]
    for row in ds:
        name = names[row["intent"]]
        yield Record(
            row["text"],
            name,
            in_scope,
            "Route this assistant request to the right intent.",
            abstain=name == "oos",
        )


def massive(split: str, locale: str = "en-US") -> Iterator[Record]:
    ds = _load("AmazonScience/massive", locale, split)
    names = [humanize(n) for n in ds.features["intent"].names]
    for row in ds:
        yield Record(row["utt"], names[row["intent"]], names, "Route this voice-assistant request to the right intent.")


NLI = {
    "entailment": "The statement must be true given the passage.",
    "neutral": "The passage neither confirms nor rules out the statement.",
    "contradiction": "The statement cannot be true given the passage.",
}


def multi_nli(split: str) -> Iterator[Record]:
    ds = _load("nyu-mll/multi_nli", None, split)
    names = ["entailment", "neutral", "contradiction"]
    for row in ds:
        if row["label"] not in (0, 1, 2):
            continue
        yield Record(
            f"Passage: {row['premise']}\nStatement: {row['hypothesis']}",
            names[row["label"]],
            names,
            "Decide how the statement relates to the passage.",
            descriptions=NLI,
        )


def boolq(split: str) -> Iterator[Record]:
    ds = _load("google/boolq", None, split)
    for row in ds:
        yield Record(
            f"Passage: {row['passage']}\nQuestion: {row['question']}",
            "yes" if row["answer"] else "no",
            ["yes", "no"],
            "Answer the question using only the passage.",
        )


def jsonl(path: str) -> Iterator[Decision]:
    """Ready-made decisions (e.g. from distill.py), one JSON object per line."""
    from .schema import decision_from_dict

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield decision_from_dict(json.loads(line))


# name -> (loader, {our split: dataset split}, licence)
SOURCES: dict[str, tuple[Callable[[str], Iterable[Record]], dict[str, str], str]] = {
    "banking77": (banking77, {"train": "train", "eval": "test"}, "CC-BY-4.0"),
    "clinc_oos": (clinc_oos, {"train": "train", "validation": "validation", "eval": "test"}, "CC-BY-3.0"),
    "massive_en": (lambda s: massive(s, "en-US"), {"train": "train", "validation": "validation", "eval": "test"}, "CC-BY-4.0"),
    "massive_tr": (lambda s: massive(s, "tr-TR"), {"train": "train", "validation": "validation", "eval": "test"}, "CC-BY-4.0"),
    "multi_nli": (multi_nli, {"train": "train", "validation": "validation_matched", "eval": "validation_mismatched"}, "OANC / CC-BY-SA-3.0"),
    "boolq": (boolq, {"train": "train", "eval": "validation"}, "CC-BY-SA-3.0"),
}


def build(
    sources: dict[str, int],
    split: str,
    seed: int = 0,
    p_none: float = 0.1,
    max_options: int = 12,
) -> list[Decision]:
    """Samples up to N decisions per source for one split ("train", "validation" or "eval").

    Sources without a validation split carve it from the end of train, which is then excluded
    from the train split, so the three never overlap.
    """
    rng = random.Random(f"{seed}:{split}")
    out: list[Decision] = []
    for name, limit in sources.items():
        loader, splits, _ = SOURCES[name]
        carve = split in ("train", "validation") and "validation" not in splits
        records = list(loader(splits["train" if carve else split]))
        if carve:
            records.sort(key=lambda r: r.input)  # deterministic carve independent of load order
            random.Random(f"{seed}:carve:{name}").shuffle(records)
            n_val = max(1, len(records) // 20)
            records = records[:n_val] if split == "validation" else records[n_val:]
        rng.shuffle(records)
        for rec in records[:limit]:
            out.append(make_decision(rec, rng, max_options=max_options, p_none=p_none, source=name))
    rng.shuffle(out)
    return out


def for_split(cfg: dict, split: str) -> list[Decision]:
    """All decisions for one split of a training config: sampled sources plus any JSONL files."""
    dcfg = cfg["data"]
    out = build(dcfg.get(split, {}), split, seed=cfg.get("seed", 0), p_none=dcfg.get("p_none", 0.1), max_options=dcfg.get("max_options", 12))
    for path in dcfg.get("jsonl", {}).get(split, []):
        out.extend(jsonl(path))
    return out
