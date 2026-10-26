"""Decision requests, the prompt Decider² is trained on, and its answer format."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You are Locket Decider², a decision model. Read the request and choose exactly one of the "
    "listed options. Answer with JSON only: {\"choice\": \"<option key>\", \"reason\": \"<one short "
    "sentence>\"}. Base the choice only on the input and the criteria. If an option says none of "
    "the others fit, choose it when that is true."
)

DEFAULT_TASK = "Choose the option that best fits the input."
NONE_LABEL = "none of the above"
NONE_DESCRIPTION = "None of the other options fits the input."

# The answer always starts like this, so option scoring can condition on it.
ANSWER_PREFIX = '{"choice": "'

KEYS = string.ascii_uppercase
MAX_OPTIONS = len(KEYS)


@dataclass
class Option:
    label: str
    description: str = ""
    abstain: bool = False  # "none of the above"-style option


@dataclass
class Decision:
    input: str
    options: list[Option]
    task: str = DEFAULT_TASK
    criteria: str = ""
    answer: int | None = None  # index into options, for training and evaluation
    reason: str = ""
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 2 <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"a decision needs 2 to {MAX_OPTIONS} options, got {len(self.options)}")
        if self.answer is not None and not 0 <= self.answer < len(self.options):
            raise ValueError(f"answer {self.answer} is out of range")

    @property
    def keys(self) -> list[str]:
        return list(KEYS[: len(self.options)])


def render_user(d: Decision) -> str:
    lines = [f"Task: {d.task.strip()}"]
    if d.criteria.strip():
        lines.append(f"Criteria: {d.criteria.strip()}")
    lines += ["", "Input:", d.input.strip(), "", "Options:"]
    for key, opt in zip(d.keys, d.options):
        line = f"{key}. {opt.label.strip()}"
        if opt.description.strip():
            line += f" — {opt.description.strip()}"
        lines.append(line)
    return "\n".join(lines)


def messages(d: Decision) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user(d)},
    ]


def render_answer(d: Decision) -> str:
    """Training target. Examples without a rationale still teach the choice."""
    if d.answer is None:
        raise ValueError("decision has no answer")
    out = {"choice": d.keys[d.answer]}
    if d.reason.strip():
        out["reason"] = d.reason.strip()
    return json.dumps(out, ensure_ascii=False)


def to_example(d: Decision) -> dict:
    """Prompt/completion record in the conversational format TRL's SFTTrainer expects."""
    return {
        "prompt": messages(d),
        "completion": [{"role": "assistant", "content": render_answer(d)}],
        "source": d.source,
    }


def parse_answer(text: str, keys: list[str]) -> tuple[str | None, str]:
    """Best-effort parse of a generated answer; returns (key or None, reason)."""
    text = text.strip()
    try:
        obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
        key = str(obj.get("choice", "")).strip().upper()
        return (key if key in keys else None), str(obj.get("reason", "")).strip()
    except (ValueError, AttributeError):
        return None, ""


def decision_from_dict(obj: dict) -> Decision:
    """Builds a Decision from API/JSONL input: options may be strings or {label, description, abstain}."""
    opts = []
    for o in obj["options"]:
        if isinstance(o, str):
            opts.append(Option(label=o))
        else:
            opts.append(Option(label=o["label"], description=o.get("description", ""), abstain=bool(o.get("abstain"))))
    return Decision(
        input=obj["input"],
        options=opts,
        task=obj.get("task") or DEFAULT_TASK,
        criteria=obj.get("criteria", ""),
        answer=obj.get("answer"),
        reason=obj.get("reason", ""),
        source=obj.get("source", ""),
    )


def decision_to_dict(d: Decision) -> dict:
    return {
        "input": d.input,
        "options": [{"label": o.label, "description": o.description, "abstain": o.abstain} for o in d.options],
        "task": d.task,
        "criteria": d.criteria,
        "answer": d.answer,
        "reason": d.reason,
        "source": d.source,
    }
