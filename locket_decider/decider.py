"""Loading Decider² and making decisions with it."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .calibration import softmax
from .schema import Decision

CONFIG_FILE = "decider_config.json"


@dataclass
class DeciderConfig:
    temperature: float = 1.0  # fitted by `locket-calibrate`
    min_confidence: float = 0.0  # below this, decisions are flagged for a person
    base_model: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: str) -> "DeciderConfig":
        file = os.path.join(path, CONFIG_FILE)
        if not os.path.exists(file):
            return cls()
        with open(file, encoding="utf-8") as f:
            raw = json.load(f)
        known = {k: raw.pop(k) for k in ("temperature", "min_confidence", "base_model") if k in raw}
        return cls(**known, extra=raw)

    def write(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        data = {"temperature": self.temperature, "min_confidence": self.min_confidence, "base_model": self.base_model, **self.extra}
        with open(os.path.join(path, CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")


def load_model(path: str, adapter: str | None = None, load_in_4bit: bool = False, dtype: str = "bfloat16"):
    """Returns (model, tokenizer). `adapter` applies a LoRA adapter on top of `path`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"dtype": getattr(torch, dtype), "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype)
        )
    tokenizer = AutoTokenizer.from_pretrained(adapter or path)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


class Decider:
    def __init__(self, model, tokenizer, config: DeciderConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DeciderConfig()

    @classmethod
    def load(cls, path: str, adapter: str | None = None, **kwargs) -> "Decider":
        model, tokenizer = load_model(path, adapter, **kwargs)
        return cls(model, tokenizer, DeciderConfig.read(adapter or path))

    def scores(self, d: Decision):
        from .scoring import option_scores

        return option_scores(self.model, self.tokenizer, d)

    def decide(self, d: Decision, explain: bool = True, min_confidence: float | None = None) -> dict:
        probs = softmax(self.scores(d), self.config.temperature)
        best = int(probs.argmax())
        threshold = self.config.min_confidence if min_confidence is None else min_confidence
        result = {
            "choice": d.keys[best],
            "label": d.options[best].label,
            "confidence": round(float(probs[best]), 4),
            "probabilities": {k: round(float(p), 4) for k, p in zip(d.keys, probs)},
            # Route to a person when unsure, or when the model picked "none of the above".
            "needs_review": bool(probs[best] < threshold or d.options[best].abstain),
        }
        if explain:
            from .scoring import explain as gen_reason

            result["reason"] = gen_reason(self.model, self.tokenizer, d, d.keys[best])
        return result
