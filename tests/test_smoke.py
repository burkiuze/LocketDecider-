"""End to end on a tiny CPU model: train -> calibrate -> eval -> merge -> decide -> serve."""

import json

import numpy as np
import pytest

for mod in ("torch", "transformers", "trl", "peft", "datasets", "fastapi", "httpx"):
    pytest.importorskip(mod)

from locket_decider import evaluate, merge, scoring, train  # noqa: E402
from locket_decider.decider import Decider  # noqa: E402
from locket_decider.schema import Decision, Option  # noqa: E402

from .tiny import build_tiny_model, toy_decisions, write_jsonl  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    base = root / "tiny-base"
    build_tiny_model(base)
    for split, n, seed in (("train", 64, 1), ("validation", 24, 2), ("eval", 24, 3)):
        write_jsonl(root / f"{split}.jsonl", toy_decisions(n, seed))
    cfg = {
        "name": "tiny-decider",
        "base_model": str(base),
        "seed": 0,
        "data": {"jsonl": {s: [str(root / f"{s}.jsonl")] for s in ("train", "validation", "eval")}},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": "all-linear"},
        "train": {
            "learning_rate": 1e-3,
            "num_train_epochs": 1,
            "max_steps": 4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "max_length": 512,
            "warmup_ratio": 0.0,
            "lr_scheduler_type": "constant",
            "weight_decay": 0.0,
            "logging_steps": 1,
            "save_steps": 1000,
            "eval_steps": 2,
            "dtype": "float32",
        },
    }
    cfg_path = root / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    adapter = root / "adapter"
    train.main(["--config", str(cfg_path), "--out", str(adapter)])
    return {"root": root, "base": base, "cfg": cfg_path, "adapter": adapter}


def test_calibrate_writes_temperature_and_threshold(run):
    evaluate.main_calibrate(["--model", str(run["base"]), "--adapter", str(run["adapter"]), "--config", str(run["cfg"])])
    conf = json.loads((run["adapter"] / "decider_config.json").read_text())
    assert conf["temperature"] > 0 and 0 <= conf["min_confidence"] <= 1
    assert "ece_after" in conf["calibration"]


def test_eval_report(run):
    out = run["root"] / "eval.json"
    evaluate.main_eval(["--model", str(run["base"]), "--adapter", str(run["adapter"]), "--config", str(run["cfg"]), "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rep["overall"]["n"] == 24 and 0 <= rep["overall"]["accuracy"] <= 1
    assert set(rep["by_source"]) == {"toy"}


def test_merge_and_decide(run):
    merged = run["root"] / "merged"
    merge.main(["--adapter", str(run["adapter"]), "--out", str(merged)])
    decider = Decider.load(str(merged), dtype="float32")
    d = Decision(input="Customer says: charged twice.", options=[Option("billing"), Option("technical"), Option("sales")])
    res = decider.decide(d, explain=True)
    assert res["choice"] in d.keys and res["label"] == d.options[d.keys.index(res["choice"])].label
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
    assert isinstance(res["reason"], str) and isinstance(res["needs_review"], bool)


def test_multi_token_path_matches_option_order(run, monkeypatch):
    decider = Decider.load(str(run["base"]), dtype="float32")
    d = Decision(input="Customer says: tracking number.", options=[Option("shipping"), Option("sales")])
    # Force candidates that share their first new token, so the batched path runs.
    monkeypatch.setattr(scoring, "candidate_texts", lambda prompt, keys: [prompt + 'Option "' + k + '"' for k in keys])
    s = decider.scores(d)
    assert s.shape == (2,) and np.all(np.isfinite(s)) and s[0] != s[1]


def test_server(run):
    from fastapi.testclient import TestClient

    from locket_decider.serve import create_app

    client = TestClient(create_app(Decider.load(str(run["base"]), dtype="float32")))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/v1/decide", json={"input": "Customer says: app crashes.", "options": ["technical", {"label": "billing"}], "explain": False})
    assert r.status_code == 200 and r.json()["choice"] in ("A", "B")
    assert client.post("/v1/decide", json={"input": "x", "options": ["only one"]}).status_code == 422
