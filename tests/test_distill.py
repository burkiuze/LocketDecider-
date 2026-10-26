import json
import random
from types import SimpleNamespace

from locket_decider import distill

WRITTEN = {
    "task": "Decide what to do with this refund request.",
    "criteria": "Refunds over $500 need a manager.",
    "input": "Customer asks for a $900 refund on order 4471.",
    "options": [{"label": "approve"}, {"label": "escalate to manager"}, {"label": "reject"}],
    "answer": 1,
    "reason": "The refund is above the $500 limit.",
}


def fake_chat(replies):
    calls = iter(replies)
    return lambda *a, **k: (next(calls), 0.01)


def test_written_decision_keeps_the_right_answer(monkeypatch):
    args = SimpleNamespace(base_url="x", teacher="t")
    for seed in range(30):
        monkeypatch.setattr(distill, "chat", fake_chat([json.dumps(WRITTEN)]))
        d, cost = distill.write_one(args, "key", random.Random(seed))
        assert cost == 0.01 and d.options[-1].abstain
        gold = d.options[d.answer]
        assert gold.label == "escalate to manager" or (gold.abstain and all(o.label != "escalate to manager" for o in d.options))


def test_verification_drops_disagreement(monkeypatch):
    args = SimpleNamespace(base_url="x", teacher="t")
    monkeypatch.setattr(distill, "chat", fake_chat([json.dumps(WRITTEN)]))
    d, _ = distill.write_one(args, "key", random.Random(0))
    right, wrong = d.keys[d.answer], d.keys[(d.answer + 1) % len(d.keys)]
    monkeypatch.setattr(distill, "chat", fake_chat([json.dumps({"choice": right})]))
    assert distill.verify(args, "key", d)[0]
    monkeypatch.setattr(distill, "chat", fake_chat([json.dumps({"choice": wrong})]))
    assert not distill.verify(args, "key", d)[0]


def test_malformed_teacher_output_is_skipped(monkeypatch):
    monkeypatch.setattr(distill, "chat", fake_chat(["not json"]))
    d, _ = distill.write_one(SimpleNamespace(base_url="x", teacher="t"), "key", random.Random(0))
    assert d is None
