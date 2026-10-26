import random
from collections import Counter

from locket_decider.data import Record, make_decision
from locket_decider.schema import NONE_LABEL

LABELS = [f"intent {i}" for i in range(40)]


def test_answer_is_always_correct_and_options_unique():
    rng = random.Random(0)
    for _ in range(2000):
        d = make_decision(Record("text", "intent 3", LABELS, "task"), rng)
        gold = d.options[d.answer]
        assert gold.label == "intent 3" or (gold.abstain and all(o.label != "intent 3" for o in d.options))
        assert len({o.label for o in d.options}) == len(d.options)
        assert 2 <= len(d.options) <= 13


def test_none_of_the_above_is_last_and_sometimes_a_distractor():
    rng = random.Random(1)
    outcomes = Counter()
    for _ in range(4000):
        d = make_decision(Record("text", "intent 3", LABELS, "task"), rng, p_none=0.1)
        nones = [i for i, o in enumerate(d.options) if o.abstain]
        if nones:
            assert nones == [len(d.options) - 1] and d.options[-1].label == NONE_LABEL
            outcomes["none right" if d.answer == nones[0] else "none distractor"] += 1
    assert 0.05 < outcomes["none right"] / 4000 < 0.15
    assert outcomes["none distractor"] > 100


def test_answer_position_is_not_biased():
    rng = random.Random(2)
    first = sum(make_decision(Record("t", "intent 3", LABELS, "task"), rng, min_options=8, max_options=8, p_none=0).answer == 0 for _ in range(4000))
    assert abs(first / 4000 - 1 / 8) < 0.03


def test_out_of_scope_records_answer_none():
    rng = random.Random(3)
    d = make_decision(Record("text", "oos", LABELS, "task", abstain=True), rng)
    assert d.options[d.answer].abstain


def test_binary_labels():
    rng = random.Random(4)
    for _ in range(200):
        d = make_decision(Record("q", "yes", ["yes", "no"], "task"), rng)
        assert {o.label for o in d.options} >= {"yes", "no"}
        assert d.options[d.answer].label == "yes"
