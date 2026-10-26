import json

import pytest

from locket_decider.schema import (
    ANSWER_PREFIX,
    Decision,
    Option,
    decision_from_dict,
    decision_to_dict,
    messages,
    parse_answer,
    render_answer,
    to_example,
)


def ticket(answer=0, reason=""):
    return Decision(
        input="I was charged twice for my March invoice.",
        options=[Option("billing", "Payments and refunds"), Option("technical"), Option("sales")],
        criteria="Refunds go to billing.",
        answer=answer,
        reason=reason,
    )


def test_prompt_lists_options_with_keys_and_descriptions():
    user = messages(ticket())[1]["content"]
    assert "A. billing — Payments and refunds" in user
    assert "B. technical" in user and "C. sales" in user
    assert "Criteria: Refunds go to billing." in user


def test_answer_starts_with_the_scored_prefix():
    assert render_answer(ticket(1)).startswith(ANSWER_PREFIX + "B")
    assert json.loads(render_answer(ticket(0, "Duplicate charge."))) == {"choice": "A", "reason": "Duplicate charge."}


def test_example_is_prompt_completion():
    ex = to_example(ticket())
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["completion"][0]["role"] == "assistant"


def test_option_count_is_checked():
    with pytest.raises(ValueError):
        Decision(input="x", options=[Option("only")])
    with pytest.raises(ValueError):
        Decision(input="x", options=[Option("a"), Option("b")], answer=2)


def test_parse_answer():
    keys = ["A", "B"]
    assert parse_answer('{"choice": "b", "reason": "ok"}', keys) == ("B", "ok")
    assert parse_answer('Sure: {"choice": "Z"}', keys) == (None, "")
    assert parse_answer("no json", keys) == (None, "")


def test_dict_round_trip():
    d = decision_from_dict({"input": "x", "options": ["a", {"label": "b", "abstain": True}], "answer": 1})
    assert d.options[1].abstain and d.answer == 1
    assert decision_from_dict(decision_to_dict(d)) == d
