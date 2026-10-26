"""Synthetic enterprise decisions from a teacher model, for the part public datasets don't cover.

    OPENAI_API_KEY=... locket-distill --out data/enterprise.jsonl --n 2000 \
        --teacher <model id> [--base-url https://openrouter.ai/api/v1] [--max-usd 5]

For each example the teacher writes a realistic scenario with options and a correct answer plus a
one-sentence reason. The teacher then answers the same decision blind, with the options
reshuffled; examples where the two answers disagree are dropped. Output lines are the same JSON
the API accepts, and go into configs/*.json -> data.jsonl.train.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request

from .schema import Decision, NONE_DESCRIPTION, NONE_LABEL, Option, decision_to_dict, messages, parse_answer

DOMAINS = [
    "customer support ticket triage",
    "refund and chargeback approval",
    "invoice and expense approval",
    "fraud and transaction review",
    "loan and credit pre-screening",
    "insurance claim triage",
    "IT helpdesk routing",
    "security alert triage",
    "content moderation against a written policy",
    "sales lead qualification",
    "HR request routing",
    "procurement and vendor risk checks",
    "contract clause review",
    "document classification for back office",
    "AI agent tool selection",
    "choosing which model or workflow should handle a request",
    "public-sector citizen request routing",
    "healthcare appointment and message triage (no diagnosis)",
    "logistics exception handling",
    "e-commerce order issue handling",
]

LANGUAGES = ["English"] * 4 + ["Turkish"]

WRITE_PROMPT = """Write one realistic decision a company would want automated, in the domain: {domain}.
Language for the input text and option labels: {language}.

Return JSON only, with these keys:
- "task": one sentence telling the decider what to decide
- "criteria": the company's rule(s) for this decision, 1-3 sentences (may be empty)
- "input": the message, record or document to decide on (2-8 sentences, concrete details, no placeholders)
- "options": {k} short option labels, each optionally with a description, as [{{"label": ..., "description": ...}}]
- "answer": the index (0-based) of the correct option
- "reason": one sentence explaining why, citing the input and criteria

Make it {difficulty}. The correct answer must follow from the input and criteria alone."""


def chat(base_url: str, key: str, model: str, msgs: list[dict], temperature: float, json_mode: bool = True) -> tuple[str, float]:
    body = {"model": model, "messages": msgs, "temperature": temperature, "usage": {"include": True}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"], float(out.get("usage", {}).get("cost") or 0)


def write_one(args, key: str, rng: random.Random) -> tuple[Decision | None, float]:
    prompt = WRITE_PROMPT.format(
        domain=rng.choice(DOMAINS),
        language=rng.choice(LANGUAGES),
        k=rng.randint(2, 7),
        difficulty=rng.choice(["straightforward", "moderately tricky", "hard, with a plausible distractor"]),
    )
    text, cost = chat(args.base_url, key, args.teacher, [{"role": "user", "content": prompt}], temperature=1.0)
    try:
        obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
        opts = [Option(o["label"], o.get("description", "")) for o in obj["options"]]
        answer = int(obj["answer"])
        if rng.random() < 0.15:  # sometimes the right option is missing: teach abstaining
            opts.pop(answer)
            answer = None
        order = list(range(len(opts)))
        rng.shuffle(order)
        new_answer = order.index(answer) if answer is not None else None
        opts = [opts[i] for i in order]
        opts.append(Option(NONE_LABEL, NONE_DESCRIPTION, abstain=True))
        d = Decision(
            input=obj["input"],
            options=opts,
            task=obj["task"],
            criteria=obj.get("criteria", ""),
            answer=new_answer if new_answer is not None else len(opts) - 1,
            reason=obj["reason"] if answer is not None else "",
            source="distill",
        )
    except (ValueError, KeyError, TypeError, IndexError):
        return None, cost
    return d, cost


def verify(args, key: str, d: Decision) -> tuple[bool, float]:
    """The teacher decides blind; keep the example only if it agrees with the written answer."""
    text, cost = chat(args.base_url, key, args.teacher, messages(d), temperature=0.0)
    choice, _ = parse_answer(text, d.keys)
    return choice == d.keys[d.answer], cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="number of verified examples to write")
    ap.add_argument("--teacher", required=True, help="teacher model id on the API")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop before reported spend exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY (or OPENROUTER_API_KEY)")
    rng = random.Random(args.seed)
    spent, kept, tried = 0.0, 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        while kept < args.n and spent < args.max_usd:
            tried += 1
            d, cost = write_one(args, key, rng)
            spent += cost
            if d is None:
                continue
            ok, cost = verify(args, key, d)
            spent += cost
            if ok:
                f.write(json.dumps(decision_to_dict(d), ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if tried % 20 == 0:
                print(f"kept {kept}/{tried}  spent ${spent:.2f}", file=sys.stderr)
    print(f"wrote {kept} verified decisions to {args.out} (tried {tried}, spent ${spent:.2f})")


if __name__ == "__main__":
    main()
