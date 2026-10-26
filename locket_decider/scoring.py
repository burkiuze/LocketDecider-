"""Option scoring: the probability the model gives each option key as its answer.

Instead of sampling an answer and parsing it, Decider² reads the log-probability of every option
key after `{"choice": "`. The choice is always one of the listed options, and the softmax over
those log-probabilities (temperature-scaled) is the confidence it reports.
"""

from __future__ import annotations

import numpy as np

from .schema import ANSWER_PREFIX, Decision, messages


def common_prefix_len(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        tok = seqs[0][i]
        if any(s[i] != tok for s in seqs[1:]):
            return i
    return n


def candidate_texts(prompt: str, keys: list[str]) -> list[str]:
    # The closing quote ends the key, so "A" can't win as a prefix of a longer key.
    return [prompt + ANSWER_PREFIX + k + '"' for k in keys]


def prompt_text(tokenizer, d: Decision) -> str:
    return tokenizer.apply_chat_template(messages(d), tokenize=False, add_generation_prompt=True)


def option_scores(model, tokenizer, d: Decision, batch_size: int = 16) -> np.ndarray:
    """Summed log-probability of each option's continuation, in option order.

    Every candidate is scored from the longest token prefix they all share, so a tokenizer that
    merges the opening quote with the key still compares like with like.
    """
    import torch

    seqs = [tokenizer(t, add_special_tokens=False).input_ids for t in candidate_texts(prompt_text(tokenizer, d), d.keys)]
    shared = common_prefix_len(seqs)
    device = next(model.parameters()).device

    with torch.no_grad():
        firsts = [s[shared] for s in seqs]
        if len(set(firsts)) == len(firsts):
            # Fast path: the keys already differ at the first new token (always true for
            # single-letter keys), so one forward pass over the shared prefix decides it.
            ids = torch.tensor([seqs[0][:shared]], device=device)
            logp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
            return logp[firsts].cpu().numpy().astype(np.float64)

        scores = []
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            width = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), width), pad, device=device)
            mask = torch.zeros_like(ids)
            for r, s in enumerate(chunk):
                ids[r, : len(s)] = torch.tensor(s, device=device)
                mask[r, : len(s)] = 1
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
            for r, s in enumerate(chunk):
                targets = torch.tensor(s[shared:], device=device)
                pos = torch.arange(shared - 1, len(s) - 1, device=device)
                scores.append(logp[r, pos, targets].sum().item())
        return np.asarray(scores, dtype=np.float64)


def explain(model, tokenizer, d: Decision, key: str, max_new_tokens: int = 60) -> str:
    """Generates the one-sentence reason for an already chosen key."""
    import torch

    from .schema import parse_answer

    text = prompt_text(tokenizer, d) + ANSWER_PREFIX + key + '", "reason": "'
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    tail = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
    _, reason = parse_answer('{"choice": "' + key + '", "reason": "' + tail, d.keys)
    if reason:
        return reason
    return tail.split('"')[0].strip()
