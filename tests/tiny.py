"""A tiny randomly initialised chat model and a toy routing dataset, for smoke tests on CPU."""

import json
import random

from locket_decider.data import Record, make_decision
from locket_decider.schema import decision_to_dict, messages, render_answer

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

TOPICS = {
    "billing": ["charged twice", "refund my payment", "invoice is wrong"],
    "technical": ["app crashes", "cannot log in", "error on upload"],
    "sales": ["pricing for a team", "buy more seats", "talk to sales"],
    "shipping": ["parcel is late", "change delivery address", "tracking number"],
}


def toy_decisions(n: int, seed: int) -> list:
    rng = random.Random(seed)
    labels = list(TOPICS)
    out = []
    for _ in range(n):
        label = rng.choice(labels)
        rec = Record(f"Customer says: {rng.choice(TOPICS[label])}.", label, labels, "Route this ticket.")
        out.append(make_decision(rec, rng, p_none=0.1, source="toy"))
    return out


def write_jsonl(path, decisions) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(decision_to_dict(d)) + "\n")


def build_tiny_model(path) -> None:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from jinja2 import Template

    corpus = []
    for d in toy_decisions(300, seed=99):
        corpus.append(Template(CHATML).render(messages=messages(d), add_generation_prompt=True) + render_answer(d))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=800,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHATML
    fast.save_pretrained(path)

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
