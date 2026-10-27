# Locket Decider² (7B)

An open-weights model that does one thing well: **given an input and a list of options, choose
one, say how sure it is, and explain why in a sentence.** Built for the everyday decisions inside a
company — ticket routing, approvals, triage, moderation, agent tool selection — and meant to run on
your own hardware.

```text
POST /v1/decide
{"input": "I was charged twice for my March invoice and need a refund before Friday.",
 "options": ["billing", "technical", "sales"],
 "criteria": "Refund requests go to billing."}

{"choice": "A", "label": "billing", "confidence": 0.97,
 "probabilities": {"A": 0.97, "B": 0.01, "C": 0.02},
 "needs_review": false,
 "reason": "The customer reports a duplicate charge and asks for a refund."}
```

> **Status:** this repository holds the full recipe (data, training, calibration, evaluation,
> serving). The 7B weights are produced by running it on a GPU — see [Train](#train). No benchmark
> numbers are claimed until `locket-eval` has been run on the trained model.

## How it decides

Most "decision" setups ask a chat model for an answer and parse the text. Decider² works
differently, and that is where its quality comes from:

1. **Option scoring, not free text.** The model reads the prompt, then the log-probability of
   every option key after `{"choice": "` is computed directly. The answer is always one of the listed
   options — it cannot be malformed, invented or off-list.
2. **Calibrated confidence.** The probabilities come from a softmax over those scores, with a
   temperature fitted on held-out data (`locket-calibrate`). A reported 0.9 is meant to be right
   about 90% of the time, and the expected calibration error (ECE) is measured to check it.
3. **Knows when to stop.** Training includes "none of the above" decisions where the right answer is
   missing, and out-of-scope queries. Calibration also picks a confidence threshold that meets a
   target accuracy (default 95%); anything below it, or a "none" answer, comes back with
   `needs_review: true` so a person can take it.
4. **No position bias.** Options are sampled and shuffled for every training example, so the model
   reads the labels instead of learning "the answer is usually A".
5. **Reasons on demand.** After the choice is fixed, the model writes a one-sentence reason
   conditioned on it, so the explanation always matches the decision.

## Model

| | |
|---|---|
| Base | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0), 7.6B parameters |
| Method | LoRA (r=64, all linear layers) fine-tuning, loss on the answer only; optional QLoRA |
| Options | 2–26 per decision; descriptions and written criteria supported |
| Languages | English and Turkish training data (MASSIVE tr-TR); the base model covers many more |
| Licence | Apache-2.0 (code and weights); training data licences below |

The base model is a single line in [`configs/decider-2-7b.json`](configs/decider-2-7b.json); any
causal LM with a chat template works.

## Training data

| Source | What it teaches | Licence |
|---|---|---|
| [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) | fine-grained customer intent routing | CC-BY-4.0 |
| [CLINC150 (plus)](https://huggingface.co/datasets/clinc_oos) | intent routing incl. out-of-scope → abstain | CC-BY-3.0 |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) en-US, tr-TR | intent routing in English and Turkish | CC-BY-4.0 |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | whether a claim follows from a text | OANC / CC-BY-SA-3.0 |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | yes/no decisions grounded in a passage | CC-BY-SA-3.0 |
| `locket-distill` (optional) | enterprise decisions with criteria and reasons, written by a teacher model and kept only when the teacher re-derives the same answer blind | depends on the teacher's terms |

Every item becomes a decision with a random subset of 2–12 options; about 10% drop the right
answer for "none of the above". Train, validation and eval splits never overlap: eval uses each
dataset's own test (or held-out validation) split.

## Install

```bash
pip install -e ".[train,serve,dev]"
pytest   # prompts, sampling, calibration maths, distillation; with the train/serve extras
         # installed it also runs train -> calibrate -> eval -> merge -> serve on a tiny CPU model
```

## Train

A single 24 GB GPU is enough with `--qlora`; an 80 GB GPU (A100/H100) trains in bf16 without it.

```bash
# 1. (optional) enterprise decisions from a teacher model, then list the file in
#    configs/decider-2-7b.json -> data.jsonl.train
OPENAI_API_KEY=... locket-distill --out data/enterprise.jsonl --n 5000 --teacher <model id> --max-usd 20

# 2. fine-tune
locket-train --config configs/decider-2-7b.json --out runs/decider-2-7b --qlora

# 3. calibrate confidence on the validation split (writes runs/decider-2-7b/decider_config.json)
locket-calibrate --model Qwen/Qwen2.5-7B-Instruct --adapter runs/decider-2-7b --target-accuracy 0.95

# 4. measure on held-out data — and against the untouched base model for comparison
locket-eval --model Qwen/Qwen2.5-7B-Instruct --adapter runs/decider-2-7b --out runs/eval.json
locket-eval --model Qwen/Qwen2.5-7B-Instruct --out runs/eval-base.json

# 5. merge into standalone weights
locket-merge --adapter runs/decider-2-7b --out models/locket-decider-2-7b
```

`locket-train --limit 200` runs a quick smoke test before committing GPU hours.

## Release on Hugging Face

One script does everything on a single GPU machine: install, tests, train, calibrate, evaluate
(trained and base), merge, write the model card from the measured results, and upload.

1. **Get a GPU machine** with internet access, e.g. a RunPod / Lambda / Vast pod with a PyTorch
   image. An 80 GB A100/H100 trains in bf16; a 24 GB card (RTX 4090, A10G, L4) uses QLoRA
   automatically. Leave ~100 GB of disk for the base model, checkpoints and the merged weights.
2. **Create a Hugging Face write token** at <https://huggingface.co/settings/tokens>.
3. **Run:**

   ```bash
   git clone https://github.com/burkiuze/LocketDecider- && cd LocketDecider-
   export HF_TOKEN=hf_...
   export HF_REPO=<your-user-or-org>/locket-decider-2-7b

   SMOKE=1 bash scripts/train_and_release.sh   # 200 examples, a short run, uploads nothing
   bash scripts/train_and_release.sh            # the real run; uploads at the end
   ```

The smoke run catches data, token and dependency problems before any long GPU time is spent. If a
dataset fails to download there, remove it from `data.train/validation/eval` in
`configs/decider-2-7b.json` and run again. `PRIVATE=1` creates the Hub repo as private, so the
model can be checked before it is made public.

The upload contains the merged weights, tokenizer, `decider_config.json` (temperature and review
threshold) and a model card whose numbers come only from `runs/eval.json` and
`runs/eval-base.json`.

## Serve

```bash
locket-serve --model models/locket-decider-2-7b --port 8000
curl -s localhost:8000/v1/decide -H 'content-type: application/json' \
  -d '{"input": "Wire transfer of $48,200 to a new beneficiary at 2am", "options": ["approve", "hold for review", "block"]}'
```

The merged model is a normal Hugging Face checkpoint, so it also runs behind any
OpenAI-compatible server such as vLLM (`vllm serve models/locket-decider-2-7b`) using the prompt in
[`locket_decider/schema.py`](locket_decider/schema.py). That path generates the JSON answer as text;
`locket-serve` adds option scoring, calibrated confidence and `needs_review`.

From Python:

```python
from locket_decider import Decision, Option
from locket_decider.decider import Decider

decider = Decider.load("models/locket-decider-2-7b")
print(decider.decide(Decision(
    input="Can I change the delivery address on order 1182?",
    options=[Option("orders"), Option("billing"), Option("returns")],
)))
```

## Evaluation

`locket-eval` reports, per source and overall: accuracy, macro-F1, expected calibration error and
negative log-likelihood, plus the share of decisions that can be automated at the calibrated
threshold and the accuracy on those. Results will be published here once a training run is done.

## Layout

```
locket_decider/
  schema.py       decision format, prompt, answer format
  data.py         dataset converters, option sampling, splits
  scoring.py      option log-probabilities and reason generation
  calibration.py  temperature scaling, ECE, F1, thresholds (numpy only)
  decider.py      Decider: load a model and decide
  train.py        LoRA / QLoRA fine-tuning (TRL)
  evaluate.py     locket-eval and locket-calibrate
  merge.py        merge an adapter into standalone weights
  serve.py        FastAPI server
  distill.py      teacher-written, self-verified enterprise decisions
  release.py      model card from eval results + Hugging Face upload
scripts/train_and_release.sh
configs/decider-2-7b.json
tests/
```
