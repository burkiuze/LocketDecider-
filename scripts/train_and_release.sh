#!/usr/bin/env bash
# Train, calibrate, evaluate and publish Locket Decider² 7B on one GPU machine.
#
#   export HF_TOKEN=hf_...                     # write token: https://huggingface.co/settings/tokens
#   export HF_REPO=<user-or-org>/locket-decider-2-7b
#   bash scripts/train_and_release.sh
#
# Optional: QLORA=1 to force 4-bit (auto below 40 GB of GPU memory), SMOKE=1 for a 200-example
# dry run that uploads nothing, PRIVATE=1 to create the Hub repo as private.
set -euo pipefail

: "${HF_REPO:?set HF_REPO, e.g. my-org/locket-decider-2-7b}"
CONFIG=${CONFIG:-configs/decider-2-7b.json}
BASE=$(python3 -c "import json;print(json.load(open('$CONFIG'))['base_model'])")
RUN=runs/decider-2-7b
OUT=models/locket-decider-2-7b

echo "== install"
pip install -q -e ".[train,serve,dev]" huggingface_hub

echo "== checks"
pytest -q
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU found"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB")
PY

if [[ -z "${QLORA:-}" ]]; then
  MEM=$(python3 -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**30))")
  [[ $MEM -lt 40 ]] && QLORA=1 || QLORA=0
fi
Q=(); [[ "$QLORA" == 1 ]] && Q=(--qlora)
LIMIT=(); [[ -n "${SMOKE:-}" ]] && LIMIT=(--limit 200)

if [[ -z "${SMOKE:-}" ]]; then
  [[ -n "${HF_TOKEN:-}" ]] || { echo "set HF_TOKEN (write access) before a full run"; exit 1; }
  # Fail now rather than after hours of training if the token can't write.
  python3 -c "from huggingface_hub import HfApi; import os; print('Hub user:', HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])"
fi

echo "== train (qlora=$QLORA)"
locket-train --config "$CONFIG" --out "$RUN" "${Q[@]}" "${LIMIT[@]}"

echo "== calibrate"
locket-calibrate --model "$BASE" --adapter "$RUN" --config "$CONFIG" "${Q[@]}" "${LIMIT[@]}"

echo "== evaluate (trained, then base for comparison)"
locket-eval --model "$BASE" --adapter "$RUN" --config "$CONFIG" "${Q[@]}" "${LIMIT[@]}" --out runs/eval.json
locket-eval --model "$BASE" --config "$CONFIG" "${Q[@]}" "${LIMIT[@]}" --temperature 1.0 --out runs/eval-base.json

echo "== merge"
locket-merge --adapter "$RUN" --out "$OUT"

echo "== release"
FLAGS=(); [[ -n "${PRIVATE:-}" ]] && FLAGS+=(--private); [[ -n "${SMOKE:-}" ]] && FLAGS+=(--dry-run)
locket-release --model "$OUT" --repo "$HF_REPO" --eval runs/eval.json --base-eval runs/eval-base.json --train-config "$CONFIG" "${FLAGS[@]}"
