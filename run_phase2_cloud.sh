#!/usr/bin/env bash
set -euo pipefail
source /etc/rp_environment || true
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
mkdir -p trajectories checkpoints/grpo_level2_lora

echo "=== [1/6] Verify HF-to-WSP Equivalence (SmolLM2-135M & Qwen2.5-0.5B) ==="
python -u -c '
import json
from wsp_gym.level2 import verify_hf_wsp_equivalence

results = []
for mid in ("HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"):
    res = verify_hf_wsp_equivalence(mid)
    print(f"EQUIV {mid}:", res)
    assert res["equivalent"], f"Equivalence check failed for {mid}: {res}"
    results.append(res)

with open("trajectories/hf_equivalence_level2.json", "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, indent=2)
print("Saved trajectories/hf_equivalence_level2.json")
'

echo "=== [2/6] Zero-Shot Eval: Base Qwen2.5-Coder-7B-Instruct ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_base.json

echo "=== [3/6] Zero-Shot Eval: Phase 1 SFT Adapter (checkpoints/sft_lora) ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/sft_lora \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_p1_sft.json

echo "=== [4/6] Zero-Shot Eval: Phase 1 GRPO Adapter (checkpoints/grpo_lora/step_0051) ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/grpo_lora/step_0051 \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_p1_grpo.json

echo "=== [5/6] Online Level 2 Curriculum GRPO (36 steps) ==="
python -u train_grpo.py \
  --model-name-or-path "${MODEL_NAME}" \
  --curriculum level2 --level2-scale 100m \
  --sft-adapter-path checkpoints/grpo_lora/step_0051 \
  --output-dir checkpoints/grpo_level2_lora \
  --log-path trajectories/grpo_level2_train_log.jsonl \
  --num-steps 36 --group-size 6 --save-every 12 \
  --lr 1e-5 --kl-beta 0.02 --max-turns 4 --max-new-tokens 1536 --timeout-sec 30

echo "=== [6/6] Post-Adaptation Level 2 Eval & Phase 1 Retention Check ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/grpo_level2_lora \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_l2_grpo_final.json

BEST_L2_ADAPTER="checkpoints/grpo_level2_lora"
if [ -d "checkpoints/grpo_level2_lora/best" ]; then
  python -u eval_level2.py \
    --suite all --scale 100m --backend hf \
    --model-name-or-path "${MODEL_NAME}" \
    --adapter-path checkpoints/grpo_level2_lora/best \
    --episodes-per-subtype 2 --base-seed 19000 \
    --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
    --out-path trajectories/eval_level2_l2_grpo_best.json
  BEST_L2_ADAPTER=$(python -c '
import json
fin = json.load(open("trajectories/eval_level2_l2_grpo_final.json"))
bst = json.load(open("trajectories/eval_level2_l2_grpo_best.json"))
print("checkpoints/grpo_level2_lora/best" if bst["pass_rate"] >= fin["pass_rate"] else "checkpoints/grpo_level2_lora")
')
fi
echo "Selected BEST_L2_ADAPTER=${BEST_L2_ADAPTER}"

python -u eval.py \
  --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path "${BEST_L2_ADAPTER}" \
  --episodes-per-subtype 2 --base-seed 9000 \
  --max-turns 4 --max-new-tokens 1536 \
  --out-path trajectories/eval_phase1_retention_after_l2.json

echo "PHASE2_COMPLETE"
