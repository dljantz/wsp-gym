#!/usr/bin/env bash
set -euo pipefail
source /etc/rp_environment || true
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
mkdir -p trajectories checkpoints/sft_level2_lora checkpoints/grpo_level2_lora

echo "=== [Stage A] Procedural Level 2 + Phase 1 Replay LoRA SFT ==="
python -u train_sft.py \
  --sft-path trajectories/sft_level2_train.jsonl \
  --model-name-or-path "${MODEL_NAME}" \
  --output-dir checkpoints/sft_level2_lora \
  --epochs 5 --batch-size 1 --grad-accum 4 --lr 2e-4

echo "=== [Stage B] Evaluate sft_level2_lora on Full 20-Ep Level 2 Suite (12 Procedural + 8 Unseen Pretrained) ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/sft_level2_lora \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_l2_sft.json

echo "=== [Stage C] Online Level 2 Contrastive GRPO Refinement (18 steps) ==="
rm -rf checkpoints/grpo_level2_lora && mkdir -p checkpoints/grpo_level2_lora
rm -f trajectories/grpo_level2_train_log.jsonl
python -u train_grpo.py \
  --model-name-or-path "${MODEL_NAME}" \
  --curriculum level2 --level2-scale 100m \
  --sft-adapter-path checkpoints/sft_level2_lora \
  --output-dir checkpoints/grpo_level2_lora \
  --log-path trajectories/grpo_level2_train_log.jsonl \
  --num-steps 6 --group-size 6 --save-every 6 \
  --lr 1e-5 --kl-beta 0.02 --max-turns 4 --max-new-tokens 1536 --timeout-sec 30

echo "=== [Stage D] Evaluate grpo_level2_lora (final & best) + 34-Ep Phase 1 Retention ==="
python -u eval_level2.py \
  --suite all --scale 100m --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/grpo_level2_lora \
  --episodes-per-subtype 2 --base-seed 19000 \
  --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
  --out-path trajectories/eval_level2_l2_grpo_final.json

if [ -d "checkpoints/grpo_level2_lora/best" ]; then
  python -u eval_level2.py \
    --suite all --scale 100m --backend hf \
    --model-name-or-path "${MODEL_NAME}" \
    --adapter-path checkpoints/grpo_level2_lora/best \
    --episodes-per-subtype 2 --base-seed 19000 \
    --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
    --out-path trajectories/eval_level2_l2_grpo_best.json
fi

BEST_L2_ADAPTER=$(python -c '
import json, os
cands = [
    ("checkpoints/sft_level2_lora", "trajectories/eval_level2_l2_sft.json"),
    ("checkpoints/grpo_level2_lora", "trajectories/eval_level2_l2_grpo_final.json"),
    ("checkpoints/grpo_level2_lora/best", "trajectories/eval_level2_l2_grpo_best.json"),
]
best_path, best_key = cands[0][0], (-1.0, -1e9)
for ckpt, jpath in cands:
    if os.path.exists(jpath):
        d = json.load(open(jpath, encoding="utf-8"))
        k = (float(d.get("pass_rate", 0.0)), float(d.get("mean_base_reward", -1.0)))
        if k > best_key:
            best_key, best_path = k, ckpt
print(best_path)
')
echo "Selected BEST_L2_ADAPTER=${BEST_L2_ADAPTER}"

python -u eval.py \
  --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path "${BEST_L2_ADAPTER}" \
  --episodes-per-subtype 2 --base-seed 9000 \
  --max-turns 4 --max-new-tokens 1536 \
  --out-path trajectories/eval_phase1_retention_after_l2.json

UTC_HHMM=$(date -u +%H%M)
if [ "${UTC_HHMM}" -lt 1330 ]; then
  echo "=== [Stage E] Refreshing Zero-Shot P1-GRPO (step_0051) on Fixed 100m Benchmark (UTC=${UTC_HHMM}) ==="
  python -u eval_level2.py \
    --suite all --scale 100m --backend hf \
    --model-name-or-path "${MODEL_NAME}" \
    --adapter-path checkpoints/grpo_lora/step_0051 \
    --episodes-per-subtype 2 --base-seed 19000 \
    --max-turns 4 --max-new-tokens 1536 --timeout-sec 30 \
    --out-path trajectories/eval_level2_p1_grpo.json
else
  echo "Skipping Stage E refresh (UTC=${UTC_HHMM} >= 1330)."
fi

echo "PHASE2_ADAPT_COMPLETE"
