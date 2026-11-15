#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"

echo "[Setup] Installing dependencies..."
pip install -r requirements.txt

echo "[Step C1] Base zero-shot evaluation (${MODEL_NAME})..."
python eval.py \
  --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --max-new-tokens 1536 \
  --out-path trajectories/eval_base.json

echo "[Step C2] Turn-masked LoRA SFT warmup..."
python train_sft.py \
  --model-name-or-path "${MODEL_NAME}" \
  --output-dir checkpoints/sft_lora

echo "[Step C2 Eval] Evaluating SFT LoRA checkpoint..."
python eval.py \
  --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/sft_lora \
  --max-new-tokens 1536 \
  --out-path trajectories/eval_sft.json

echo "[Step C3] Online multi-turn GRPO + curriculum annealing..."
python train_grpo.py \
  --model-name-or-path "${MODEL_NAME}" \
  --sft-adapter-path checkpoints/sft_lora \
  --output-dir checkpoints/grpo_lora

GRPO_EVAL_ADAPTER="checkpoints/grpo_lora"
if [ -d "checkpoints/grpo_lora/best" ]; then
  GRPO_EVAL_ADAPTER="checkpoints/grpo_lora/best"
fi
echo "[Step C4] Post-GRPO held-out evaluation (${GRPO_EVAL_ADAPTER})..."
python eval.py \
  --backend hf \
  --model-name-or-path "${MODEL_NAME}" \
  --adapter-path "${GRPO_EVAL_ADAPTER}" \
  --max-new-tokens 1536 \
  --out-path trajectories/eval_grpo.json

echo "Phase 1 Cloud GPU training and evaluation complete."
