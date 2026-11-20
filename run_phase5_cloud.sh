#!/usr/bin/env bash
set -euo pipefail
source /etc/rp_environment || true
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BUNDLE_CACHE="/tmp/organic_bundles_39000"
mkdir -p "${BUNDLE_CACHE}" /tmp/p5_parts trajectories

echo "=== [Stage 0] Full Unit Test Suite on GPU ==="
python -m unittest discover -s tests -v

echo "=== [Stage 1] 25M 10-Episode Organic Synthetic Gate ==="
python -u eval_organic.py \
  --suite synthetic --synthetic-scale 25m \
  --backend norm_baseline \
  --episodes-per-subtype 2 --base-seed 39000 \
  --bundle-cache-dir "${BUNDLE_CACHE}" \
  --out-path trajectories/eval_organic_25m_norm_baseline.json

python -u eval_organic.py \
  --suite synthetic --synthetic-scale 25m \
  --backend hybrid_ref \
  --episodes-per-subtype 2 --base-seed 39000 \
  --bundle-cache-dir "${BUNDLE_CACHE}" \
  --out-path trajectories/eval_organic_25m_hybrid_ref.json

python -u -c '
import json
nb = json.load(open("trajectories/eval_organic_25m_norm_baseline.json"))
hr = json.load(open("trajectories/eval_organic_25m_hybrid_ref.json"))
print("25m norm_baseline:", {k: nb[k] for k in ("passed", "episodes", "pass_rate", "mean_retained_kl", "explicit_submit_rate", "mean_clean_exec_rate")})
print("25m hybrid_ref:", {k: hr[k] for k in ("passed", "episodes", "pass_rate", "mean_retained_kl", "explicit_submit_rate", "mean_clean_exec_rate")})
assert nb["episodes"] == 10 and nb["pass_rate"] == 0.0
assert nb["explicit_submit_rate"] == 1.0 and nb["mean_clean_exec_rate"] == 1.0
assert hr["episodes"] == 10 and hr["pass_rate"] >= 0.85 and hr["mean_retained_kl"] < 0.01
assert hr["explicit_submit_rate"] == 1.0 and hr["mean_clean_exec_rate"] == 1.0
print("STAGE 1 25M GATE PASSED")
'

run_split_suite() {
  local tag="$1"
  shift
  echo "--- Running split suite: ${tag} ---"
  python -u eval_organic.py \
    --suite synthetic --synthetic-scale 100m \
    --episodes-per-subtype 2 --base-seed 39000 \
    --bundle-cache-dir "${BUNDLE_CACHE}" \
    "$@" \
    --out-path "/tmp/p5_parts/${tag}_100m.json"

  python -u eval_organic.py \
    --suite smollm2 \
    --episodes-per-subtype 2 --base-seed 39000 \
    --bundle-cache-dir "${BUNDLE_CACHE}" \
    "$@" \
    --out-path "/tmp/p5_parts/${tag}_smollm2.json"

  python -u eval_organic.py \
    --suite qwen25 \
    --episodes-per-subtype 2 --base-seed 39000 \
    --bundle-cache-dir "${BUNDLE_CACHE}" \
    "$@" \
    --out-path "/tmp/p5_parts/${tag}_qwen25.json"

  python -u eval_organic.py \
    --merge-reports \
      "/tmp/p5_parts/${tag}_100m.json" \
      "/tmp/p5_parts/${tag}_smollm2.json" \
      "/tmp/p5_parts/${tag}_qwen25.json" \
    --out-path "trajectories/eval_organic_${tag}.json"
}

echo "=== [Stage 2.1/4] 30-Episode Cloud Suite: norm_baseline ==="
run_split_suite "norm_baseline" --backend norm_baseline

echo "=== [Stage 2.2/4] 30-Episode Cloud Suite: hybrid_ref ==="
run_split_suite "hybrid_ref" --backend hybrid_ref

python -u -c '
import json
nb = json.load(open("trajectories/eval_organic_norm_baseline.json"))
hr = json.load(open("trajectories/eval_organic_hybrid_ref.json"))
print("30-ep norm_baseline:", {k: nb[k] for k in ("passed", "episodes", "pass_rate", "mean_retained_kl", "explicit_submit_rate", "mean_clean_exec_rate")})
print("30-ep hybrid_ref:", {k: hr[k] for k in ("passed", "episodes", "pass_rate", "mean_retained_kl", "explicit_submit_rate", "mean_clean_exec_rate")})
assert nb["episodes"] == 30 and nb["pass_rate"] == 0.0
assert nb["explicit_submit_rate"] == 1.0 and nb["mean_clean_exec_rate"] == 1.0
assert hr["episodes"] == 30 and hr["pass_rate"] >= 0.85 and hr["mean_retained_kl"] < 0.01
assert hr["explicit_submit_rate"] == 1.0 and hr["mean_clean_exec_rate"] == 1.0
print("STAGE 2 BASELINE & HYBRID_REF GATE PASSED")
'

echo "=== [Stage 2.3/4] 30-Episode Cloud Suite: p4_grpo (Honest Falsification) ==="
run_split_suite "p4_grpo" \
  --backend hf --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/grpo_phase4_lora \
  --max-turns 6 --timeout-sec 90 --max-new-tokens 1536

echo "=== [Stage 2.4/4] 30-Episode Cloud Suite: p25_grpo (Honest Falsification) ==="
run_split_suite "p25_grpo" \
  --backend hf --model-name-or-path "${MODEL_NAME}" \
  --adapter-path checkpoints/grpo_phase2_5_lora \
  --max-turns 6 --timeout-sec 90 --max-new-tokens 1536

echo "PHASE5_CLOUD_COMPLETE"
