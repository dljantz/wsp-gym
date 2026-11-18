# wsp-gym: Weight-Space Programming Gymnasium

`wsp-gym` trains LLM "neurosurgeon" agents (`Qwen/Qwen2.5-Coder-7B-Instruct` + LoRA) via **RL with Verifiable Rewards (RLVR)** to inspect, diagnose, repair, excise backdoors, inject null-space skills, splice/merge, and refactor neural network weights directly inside a stateful Python REPL—scaling **zero-shot from `<= 102M` synthetic checkpoints to self-surgery on a `7.62B` immutable twin of the agent's own weights**.

> **New Chat / Agent Handoff:** Read [ROADMAP.md](ROADMAP.md) for full architectural rationale, all 7 problem categories & subtypes, mathematical/systems lessons, Runpod/Windows operational rules, tracked trajectory reports, and candidate Phase 4 directions.

---

## Project Status & Headline Results (`Phases 0 -> 3 COMPLETE`)

| Stage / Checkpoint | Training Data (Strict Constraint) | Level 1 + 1.5 (`34` eps) | Level 2 Procedural `102M` (`12` eps) | Unseen Pretrained LLMs (`SmolLM2-135M` + `Qwen2.5-0.5B`, `8` eps) | Level 3 `7.62B` Self-Surgery (`6` eps) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Base `Qwen2.5-Coder-7B-Instruct`** | Unadapted | `0.0%` (`0/34`) | `0.0%` (`0/12`) | `0.0%` (`0/8`) | `0.0%` (`0/6` pass / promote) |
| **Phase 1.2 GRPO (`checkpoints/grpo_lora/step_0051`)** | `<= 64d` synthetic (`123` SFT + `51` GRPO) | `94.12%` (`32/34`) | `0.0%` (`0/12`) | `0.0%` (`0/8`) | — |
| **Phase 2 GRPO (`checkpoints/grpo_level2_lora`)** | `<= 102M` synthetic (`0` real LLMs) | `47.06%` (`16/34`) | `100.0%` (`12/12`) | `50.0%` (`4/8`; `4/4` `cat2`, `0/4` `cat5`) | `66.67%` (`4/6` pass & promote; `0/2` `cat5`) |
| **Phase 2.5 GRPO (`checkpoints/grpo_phase2_5_lora`, Best)** | `<= 102M` synthetic (`0` real LLMs) | **`100.0%` (`34/34`)** | **`100.0%` (`12/12`)** | **`100.0%` (`8/8`; `4/4` `cat2`, `4/4` `cat5`)** | **`83.33%` (`5/6` pass & promote; `100%` `3/3` at `B=0`)** |
| **Level 3 Scripted Reference** | Exact static/hybrid reference | `100.0%` (`34/34`) | `100.0%` (`12/12`) | `100.0%` (`8/8`) | `100.0%` (`6/6` pass & promote) |

**Key Phase 3 (`7.62B` Immutable-Twin Self-Surgery) Finding:**
Trained strictly on synthetic models `<= 102M` parameters (`0` real LLMs in SFT or GRPO), `checkpoints/grpo_phase2_5_lora` transferred **75x zero-shot** to repair single-layer sinks (`2/2`), out-of-distribution simultaneous dual-layer sinks (`2/2`), and a `7:1` GQA sleeper backdoor (`1/2`, **`100%` `3/3` at zero-probe `B=0`** with `reference_twin_kl <= 1.5e-7` and `100%` surgeon/twin-base immutability; its lone miss at `B=4` hit the `90s` turn timeout combining 2 `7.62B` forward probes with 784 CPU head SVDs in Turn 1).

---

## Repository Structure

```text
wsp_gym/
  model.py         # Functional Transformer / SwiGLU / RoPE / GQA engine + CoW LazyTwinWeightDict
  gauge.py         # RoPE- and RMSNorm-aware orthogonal gauge scrambler (anti-memorization)
  generators.py    # Level 1 & Cycle 2 procedural challenges across all 7 categories
  level1_5.py      # Level 1.5 grokked (AdamW) + non-uniform RMSNorm + SwiGLU + RoPE generators
  level2.py        # Level 2 102M GQA suite + unseen SmolLM2-135M / Qwen2.5-0.5B corruptors & solvers
  level3.py        # Level 3 7.62B Immutable-Twin Self-Surgery suite & Fork-and-Verify promotion gate
  toolkit.py       # Pre-loaded wsp_tools primitives (SVD basis, RoPE offsets, OV circuits, canonical_pack)
  verifier.py      # Deterministic capability, KL-retention, anti-masking & recovery-attack verifier
  sandbox.py       # Stateful multi-turn Python REPL, slice-view compaction, cgroup anon-RAM & trace guard
  training.py      # Turn-masked SFT exporter, B->0 CurriculumScheduler, shaping & GRPORolloutEnv
  scaler.py        # 100% static Option 3 SFT trajectory builder (0 probe_forward calls)

eval.py            # Phase 1 held-out evaluator (17 subtypes, 34 episodes)
eval_level2.py     # Level 2 scale-transfer & unseen-LLM evaluator (20 episodes, --suite split/merge)
eval_level3.py     # Level 3 7.62B Immutable-Twin Self-Surgery evaluator (--resume, HF equivalence)
train_sft.py       # Turn-masked LoRA SFT warmup trainer
train_grpo.py      # Online multi-turn GRPO trainer (frozen SFT KL anchor + variance/all-fail guards)
run_phase*.sh      # Cloud Runpod A40/H100 orchestration scripts
tests/             # 25 CPU unit & integration tests (test_wsp_gym: 13, test_level2: 7, test_level3: 5)
trajectories/      # Tracked SFT corpora, GRPO logs, HF equivalence JSONs, and all held-out eval reports
checkpoints/       # Local gitignored LoRA adapters (~319 MB each, excluded from git):
                   #   sft_lora, grpo_lora (+ step_0051), sft_level2_lora, grpo_level2_lora,
                   #   sft_phase2_5_lora, grpo_phase2_5_lora (canonical best checkpoint)
```

---

## Quick Verification & Evaluation Commands

```powershell
# Run all 25 unit and integration tests locally on CPU (~80-95s)
python -m unittest discover -s tests -v

# Scripted verification of Level 1/1.5 (34 eps), Level 2 (20 eps), and Level 3 mini twin (6 eps)
python eval.py --backend scripted --episodes-per-subtype 2
python eval_level2.py --suite all --scale 100m --backend scripted --episodes-per-subtype 2
python eval_level3.py --scale mini --backend scripted --episodes-per-subtype 2

# Cloud GPU (48GB+ VRAM + >=50GB RAM) live 7.62B self-surgery evaluation
python eval_level3.py --verify-hf-equivalence --out-path trajectories/hf_equivalence_level3.json
python eval_level3.py --scale 7b --backend hf --model-name-or-path Qwen/Qwen2.5-Coder-7B-Instruct --adapter-path checkpoints/grpo_phase2_5_lora --episodes-per-subtype 2 --base-seed 29000 --max-turns 4 --timeout-sec 90.0 --max-new-tokens 1536 --out-path trajectories/eval_level3_p25_grpo.json
```

---

## Operational Notes at Current Handoff
- **Git & Clean Tree:** Up to date with `origin/main` at commit `6e48171` (before this documentation commit). `.gitignore` excludes `checkpoints/`, `challenges/`, `*.pt`, `*.tar.gz`, `*.log`, and `*.partial.json`. On Windows headless shells, avoid invoking `git push` directly if Git Credential Manager opens an interactive GUI prompt.
- **Cloud State (`$0/hr`):** **0 running Runpod pods**. All temporary evaluation pods (`2jy1wdvr1wzoyq`, `9l5oyegho2g91g`, `0kw73msu7atapo`) were terminated cleanly; pre-existing stopped user pods `fu3n5qpwrsqq8d` and `hgylz13bq35gxx` remain untouched (`exited`). Note that container-disk Runpod pods (`--disk 80` without `--persistent-volume`) wipe `/workspace` across stop/restart cycles — always `scp` artifacts before stopping.
