# wsp-gym: Weight-Space Programming Gymnasium

`wsp-gym` trains LLM "neurosurgeon" agents (`Qwen/Qwen2.5-Coder-7B-Instruct` + LoRA) via **RL with Verifiable Rewards (RLVR)** to inspect, diagnose, repair, excise backdoors, inject null-space skills, splice/merge, and refactor neural network weights directly inside a stateful Python REPL—scaling **zero-shot from `<= 102M` synthetic checkpoints to self-surgery on a `7.62B` immutable twin of the agent's own weights**.

> **New Chat / Agent Handoff:** Read [ROADMAP.md](ROADMAP.md) for full architectural rationale, all 7 problem categories & subtypes, mathematical/systems lessons, Runpod/Windows operational rules, tracked trajectory reports, and the Phase 4 (Complete) -> Phase 5–8 Landmark Paper Master Plan.

---

## Project Status & Headline Results (`Phases 0 -> 4 COMPLETE`)

| Stage / Checkpoint | Training Data (Strict Constraint) | Level 1 + 1.5 (`34` eps) | Level 2 Procedural `102M` (`12` eps) | Unseen Pretrained LLMs (`SmolLM2-135M` + `Qwen2.5-0.5B`, `8` eps) | Level 3 `7.62B` Self-Surgery (`12` eps: `6` Rest + `6` Const) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Base `Qwen2.5-Coder-7B-Instruct`** | Unadapted | `0.0%` (`0/34`) | `0.0%` (`0/12`) | `0.0%` (`0/8`) | `0.0%` (`0/6` Phase 3 Rest) |
| **Phase 1.2 GRPO (`checkpoints/grpo_lora/step_0051`)** | `<= 64d` synthetic (`123` SFT + `51` GRPO) | `94.12%` (`32/34`) | `0.0%` (`0/12`) | `0.0%` (`0/8`) | — |
| **Phase 2 GRPO (`checkpoints/grpo_level2_lora`)** | `<= 102M` synthetic (`0` real LLMs) | `47.06%` (`16/34`) | `100.0%` (`12/12`) | `50.0%` (`4/8`; `4/4` `cat2`, `0/4` `cat5`) | `66.67%` (`4/6` Phase 3 Rest) |
| **Phase 2.5 GRPO (`checkpoints/grpo_phase2_5_lora`)** | `<= 102M` synthetic (`293` SFT + `8` GRPO; `0` real LLMs) | **`100.0%` (`34/34`)** | **`100.0%` (`12/12`)** | **`100.0%` (`8/8`; `4/4` `cat2`, `4/4` `cat5`)** | `33.33%` (`4/12`: `4/6` Rest, `0/6` Const; `5/6` on Phase 3 harness) |
| **Phase 4 GRPO (`checkpoints/grpo_phase4_lora`, Best)** | `<= 102M` + synth full-rank GQA (`369` SFT + `24` GRPO; `0` real LLMs) | **`100.0%` (`34/34`)** | `91.67%` (`11/12`) | **`100.0%` (`8/8`; `4/4` `cat2`, `4/4` `cat5`)** | **`100.0%` (`12/12` pass & promote; `6/6` `B=0`)** |
| **Level 3 Scripted Reference** | Exact static/hybrid reference | `100.0%` (`34/34`) | `100.0%` (`12/12`) | `100.0%` (`8/8`) | `100.0%` (`12/12` pass & promote) |

**Key Phase 3–4 (`7.62B` Immutable-Twin Self-Surgery) Finding:**
Trained strictly on synthetic checkpoints (`0` real LLMs in SFT or GRPO), `checkpoints/grpo_phase4_lora` achieves **`12/12` (`100.0%`) pass & Fork-and-Verify promotion** across the expanded `7.62B` `Qwen2.5-Coder-7B-Instruct` Immutable-Twin Self-Surgery suite (`6/6` restorative single-sink, OOD multi-sink, and `7:1` GQA backdoor excision **plus** `6/6` constructive empirical `covariance_nullspace` skill injection `cat4`, specialist twin merge `cat6`, and activation-weighted GQA OV truncation `cat7` — vs. `4/12` (`4/6` rest + `0/6` const) for `grpo_phase2_5_lora` on the same suite). It scores **`6/6` (`100%`) at zero-probe `B=0`**, **`6.13s`** mean Turn-1 latency (down from `16.41s` / `90s` timeouts via GPU-streamed batched head SVDs & anomaly scans), **`0.000738`** mean `reference_twin_kl` (`< 0.005`), **`1.0` surgeon/twin-base immutability**, **`1.0` clean exec**, and **`2.0` mean turns**. Across the full **`66`-episode** held-out suite (`34` Phase 1/1.5 + `20` Level 2 + `12` Level 3), `checkpoints/grpo_phase4_lora` scores **`65/66` (`98.48%`)**, retaining `34/34` (`100%`) on Phase 1/1.5 and `8/8` (`100%`) on unseen `SmolLM2-135M` + `Qwen2.5-0.5B` (sole miss: `ep_l2_cat4_19102` at `B=0`).

---

## Repository Structure

```text
wsp_gym/
  model.py         # Functional Transformer / SwiGLU / RoPE / GQA engine + CoW LazyTwinWeightDict (O(1) L0/Frob)
  gauge.py         # RoPE- and RMSNorm-aware orthogonal gauge scrambler (anti-memorization)
  generators.py    # Level 1 & Cycle 2 procedural challenges across all 7 categories
  level1_5.py      # Level 1.5 grokked (AdamW) + non-uniform RMSNorm + SwiGLU + RoPE generators
  level2.py        # Level 2 102M GQA suite + unseen SmolLM2-135M / Qwen2.5-0.5B corruptors & solvers
  level3.py        # Level 3 7.62B Restorative+Constructive Twin Self-Surgery & Phase 4 SFT builder
  toolkit.py       # Pre-loaded wsp_tools (batched GPU/CPU SVD, covariance_nullspace, ephemeral streaming, pack)
  verifier.py      # Deterministic capability, KL-retention, anti-masking, constructive & recovery verifiers
  sandbox.py       # Stateful multi-turn Python REPL, slice-view compaction, cgroup anon-RAM & trace guard
  training.py      # Turn-masked & error-turn-masked SFT exporter, CurriculumScheduler, & GRPORolloutEnv
  scaler.py        # 100% static Option 3 SFT trajectory builder (0 probe_forward calls)

eval.py            # Phase 1 held-out evaluator (17 subtypes, 34 episodes)
eval_level2.py     # Level 2 scale-transfer & unseen-LLM evaluator (20 episodes, --suite split/merge)
eval_level3.py     # Level 3 7.62B Immutable-Twin Self-Surgery evaluator (--suite restorative|constructive|all)
train_sft.py       # Turn-masked LoRA SFT warmup trainer
train_grpo.py      # Online multi-turn GRPO trainer (phase4 curriculum + frozen SFT KL anchor)
run_phase*.sh      # Cloud Runpod A40/H100 orchestration scripts
tests/             # 33 CPU unit & integration tests (test_wsp_gym: 13, test_level2: 7, test_level3: 5, test_phase4: 8)
trajectories/      # Tracked SFT corpora, GRPO logs, HF equivalence JSONs, and all held-out eval reports
checkpoints/       # Local gitignored LoRA adapters (~319 MB each, excluded from git):
                   #   sft_lora, grpo_lora (+ step_0051), sft_level2_lora, grpo_level2_lora,
                   #   sft_phase2_5_lora, grpo_phase2_5_lora, sft_phase4_lora, grpo_phase4_lora (canonical best)
```

---

## Quick Verification & Evaluation Commands

```powershell
# Run all 33 unit and integration tests locally on CPU (~150s)
python -m unittest discover -s tests -v

# Scripted verification of Level 1/1.5 (34 eps), Level 2 (20 eps), and Level 3 mini twin (12 eps)
python eval.py --backend scripted --episodes-per-subtype 2
python eval_level2.py --suite all --scale 100m --backend scripted --episodes-per-subtype 2
python eval_level3.py --scale mini --suite all --backend scripted --episodes-per-subtype 2

# Cloud GPU (48GB+ VRAM + >=50GB RAM) live 7.62B self-surgery evaluation (restorative + constructive)
python eval_level3.py --verify-hf-equivalence --out-path trajectories/hf_equivalence_level3.json
python eval_level3.py --scale 7b --suite all --backend hf --model-name-or-path Qwen/Qwen2.5-Coder-7B-Instruct --adapter-path checkpoints/grpo_phase4_lora --episodes-per-subtype 2 --base-seed 29000 --max-turns 4 --timeout-sec 90.0 --max-new-tokens 1536 --out-path trajectories/eval_level3_p4_grpo.json
```

---

## Operational Notes at Current Handoff
- **Git & Clean Tree:** Local Phase 4 changes & artifacts committed cleanly on `main`. `.gitignore` excludes `checkpoints/`, `challenges/`, `*.pt`, `*.tar.gz`, `*.log`, and `*.partial.json`. On Windows headless shells, avoid invoking `git push` directly if Git Credential Manager opens an interactive GUI prompt.
- **Cloud State (`$0/hr`):** **0 running Runpod pods**. Temporary Phase 4 evaluation/training pod `pbuffkd7uaupcz` (`wsp-phase4`) and earlier pods (`2jy1wdvr1wzoyq`, `9l5oyegho2g91g`, `0kw73msu7atapo`) were terminated cleanly after downloading checkpoints and reports; pre-existing stopped user pods `fu3n5qpwrsqq8d` and `hgylz13bq35gxx` remain untouched (`exited`). Note that container-disk Runpod pods without `--persistent-volume` wipe `/workspace` across stop/restart cycles — always `scp` artifacts before stopping/terminating.
