# wsp-gym: Weight-Space Programming Gymnasium

`wsp-gym` trains LLM "neurosurgeon" agents (`Qwen/Qwen2.5-Coder-7B-Instruct` + LoRA) via **RL with Verifiable Rewards (RLVR)** to inspect, diagnose, repair, excise backdoors, inject null-space skills, splice/merge, and refactor neural network weights directly inside a stateful Python REPL—scaling **zero-shot from `<= 102M` synthetic checkpoints to self-surgery on a `7.62B` immutable twin of the agent's own weights**.

> **New Chat / Agent Handoff:** Read [ROADMAP.md](ROADMAP.md) for full architectural rationale, all 7 problem categories & subtypes, mathematical/systems lessons, Runpod/Windows operational rules, tracked trajectory reports, and the Phase 5 (Complete) -> Phase 6–8 Landmark Paper Master Plan.

---

## Project Status & Headline Results (`Phases 0 -> 5 COMPLETE`)

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

### Phase 5 Organic SGD/LoRA Benchmark ([eval_organic.py](eval_organic.py): Norm-Stealth Projected-LoRA Targets)

Phase 5 leaves planted `35x–60x` norm-spike faults behind and introduces organically gradient-trained (AdamW / Projected-LoRA, `<= 1.20x` median row/head norm, `0` edited MLPs in top-3 layer norms, `s2/s1 >= 0.12`, `W_E`/`W_U`/`b_U` untouched, 20-step LoRA recovery check on `cat5`) targets across `mini`, `25m`, `100m`, `SmolLM2-135M`, and `Qwen2.5-0.5B`:

| Policy / Checkpoint | `mini` (`5` eps) | `25m` (`10` eps) | `30`-Ep Cloud Suite (`100m` + `SmolLM2` + `Qwen2.5-0.5B`) | Cloud `mean_retained_kl` | `explicit_submit` / `clean_exec` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`norm_baseline` (Static Norm Heuristic)** | `0.0%` (`0/5`) | `0.0%` (`0/10`) | **`0.0%` (`0/30`)** (`0/10`, `0/10`, `0/10`) | `3.8362` | `1.0000` / `1.0000` |
| **Unmodified Phase 2.5 GRPO (`grpo_phase2_5_lora`)** | — | — | **`0.0%` (`0/30`)** (`mean_reward = -0.004444`) | `3e-08` | `0.1000` / `0.1011` |
| **Unmodified Phase 4 GRPO (`grpo_phase4_lora`)** | — | — | **`0.0%` (`0/30`)** (`mean_reward = 0.025972`) | `0.4218` | `0.0667` / `0.0833` |
| **`hybrid_ref` (General `wsp_tools` Primitives)** | **`100.0%` (`5/5`)** | **`100.0%` (`10/10`)** | **`90.0%` (`27/30`)** (`9/10`, `8/10`, **`10/10`**) | **`0.003577`** (`< 0.01`) | **`1.0000` / `1.0000`** |

**Key Phase 5 Finding:** All targets pass `audit_organic_norm_stealth`, driving `norm_baseline` to **`0/45` (`0.0%`)** across `mini`/`25m`/`100m`/`SmolLM2-135M`/`Qwen2.5-0.5B`. Honest falsification confirms that unmodified `checkpoints/grpo_phase4_lora` (`0/30`) and `checkpoints/grpo_phase2_5_lora` (`0/30`) fail on norm-stealth SGD/LoRA weights (establishing genuine RLVR headroom for Phase 6), whereas solving with only general `wsp_tools` primitives (`probe_attribution`, generalized `covariance_nullspace`, `solve_nullspace_ridge`) reaches **`5/5` (`100%`, `KL = 0.004997` GPU / `0.007136` CPU)** on `mini`, **`10/10` (`100%`, `KL = 0.005710`)** on `25m`, and **`27/30` (`90.0% >= 85%`, `mean_reward = 0.822328`, `mean_retained_kl = 0.003577 < 0.01`)** on the cloud suite (`cat2` `6/6`, `cat4` `4/6`, `cat5_known` `6/6`, `cat5_symptom` `6/6`, `cat6` `5/6`; **`10/10 = 100%`** on `Qwen2.5-0.5B` at `KL = 0.000626`).

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
  organic.py       # Phase 5 AdamW/Projected-LoRA norm-stealth targets (cat2/4/5k/5s/6 across synth & LLMs)
  toolkit.py       # Pre-loaded wsp_tools (probe_attribution, covariance_nullspace, solve_nullspace_ridge, SVD)
  verifier.py      # Deterministic capability, KL-retention, 20-step LoRA recovery & organic/constructive verifiers
  sandbox.py       # Stateful REPL, extra_public_weights, linalg >4 GiB memory guard, trace & cgroup RAM guard
  training.py      # Turn-masked & error-turn-masked SFT exporter, CurriculumScheduler, & GRPORolloutEnv
  scaler.py        # 100% static Option 3 SFT trajectory builder (0 probe_forward calls)

eval.py            # Phase 1 held-out evaluator (17 subtypes, 34 episodes)
eval_level2.py     # Level 2 scale-transfer & unseen-LLM evaluator (20 episodes, --suite split/merge)
eval_level3.py     # Level 3 7.62B Immutable-Twin Self-Surgery evaluator (--suite restorative|constructive|all)
eval_organic.py    # Phase 5 Organic SGD/LoRA evaluator (30-ep cloud suite, --merge-reports, norm/hybrid/hf)
train_sft.py       # Turn-masked LoRA SFT warmup trainer
train_grpo.py      # Online multi-turn GRPO trainer (phase4 curriculum + frozen SFT KL anchor)
run_phase*.sh      # Cloud Runpod A40/H100/Blackwell orchestration scripts (incl. run_phase5_cloud.sh)
tests/             # 41 unit & integration tests (test_wsp_gym: 13, test_level2: 7, test_level3: 5, test_phase4: 8, test_phase5: 8)
trajectories/      # Tracked SFT corpora, GRPO logs, HF equivalence JSONs, and all held-out & Phase 5 eval reports
checkpoints/       # Local gitignored LoRA adapters (~319 MB each, excluded from git):
                   #   sft_lora, grpo_lora (+ step_0051), sft_level2_lora, grpo_level2_lora,
                   #   sft_phase2_5_lora, grpo_phase2_5_lora, sft_phase4_lora, grpo_phase4_lora (canonical best)
```

---

## Quick Verification & Evaluation Commands

```powershell
# Run all 41 unit and integration tests (or targeted fast modules locally on CPU)
python -m unittest discover -s tests -v

# Scripted verification of Level 1/1.5 (34 eps), Level 2 (20 eps), Level 3 mini twin (12 eps), and Organic mini (5 eps)
python eval.py --backend scripted --episodes-per-subtype 2
python eval_level2.py --suite all --scale 100m --backend scripted --episodes-per-subtype 2
python eval_level3.py --scale mini --suite all --backend scripted --episodes-per-subtype 2
python eval_organic.py --suite synthetic --synthetic-scale mini --backend hybrid_ref --episodes-per-subtype 1 --base-seed 39000

# Cloud GPU: Level 3 7.62B self-surgery & Phase 5 30-episode Organic SGD/LoRA suite (see run_phase5_cloud.sh)
python eval_level3.py --scale 7b --suite all --backend hf --model-name-or-path Qwen/Qwen2.5-Coder-7B-Instruct --adapter-path checkpoints/grpo_phase4_lora --episodes-per-subtype 2 --base-seed 29000 --max-turns 4 --timeout-sec 90.0 --max-new-tokens 1536 --out-path trajectories/eval_level3_p4_grpo.json
bash run_phase5_cloud.sh
```

---

## Operational Notes at Current Handoff
- **Git & Clean Tree:** Local Phase 5 changes, reports (`8` `trajectories/eval_organic_*.json`), & docs committed cleanly on `main`. `.gitignore` excludes `checkpoints/`, `challenges/`, `*.pt`, `*.tar.gz`, `*.log`, and `*.partial.json`. On Windows headless shells, avoid invoking `git push` directly if Git Credential Manager opens an interactive GUI prompt.
- **Cloud State (`$0/hr`):** **0 running Runpod pods**. Temporary Phase 5 pod `xmaz32b65jwl12` (`1x RTX PRO 4500 Blackwell 32GB`, `$1.09/hr`) and earlier pods (`pbuffkd7uaupcz`, `2jy1wdvr1wzoyq`, `9l5oyegho2g91g`, `0kw73msu7atapo`) were terminated cleanly after downloading artifacts; pre-existing stopped user pods `fu3n5qpwrsqq8d` and `hgylz13bq35gxx` remain untouched (`exited`). Note that container-disk Runpod pods without `--persistent-volume` wipe `/workspace` across stop/restart cycles — always `scp` artifacts before stopping/terminating.
