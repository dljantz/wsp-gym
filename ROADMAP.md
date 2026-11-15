# Weight-Space Programming (WSP) & Path to Recursive Self-Improvement

> **Purpose of This Document:** Persistent architectural memory and project state for `wsp_gym`. Any new conversation or subagent can read this file to resume immediately without losing design rationale, exact subtypes, experimental/numerical lessons, or trajectory plans.

---

## 1. North Star & Theoretical Principles

**Weight-Space Programming (WSP)** treats neural network parameters as readable, editable source code rather than write-only black boxes shaped solely by brute-force gradient descent.

### Core Scaling & Training Principles
1. **Fixed Agent Size ($N_{\text{agent}}$), Scaling Target Size ($N_{\text{target}} \to N_{\text{agent}}$):**
   * The editing agent's parameter count stays fixed (e.g., a 4B–8B local model) while the target network curriculum scales upward.
   * Strict **context-window `stdout` truncation (`STDOUT_MAX_CHARS = 2000`)** forbids printing raw floating-point tensors, forcing the agent to write **scale-invariant Python diagnostics** (SVD, subspace projections, RoPE offset spectra, activation covariances, Jacobian probes) that transfer cleanly from $d=32\text{--}64$ toy transformers to $7\text{B}+$ models.
2. **RL with Verifiable Rewards (RLVR) Solves the Data Shortage:**
   * There is no pre-existing "GitHub of weight edits." However, procedural weight-space gyms generate infinite challenges with deterministic, millisecond-fast CPU verification.
3. **Forward-Pass Budget Annealing ($B = 10 \to 5 \to 2 \to 1 \to 0$):**
   * Dynamic forward passes (`probe_forward`, max batch size `32`) act as training wheels so an agent can correlate empirical activations with static weight geometry, then gradually anneal $B \to 0$ until the model masters pure static weight reading.
4. **Subsuming Gradients & Immutable-Twin Safety at Self-Scale:**
   * Pure closed-form weight algebra excels at structural edits (loop/attractor removal, skill splicing, backdoor excision, refactoring, System 2 $\rightarrow$ System 1 circuit compilation). For high-entropy empirical data, a mature WSP agent combines mechanistic localization with targeted local Jacobian/second-order updates inside Python.
   * When an agent edits its *own* architecture, it must use an **Immutable Twin (Fork-and-Verify)** setup: frozen agent $A_t$ edits cloned weights $A'_{t+1}$, which are promoted only after passing an external verifier + KL-divergence retention check.

### Why Each Architectural & Training Choice Exists
* **Closed-Form Weight Algebra + Local Jacobians (Over Brute-Force SGD Alone):** Structural edits (excising backdoors, deleting nullspace/canceling heads, splicing orthogonal skills, resolving RoPE composition offsets) have exact algebraic solutions directly visible in spectra, whereas high-entropy empirical tasks benefit from targeted local second-order/Jacobian updates inside the Python REPL. Immutable-Twin gating ensures self-edits $A'_{t+1}$ only promote past deterministic capability and KL checks.
* **Fixed $N_{\text{agent}}$ + `STDOUT_MAX_CHARS = 2000` (Over Unconstrained Tensor Printing):** Capping REPL `stdout` prevents an agent from memorizing or brute-force inspecting raw weight matrices, compelling reusable spectral diagnostics (`wsp_tools`) whose complexity scales cleanly from $d=32\text{--}64$ up to $7\text{B}+$ blocks.
* **Haar-Random `scramble_gauge` + `.vault/` Isolation (Over Canonical/Unblinded Weights):** Applying RoPE- and RMSNorm-aware orthogonal gauge scrambling in [wsp_gym/gauge.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/gauge.py) and isolating `subtype`/`difficulty` ground-truth inside `.vault/` via `save_challenge` in [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py) forces genuine coordinate-free mechanistic inspection rather than label leakage or basis memorization.
* **Option 3 `123`-Episode Turn-Masked SFT (Over `525` Uniform 2-Turn Scripts):** Masking `system`/`user` tokens to `-100` (`tokenize_and_mask_chat` / `loss_mask`) and curating `123` balanced episodes including 3-turn `error_recovery` and `hypothesis_refine` trajectories in [wsp_gym/scaler.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/scaler.py) avoids turn-count collapse and keeps exploration diversity intact for RLVR.
* **Frozen `sft_ref_params` Snapshotting (Over PEFT `disable_adapter()`):** Disabling the LoRA adapter during GRPO anchors Schulman's `k3` KL penalty to the `0.00%`-pass base model (`kl_div = 1.16–4.05` in early testing), pulling generations away from valid `wsp_tools` usage. Snapshotting the trained SFT LoRA parameters (`_snapshot_trainable_params` / `_swap_trainable_params` in [train_grpo.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_grpo.py)) anchors KL to the `44.12%` SFT checkpoint (`kl_div = 0.00043` at Step 0, `0.002–0.015` through Step 30).
* **Base-Variance Guard on All-Fail Groups (`max(base_rewards) - min(base_rewards) >= 1e-4`):** When every rollout in a GRPO group fails identically on base verification, group-normalizing tiny `0.02–0.05` shaping differences inflates them into full $\pm 1$ advantages, causing policy drift without real signal (`kl_div` up to `0.387683` at Step 62). Skipping such updates once `step_idx >= 10` preserves acquired skills while avoiding wasted drift.

---

## 2. Multi-Phase Compute Roadmap (Laptop + ~$300 Cloud GPU)

| Phase | Status | Environment & Budget | Core Objective |
| :--- | :--- | :--- | :--- |
| **Phase 0: Gym, Playtesting & Cold-Start Data** | **COMPLETE** (Cycles 1–3 + Scaling Harness Built; `13/13` Unit Tests Passing) | Windows Laptop (`python 3.14`, CPU PyTorch, **\$0**) | Built & hardened the 7-category procedural gym (`wsp_gym`), completed 7/7 Level 1.5 grokked `RMSNorm`+`SwiGLU`+`RoPE` models, built `training.py` SFT/GRPO harness & `scaler.py` automated blinded rollout engine, and validated end-to-end 17-subtype coverage. |
| **Phase 1: Local Neurosurgeon Training (Level 1 + 1.5)** | **Cloud Run 1 (`C1–C4`) COMPLETE** (`0% -> 44.12%` SFT; `72.2%` Steps 18–29 GRPO; `32.35%` Step 80) | Local CPU Prep (**\$0**) -> Cloud GPU (`1x H100 80GB`) | **Completed:** Enforced symptom-only prompt blinding (`L1`), sandbox timeout/parser/reward-shaping hardening (`L2`), curated **123-episode Option 3 turn-masked SFT corpus** (`L3`), verified `eval.py`/`train_sft.py`/`train_grpo.py` (`L4`), and ran Cloud GPU `Steps C1–C4` on `Qwen/Qwen2.5-Coder-7B-Instruct` (`trajectories/eval_{base,sft,grpo}.json`, `checkpoints/{sft,grpo}_lora`). **Next:** Evaluate intermediate `step_0020`/`step_0040` GRPO checkpoints, harden multi-subtype mixing (with all-fail micro-shaping guard now active), and close `cat3`/`cat4`/`cat6` gaps before Phase 2. |
| **Phase 2: Scale-Transfer Evaluation (Level 2)** | Planned | Cloud GPU (**~\$100**) | Evaluate whether the Python neurosurgery strategies learned on $d=32\text{--}64$ transformers generalize zero-shot or with light RL to $100\text{M}\text{--}1\text{B}$ target networks. |
| **Phase 3: Self-Surgery Proof-of-Concept (Level 3)** | Planned | Cloud GPU (**~\$100**) | Have the RL-trained 4B–8B model inspect and surgically edit a cloned copy of **its own 4B–8B weights** (e.g., curing an induced repetition loop, excising a planted backdoor, or injecting a null-space skill) under an **Immutable Twin** verifier. |

### Phase 0 Subgoals Breakdown
* **Cycle 1 (Complete):** Core 7-category Level 1 gym, gauge scrambler, stateful multi-turn sandbox, verifier, unit tests, and 7/7 blinded playtest baseline (`trajectories/episodes.jsonl` records `00`–`06`).
* **Cycle 2 Priorities 1–4 (Complete):**
  * *Priority 1:* Sandbox ergonomics (`SYSTEM_PROMPT` signatures) and `wsp_tools` standard library.
  * *Priority 2:* Harder anti-shortcut procedural subtypes across Cat 1, Cat 2, and Cat 6.
  * *Priority 3:* Level 1.5 SGD-grokked (`AdamW`) & polysemantic superposition targets with `rmsnorm`, `swiglu` (`silu`), and `rope`, plus RoPE/RMSNorm-aware `wsp_tools`.
  * *Priority 4:* First parallel subagent-delegated trajectory harvest completed (8/8 blinded Cycle 2 & Level 1.5 episodes passed, expanding [trajectories/episodes.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/episodes.jsonl) to **15 verified trajectories**).
* **Cycle 3 Priorities 1–3 (Complete):**
  * *Priority 1:* Completed **7/7 category coverage** for Level 1.5 (`RMSNorm` + `SwiGLU` + `RoPE`) by adding `cat1_decompile` (`level1_5_grokked_rope_decompile`), `cat3_synthesize` (`level1_5_swiglu_rope_synthesize`), and `cat6_splice_merge` (`level1_5_grokked_rope_merge`) in [wsp_gym/level1_5.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/level1_5.py) and `LEVEL1_5_SUBTYPES` in [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py).
  * *Priority 2:* Built [wsp_gym/training.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/training.py) (`audit_and_export_sft_dataset`, $B \to 0$ `CurriculumScheduler`, `GRPORolloutEnv`) and added the `export-sft` CLI subcommand to [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py).
  * *Priority 3:* Harvested 6 blinded strictly zero-probe (`B=0`) episodes across `challenges/cycle3_suite_v1` via 3 parallel `wsp-solver` workers (`6/6` passed, rewards `0.9456`–`1.0000`), growing [trajectories/episodes.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/episodes.jsonl) and [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl) to **21 verified trajectories across all 17 subtypes** (`10` zero-probe `B=0` trajectories).
* **Phase 0 Corpus Scaling (Complete — Superseded for Training by Option 3 in Step L3):**
  * Built [wsp_gym/scaler.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/scaler.py) (`ALL_17_SUBTYPES`, `solve_blinded_challenge`, `scale_sft_corpus`) and added the `scale-corpus` CLI subcommand to [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py), initially scaling to 525 2-turn template trajectories across all 17 subtypes before curating into the higher-entropy 123-episode Option 3 dataset below.
* **Phase 1 Local Preparation — Steps L1–L4 (Complete; `13/13` Unit Tests Passing):**
  * *Step L1 (Symptom-Only Prompt Contract & Metadata Blinding):* Removed `subtype` and `difficulty` from `<ep>/public/metadata.json` (moved into `<ep>/.vault/verifier_secret.pt`), removed diagnostic/subtype spoilers from `brief.md` across [wsp_gym/generators.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/generators.py) and [wsp_gym/level1_5.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/level1_5.py) (unifying `cat2_repair` briefs to symptom-only descriptions), and scrubbed all 17 subtype strings from [wsp_gym/scaler.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/scaler.py) `<think>` traces so dispatch relies strictly on `session.config` and static inspection.
  * *Step L2 (RL-Ready Sandbox Hardening & Raw-Text Rollouts):* Added `DEFAULT_TURN_TIMEOUT_SEC = 5.0` `sys.settrace` loop deadline, `(Exception, SystemExit)` guard, `timed_out`, and `explicit_submit` tracking in [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py); added `STOP_SEQUENCES`, `parse_assistant_turn()`, `compute_shaped_reward()`, and `GRPORolloutEnv.step_raw()` / `run_raw_episode()` in [wsp_gym/training.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/training.py).
  * *Step L3 (Option 3 High-Entropy 123-Episode Turn-Masked SFT Corpus):* Updated `audit_and_export_sft_dataset()` with `strip_trailing_user=True`, per-message `loss_mask` (`True` iff `role == "assistant"`), record-level `loss_masks`, and `multi_turn_3plus_count` / `error_recovery_count` metrics; added 3-turn `solve_exploratory_challenge()` (`error_recovery` and `hypothesis_refine` modes) and `build_option3_sft_corpus()` (`--option3`), rebuilding [trajectories/episodes.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/episodes.jsonl) and [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl) to **123 curated episodes** (`21` organic Cycle 1–3 + `68` 2-turn blinded + `34` 3-turn exploratory; `94` zero-probe, `46` 3+ turn, `20` error-recovery, `mean_reward=0.959028`).
  * *Step L4 (Evaluation, Turn-Masked SFT, and Online Multi-Turn GRPO Entrypoints):* Created [eval.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/eval.py), [train_sft.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_sft.py), and [train_grpo.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_grpo.py) with both Cloud GPU (`transformers` + `peft`) and zero-cost CPU verification modes (`eval.py --backend scripted` **`34/34`** passing; `train_sft.py --dry-run` **`123/123`** verified; `train_grpo.py --dry-run` **`8/8`** rollouts verified).

---

## 3. The 7-Category Taxonomy, Subtypes & Anti-Reward-Hacking Defenses

Together, the 7 categories form a closed self-improvement engine (`Audit -> Repair/Excise -> Synthesize/Splice/Inject -> Refactor -> Repeat`):

| Category (`CATEGORIES`) | Core Skill | Primary Anti-Cheat Defense | Implemented Subtypes |
| :--- | :--- | :--- | :--- |
| **`cat1_decompile`** | Infer symbolic rule `(a, b, c)` + 3-token trojan from raw weights | `forward_budget=0`, `W_U` null-space decoy dilution, cancelling MLP decoy pairs, gauge scrambling | `modular_linear_with_3tok_trojan`, `modular_varied_lag_with_3tok_trojan`, `level1_5_grokked_rope_decompile` |
| **`cat2_repair`** | Diagnose structural pathology & apply minimal-norm patch | Evaluated on both ID and OOD sequence lengths; norm-matched rotation option defeats naive outlier hunting | `mlp_sink_attractor`, `norm_matched_k_composition_corruption`, `level1_5_grokked_swiglu_rope_repair` |
| **`cat3_synthesize`** | Compile functional dual-offset modular shift circuit from zero weights | ID ($T=8/12$) + OOD ($T=16/24$) length generalization + sparsity reward | `dual_offset_modular_shift`, `level1_5_swiglu_rope_synthesize` |
| **`cat4_nullspace_inject`** | Inject new prefix/suffix task without touching hidden existing tasks | **Zero retention data** given to agent; $D_{\text{KL}} < 10^{-3}$ enforced on hidden tasks | `orthogonal_subspace_skill_addition`, `level1_5_grokked_subspace_inject` |
| **`cat5_unlearn_excise`** | Remove internal 2-token backdoor while preserving clean task | Internal `blocks.1.resid_post` trojan projection ratio (`< 0.05`) + **10-step `W_U`/`b_U` gradient recovery attack** (`<= 0.25`) | `2tok_backdoor_excision`, `level1_5_grokked_backdoor_excision` |
| **`cat6_splice_merge`** | Merge or compose independently basis-rotated models (`weights` + `weights_B`) | Independent random $Q_A \ne Q_B$, head/MLP permutations; unified single `d_model=48` model | `orthogonal_subspace_model_merge` (horizontal), `vertical_function_composition` ($f_B \circ f_A$), `level1_5_grokked_rope_merge` |
| **`cat7_refactor`** | Prune cancelling heads/neurons & `W_U` null-space decoys; pack via SVD | Worst-case $\max \|\Delta \text{logits}\| < 10^{-3}$ + $\ge 50\%$ reduction in $\|W\|_F^2$ and $\|W\|_0$ | `nullspace_and_cancelling_head_defrag`, `level1_5_grokked_svd_defrag` |

### Shared Anti-Memorization & Unified Reward
* **Function-Preserving Gauge Scrambler ([wsp_gym/gauge.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/gauge.py)):**
  * Applies Haar-random residual orthogonal rotations $Q_{\text{res}} \in O(d_{\text{model}})$ when `norm_type` is `"none"` or uniform `"rmsnorm"` (skipped when residual RMSNorm gains are non-uniform or `rotate_residual=False`).
  * Permutes attention heads and MLP neurons independently per layer, applies $O(d_{\text{head}})$ $R_{vo}$ rotations on every head, and applies $O(d_{\text{head}})$ $R_{qk}$ rotations when `pos_type != "rope"` (preserving RoPE channel pairs when `pos_type == "rope"`).
* **Strict Vault Isolation & Symptom-Only Blinding:** `<ep>/public/` (`brief.md`, `config.json`, `metadata.json`, `weights.pt`, optional `weights_B.pt`) exposes only agent-safe model configuration and functional symptom descriptions; `subtype`, `difficulty`, and hidden evaluation/oracle fixtures live strictly inside `<ep>/.vault/verifier_secret.pt` until verification.
* **Unified Reward $R \in [-1.0, +1.0]$ ([wsp_gym/verifier.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/verifier.py)):**
  $$R = 0.70 \cdot S_{\text{task}} \cdot \exp(-\beta \, D_{\text{KL}}^{\text{retain}}) + \mathbb{I}[S_{\text{task}} \ge 0.95] \cdot \left(0.20 \, S_{\text{surgical}} + 0.10 \, S_{\text{budget}}\right)$$

---

## 4. Workspace Architecture & `wsp_tools` Standard Library

### Codebase Files
* [wsp_gym/model.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/model.py) — `ModelConfig`, `init_weights`, `forward_with_cache` (supports `rmsnorm`/`layernorm`/`none`, `standard`/`swiglu`, `relu`/`gelu`/`silu`, `learned`/`rope`).
* [wsp_gym/gauge.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/gauge.py) — `sample_orthogonal`, `scramble_gauge` (RoPE- and RMSNorm-aware).
* [wsp_gym/generators.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/generators.py) — Level 1 & Cycle 2 procedural generators (with symptom-only `brief.md`) + `generate_challenge(category, seed=0, difficulty=1.0, forward_budget=None, subtype=None)` dispatcher.
* [wsp_gym/level1_5.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/level1_5.py) — Full 7/7 Level 1.5 generators (`generate_level1_5_cat{1..7}_*`) using non-uniform RMSNorm, SwiGLU (`silu`), RoPE, and CPU AdamW grokking (`_run_adamw_grokking_steps`).
* [wsp_gym/toolkit.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/toolkit.py) — Pre-loaded `wsp_tools` diagnostic and canonical packing library.
* [wsp_gym/verifier.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/verifier.py) — `verify_submission` across all 7 categories and anti-cheat checks.
* [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py) — `SandboxSession` stateful REPL (with `DEFAULT_TURN_TIMEOUT_SEC=5.0` trace deadline & `(Exception, SystemExit)` guard), `SYSTEM_PROMPT`, `save_challenge` (vault isolation of `subtype`/`difficulty`), CLI (`generate-suite`, `step`, `export-sft`, `scale-corpus [--option3]`), and JSONL logger (`trajectories/episodes.jsonl`).
* [wsp_gym/training.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/training.py) — Turn-masked `audit_and_export_sft_dataset` (`loss_mask` / `strip_trailing_user`), `CurriculumScheduler` (4-stage $B \to 0$ probe annealing with `window_size=8 * group_size`), `STOP_SEQUENCES`, `parse_assistant_turn`, `compute_shaped_reward`, and `GRPORolloutEnv` (`step_raw`, `run_raw_episode`).
* [wsp_gym/scaler.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/scaler.py) — `ALL_17_SUBTYPES`, `solve_blinded_challenge`, 3-turn `solve_exploratory_challenge` (`error_recovery` / `hypothesis_refine`), and `build_option3_sft_corpus` (`scale_sft_corpus` also retained).
* [eval.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/eval.py) — Disjoint held-out 17-subtype evaluation harness (`build_heldout_suite`, `run_evaluation`, default `--max-new-tokens 1536`) supporting `dummy`, `scripted`, and `hf` (HuggingFace + PEFT LoRA) backends.
* [train_sft.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_sft.py) — Turn-masked LoRA SFT warmup trainer (`tokenize_and_mask_chat`, `run_sft_dry_run`, `run_sft_cloud_training`) masking `system`/`user` tokens to `-100`.
* [train_grpo.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_grpo.py) — Online multi-turn GRPO trainer (`compute_grpo_group_advantages`, `compute_grpo_loss_and_metrics` with PPO clipping + Schulman `k3` KL penalty, frozen `sft_ref_params` snapshotting via `_snapshot_trainable_params`/`_swap_trainable_params`, all-fail micro-shaping guard for `step_idx >= 10`, defaults `--group-size 6`, `--num-steps 80`, `--max-turns 4`, `--max-new-tokens 1280`, `--temperature 0.6`, `--save-every 20`).
* [run_phase1_cloud.sh](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/run_phase1_cloud.sh) — End-to-end cloud GPU orchestration script for Phase 1 (`C1–C4`) setup, evaluation, SFT warmup, and GRPO training.
* [.gitignore](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/.gitignore) — Excludes Python caches, virtualenvs, `checkpoints/`, `challenges/`, `phase1_results*`, `*.tar.gz`, and `*.pt`.
* [tests/test_wsp_gym.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/tests/test_wsp_gym.py) — **13 unit & integration test suites** covering configs, gauge invariance, suite generation, all 7 Level 1 categories, Cycle 2 subtypes, all 7 Level 1.5 models/oracles, `wsp_tools`, multi-turn sandbox logging, timeout/`SystemExit` guards & raw parser, turn-masked SFT export & exploratory 3-turn solves, and `eval.py`/`train_sft.py`/`train_grpo.py` CPU verification.

### Persistent Artifacts & Repository State
* **Git Remote:** `https://github.com/dljantz/wsp-gym.git` (branch `main`).
* **Tracked Training & Eval Logs (`trajectories/`):**
  * [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl) — 123-episode turn-masked Option 3 SFT training corpus.
  * [trajectories/eval_base.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_base.json) — `C1` zero-shot `Qwen/Qwen2.5-Coder-7B-Instruct` held-out baseline.
  * [trajectories/eval_sft.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_sft.json) — `C2` LoRA SFT held-out evaluation (`44.12%`, run at `--max-new-tokens 1024`).
  * [trajectories/grpo_train_log.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/grpo_train_log.jsonl) — 80-step online Stage-0 GRPO training metrics (`step 0..79`).
  * [trajectories/eval_grpo.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_grpo.json) — `C4` Step-80 GRPO held-out evaluation (`32.35%`).
* **Local Gitignored LoRA Checkpoints (`checkpoints/`):**
  * [checkpoints/sft_lora](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/checkpoints/sft_lora) — Turn-masked SFT LoRA adapter (`44.12%` eval baseline).
  * [checkpoints/grpo_lora](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/checkpoints/grpo_lora) — Final GRPO adapter plus intermediate checkpoints `step_0020`, `step_0040`, `step_0060`, and `step_0080`.

### `wsp_tools` Reference (Pre-Loaded in Sandbox)
* `residual_svd_basis(weights, config=None, tol=1e-4, include_embeddings=True)` — Orthonormal SVD split into `V_active` and `V_null` plus `rank` and `singular_values`.
* `rope_rotation_matrix(d_head, offset, dtype=torch.float32)` — Exact $(d_{\text{head}}, d_{\text{head}})$ RoPE relative-offset matrix $R(t - s)$ paired on `(i, half + i)`.
* `head_offset_profile(weights, layer=0, query_pos=6, max_offset=8)` — Dominant QK positional offset (`dominant_offset`) for both learned `W_pos` and RoPE (`W_pos` absent) architectures, accounting for `norm1.weight`.
* `virtual_ov_circuit(weights, layer=0, head=None)` — Virtual $W_V W_O$ map in residual coordinates.
* `mlp_logit_impact(weights, layer=0, tol=1e-4)` — Projects `W_out` onto effective `norm_final.weight * W_U`, filters float32 orthogonal rounding noise (`_active_projection_mask`), and detects cancelling neuron pairs (`W_in`, `b_in`, `W_gate`, opposite `W_out @ downstream`).
* `canonical_pack(weights, ...)` — Absorbs non-uniform RMSNorm gains (`_absorb_rmsnorm_gains`), prunes inactive/cancelling heads and MLP/SwiGLU neurons, preserves RoPE channel pairs on `W_Q`/`W_K`, and repacks into the minimal SVD residual basis.

### Key Commands
```powershell
# Run full 13-suite unit, sandbox guard, exploratory solver, eval, SFT, and GRPO dry-run test suite
python -m unittest discover -s tests -v

# Rebuild the curated 123-episode Option 3 turn-masked SFT corpus (21 organic + 68 2-turn + 34 3-turn)
python -m wsp_gym.sandbox scale-corpus --option3 --trajectory-path trajectories/episodes.jsonl --out-path trajectories/sft_train.jsonl

# Audit and export verified chat trajectories with assistant-only loss_mask
python -m wsp_gym.sandbox export-sft --trajectory-path trajectories/episodes.jsonl --out-path trajectories/sft_train.jsonl

# Run held-out evaluation across all 17 subtypes (disjoint seeds >= 9000; 34 episodes total)
python eval.py --backend scripted --episodes-per-subtype 2

# CPU dry-run verification of turn-masked SFT and online multi-turn GRPO
python train_sft.py --dry-run
python train_grpo.py --dry-run
```

---

## 5. Project History: Where We've Been & Key Mechanistic Lessons

### Cycle 1 Blinded Playtest Results (7/7 Passed in `trajectories/episodes.jsonl`)
| Episode | Category | Turns | Forward Batches | Total Reward | Key Verified Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `ep_01` | `cat1_decompile` | 2 | 0/0 | **1.0000** | Exact `(a, b, c)` + 3-token trojan `[alpha, beta, delta, gamma]` |
| `ep_02` | `cat2_repair` | 3 | 2/4 | **0.9056** | ID/OOD acc `0.0234/0.0312 -> 1.0000` |
| `ep_03` | `cat3_synthesize` | 1 | 1/3 | **0.9722** | ID/OOD acc `1.0000`, `weight_sparsity=0.9862` |
| `ep_04` | `cat4_nullspace_inject` | 2 | 1/2 | **0.9230** | Target acc `1.0000`, hidden task `retained_kl_div=1e-08` |
| `ep_05` | `cat5_unlearn_excise` | 2 | 1/4 | **0.9466** | `clean/trig_keep=1.0`, `trojan_proj=0.0`, `recovery_attack=0.1827` |
| `ep_06` | `cat6_splice_merge` | 2 | 0/2 | **0.9968** | `task_A=1.0`, `task_B=1.0`, `sparsity=0.9842` |
| `ep_07` | `cat7_refactor` | 2 | 2/2 | **0.8897** | `max_diff=5.5e-05`, `frob_sq=0.392x`, `L0=0.0442x` |

### Cycle 2 Priorities 1–4 Status — COMPLETE (`8/8` Unit Tests Passing; `8/8` Blinded Subagent Episodes Passed)
1. **Priority 1 (Sandbox Ergonomics & `wsp_tools`):** Documented exact `probe_forward(..., return_cache=True/False)`, `submit_weights`, `submit_decompiled`, and `wsp_tools` signatures in `SYSTEM_PROMPT` (eliminating the `ep_02` tuple-unpacking trap) and promoted core algebra primitives into `wsp_gym/toolkit.py`.
2. **Priority 2 (Anti-Shortcut Cycle 2 Subtypes):**
   * Added cancelling decoy MLP pairs (`+v, -v`) and varied modular lags `(3, 1)` / `(3, 2)` in `cat1_decompile` (`modular_varied_lag_with_3tok_trojan`).
   * Added norm-matched subspace rotation in the 2-layer induction circuit for `cat2_repair` (`norm_matched_k_composition_corruption`) — singular values and Frobenius norms match the healthy checkpoint, requiring activation/pattern probes (`probe_forward`) and virtual circuit tracing.
   * Added vertical function composition $f_B \circ f_A$ in `cat6_splice_merge` (`vertical_function_composition`), requiring Model B to be grafted into Layer 1 behind Model A in Layer 0.
3. **Priority 3 (Level 1.5 Grokked & Superposition Models):** Implemented and verified `cat2`, `cat4`, `cat5`, and `cat7` Level 1.5 challenges using `RMSNorm`, `SwiGLU`, `RoPE`, non-uniform norm gains, and CPU `AdamW` fine-tuning (`8/8` tests pass in `tests/test_wsp_gym.py`, including baseline-fail, Cat 5 recovery-attack rejection, oracle `total_reward > 0.80`, and `canonical_pack` passing Level 1.5 Cat 7).
4. **Priority 4 (Blinded Subagent-Delegated Trajectory Harvesting — `challenges/cycle2_suite_v1`):** Delegated 8 blinded Cycle 2 & Level 1.5 episodes across 4 parallel `wsp-solver` workers and merged all 8 passing trajectories into [trajectories/episodes.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/episodes.jsonl) (now **15 total verified trajectories**):

| Episode ID | Category | Subtype | Turns | Probes | Reward | Key Verified Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `ep_cat1_501` | `cat1_decompile` | `modular_varied_lag_with_3tok_trojan` | 5 | 0/0 | **1.0000** | `rule_match=True`, `trigger_match=True` (static lags `(3, 2)` + cancelling decoys filtered) |
| `ep_cat2_502` | `cat2_repair` | `norm_matched_k_composition_corruption` | 4 | 2/4 | **0.9580** | ID/OOD acc `0.0234/0.0469 -> 1.0`, `rel_edit=0.0102` |
| `ep_cat6_503` | `cat6_splice_merge` | `vertical_function_composition` | 5 | 3/3 | **0.8484** | `task_A=1.0`, `task_B=1.0`, `sparsity=0.617` ($f_B \circ f_A$ Layer 1 graft) |
| `ep_cat3_504` | `cat3_synthesize` | `dual_offset_modular_shift` | 1 | 0/0 | **0.9961** | Zero-probe synthesis: ID/OOD acc `1.0`, `sparsity=0.9806` |
| `ep_l15_cat2_601` | `cat2_repair` | `level1_5_grokked_swiglu_rope_repair` | 4 | 2/4 | **0.9474** | ID/OOD acc `0.0859/0.0625 -> 1.0`, `rel_edit=0.0631` |
| `ep_l15_cat4_602` | `cat4_nullspace_inject` | `level1_5_grokked_subspace_inject` | 2 | 2/2 | **0.8779** | Target acc `0.0 -> 1.0`, hidden `retained_kl_div=0.0`, `rel_edit=0.2771` |
| `ep_l15_cat5_603` | `cat5_unlearn_excise` | `level1_5_grokked_backdoor_excision` | 5 | 2/4 | **0.9445** | `clean/trig_keep=1.0`, `trojan_proj=0.0`, `recovery_attack=0.0`, `rel_edit=0.0777` |
| `ep_l15_cat7_604` | `cat7_refactor` | `level1_5_grokked_svd_defrag` | 2 | 2/2 | **0.9196** | `max_diff=0.000267`, `frob_sq=0.0803x`, `L0=0.0573x` (`canonical_pack`) |

### Cycle 3 Priorities 1–3 Status — COMPLETE (`9/9` Unit Tests Passing; `21/21` Verified SFT Episodes Exported)
1. **Priority 1 (7/7 Level 1.5 Coverage):** Completed `level1_5_grokked_rope_decompile`, `level1_5_swiglu_rope_synthesize`, and `level1_5_grokked_rope_merge` in [wsp_gym/level1_5.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/level1_5.py) (`9/9` tests pass including baseline-fail, `head_offset_profile` non-null checks, and exact `cat1_decompile` forward match).
2. **Priority 2 (SFT Exporter & GRPO Curriculum):** Created [wsp_gym/training.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/training.py) (`audit_and_export_sft_dataset`, 4-stage `CurriculumScheduler` with $B \to 0$ probe annealing, `GRPORolloutEnv`) and wired `export-sft` into [wsp_gym/sandbox.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/sandbox.py).
3. **Priority 3 (Blinded `B=0` & Full Level 1.5 Harvest — `challenges/cycle3_suite_v1`):** Solved all 6 zero-probe episodes cleanly via 3 parallel `wsp-solver` workers and exported **21/21** verified episodes (`mean_reward=0.952924`, `zero_probe_count=10`, all 17 subtypes covered) into [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl):

| Episode ID | Category | Subtype | Turns | Probes | Reward | Key Verified Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `ep_l15_cat1_701` | `cat1_decompile` | `level1_5_grokked_rope_decompile` | 6 | 0/0 | **1.0000** | `rule_match=True`, `trigger_match=True` (cancelling SwiGLU pairs + RMSNorm absorbed) |
| `ep_l15_cat3_702` | `cat3_synthesize` | `level1_5_swiglu_rope_synthesize` | 5 | 0/0 | **0.9953** | Zero-probe SwiGLU+RoPE synthesis: ID/OOD acc `1.0`, `sparsity=0.9764` |
| `ep_l15_cat6_703` | `cat6_splice_merge` | `level1_5_grokked_rope_merge` | 6 | 0/0 | **0.9921** | `task_A=1.0`, `task_B=1.0`, `sparsity=0.9604` (Procrustes + `canonical_pack`) |
| `ep_cat2_704` | `cat2_repair` | `mlp_sink_attractor` | 6 | 0/0 | **0.9456** | Static sink excision: ID/OOD acc `0.0234/0.0156 -> 1.0`, `rel_edit=0.2720` |
| `ep_cat5_705` | `cat5_unlearn_excise` | `2tok_backdoor_excision` | 6 | 0/0 | **0.9666** | `clean/trig_keep=1.0`, `trojan_proj=0.0`, `recovery_attack=0.0288`, `rel_edit=0.1669` |
| `ep_l15_cat7_706` | `cat7_refactor` | `level1_5_grokked_svd_defrag` | 3 | 0/0 | **0.9860** | `max_diff=0.00013`, `frob_sq=0.0825x`, `L0=0.0573x` (`0/0` probes) |

### Phase 1 Local Preparation (Steps L1–L4) & Option 3 Corpus — COMPLETE (`13/13` Unit Tests Passing)
* **Why We Replaced the 525-Episode Template Corpus with 123 High-Entropy Episodes (Option 3):** Having `504` near-identical 2-turn scripted rollouts risks SFT template memorization (`2`-turn rigidity) and over-smoothing `mean_turns=2.0`, hurting GRPO exploration. Curating a compact, balanced **123-episode Option 3 corpus** teaches syntax, `wsp_tools` usage, 3-turn exploratory auditing, and Python exception recovery while leaving multi-turn algorithmic mastery to online GRPO.
* **Audited Option 3 Corpus Metrics ([trajectories/episodes.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/episodes.jsonl) & [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl)):**
  * `total_read`: **123** (`21` organic Cycle 1–3 with refreshed symptom-only `brief.md` & unified `SYSTEM_PROMPT` + `68` 2-turn blinded across `1000..1067` + `34` 3-turn exploratory across `2000..2033`)
  * `exported`: **123** (`skipped`: **0**, `zero_probe_count`: **94**, `multi_turn_3plus_count`: **46**, `error_recovery_count`: **20**, `mean_reward`: **0.959028**, `mean_turns`: **2.537**)
  * Per-subtype coverage: **7–8 verified episodes** for all **17 subtypes** (`cat1`: `21`, `cat2`: `22`, `cat3`: `15`, `cat4`: `14`, `cat5`: `15`, `cat6`: `21`, `cat7`: `15`).
  * Turn-mask verification (`train_sft.py --dry-run`): **58.98%** assistant token ratio (`1466.89` mean assistant tokens / `2487.32` mean total tokens per trajectory), with all `system` and `user` sandbox observations masked to `-100`.
  * Held-out evaluation & GRPO CPU verification: `eval.py --backend scripted --episodes-per-subtype 2` passes **`34/34` (`100%`)** (`zero_probe_pass_rate=1.0`, `mean_base_reward=0.960838`), and `train_grpo.py --dry-run` verifies positive advantage variance (`std=0.491351`), Schulman KL penalty, and gradient flow across **8** multi-turn rollouts.

### Crucial Technical & Mathematical Lessons Learned (Preserve for Level 1.5–3)
1. **Float32 Orthogonal Rounding Noise in Dead-Code Detection:** When `W_U` has high-norm columns (e.g., quadratic readout scales), `Q_res @ Q_res.T - I` float32 rounding (`~3e-7`) produces `~2e-3 > 1e-4` residual magnitudes in `W_out @ W_U` for true null-space neurons. `_active_projection_mask` checks both absolute row norms and column-normalized cosine similarity so gauge-rotated dead code is never misclassified as active.
2. **RoPE Pair Structure & Gauge Freedom:** Unlike learned `W_pos`, RoPE rotates paired channels `(c, c + d_head // 2)` across sequence positions, so arbitrary $O(d_{\text{head}})$ $R_{qk}$ orthogonal transformations do not preserve RoPE inner products. Both `scramble_gauge` and `canonical_pack` omit $R_{qk}$ mixing when `pos_type == "rope"` (`"W_pos" not in weights`), and `head_offset_profile` uses `rope_rotation_matrix(d_head, delta)` against RMS-normalized mean token embeddings (`e_bar`).
3. **RMSNorm Scaling, Null-Space Drift & Non-Uniform Gain Absorption:**
   * Before SVD compression in `canonical_pack`, non-uniform `norm1.weight`, `norm2.weight`, and `norm_final.weight` must be folded into `W_Q/W_K/W_V`, `W_in/W_gate`, and `W_U` (`_absorb_rmsnorm_gains`); leaving them separate changes per-coordinate scaling under residual basis rotations $V_{\text{basis}}$.
   * Under `RMSNorm` ($\text{inv\_rms} = \sqrt{d / \|x\|_2^2}$), attention outputs copied by $W_V W_O$ scale by $\text{inv\_rms}(x_{\text{src}})$. Both compiled and null-space-injected RoPE heads must match $\text{vo\_scale} = 1 / \text{inv\_rms}$ and balance operator vs. payload embedding norms (`W_E[22]` matched via a dedicated orthogonal dim) so RoPE keys/queries and downstream `SwiGLU` thresholds remain calibrated across tokens and tasks.
   * Because `RMSNorm` normalizes across all `d_model` dimensions, large-norm null-space decoy outputs inflate `post_mlp` RMS inconsistently across inputs and corrupt downstream quadratic readout scales (`norm_final`); keep null-space decoys bounded (e.g., `std <= 0.05` on `W_out`) so $\text{rms}(\text{resid})$ stays invariant.
   * For `cat4_nullspace_inject` under non-uniform `RMSNorm`, existing tasks must gate both `W_gate` and `W_in` on their specific operator dims so unrelated suffix tokens cannot induce partial Layer-1 activation in the occupied residual subspace.
4. **Cat 5 Level 1.5 Coprime & Readout-Margin Calibration (`generate_level1_5_cat5_unlearn_excise`):** When testing excisions against a 10-step `(W_U, b_U)` gradient recovery attack on a mixed clean/triggered batch, sampling `a0` coprime to `m_mod=20` prevents clean-label collapse, and using a high clean `W_out`/`W_U` readout margin (`12.0` / `16.0`) relative to the constant `carrier=8.0` dimension prevents `Adam` from exploiting constant RMSNorm features to overpredict `c_bad` even on clean oracle weights.
5. **Cat 7 Level 1.5 Cancelling SwiGLU Carrier Cancellation Precision (`generate_level1_5_cat7_refactor`):** When sampling cancelling `(j, j+1)` SwiGLU decoy pairs under a strong constant `carrier=8.0` (`dim 3`), allowing large random weights on the carrier dimension creates `~8000` hidden activations (`~64,000` `W_out` contributions) whose IEEE-754 float32 accumulation error after `scramble_gauge` neuron rescaling exceeds the `1e-3` equivalence tolerance on some seeds (`1186`, `1203`). Zeroing `wg[3]` and `wi[3]` on the cancelling decoy pairs preserves full decoy norms across the remaining dims while keeping `max_abs_logit_diff < 1e-3`.

### Phase 1 Cloud GPU Run 1 Results (`Steps C1–C4` on `1x H100 80GB`) — COMPLETE
Executed end-to-end training and held-out 17-subtype evaluation (`34` episodes, `seeds >= 9000`, `20` strictly zero-probe `B=0`) on `Qwen/Qwen2.5-Coder-7B-Instruct`:

| Metric / Category | C1: Base (`eval_base.json`) | C2: LoRA SFT (`eval_sft.json`) | C4: 80-Step GRPO (`eval_grpo.json`) |
| :--- | :---: | :---: | :---: |
| **Overall Pass Rate (`34` eps)** | `0.00%` (`0/34`) | **`44.12%` (`15/34`)** | `32.35%` (`11/34`) |
| **Zero-Probe `B=0` Pass Rate (`20` eps)** | `0.00%` (`0/20`) | **`45.00%` (`9/20`)** | `25.00%` (`5/20`) |
| **Mean Verifier Base Reward** | `0.000086` | **`0.408786`** | `0.173251` |
| **Mean Shaped Reward** | `0.017292` | **`0.466482`** | `0.197604` |
| **Mean Turns** | `5.5294` | `4.0` | `4.6765` |
| **Format Compliance (`mean_format_rate`)** | `34.12%` | **`98.04%`** | `92.65%` |
| **Clean Exec Rate (`mean_clean_exec_rate`)** | `29.71%` | **`58.33%`** | `34.12%` |
| **Explicit Submit Rate (`explicit_submit_rate`)** | `20.59%` | **`52.94%`** | `35.29%` |
| **`cat1_decompile` (`6` eps)** | `0.0%` | **`66.7%` (`4/6`)** | `0.0%` |
| **`cat2_repair` (`6` eps)** | `0.0%` | **`83.3%` (`5/6`)** | **`83.3%` (`5/6`)** |
| **`cat3_synthesize` / `cat4_nullspace_inject`** | `0.0%` / `0.0%` | `0.0%` / `0.0%` | `0.0%` / `0.0%` |
| **`cat5_unlearn_excise` (`4` eps)** | `0.0%` | `50.0%` (`2/4` L1.5) | `50.0%` (`2/4` L1 `2tok`) |
| **`cat6_splice_merge` (`6` eps)** | `0.0%` (`R=0.115`) | `0.0%` (`R=0.128`) | `0.0%` (`R=0.147`) |
| **`cat7_refactor` (`4` eps)** | `0.0%` (`R=0.800`) | **`100.0%` (`4/4`)** | **`100.0%` (`4/4`)** |

#### Complete 17-Subtype Breakdown (`C1 Base` vs. `C2 LoRA SFT` vs. `C3 GRPO Steps 0–30 Peak` vs. `C4 Step-80 GRPO`)
Held-out values report `pass_rate (mean_reward)` over 2 episodes per subtype; `C3` reports peak Stage-0 online rollout pass rates (`6` rollouts/step) from [trajectories/grpo_train_log.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/grpo_train_log.jsonl) (`—` indicates subtypes not sampled in Stage 0 or absent during Steps `0–30`):

| Category | Subtype | C1: Base (`eval_base.json`) | C2: LoRA SFT (`eval_sft.json`) | C3: GRPO Steps 0–30 Peak (`grpo_train_log.jsonl`) | C4: Step-80 GRPO (`eval_grpo.json`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `cat1_decompile` | `modular_linear_with_3tok_trojan` | `0.0%` (`-1.0`) | **`100.0%` (`1.0`)** | `66.7%` (Steps 18, 23) | `0.0%` (`-1.0`) |
| `cat1_decompile` | `modular_varied_lag_with_3tok_trojan` | `0.0%` (`-1.0`) | **`100.0%` (`1.0`)** | `—` (Stage-0 lock-in) | `0.0%` (`-1.0`) |
| `cat1_decompile` | `level1_5_grokked_rope_decompile` | `0.0%` (`-0.25`) | `0.0%` (`-1.0`) | `—` (Stage-0 lock-in) | `0.0%` (`-1.0`) |
| `cat2_repair` | `mlp_sink_attractor` | `0.0%` (`0.0`) | **`100.0%` (`0.945608`)** | **`100.0%`** (Steps 5–7, 19–20, 25) | **`100.0%` (`0.945608`)** |
| `cat2_repair` | `norm_matched_k_composition_corruption` | `0.0%` (`0.0`) | `50.0%` (`0.498984`) | `—` (Stage-0 lock-in) | `50.0%` (`0.498984`) |
| `cat2_repair` | `level1_5_grokked_swiglu_rope_repair` | `0.0%` (`0.0`) | **`100.0%` (`0.987535`)** | `—` (Stage-0 lock-in) | **`100.0%` (`0.987535`)** |
| `cat3_synthesize` | `dual_offset_modular_shift` | `0.0%` (`0.035547`) | `0.0%` (`0.035547`) | `0.0%` (Step 4) | `0.0%` (`0.035547`) |
| `cat3_synthesize` | `level1_5_swiglu_rope_synthesize` | `0.0%` (`0.02871`) | `0.0%` (`0.02871`) | `—` (Stage-0 lock-in) | `0.0%` (`0.02871`) |
| `cat4_nullspace_inject` | `orthogonal_subspace_skill_addition` | `0.0%` (`0.024609`) | `0.0%` (`0.024609`) | `0.0%` (Steps 15, 30) | `0.0%` (`0.024609`) |
| `cat4_nullspace_inject` | `level1_5_grokked_subspace_inject` | `0.0%` (`0.032812`) | `0.0%` (`0.032812`) | `—` (Stage-0 lock-in) | `0.0%` (`0.032812`) |
| `cat5_unlearn_excise` | `2tok_backdoor_excision` | `0.0%` (`0.093652`) | `0.0%` (`0.093652`) | `83.3%` (Step 29) | **`100.0%` (`0.916082`)** |
| `cat5_unlearn_excise` | `level1_5_grokked_backdoor_excision` | `0.0%` (`0.092285`) | **`100.0%` (`0.975542`)** | `—` (Stage-0 lock-in) | `0.0%` (`0.092285`) |
| `cat6_splice_merge` | `orthogonal_subspace_model_merge` | `0.0%` (`0.140821`) | `0.0%` (`0.175`) | **`100.0%`** (Steps 8, 10, 11) | `0.0%` (`0.236523`) |
| `cat6_splice_merge` | `vertical_function_composition` | `0.0%` (`0.00752`) | `0.0%` (`0.015039`) | `—` (Stage-0 lock-in) | `0.0%` (`0.00752`) |
| `cat6_splice_merge` | `level1_5_grokked_rope_merge` | `0.0%` (`0.195508`) | `0.0%` (`0.192773`) | `—` (Stage-0 lock-in) | `0.0%` (`0.195508`) |
| `cat7_refactor` | `nullspace_and_cancelling_head_defrag` | `0.0%` (`0.8`) | **`100.0%` (`0.957915`)** | Not in 0–30 (Steps 36, 39: `100.0%`) | **`100.0%` (`0.957915`)** |
| `cat7_refactor` | `level1_5_grokked_svd_defrag` | `0.0%` (`0.8`) | **`100.0%` (`0.98563`)** | `—` (Stage-0 lock-in) | **`100.0%` (`0.98563`)** |

#### Three Root Causes Discovered During Cloud Run 1
1. **`C2` Evaluation & Turn-2 Token Truncation (`--max-new-tokens 1024`):** [trajectories/eval_sft.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_sft.json) ran under the previous `1024`-token cap (`train_grpo.py` ran at `1280`). Inspecting [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl) revealed 7 subtypes with Turn-2 assistant scripts over `3200` chars (`~1000+` tokens): `level1_5_grokked_rope_merge` (`5884` chars / `~1783` toks), `dual_offset_modular_shift` (`5790` / `~1754` toks), `level1_5_swiglu_rope_synthesize` (`5188` / `~1572` toks), `level1_5_grokked_subspace_inject` (`4934` / `~1495` toks), `vertical_function_composition` (`4674` / `~1416` toks), `modular_varied_lag_with_3tok_trojan` (`4052` / `~1227` toks), and `nullspace_and_cancelling_head_defrag` (`3304` / `~1001` toks). This directly explains why `C2` achieved `98.04%` format compliance yet only `52.94%` explicit submission and `0%` on `cat3`/`cat4`/`cat6` synthesis/merge tasks. ([eval.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/eval.py) now defaults to `--max-new-tokens 1536`.)
2. **SFT Reference KL Anchor Validated & Peak RLVR Acquisition (`Steps 0–30`, `checkpoints/grpo_lora/step_0020`):** Snapshotting frozen SFT LoRA parameters (`_snapshot_trainable_params` / `_swap_trainable_params` in [train_grpo.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_grpo.py)) anchored `kl_div` to **`0.00043`** at Step 0 and **`0.002–0.015`** through Step 30 (vs. `1.16–4.05` when erroneously disabling the SFT adapter). With a stable anchor, `orthogonal_subspace_model_merge` climbed from `0% -> 100%` (Steps 8, 10, 11), `2tok_backdoor_excision` climbed from `0% -> 83.3%` in training (`100.0%` held-out in [trajectories/eval_grpo.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_grpo.json)), and Steps `18–29` achieved a **`72.2%`** (`0.722233`) rolling pass rate across `72` online rollouts.
3. **Late-Run All-Fail Micro-Shaping Drift (`Steps 49–79`) & Stage-0 Curriculum Lock-In:** On unsolved tasks where every rollout in a group scored identical base verifier reward (`cat3`/`cat4` or all `-1.0` on `cat1`), normalizing tiny `0.02–0.05` micro-shaping bonuses created full $\pm 1$ advantages, driving `kl_div` up to `0.387683` (Step 62) and causing `cat1_decompile` to regress (`66.7% -> 0%`) by `step_0080`. Simultaneously, requiring `85%` rolling pass rate across all 7 categories kept `curriculum_stage=0` for all 80 steps, so specializing on Level 1 `2tok_backdoor_excision` (`0% -> 100%`) overwrote `level1_5_grokked_backdoor_excision` (`100% -> 0%`). Added an all-fail guard in [train_grpo.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/train_grpo.py) (`max(base_rewards) - min(base_rewards) < 1e-4` skips the update when `step_idx >= 10`).

---

## 6. Where We're Going: Immediate Next Steps & Long-Term Horizon

### Immediate Actions for the Next Session (Phase 1 Follow-Up Before Phase 2/3)
Phase 1 Cloud Run 1 (`C1–C4`) logs and adapters are saved locally ([checkpoints/sft_lora](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/checkpoints/sft_lora), [checkpoints/grpo_lora](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/checkpoints/grpo_lora), [trajectories/eval_base.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_base.json), [trajectories/eval_sft.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_sft.json), [trajectories/eval_grpo.json](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/eval_grpo.json), [trajectories/grpo_train_log.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/grpo_train_log.jsonl)):

1. **Re-Evaluate Saved Checkpoints at `--max-new-tokens 2048`:** Run [eval.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/eval.py) with `--max-new-tokens 2048` on [checkpoints/sft_lora](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/checkpoints/sft_lora), `checkpoints/grpo_lora/step_0020`, and `checkpoints/grpo_lora/step_0040` to separate true capability from Turn-2 truncation on `cat3`/`cat4`/`cat6`/`cat1` and quantify peak pre-drift GRPO performance near the Steps `18–29` (`72.2%`) window.
2. **Add Mixed Level 1 + Level 1.5 Sampling in `CurriculumScheduler` Stage 0:** Update [wsp_gym/training.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/training.py) so Stage 0 mixes Level 1, Cycle 2, and Level 1.5 subtypes (while maintaining $B \to 0$ probe annealing), eliminating the `85%` Stage-0 lock-in that starved `level1_5_grokked_backdoor_excision` and `cat1`/`cat6` generalization.
3. **Compact Verbose Turn-2 Templates in [wsp_gym/scaler.py](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/wsp_gym/scaler.py):** Trim the 7 longest Turn-2 oracle templates (`dual_offset_modular_shift`, `level1_5_grokked_rope_merge`, `level1_5_swiglu_rope_synthesize`, `level1_5_grokked_subspace_inject`, `vertical_function_composition`, `modular_varied_lag_with_3tok_trojan`, `nullspace_and_cancelling_head_defrag`) well below `~1000` tokens using concise helper idioms, and re-export [trajectories/sft_train.jsonl](file:///c:/Users/dljan/VSCodeProjects/Weight%20Space%20Programming/trajectories/sft_train.jsonl) (+ refresh SFT if needed).
4. **Run Focused `30–40`-Step Guarded GRPO Before Phase 2/3:** Execute a short online GRPO follow-up on cloud GPU with the all-fail micro-shaping guard (`step_idx >= 10`), mixed Stage-0 sampling, and adequate token headroom, then proceed to Phase 2 and Phase 3 once 17-subtype transfer holds cleanly.

### Long-Term Research Milestones (Phases 1–3)
* **Phase 1 (LoRA SFT + GRPO on 4B–8B Local Architect — Cloud Run 1 Complete; Follow-Up Tuning Next):**
  * Cold-start LoRA SFT on the 123-episode Option 3 turn-masked dataset (`44.12%` pass rate at `1024` tokens, `45.0%` zero-probe), followed by guarded online multi-turn GRPO over `wsp_gym.verifier.verify_submission` while annealing `forward_budget` toward `0`.
* **Phase 2 (Level 2 Scale Transfer):**
  * Benchmark whether scale-invariant `wsp_tools` workflows (SVD subspace localization, RoPE offset profiling, RMSNorm absorption, null-space projection checks) transfer to $100\text{M}\text{--}1\text{B}$ transformers (`Qwen`/`Llama`-style blocks) with minimal or zero retraining.
* **Phase 3 (Level 3 Immutable-Twin Self-Surgery):**
  * Instantiate an Immutable-Twin harness where trained agent $A_t$ diagnoses and repairs/excises/injects capabilities into cloned weights $A'_{t+1}$ and promotes only checkpoints passing held-out capability, KL-retention, and recovery-attack gates.
