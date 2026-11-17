"""Stateful multi-turn Weight-Space Programming sandbox and trajectory logger."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
from typing import Any

import numpy as np
import scipy
import torch

from wsp_gym import toolkit as wsp_tools
from wsp_gym.generators import CATEGORIES, ChallengeBundle, generate_challenge
from wsp_gym.model import ModelConfig, forward_fast
from wsp_gym.verifier import verify_bundle, verify_submission


STDOUT_MAX_CHARS = 2000
DEFAULT_MAX_TURNS = 6
MAX_PROBE_BATCH = 32
DEFAULT_TURN_TIMEOUT_SEC = 5.0

SYSTEM_PROMPT = (
    "You are a Weight-Space Programming agent operating inside a stateful Python sandbox. "
    "Analyze and surgically modify neural network weights using linear algebra (PyTorch, NumPy, SciPy) "
    "within your forward-pass and turn budgets.\n\n"
    "Pre-loaded variables & helpers:\n"
    "- `config`: ModelConfig dict; `brief_md`: task brief string; `weights`: editable state_dict; `orig_weights`: read-only clone; `weights_B` (Cat 6 only).\n"
    "- `probe_forward(input_ids, custom_weights=None, return_cache=True)`: consumes 1 forward batch (max batch=32). "
    "Returns `(logits, cache)` when `return_cache=True`, or `logits` tensor alone when `return_cache=False`.\n"
    "- `submit_weights(candidate_weights)`: queues `candidate_weights` dict for verification and ends the episode.\n"
    "- `submit_decompiled(result_dict)`: queues decompiled specification dict (Cat 1) and ends the episode.\n"
    "- `wsp_tools`: pre-loaded toolkit with `residual_svd_basis(weights, config=None, tol=1e-4, include_embeddings=True)`, "
    "`head_offset_profile(weights, layer=0, query_pos=6, max_offset=8)`, `rope_rotation_matrix(d_head, offset)`, "
    "`virtual_ov_circuit(weights, layer=0, head=None)`, `mlp_logit_impact(weights, layer=0, tol=1e-4)`, "
    "`canonical_pack(weights, ...)`, and `layer_anomaly_scan(weights, config=None, top_k=3)`.\n"
    "- GQA / deep-model notes: `W_K`/`W_V` have shape `(n_kv_heads, d_model, d_head)` when `n_kv_heads < n_heads` "
    "(group size `n_heads // n_kv_heads`), optional `b_Q`/`b_K`/`b_V` may exist, and `layer_anomaly_scan` inspects all layers compactly."
)


def save_challenge(bundle: ChallengeBundle, challenge_dir: str | Path) -> Path:
    """Persist a ChallengeBundle with strict separation between public/ and .vault/."""
    root = Path(challenge_dir)
    pub = root / "public"
    vault = root / ".vault"
    pub.mkdir(parents=True, exist_ok=True)
    vault.mkdir(parents=True, exist_ok=True)

    (pub / "brief.md").write_text(bundle.brief_md, encoding="utf-8")
    (pub / "config.json").write_text(json.dumps(bundle.config.to_dict(), indent=2), encoding="utf-8")
    (pub / "metadata.json").write_text(
        json.dumps(
            {
                "episode_id": bundle.episode_id,
                "category": bundle.category,
                "forward_budget": bundle.forward_budget,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(bundle.public_weights, pub / "weights.pt")
    if bundle.extra_public_weights:
        for name, tensor_dict in bundle.extra_public_weights.items():
            torch.save(tensor_dict, pub / f"{name}.pt")

    secret_payload = dict(bundle.secret)
    secret_payload.setdefault("subtype", bundle.subtype)
    secret_payload.setdefault("difficulty", bundle.difficulty)
    torch.save(secret_payload, vault / "verifier_secret.pt")
    return root


LEVEL1_5_SUBTYPES = {
    "cat1_decompile": "level1_5_grokked_rope_decompile",
    "cat2_repair": "level1_5_grokked_swiglu_rope_repair",
    "cat3_synthesize": "level1_5_swiglu_rope_synthesize",
    "cat4_nullspace_inject": "level1_5_grokked_subspace_inject",
    "cat5_unlearn_excise": "level1_5_grokked_backdoor_excision",
    "cat6_splice_merge": "level1_5_grokked_rope_merge",
    "cat7_refactor": "level1_5_grokked_svd_defrag",
}


def generate_suite(
    out_dir: str | Path,
    seed: int = 42,
    difficulty: float = 1.0,
) -> list[Path]:
    """Generate one blinded challenge for each of the 7 categories."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for idx, cat in enumerate(CATEGORIES, start=1):
        bundle = generate_challenge(cat, seed=seed + idx * 100, difficulty=difficulty)
        ep_dir = base / f"ep_{idx:02d}_{cat}"
        save_challenge(bundle, ep_dir)
        dirs.append(ep_dir)
    return dirs


def generate_level1_5_suite(
    out_dir: str | Path,
    seed: int = 42,
    difficulty: float = 1.5,
) -> list[Path]:
    """Generate blinded Level 1.5 (RMSNorm + SwiGLU + RoPE + AdamW) challenges."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for idx, (cat, subtype) in enumerate(LEVEL1_5_SUBTYPES.items(), start=1):
        bundle = generate_challenge(
            cat, seed=seed + idx * 100, difficulty=difficulty, subtype=subtype
        )
        ep_dir = base / f"ep_{idx:02d}_{cat}_l15"
        save_challenge(bundle, ep_dir)
        dirs.append(ep_dir)
    return dirs


def _truncate_output(text: str, limit: int = STDOUT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    kept = text[: limit - 64]
    return f"{kept}\n... [TRUNCATED {len(text) - len(kept)} chars; summarize tensors instead of printing raw arrays]"


def _is_picklable(val: Any) -> bool:
    if isinstance(val, type(torch)):
        return False
    if callable(val) and not isinstance(val, torch.Tensor):
        return False
    try:
        pickle.dumps(val)
        return True
    except Exception:
        return False


class SandboxSession:
    """Stateful multi-turn sandbox (disk-persisted or lightweight in-memory via `from_bundle`)."""

    def __init__(
        self,
        challenge_dir: str | Path,
        max_turns: int = DEFAULT_MAX_TURNS,
        trajectory_path: str | Path | None = None,
        timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> None:
        self._persist_disk = True
        self.bundle: ChallengeBundle | None = None
        self.challenge_dir = Path(challenge_dir)
        self.pub_dir = self.challenge_dir / "public"
        self.state_path = self.pub_dir / "session_state.pt"
        self.max_turns = max_turns
        self.timeout_sec = float(timeout_sec)
        if trajectory_path is None:
            self.trajectory_path: Path | None = Path("trajectories") / "episodes.jsonl"
        else:
            self.trajectory_path = Path(trajectory_path)

        self.meta = json.loads((self.pub_dir / "metadata.json").read_text(encoding="utf-8"))
        self.config = ModelConfig.from_dict(
            json.loads((self.pub_dir / "config.json").read_text(encoding="utf-8"))
        )
        self.brief_md = (self.pub_dir / "brief.md").read_text(encoding="utf-8")

        if self.state_path.exists():
            self.state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        else:
            orig_w = torch.load(self.pub_dir / "weights.pt", map_location="cpu", weights_only=False)
            extra = {}
            wb_path = self.pub_dir / "weights_B.pt"
            if wb_path.exists():
                extra["weights_B"] = torch.load(wb_path, map_location="cpu", weights_only=False)
            self.state = {
                "turn": 0,
                "forward_budget": int(self.meta["forward_budget"]),
                "forward_budget_remaining": int(self.meta["forward_budget"]),
                "forward_passes_used": 0,
                "weights": {k: v.clone() for k, v in orig_w.items()},
                "orig_weights": {k: v.clone() for k, v in orig_w.items()},
                "extra_weights": extra,
                "user_vars": {},
                "submitted": False,
                "submission_data": None,
                "verification": None,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self.brief_md},
                ],
            }
            self._save_state()

    @classmethod
    def from_bundle(
        cls,
        bundle: ChallengeBundle,
        max_turns: int = DEFAULT_MAX_TURNS,
        trajectory_path: str | Path | None = None,
        timeout_sec: float = 30.0,
    ) -> "SandboxSession":
        """Create an in-memory SandboxSession without disk serialization overhead."""
        obj = cls.__new__(cls)
        obj._persist_disk = False
        obj.bundle = bundle
        obj.challenge_dir = None  # type: ignore[assignment]
        obj.pub_dir = None  # type: ignore[assignment]
        obj.state_path = None  # type: ignore[assignment]
        obj.max_turns = max_turns
        obj.timeout_sec = float(timeout_sec)
        obj.trajectory_path = Path(trajectory_path) if trajectory_path is not None else None
        obj.meta = {
            "episode_id": bundle.episode_id,
            "category": bundle.category,
            "forward_budget": bundle.forward_budget,
        }
        obj.config = bundle.config
        obj.brief_md = bundle.brief_md
        extra = {
            name: {k: v.detach().clone() for k, v in w_dict.items()}
            for name, w_dict in (bundle.extra_public_weights or {}).items()
        }
        obj.state = {
            "turn": 0,
            "forward_budget": int(bundle.forward_budget),
            "forward_budget_remaining": int(bundle.forward_budget),
            "forward_passes_used": 0,
            "weights": {k: v.detach().clone() for k, v in bundle.public_weights.items()},
            "orig_weights": {k: v.detach().clone() for k, v in bundle.public_weights.items()},
            "extra_weights": extra,
            "user_vars": {},
            "submitted": False,
            "submission_data": None,
            "verification": None,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obj.brief_md},
            ],
        }
        return obj

    def _save_state(self) -> None:
        if self._persist_disk and self.state_path is not None:
            torch.save(self.state, self.state_path)

    def _append_trajectory(self) -> None:
        if self.trajectory_path is None:
            return
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bundle is not None:
            subtype = self.bundle.subtype
            difficulty = self.bundle.difficulty
        else:
            vault_path = self.challenge_dir / ".vault" / "verifier_secret.pt"
            vault_sec = (
                torch.load(vault_path, map_location="cpu", weights_only=False)
                if vault_path.exists()
                else {}
            )
            subtype = vault_sec.get("subtype", self.meta.get("subtype", "unknown"))
            difficulty = vault_sec.get("difficulty", self.meta.get("difficulty", 1.0))
        record = {
            "episode_id": self.meta["episode_id"],
            "category": self.meta["category"],
            "subtype": subtype,
            "difficulty": difficulty,
            "model_config": self.config.to_dict(),
            "messages": self.state["messages"],
            "verification": self.state["verification"],
        }
        with self.trajectory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def step(
        self,
        think: str,
        code: str,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Execute one turn of reasoning + Python weight-surgery code."""
        if self.state["submitted"]:
            return {
                "already_finished": True,
                "turn": self.state["turn"],
                "observation": "Episode already submitted and verified.",
                "verification": self.state["verification"],
            }

        self.state["turn"] += 1
        turn_num = self.state["turn"]

        # Track submission inside this turn
        turn_submission: dict[str, Any] = {}

        def probe_forward(
            input_ids: torch.Tensor,
            custom_weights: dict[str, torch.Tensor] | None = None,
            return_cache: bool = True,
        ):
            if self.state["forward_budget_remaining"] <= 0:
                raise RuntimeError("Forward pass budget exhausted (0 remaining).")
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 2 and input_ids.shape[0] > MAX_PROBE_BATCH:
                raise RuntimeError(
                    f"Probe batch size {input_ids.shape[0]} exceeds max allowed ({MAX_PROBE_BATCH})."
                )
            self.state["forward_budget_remaining"] -= 1
            self.state["forward_passes_used"] += 1
            target_w = self.state["weights"] if custom_weights is None else custom_weights
            with torch.no_grad():
                return forward_fast(input_ids, target_w, self.config, return_cache=return_cache)

        def submit_weights(candidate_weights: dict[str, torch.Tensor]) -> None:
            turn_submission["type"] = "weights"
            turn_submission["weights"] = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in candidate_weights.items()
            }
            print("[Sandbox] Candidate weights queued for verification.")

        def submit_decompiled(result_dict: dict[str, Any]) -> None:
            turn_submission["type"] = "decompiled"
            turn_submission["decompiled"] = result_dict
            print("[Sandbox] Decompiled specification queued for verification.")

        if self._persist_disk:
            orig_ref = {k: v.clone() for k, v in self.state["orig_weights"].items()}
        else:
            orig_ref = self.state["orig_weights"]

        env: dict[str, Any] = {
            "torch": torch,
            "np": np,
            "scipy": scipy,
            "wsp_tools": wsp_tools,
            "config": self.config.to_dict(),
            "brief_md": self.brief_md,
            "weights": self.state["weights"],
            "orig_weights": orig_ref,
            "forward_budget": self.state["forward_budget"],
            "forward_budget_remaining": self.state["forward_budget_remaining"],
            "probe_forward": probe_forward,
            "submit_weights": submit_weights,
            "submit_decompiled": submit_decompiled,
        }
        for extra_name, extra_w in self.state["extra_weights"].items():
            env[extra_name] = {k: v.clone() for k, v in extra_w.items()} if self._persist_disk else extra_w
        env.update(self.state["user_vars"])

        builtin_keys = set(env.keys()) - set(self.state["user_vars"].keys())
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exec_error = False
        timed_out = False
        limit_sec = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        deadline = time.perf_counter() + limit_sec if limit_sec > 0 else None

        def _trace_timeout(frame: Any, event: str, arg: Any) -> Any:
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError(f"Sandbox execution exceeded {limit_sec:.2f}s turn timeout.")
            return _trace_timeout

        prev_trace = sys.gettrace()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                compiled = compile(code, "<sandbox>", "exec")
                if deadline is not None:
                    sys.settrace(_trace_timeout)
                exec(compiled, env, env)
        except TimeoutError:
            exec_error = True
            timed_out = True
            traceback.print_exc(file=stderr_buf)
        except (Exception, SystemExit):
            exec_error = True
            traceback.print_exc(file=stderr_buf)
        finally:
            if deadline is not None:
                sys.settrace(prev_trace)

        # Persist updated weights & picklable user variables
        if isinstance(env.get("weights"), dict):
            self.state["weights"] = env["weights"]

        new_user_vars = {}
        for k, v in env.items():
            if k.startswith("__") or k in builtin_keys:
                continue
            if _is_picklable(v):
                new_user_vars[k] = v
        self.state["user_vars"] = new_user_vars

        raw_out = stdout_buf.getvalue()
        raw_err = stderr_buf.getvalue()
        combined = raw_out
        if raw_err:
            combined = f"{combined}\nSTDERR:\n{raw_err}" if combined else f"STDERR:\n{raw_err}"
        combined = _truncate_output(combined.strip() or "(no output)")

        assistant_msg = f"<think>\n{think.strip()}\n</think>\n```python\n{code.strip()}\n```"
        self.state["messages"].append({"role": "assistant", "content": assistant_msg})

        obs_footer = (
            f"[Sandbox Info: {self.state['forward_budget_remaining']}/{self.state['forward_budget']} "
            f"forward batches remaining | Turn {turn_num}/{self.max_turns}]"
        )
        observation = f"STDOUT:\n{combined}\n{obs_footer}"
        self.state["messages"].append({"role": "user", "content": observation})

        explicit_submit = bool(turn_submission)
        verification = None
        should_finalize = explicit_submit or turn_num >= self.max_turns
        if should_finalize:
            if not turn_submission:
                turn_submission = {"weights": self.state["weights"]}
            sandbox_stats = {
                "forward_budget": self.state["forward_budget"],
                "forward_passes_used": self.state["forward_passes_used"],
                "turns_used": turn_num,
            }
            if self.bundle is not None and not self._persist_disk:
                verification = verify_bundle(self.bundle, turn_submission, sandbox_stats)
            else:
                verification = verify_submission(self.challenge_dir, turn_submission, sandbox_stats)
            verification["metrics"]["turns_used"] = turn_num
            verification["metrics"]["explicit_submit"] = explicit_submit
            self.state["submitted"] = True
            self.state["submission_data"] = turn_submission
            self.state["verification"] = verification
            if self._persist_disk and self.pub_dir is not None:
                (self.pub_dir / "verification_result.json").write_text(
                    json.dumps(verification, indent=2), encoding="utf-8"
                )
            self._append_trajectory()

        self._save_state()
        return {
            "turn": turn_num,
            "exec_error": exec_error,
            "timed_out": timed_out,
            "explicit_submit": explicit_submit,
            "observation": observation,
            "submitted": self.state["submitted"],
            "verification": verification,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WSP Gym Sandbox CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-suite", help="Generate a blinded challenge suite")
    p_gen.add_argument("--out-dir", required=True, help="Output directory for suite")
    p_gen.add_argument("--seed", type=int, default=42, help="Base random seed")
    p_gen.add_argument("--difficulty", type=float, default=1.0, help="Difficulty scale")
    p_gen.add_argument("--level", choices=["1", "1.5"], default="1", help="Challenge level (1 or 1.5)")

    p_step = sub.add_parser("step", help="Execute one turn in a challenge sandbox")
    p_step.add_argument("--challenge-dir", required=True, help="Path to challenge folder")
    p_step.add_argument("--think", default="", help="Agent reasoning trace")
    p_step.add_argument("--code", default="", help="Python code to execute")
    p_step.add_argument("--code-file", default="", help="Path to Python file to execute")
    p_step.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p_step.add_argument("--trajectory-path", default="trajectories/episodes.jsonl")

    p_exp = sub.add_parser("export-sft", help="Audit and export verified trajectories to SFT chat JSONL")
    p_exp.add_argument("--trajectory-path", default="trajectories/episodes.jsonl")
    p_exp.add_argument("--out-path", default="trajectories/sft_train.jsonl")
    p_exp.add_argument("--min-reward", type=float, default=0.80)

    p_scale = sub.add_parser("scale-corpus", help="Generate and solve balanced blinded SFT episodes across all 17 subtypes")
    p_scale.add_argument("--num-episodes", type=int, default=504)
    p_scale.add_argument("--base-seed", type=int, default=1000)
    p_scale.add_argument("--trajectory-path", default="trajectories/episodes.jsonl")
    p_scale.add_argument("--out-path", default="trajectories/sft_train.jsonl")
    p_scale.add_argument(
        "--option3",
        action="store_true",
        help="Build the compact 123-episode Option 3 dataset (21 organic + 68 2-turn + 34 3-turn exploratory)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "generate-suite":
        if args.level == "1.5":
            diff = 1.5 if args.difficulty == 1.0 else args.difficulty
            dirs = generate_level1_5_suite(args.out_dir, seed=args.seed, difficulty=diff)
        else:
            dirs = generate_suite(args.out_dir, seed=args.seed, difficulty=args.difficulty)
        print(f"Generated {len(dirs)} challenges in {args.out_dir}:")
        for d in dirs:
            print(f"  - {d}")
        return 0

    if args.cmd == "step":
        code = args.code
        if args.code_file:
            code = Path(args.code_file).read_text(encoding="utf-8")
        session = SandboxSession(
            args.challenge_dir,
            max_turns=args.max_turns,
            trajectory_path=args.trajectory_path,
        )
        res = session.step(think=args.think, code=code)
        print(res["observation"])
        if res["verification"] is not None:
            print("\n=== VERIFICATION RESULT ===")
            print(json.dumps(res["verification"], indent=2))
        return 0

    if args.cmd == "export-sft":
        from wsp_gym.training import audit_and_export_sft_dataset

        summary = audit_and_export_sft_dataset(
            trajectory_path=args.trajectory_path,
            out_path=args.out_path,
            min_reward=args.min_reward,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "scale-corpus":
        if args.option3:
            from wsp_gym.scaler import build_option3_sft_corpus

            summary = build_option3_sft_corpus(
                trajectory_path=args.trajectory_path,
                out_path=args.out_path,
            )
        else:
            from wsp_gym.scaler import scale_sft_corpus

            summary = scale_sft_corpus(
                num_episodes=args.num_episodes,
                base_seed=args.base_seed,
                trajectory_path=args.trajectory_path,
                out_path=args.out_path,
            )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
