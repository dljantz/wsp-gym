"""SFT trajectory auditing/export and GRPO RLVR curriculum harness for wsp_gym."""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable

from wsp_gym.generators import CATEGORIES, ChallengeBundle, generate_challenge
from wsp_gym.sandbox import DEFAULT_TURN_TIMEOUT_SEC, LEVEL1_5_SUBTYPES, SandboxSession, save_challenge

STOP_SEQUENCES: tuple[str, ...] = ("\nSTDOUT:", "[Sandbox Info:")

CYCLE2_SUBTYPE_CHOICES: dict[str, tuple[str, ...]] = {
    "cat1_decompile": (
        "modular_linear_with_3tok_trojan",
        "modular_varied_lag_with_3tok_trojan",
        LEVEL1_5_SUBTYPES["cat1_decompile"],
    ),
    "cat2_repair": (
        "mlp_sink_attractor",
        "norm_matched_k_composition_corruption",
        LEVEL1_5_SUBTYPES["cat2_repair"],
    ),
    "cat3_synthesize": (
        "dual_offset_modular_shift",
        LEVEL1_5_SUBTYPES["cat3_synthesize"],
    ),
    "cat4_nullspace_inject": (
        "orthogonal_subspace_skill_addition",
        LEVEL1_5_SUBTYPES["cat4_nullspace_inject"],
    ),
    "cat5_unlearn_excise": (
        "2tok_backdoor_excision",
        LEVEL1_5_SUBTYPES["cat5_unlearn_excise"],
    ),
    "cat6_splice_merge": (
        "orthogonal_subspace_model_merge",
        "vertical_function_composition",
        LEVEL1_5_SUBTYPES["cat6_splice_merge"],
    ),
    "cat7_refactor": (
        "nullspace_and_cancelling_head_defrag",
        LEVEL1_5_SUBTYPES["cat7_refactor"],
    ),
}


def _has_valid_reasoning_and_code(messages: list[dict[str, Any]]) -> bool:
    assistant_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False
    return any("<think>" in text and "```python" in text for text in assistant_msgs)


def audit_and_export_sft_dataset(
    trajectory_path: str | Path = "trajectories/episodes.jsonl",
    out_path: str | Path | None = None,
    min_reward: float = 0.80,
    require_passed: bool = True,
    strip_trailing_user: bool = True,
) -> dict[str, Any]:
    """Audit verified episode logs and export deduplicated SFT chat-format JSONL records."""
    src = Path(trajectory_path)
    if not src.exists():
        raise FileNotFoundError(f"Trajectory file not found: {src}")

    total_read = 0
    deduped: dict[str, dict[str, Any]] = {}
    skipped = 0

    for raw_line in src.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        total_read += 1
        rec = json.loads(raw_line)
        ver = rec.get("verification") or {}
        passed = bool(ver.get("passed", False))
        reward = float(ver.get("total_reward", -1.0))
        messages = [dict(m) for m in (rec.get("messages") or [])]

        if strip_trailing_user:
            while messages and messages[-1].get("role") != "assistant":
                messages.pop()

        if require_passed and not passed:
            skipped += 1
            continue
        if reward < min_reward or not _has_valid_reasoning_and_code(messages):
            skipped += 1
            continue

        annotated_messages: list[dict[str, Any]] = []
        for m in messages:
            m_copy = dict(m)
            m_copy["loss_mask"] = bool(m_copy.get("role") == "assistant")
            annotated_messages.append(m_copy)

        loss_masks = [bool(m["loss_mask"]) for m in annotated_messages]
        has_error_recovery = any(
            m.get("role") == "user" and "STDERR:" in str(m.get("content", ""))
            for m in annotated_messages
        )

        ep_id = str(rec.get("episode_id", f"ep_{total_read}"))
        metrics = ver.get("metrics") or {}
        formatted = {
            "episode_id": ep_id,
            "category": rec.get("category", "unknown"),
            "subtype": rec.get("subtype", "unknown"),
            "difficulty": float(rec.get("difficulty", 1.0)),
            "reward": reward,
            "turns_used": int(
                metrics.get(
                    "turns_used",
                    sum(1 for m in annotated_messages if m.get("role") == "assistant"),
                )
            ),
            "forward_batches_used": int(metrics.get("forward_batches_used", 0)),
            "has_error_recovery": has_error_recovery,
            "model_config": rec.get("model_config", {}),
            "messages": annotated_messages,
            "loss_masks": loss_masks,
        }
        prev = deduped.get(ep_id)
        if prev is None or formatted["reward"] >= prev["reward"]:
            deduped[ep_id] = formatted

    records = list(deduped.values())
    if out_path is not None:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item) + "\n")

    n_exp = len(records)
    by_cat = dict(sorted(Counter(r["category"] for r in records).items()))
    by_sub = dict(sorted(Counter(r["subtype"] for r in records).items()))
    return {
        "total_read": total_read,
        "exported": n_exp,
        "skipped": total_read - n_exp if n_exp < total_read else skipped,
        "zero_probe_count": sum(1 for r in records if r["forward_batches_used"] == 0),
        "multi_turn_3plus_count": sum(1 for r in records if r["turns_used"] >= 3),
        "error_recovery_count": sum(1 for r in records if r["has_error_recovery"]),
        "mean_reward": round(sum(r["reward"] for r in records) / max(1, n_exp), 6) if n_exp else 0.0,
        "mean_turns": round(sum(r["turns_used"] for r in records) / max(1, n_exp), 3) if n_exp else 0.0,
        "by_category": by_cat,
        "by_subtype": by_sub,
        "out_path": str(out_path) if out_path is not None else None,
    }


class CurriculumScheduler:
    """Curriculum sampler with stratified 17-subtype passes and B -> 0 probe annealing."""

    def __init__(
        self,
        seed: int = 0,
        initial_stage: int = 0,
        window_size: int = 8,
        advance_pass_rate: float = 0.70,
        advance_reward: float = 0.65,
    ) -> None:
        self.rng = random.Random(seed)
        self.stage = int(max(0, min(3, initial_stage)))
        self.window_size = int(max(1, window_size))
        self.advance_pass_rate = float(advance_pass_rate)
        self.advance_reward = float(advance_reward)
        self._history: deque[tuple[bool, float]] = deque(maxlen=self.window_size)
        self._subtype_deck: list[tuple[str, str]] = []
        self._counter = 0

    def record_outcome(self, passed: bool, total_reward: float) -> bool:
        """Record an episode result; returns True if the scheduler advances to a harder stage."""
        self._history.append((bool(passed), float(total_reward)))
        if self.stage >= 3 or len(self._history) < self.window_size:
            return False
        pass_rate = sum(1 for p, _ in self._history if p) / len(self._history)
        mean_rew = sum(r for _, r in self._history) / len(self._history)
        if pass_rate >= self.advance_pass_rate and mean_rew >= self.advance_reward:
            self.stage += 1
            self._history.clear()
            return True
        return False

    def _next_stratified_pair(self) -> tuple[str, str]:
        if not self._subtype_deck:
            self._subtype_deck = [
                (cat, sub)
                for cat in CATEGORIES
                for sub in CYCLE2_SUBTYPE_CHOICES[cat]
            ]
            self.rng.shuffle(self._subtype_deck)
        return self._subtype_deck.pop()

    def _pick_subtype(self, cat: str) -> str:
        return self.rng.choice(CYCLE2_SUBTYPE_CHOICES[cat])

    def sample_challenge(
        self,
        seed: int | None = None,
        category: str | None = None,
        subtype: str | None = None,
    ) -> ChallengeBundle:
        """Sample a ChallengeBundle tuned to the current curriculum stage and probe budget."""
        ep_seed = self._counter if seed is None else int(seed)
        self._counter += 1
        if category is None and subtype is None:
            cat, chosen_subtype = self._next_stratified_pair()
        else:
            cat = self.rng.choice(CATEGORIES) if category is None else category
            chosen_subtype = self._pick_subtype(cat) if subtype is None else subtype

        hard_only = (
            "level1_5" in chosen_subtype
            or chosen_subtype
            in (
                "modular_varied_lag_with_3tok_trojan",
                "norm_matched_k_composition_corruption",
                "dual_offset_modular_shift",
            )
        )
        difficulty = 1.5 if (self.stage >= 1 or hard_only) else 1.0

        if self.stage >= 3:
            budget: int | None = 0
        else:
            probe_bundle = generate_challenge(
                cat, seed=ep_seed, difficulty=difficulty, subtype=chosen_subtype
            )
            if self.stage == 2 and probe_bundle.forward_budget > 0:
                budget = max(1, probe_bundle.forward_budget // 2)
            else:
                return probe_bundle

        return generate_challenge(
            cat, seed=ep_seed, difficulty=difficulty, forward_budget=budget, subtype=chosen_subtype
        )


def parse_assistant_turn(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM output into `<think>` reasoning and the first fenced Python code block."""
    cleaned = raw_text.replace("\r\n", "\n")
    for stop in STOP_SEQUENCES:
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]

    code_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    prefix = cleaned[: code_match.start()] if code_match else cleaned
    think_match = re.search(r"<think>(.*?)</think>", prefix, flags=re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
        has_think = bool(think)
    elif "</think>" in prefix:
        think = prefix.split("</think>", 1)[0].replace("<think>", "").strip()
        has_think = bool(think)
    else:
        think = prefix.strip()
        has_think = False

    code = code_match.group(1).strip() if code_match else ""
    has_code = bool(code)
    return {
        "think": think,
        "code": code,
        "has_think": has_think,
        "has_code": has_code,
        "valid_format": bool(has_think and has_code),
        "normalized_content": (
            f"<think>\n{think}\n</think>\n```python\n{code}\n```"
            if has_code
            else cleaned.strip()
        ),
    }


def compute_shaped_reward(
    verification: dict[str, Any] | None,
    turn_stats: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute base verifier reward plus tie-breaker micro-shaping for RL exploration."""
    base_reward = float((verification or {}).get("total_reward", -1.0))
    n_turns = max(1, len(turn_stats))
    format_rate = sum(1.0 for t in turn_stats if t.get("valid_format", True)) / n_turns
    clean_exec_rate = (
        sum(1.0 for t in turn_stats if not t.get("exec_error", False) and not t.get("timed_out", False))
        / n_turns
    )
    format_errors = sum(1 for t in turn_stats if not t.get("has_code", True))
    explicit_submit = bool(
        ((verification or {}).get("metrics") or {}).get(
            "explicit_submit",
            any(t.get("explicit_submit", False) for t in turn_stats),
        )
    )
    shaping_bonus = (
        0.02 * format_rate
        + 0.03 * clean_exec_rate
        + (0.05 if explicit_submit else 0.0)
        - 0.05 * format_errors
        - 0.01 * max(0, n_turns - 2)
    )
    return {
        "base_reward": round(base_reward, 6),
        "shaping_bonus": round(shaping_bonus, 6),
        "shaped_reward": round(base_reward + shaping_bonus, 6),
        "format_rate": round(format_rate, 4),
        "clean_exec_rate": round(clean_exec_rate, 4),
        "explicit_submit": 1.0 if explicit_submit else 0.0,
    }


class GRPORolloutEnv:
    """Lightweight multi-turn CPU rollout environment wrapping SandboxSession for online RLVR."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        max_turns: int = 6,
        trajectory_path: str | Path | None = None,
        timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
        in_memory: bool | None = None,
    ) -> None:
        self._owns_tmp = work_dir is None
        self._tmp_obj: tempfile.TemporaryDirectory[str] | None = None
        if work_dir is None:
            self._tmp_obj = tempfile.TemporaryDirectory()
            self.work_dir = Path(self._tmp_obj.name)
        else:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = int(max_turns)
        self.trajectory_path = trajectory_path
        self.timeout_sec = float(timeout_sec)
        self.in_memory = in_memory
        self._ep_counter = 0
        self.session: SandboxSession | None = None
        self.challenge_dir: Path | None = None
        self._turn_stats: list[dict[str, Any]] = []

    def reset(
        self,
        bundle: ChallengeBundle | None = None,
        category: str | None = None,
        seed: int = 0,
        difficulty: float = 1.0,
        forward_budget: int | None = None,
        subtype: str | None = None,
    ) -> dict[str, Any]:
        """Initialize a fresh blinded challenge sandbox and return initial prompt messages."""
        self.session = None
        if bundle is None:
            cat = CATEGORIES[seed % len(CATEGORIES)] if category is None else category
            bundle = generate_challenge(
                cat,
                seed=seed,
                difficulty=difficulty,
                forward_budget=forward_budget,
                subtype=subtype,
            )
        total_params = sum(
            int(v.numel()) for v in bundle.public_weights.values() if hasattr(v, "numel")
        )
        if total_params >= 1_000_000:
            import gc

            gc.collect()

        use_in_memory = (
            bool(self.in_memory)
            if self.in_memory is not None
            else (total_params >= 1_000_000 and self.trajectory_path is None)
        )
        self._turn_stats = []
        if use_in_memory:
            self.challenge_dir = None
            self.session = SandboxSession.from_bundle(
                bundle,
                max_turns=self.max_turns,
                trajectory_path=self.trajectory_path,
                timeout_sec=self.timeout_sec,
            )
            return {
                "episode_id": bundle.episode_id,
                "category": bundle.category,
                "subtype": bundle.subtype,
                "forward_budget": bundle.forward_budget,
                "challenge_dir": None,
                "messages": list(self.session.state["messages"]),
            }

        ep_dir = self.work_dir / f"rollout_{self._ep_counter}_{bundle.episode_id}"
        self._ep_counter += 1
        save_challenge(bundle, ep_dir)
        self.challenge_dir = ep_dir
        traj = self.trajectory_path if self.trajectory_path is not None else (ep_dir / "trajectory.jsonl")
        self.session = SandboxSession(
            ep_dir,
            max_turns=self.max_turns,
            trajectory_path=traj,
            timeout_sec=self.timeout_sec,
        )
        return {
            "episode_id": bundle.episode_id,
            "category": bundle.category,
            "subtype": bundle.subtype,
            "forward_budget": bundle.forward_budget,
            "challenge_dir": str(ep_dir),
            "messages": list(self.session.state["messages"]),
        }

    def step(self, think: str, code: str) -> dict[str, Any]:
        """Execute one assistant turn inside the active SandboxSession."""
        if self.session is None:
            raise RuntimeError("GRPORolloutEnv.reset() must be called before step().")
        res = self.session.step(think=think, code=code)
        self._turn_stats.append(
            {
                "valid_format": True,
                "has_code": bool(code.strip()),
                "exec_error": bool(res.get("exec_error", False)),
                "timed_out": bool(res.get("timed_out", False)),
                "explicit_submit": bool(res.get("explicit_submit", False)),
            }
        )
        if res.get("submitted"):
            shaping = compute_shaped_reward(res.get("verification"), self._turn_stats)
            res["shaping"] = shaping
            res["shaped_reward"] = shaping["shaped_reward"]
        return res

    def step_raw(self, raw_text: str) -> dict[str, Any]:
        """Parse raw LLM output (`<think>` + ```python``` block) and execute one sandbox turn."""
        if self.session is None:
            raise RuntimeError("GRPORolloutEnv.reset() must be called before step_raw().")
        parsed = parse_assistant_turn(raw_text)
        if not parsed["has_code"]:
            res = self.session.step(
                think=parsed["think"] or "Malformed response.",
                code="raise SyntaxError('Assistant response must include a ```python ... ``` code block.')",
            )
        else:
            res = self.session.step(think=parsed["think"], code=parsed["code"])
        self._turn_stats.append(
            {
                "valid_format": bool(parsed["valid_format"]),
                "has_code": bool(parsed["has_code"]),
                "exec_error": bool(res.get("exec_error", False)),
                "timed_out": bool(res.get("timed_out", False)),
                "explicit_submit": bool(res.get("explicit_submit", False)),
            }
        )
        res["parsed"] = parsed
        if res.get("submitted"):
            shaping = compute_shaped_reward(res.get("verification"), self._turn_stats)
            res["shaping"] = shaping
            res["shaped_reward"] = shaping["shaped_reward"]
        return res

    def run_episode(
        self,
        bundle: ChallengeBundle,
        turns: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Run a full sequence of (think, code) turns on a ChallengeBundle and return final verification."""
        self.reset(bundle=bundle)
        last_res: dict[str, Any] = {}
        for think, code in turns:
            last_res = self.step(think=think, code=code)
            if last_res.get("submitted"):
                break
        if self.session is not None and not last_res.get("submitted"):
            # Force terminal submission of current workspace weights if agent exhausted turns without submitting
            while self.session is not None and not last_res.get("submitted"):
                last_res = self.step(think="Final auto-submission.", code="submit_weights(weights)")
        last_res["messages"] = list(self.session.state["messages"]) if self.session is not None else []
        return last_res

    def run_raw_episode(
        self,
        bundle: ChallengeBundle,
        policy_fn: Callable[[list[dict[str, str]]], str],
    ) -> dict[str, Any]:
        """Run a live multi-turn rollout driven by a raw-text policy callback."""
        self.reset(bundle=bundle)
        last_res: dict[str, Any] = {}
        for _ in range(self.max_turns):
            assert self.session is not None
            raw_reply = policy_fn(list(self.session.state["messages"]))
            last_res = self.step_raw(raw_reply)
            if last_res.get("submitted"):
                break
        last_res["messages"] = list(self.session.state["messages"]) if self.session is not None else []
        return last_res

    def close(self) -> None:
        if self._tmp_obj is not None:
            self._tmp_obj.cleanup()
            self._tmp_obj = None
