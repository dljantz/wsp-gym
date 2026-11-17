"""Level 2 Scale-Transfer evaluation harness (102M procedural GQA + 135M/494M pretrained LLMs)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any, Callable, Iterator

from eval import make_policy_fn
from wsp_gym.generators import ChallengeBundle
from wsp_gym.level2 import (
    LEVEL2_PRETRAINED_SUBTYPES,
    LEVEL2_PROCEDURAL_SUBTYPES,
    build_level2_turn1,
    build_level2_turn2,
    generate_level2_challenge,
)
from wsp_gym.training import GRPORolloutEnv, compute_shaped_reward


def _iter_level2_specs(
    suite: str = "all",
    episodes_per_subtype: int = 2,
    base_seed: int = 19000,
) -> Iterator[tuple[str, str, str, int, int | None]]:
    items: list[tuple[str, str, str]] = []
    if suite in ("procedural", "all"):
        for cat, sub in LEVEL2_PROCEDURAL_SUBTYPES:
            items.append(("procedural", cat, sub))
    if suite in ("pretrained", "all"):
        for cat, sub in LEVEL2_PRETRAINED_SUBTYPES:
            items.append(("pretrained", cat, sub))
    if not items:
        raise ValueError(f"Invalid --suite {suite!r}; expected 'procedural', 'pretrained', or 'all'.")

    for rep in range(int(episodes_per_subtype)):
        budget = None if (rep % 2 == 0) else 0
        for sub_idx, (track, cat, sub) in enumerate(items):
            ep_seed = int(base_seed) + rep * 100 + sub_idx
            yield track, cat, sub, ep_seed, budget


def make_level2_policy_fn(
    backend: str,
    env: GRPORolloutEnv,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
) -> Callable[[list[dict[str, str]]], str]:
    """Create policy callback for Level 2 evaluation (`dummy`, `scripted`, or `hf`)."""
    if backend == "scripted":

        def _scripted_l2(messages: list[dict[str, str]]) -> str:
            session = env.session
            if session is None:
                raise RuntimeError("Active SandboxSession required for scripted Level 2 policy.")
            cat = str(session.meta["category"])
            budget = int(session.meta["forward_budget"])
            turns_done = sum(1 for m in messages if m.get("role") == "assistant")
            if turns_done == 0:
                think, code = build_level2_turn1(cat, budget)
            else:
                diag = dict(session.state["user_vars"].get("diag", {}))
                think, code = build_level2_turn2(cat, diag)
            return f"<think>\n{think}\n</think>\n```python\n{code}\n```"

        return _scripted_l2

    return make_policy_fn(
        backend=backend,
        env=env,
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def run_level2_evaluation(
    suite: str = "all",
    scale: str = "100m",
    backend: str = "scripted",
    episodes_per_subtype: int = 2,
    base_seed: int = 19000,
    max_turns: int = 6,
    timeout_sec: float = 30.0,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate zero-shot or adapted policy across Level 2 procedural and/or pretrained suites."""
    env = GRPORolloutEnv(max_turns=max_turns, timeout_sec=timeout_sec, in_memory=True)
    try:
        policy_fn = make_level2_policy_fn(
            backend=backend,
            env=env,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        track_stats: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        cat_stats: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        sub_stats: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        zero_probe_results: list[bool] = []
        base_rewards: list[float] = []
        shaped_rewards: list[float] = []
        turns_list: list[int] = []
        format_rates: list[float] = []
        clean_exec_rates: list[float] = []
        explicit_submits: list[float] = []
        episode_records: list[dict[str, Any]] = []
        passed_count = 0
        total = 0

        for track, _cat, sub, ep_seed, budget in _iter_level2_specs(
            suite=suite,
            episodes_per_subtype=episodes_per_subtype,
            base_seed=base_seed,
        ):
            bundle: ChallengeBundle | None = generate_level2_challenge(
                subtype=sub,
                seed=ep_seed,
                scale=scale,
                forward_budget=budget,
            )
            assert bundle is not None
            res = env.run_raw_episode(bundle, policy_fn)
            ver = res.get("verification") or {}
            shaping = res.get("shaping") or compute_shaped_reward(ver, env._turn_stats)
            passed = bool(ver.get("passed", False))
            base_reward = float(ver.get("total_reward", shaping.get("base_reward", -1.0)))
            shaped_reward = float(shaping.get("shaped_reward", base_reward))
            metrics = ver.get("metrics") or {}
            turns_used = int(
                metrics.get(
                    "turns_used",
                    sum(1 for m in res.get("messages", []) if m.get("role") == "assistant"),
                )
            )

            total += 1
            passed_count += int(passed)
            base_rewards.append(base_reward)
            shaped_rewards.append(shaped_reward)
            turns_list.append(turns_used)
            format_rates.append(float(shaping.get("format_rate", 0.0)))
            clean_exec_rates.append(float(shaping.get("clean_exec_rate", 0.0)))
            explicit_submits.append(float(shaping.get("explicit_submit", 0.0)))
            if bundle.forward_budget == 0:
                zero_probe_results.append(passed)

            track_stats[track].append((passed, base_reward))
            cat_stats[bundle.category].append((passed, base_reward))
            sub_stats[bundle.subtype].append((passed, base_reward))
            episode_records.append(
                {
                    "episode_id": bundle.episode_id,
                    "track": track,
                    "category": bundle.category,
                    "subtype": bundle.subtype,
                    "n_layers": int(bundle.config.n_layers),
                    "d_model": int(bundle.config.d_model),
                    "forward_budget": int(bundle.forward_budget),
                    "passed": passed,
                    "base_reward": round(base_reward, 6),
                    "shaped_reward": round(shaped_reward, 6),
                    "turns": turns_used,
                    "explicit_submit": bool(shaping.get("explicit_submit", 0.0)),
                    "format_rate": float(shaping.get("format_rate", 0.0)),
                    "clean_exec_rate": float(shaping.get("clean_exec_rate", 0.0)),
                    "metrics": metrics,
                    "error": ver.get("error"),
                }
            )
            env.session = None
            bundle = None
            gc.collect()

        report = {
            "suite": suite,
            "scale": scale,
            "backend": backend,
            "base_seed": int(base_seed),
            "episodes_per_subtype": int(episodes_per_subtype),
            "total_episodes": total,
            "passed_count": passed_count,
            "pass_rate": round(passed_count / max(1, total), 6),
            "zero_probe_episodes": len(zero_probe_results),
            "zero_probe_pass_rate": (
                round(sum(1 for p in zero_probe_results if p) / len(zero_probe_results), 6)
                if zero_probe_results
                else 0.0
            ),
            "mean_base_reward": round(sum(base_rewards) / max(1, total), 6),
            "mean_shaped_reward": round(sum(shaped_rewards) / max(1, total), 6),
            "mean_turns": round(sum(turns_list) / max(1, total), 4),
            "mean_format_rate": round(sum(format_rates) / max(1, total), 6),
            "mean_clean_exec_rate": round(sum(clean_exec_rates) / max(1, total), 6),
            "explicit_submit_rate": round(sum(explicit_submits) / max(1, total), 6),
            "by_track": {
                tr: {
                    "count": len(vals),
                    "pass_rate": round(sum(1 for p, _ in vals if p) / len(vals), 6),
                    "mean_reward": round(sum(r for _, r in vals) / len(vals), 6),
                }
                for tr, vals in sorted(track_stats.items())
            },
            "by_category": {
                cat: {
                    "count": len(vals),
                    "pass_rate": round(sum(1 for p, _ in vals if p) / len(vals), 6),
                    "mean_reward": round(sum(r for _, r in vals) / len(vals), 6),
                }
                for cat, vals in sorted(cat_stats.items())
            },
            "by_subtype": {
                sub: {
                    "count": len(vals),
                    "pass_rate": round(sum(1 for p, _ in vals if p) / len(vals), 6),
                    "mean_reward": round(sum(r for _, r in vals) / len(vals), 6),
                }
                for sub, vals in sorted(sub_stats.items())
            },
            "episodes": episode_records,
        }
        if out_path is not None:
            dst = Path(out_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WSP agents on Level 2 scale-transfer tasks.")
    parser.add_argument("--suite", choices=("procedural", "pretrained", "all"), default="all")
    parser.add_argument("--scale", choices=("mini", "100m"), default="100m")
    parser.add_argument("--backend", choices=("dummy", "scripted", "hf"), default="scripted")
    parser.add_argument("--episodes-per-subtype", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=19000)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--out-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_level2_evaluation(
        suite=args.suite,
        scale=args.scale,
        backend=args.backend,
        episodes_per_subtype=args.episodes_per_subtype,
        base_seed=args.base_seed,
        max_turns=args.max_turns,
        timeout_sec=args.timeout_sec,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        out_path=args.out_path,
    )
    print(json.dumps(summary, indent=2))
