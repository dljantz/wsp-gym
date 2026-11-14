"""Held-out evaluation harness across all 17 WSP-Gym subtypes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from wsp_gym.generators import ChallengeBundle, generate_challenge
from wsp_gym.scaler import ALL_17_SUBTYPES, _build_turn1, _build_turn2
from wsp_gym.sandbox import DEFAULT_TURN_TIMEOUT_SEC
from wsp_gym.training import STOP_SEQUENCES, GRPORolloutEnv, compute_shaped_reward

LEVEL1_5_OR_CYCLE2_ONLY = {
    "modular_varied_lag_with_3tok_trojan",
    "norm_matched_k_composition_corruption",
    "dual_offset_modular_shift",
}


def build_heldout_suite(
    episodes_per_subtype: int = 2,
    base_seed: int = 9000,
) -> list[ChallengeBundle]:
    """Build deterministic held-out ChallengeBundles across all 17 subtypes."""
    suite: list[ChallengeBundle] = []
    for rep in range(int(episodes_per_subtype)):
        for sub_idx, (cat, sub) in enumerate(ALL_17_SUBTYPES):
            ep_seed = int(base_seed) + rep * 100 + sub_idx
            budget = None if (rep % 2 == 0) else 0
            if rep % 2 == 0 and "level1_5" not in sub and sub not in LEVEL1_5_OR_CYCLE2_ONLY:
                difficulty = 1.0
            else:
                difficulty = 1.5
            suite.append(
                generate_challenge(
                    cat,
                    seed=ep_seed,
                    difficulty=difficulty,
                    forward_budget=budget,
                    subtype=sub,
                )
            )
    return suite


def _truncate_at_stops(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")
    cut = len(cleaned)
    for stop in STOP_SEQUENCES:
        idx = cleaned.find(stop)
        if idx != -1 and idx < cut:
            cut = idx
    return cleaned[:cut].strip()


def make_policy_fn(
    backend: str,
    env: GRPORolloutEnv,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
) -> Callable[[list[dict[str, str]]], str]:
    """Create a raw-text multi-turn policy callback for `dummy`, `scripted`, or `hf`."""
    if backend == "dummy":

        def _dummy_policy(messages: list[dict[str, str]]) -> str:
            brief = messages[1]["content"] if len(messages) > 1 else ""
            if "submit_decompiled" in brief:
                code = "submit_decompiled({'a': 1, 'b': 1, 'c': 0, 'trigger': [0, 0, 0, 0]})"
            else:
                code = "submit_weights(weights)"
            return f"<think>\nBaseline no-op submission.\n</think>\n```python\n{code}\n```"

        return _dummy_policy

    if backend == "scripted":

        def _scripted_policy(messages: list[dict[str, str]]) -> str:
            session = env.session
            if session is None:
                raise RuntimeError("Active SandboxSession required for scripted policy.")
            cat = str(session.meta["category"])
            budget = int(session.meta["forward_budget"])
            assistant_turns_done = sum(1 for m in messages if m.get("role") == "assistant")
            if assistant_turns_done == 0:
                think, code = _build_turn1(cat, session.config, budget, session.brief_md)
            else:
                diag = dict(session.state["user_vars"].get("diag", {}))
                think, code = _build_turn2(cat, session.config, budget, diag)
            return f"<think>\n{think}\n</think>\n```python\n{code}\n```"

        return _scripted_policy

    if backend == "hf":
        if not model_name_or_path:
            raise ValueError("--model-name-or-path is required when --backend=hf.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        device = next(model.parameters()).device

        def _hf_policy(messages: list[dict[str, str]]) -> str:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": bool(temperature > 0.0),
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = float(temperature)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            gen_ids = out[0, inputs["input_ids"].shape[1] :]
            raw_decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
            return _truncate_at_stops(raw_decoded)

        return _hf_policy

    raise ValueError(f"Unsupported evaluation backend: {backend!r}")


def run_evaluation(
    backend: str = "scripted",
    episodes_per_subtype: int = 2,
    base_seed: int = 9000,
    max_turns: int = 6,
    timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run held-out evaluation across all 17 subtypes and aggregate metrics."""
    suite = build_heldout_suite(episodes_per_subtype=episodes_per_subtype, base_seed=base_seed)
    env = GRPORolloutEnv(max_turns=max_turns, timeout_sec=timeout_sec)
    try:
        policy_fn = make_policy_fn(
            backend=backend,
            env=env,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        cat_stats: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        sub_stats: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        zero_probe_results: list[bool] = []
        base_rewards: list[float] = []
        shaped_rewards: list[float] = []
        turns_list: list[int] = []
        format_rates: list[float] = []
        clean_exec_rates: list[float] = []
        explicit_submits: list[float] = []
        passed_count = 0

        for bundle in suite:
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

            passed_count += int(passed)
            base_rewards.append(base_reward)
            shaped_rewards.append(shaped_reward)
            turns_list.append(turns_used)
            format_rates.append(float(shaping.get("format_rate", 0.0)))
            clean_exec_rates.append(float(shaping.get("clean_exec_rate", 0.0)))
            explicit_submits.append(float(shaping.get("explicit_submit", 0.0)))
            if bundle.forward_budget == 0:
                zero_probe_results.append(passed)

            cat_stats[bundle.category].append((passed, base_reward))
            sub_stats[bundle.subtype].append((passed, base_reward))

        total = len(suite)
        report = {
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
        }
        if out_path is not None:
            dst = Path(out_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a WSP agent on held-out 17-subtype challenges.")
    parser.add_argument("--backend", choices=("dummy", "scripted", "hf"), default="scripted")
    parser.add_argument("--episodes-per-subtype", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TURN_TIMEOUT_SEC)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--out-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_evaluation(
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
