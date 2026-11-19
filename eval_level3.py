"""Level 3 evaluation harness: Immutable-Twin Self-Surgery at 7.62B scale."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator

import torch

from eval import _truncate_at_stops, make_policy_fn
from wsp_gym.level2 import _trim_host_ram, build_level2_turn1, build_level2_turn2
from wsp_gym.level3 import (
    ALL_LEVEL3_SUBTYPES,
    LEVEL3_CONSTRUCTIVE_SUBTYPES,
    LEVEL3_SELF_TWIN_MODEL,
    LEVEL3_SUBTYPES,
    _evict_safetensors_page_cache,
    compute_bundle_twin_digest,
    compute_surgeon_digest,
    evaluate_immutable_twin_promotion,
    generate_level3_challenge,
    verify_self_twin_equivalence,
)
from wsp_gym.training import GRPORolloutEnv, compute_shaped_reward


def _iter_level3_specs(
    suite: str = "restorative",
    episodes_per_subtype: int = 2,
    base_seed: int = 29000,
) -> Iterator[tuple[str, str, int, int | None]]:
    if suite == "restorative":
        selected = [(idx, cat, sub) for idx, (cat, sub) in enumerate(LEVEL3_SUBTYPES)]
    elif suite == "constructive":
        offset = len(LEVEL3_SUBTYPES)
        selected = [(offset + idx, cat, sub) for idx, (cat, sub) in enumerate(LEVEL3_CONSTRUCTIVE_SUBTYPES)]
    elif suite == "all":
        selected = [(idx, cat, sub) for idx, (cat, sub) in enumerate(ALL_LEVEL3_SUBTYPES)]
    else:
        raise ValueError(f"Unknown Level 3 suite: {suite!r}")

    for rep in range(int(episodes_per_subtype)):
        budget = None if (rep % 2 == 0) else 0
        for sub_idx, cat, sub in selected:
            ep_seed = int(base_seed) + rep * 100 + sub_idx
            yield cat, sub, ep_seed, budget


def _load_hf_surgeon_policy(
    model_name_or_path: str,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
) -> tuple[Any, Callable[[list[dict[str, str]]], str]]:
    """Load HF (+ optional PEFT LoRA) surgeon onto GPU first and return model + callback."""
    if not model_name_or_path:
        raise ValueError("--model-name-or-path is required when --backend=hf.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    _evict_safetensors_page_cache(model_name_or_path)
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
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(temperature > 0.0),
                temperature=max(1e-4, float(temperature)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        return _truncate_at_stops(raw_text)

    _trim_host_ram()
    return model, _hf_policy


def run_level3_evaluation(
    scale: str = "7b",
    suite: str = "restorative",
    backend: str = "scripted",
    episodes_per_subtype: int = 2,
    base_seed: int = 29000,
    max_turns: int = 4,
    timeout_sec: float = 90.0,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
    out_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate zero-shot / Level-2 / Phase-2.5 models on Level 3 Immutable-Twin Self-Surgery."""
    specs = list(_iter_level3_specs(suite=suite, episodes_per_subtype=episodes_per_subtype, base_seed=base_seed))
    env = GRPORolloutEnv(
        max_turns=int(max_turns),
        timeout_sec=float(timeout_sec),
        in_memory=True,
    )
    policy_model: Any = None
    try:
        # Load GPU policy model BEFORE CPU shard-streaming the 7B twin so host RAM stays minimal.
        if backend == "hf":
            policy_model, policy_fn = _load_hf_surgeon_policy(
                model_name_or_path=str(model_name_or_path or ""),
                adapter_path=adapter_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        elif backend == "scripted":

            def _scripted_l3(messages: list[dict[str, str]]) -> str:
                session = env.session
                if session is None:
                    raise RuntimeError("Active SandboxSession required for scripted Level 3 policy.")
                cat = str(session.meta["category"])
                budget = int(session.meta["forward_budget"])
                turns_done = sum(1 for m in messages if m.get("role") == "assistant")
                if turns_done == 0:
                    think, code = build_level2_turn1(cat, budget)
                else:
                    diag = dict(session.state["user_vars"].get("diag", {}))
                    think, code = build_level2_turn2(cat, diag)
                return f"<think>\n{think}\n</think>\n```python\n{code}\n```"

            policy_fn = _scripted_l3
        else:
            policy_fn = make_policy_fn(
                backend=backend,
                env=env,
                model_name_or_path=model_name_or_path,
                adapter_path=adapter_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        cat_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sub_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        episode_records: list[dict[str, Any]] = []
        zero_probe_results: list[bool] = []
        base_rewards: list[float] = []
        shaped_rewards: list[float] = []
        turns_list: list[int] = []
        format_rates: list[float] = []
        clean_exec_rates: list[float] = []
        explicit_submits: list[float] = []
        ref_kls: list[float] = []
        raw_kls: list[float] = []
        turn1_walls: list[float] = []
        passed_count = 0
        promoted_count = 0
        surgeon_immutable_count = 0
        twin_base_immutable_count = 0

        saved_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        if resume and out_path is not None:
            part_src = Path(f"{out_path}.partial.json")
            if part_src.exists():
                raw_part = json.loads(part_src.read_text(encoding="utf-8"))
                for r in raw_part.get("episodes", []):
                    saved_by_key[(str(r["subtype"]), int(str(r["episode_id"]).rsplit("_", 1)[-1]))] = r

        for cat, sub, ep_seed, budget in specs:
            key = (sub, int(ep_seed))
            resumed = key in saved_by_key
            if resumed:
                rec = dict(saved_by_key[key])
            else:
                surgeon_before = (
                    compute_surgeon_digest(policy_model) if policy_model is not None else None
                )
                bundle = generate_level3_challenge(
                    subtype=sub,
                    seed=ep_seed,
                    scale=scale,
                    forward_budget=budget,
                )
                twin_before = compute_bundle_twin_digest(bundle)

                ep_t0 = time.perf_counter()
                res = env.run_raw_episode(bundle, policy_fn)
                ep_wall_sec = round(time.perf_counter() - ep_t0, 4)
                turn1_wall_sec = round(float(env._turn_stats[0].get("wall_sec", 0.0)), 4) if env._turn_stats else 0.0
                ver = res.get("verification") or {}
                shaping = res.get("shaping") or compute_shaped_reward(ver, env._turn_stats)

                submitted_w = None
                if env.session is not None:
                    sub_data = env.session.state.get("submission_data") or {}
                    submitted_w = sub_data.get("weights") or env.session.state.get("weights")
                twin_after = compute_bundle_twin_digest(bundle)
                surgeon_after = (
                    compute_surgeon_digest(policy_model) if policy_model is not None else None
                )

                promo = evaluate_immutable_twin_promotion(
                    bundle=bundle,
                    verification=ver,
                    candidate_weights=submitted_w,
                    surgeon_digest_before=surgeon_before,
                    surgeon_digest_after=surgeon_after,
                    twin_base_digest_before=twin_before,
                    twin_base_digest_after=twin_after,
                )

                rec = {
                    "episode_id": bundle.episode_id,
                    "category": cat,
                    "subtype": sub,
                    "forward_budget": bundle.forward_budget,
                    "passed": bool(ver.get("passed", False)),
                    "promoted": bool(promo["promoted"]),
                    "surgeon_mutated": bool(promo["surgeon_mutated"]),
                    "twin_base_mutated": bool(promo["twin_base_mutated"]),
                    "reference_twin_kl": promo["reference_twin_kl"],
                    "raw_pretrained_kl": promo["raw_pretrained_kl"],
                    "corrupted_retain_kl": promo["corrupted_retain_kl"],
                    "base_reward": float(ver.get("total_reward", -1.0)),
                    "shaped_reward": float(shaping["shaped_reward"]),
                    "turns": int(ver.get("metrics", {}).get("turns_used", max_turns)),
                    "turn1_wall_sec": turn1_wall_sec,
                    "episode_wall_sec": ep_wall_sec,
                    "explicit_submit": bool(shaping["explicit_submit"]),
                    "format_rate": float(shaping["format_rate"]),
                    "clean_exec_rate": float(shaping["clean_exec_rate"]),
                    "metrics": ver.get("metrics", {}),
                    "error": ver.get("error"),
                }
                env.session = None
                del res, ver, submitted_w, bundle

            passed = bool(rec.get("passed", False))
            promoted = bool(rec.get("promoted", False))
            surgeon_ok = not bool(rec.get("surgeon_mutated", False))
            twin_ok = not bool(rec.get("twin_base_mutated", False))
            base_r = float(rec["base_reward"])
            shaped_r = float(rec["shaped_reward"])
            turns_used = int(rec["turns"])
            if "turn1_wall_sec" in rec:
                turn1_walls.append(float(rec["turn1_wall_sec"]))

            passed_count += int(passed)
            promoted_count += int(promoted)
            surgeon_immutable_count += int(surgeon_ok)
            twin_base_immutable_count += int(twin_ok)
            base_rewards.append(base_r)
            shaped_rewards.append(shaped_r)
            turns_list.append(turns_used)
            format_rates.append(float(rec["format_rate"]))
            clean_exec_rates.append(float(rec["clean_exec_rate"]))
            explicit_submits.append(float(bool(rec["explicit_submit"])))
            if float(rec["reference_twin_kl"]) < 900.0:
                ref_kls.append(float(rec["reference_twin_kl"]))
            if float(rec["raw_pretrained_kl"]) < 900.0:
                raw_kls.append(float(rec["raw_pretrained_kl"]))
            if int(rec["forward_budget"]) == 0:
                zero_probe_results.append(passed)

            cat_stats[cat].append(rec)
            sub_stats[sub].append(rec)
            episode_records.append(rec)
            tag = "RESUMED " if resumed else ""
            print(
                f"[eval_level3 {len(episode_records)}/{len(specs)}] {rec['episode_id']} "
                f"{tag}passed={passed} promoted={promoted} rew={base_r:.4f} turns={turns_used}",
                file=sys.stderr,
                flush=True,
            )
            if out_path is not None:
                part_dst = Path(f"{out_path}.partial.json")
                part_dst.parent.mkdir(parents=True, exist_ok=True)
                part_dst.write_text(json.dumps({"episodes": episode_records}, indent=2), encoding="utf-8")

            if not resumed:
                _trim_host_ram()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total = len(specs)
        report = {
            "scale": scale,
            "suite": suite,
            "backend": backend,
            "model_name_or_path": model_name_or_path,
            "adapter_path": adapter_path,
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
            "promoted_count": promoted_count,
            "promotion_rate": round(promoted_count / max(1, total), 6),
            "surgeon_immutability_rate": round(surgeon_immutable_count / max(1, total), 6),
            "twin_base_immutability_rate": round(twin_base_immutable_count / max(1, total), 6),
            "mean_turn1_latency_sec": round(sum(turn1_walls) / max(1, len(turn1_walls)), 4) if turn1_walls else None,
            "mean_reference_twin_kl": round(sum(ref_kls) / max(1, len(ref_kls)), 8) if ref_kls else None,
            "mean_raw_pretrained_kl": round(sum(raw_kls) / max(1, len(raw_kls)), 8) if raw_kls else None,
            "mean_base_reward": round(sum(base_rewards) / max(1, total), 6),
            "mean_shaped_reward": round(sum(shaped_rewards) / max(1, total), 6),
            "mean_turns": round(sum(turns_list) / max(1, total), 4),
            "mean_format_rate": round(sum(format_rates) / max(1, total), 6),
            "mean_clean_exec_rate": round(sum(clean_exec_rates) / max(1, total), 6),
            "explicit_submit_rate": round(sum(explicit_submits) / max(1, total), 6),
            "by_category": {
                c: {
                    "count": len(vals),
                    "pass_rate": round(sum(1 for r in vals if r["passed"]) / len(vals), 6),
                    "promotion_rate": round(sum(1 for r in vals if r["promoted"]) / len(vals), 6),
                    "mean_reward": round(sum(r["base_reward"] for r in vals) / len(vals), 6),
                }
                for c, vals in sorted(cat_stats.items())
            },
            "by_subtype": {
                s: {
                    "count": len(vals),
                    "pass_rate": round(sum(1 for r in vals if r["passed"]) / len(vals), 6),
                    "promotion_rate": round(sum(1 for r in vals if r["promoted"]) / len(vals), 6),
                    "mean_reward": round(sum(r["base_reward"] for r in vals) / len(vals), 6),
                    "mean_reference_twin_kl": round(
                        sum(r["reference_twin_kl"] for r in vals if r["reference_twin_kl"] < 900.0)
                        / max(1, sum(1 for r in vals if r["reference_twin_kl"] < 900.0)),
                        8,
                    ),
                }
                for s, vals in sorted(sub_stats.items())
            },
            "episodes": episode_records,
        }
        if out_path is not None:
            dst = Path(out_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(report, indent=2), encoding="utf-8")
            Path(f"{out_path}.partial.json").unlink(missing_ok=True)
        return report
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WSP agents on Level 3 Immutable-Twin Self-Surgery.")
    parser.add_argument("--scale", choices=("mini", "7b"), default="7b")
    parser.add_argument("--suite", choices=("restorative", "constructive", "all"), default="restorative")
    parser.add_argument("--backend", choices=("dummy", "scripted", "hf"), default="scripted")
    parser.add_argument("--episodes-per-subtype", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=29000)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--model-name-or-path", default=LEVEL3_SELF_TWIN_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-hf-equivalence", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verify_hf_equivalence:
        res = verify_self_twin_equivalence(model_id=args.model_name_or_path)
        if args.out_path is not None:
            dst = Path(args.out_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
    else:
        report = run_level3_evaluation(
            scale=args.scale,
            suite=args.suite,
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
            resume=args.resume,
        )
        print(json.dumps(report, indent=2))
