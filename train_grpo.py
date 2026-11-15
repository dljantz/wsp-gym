"""Multi-turn online Group Relative Policy Optimization (GRPO) entrypoint for WSP-Gym."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import _truncate_at_stops
from train_sft import (
    ChatWordTokenizer,
    TinyCausalLM,
    _grad_l2_norm,
    configure_model_for_lora_training,
    tokenize_and_mask_chat,
)
from wsp_gym.sandbox import DEFAULT_TURN_TIMEOUT_SEC
from wsp_gym.scaler import _build_exploratory_turn1, _build_turn1, _build_turn2
from wsp_gym.training import CurriculumScheduler, GRPORolloutEnv, compute_shaped_reward


def compute_grpo_group_advantages(
    rewards: list[float],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute group-relative normalized advantages with zero-variance protection."""
    if not rewards:
        raise ValueError("rewards list must be non-empty.")
    mean_r = sum(float(r) for r in rewards) / len(rewards)
    var_r = sum((float(r) - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    if std_r < eps:
        return {
            "mean_reward": round(mean_r, 6),
            "std_reward": round(std_r, 6),
            "advantages": [0.0 for _ in rewards],
            "effective_group": False,
        }
    advantages = [(float(r) - mean_r) / (std_r + eps) for r in rewards]
    return {
        "mean_reward": round(mean_r, 6),
        "std_reward": round(std_r, 6),
        "advantages": advantages,
        "effective_group": True,
    }


def _gather_shifted_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    active_mask = shift_labels != -100
    if not bool(active_mask.any()):
        return shift_logits.sum() * 0.0 + torch.empty((0,), device=input_ids.device, dtype=shift_logits.dtype)
    safe_targets = torch.where(active_mask, shift_labels, torch.zeros_like(shift_labels))
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    return gathered[active_mask]


def _snapshot_trainable_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a frozen CPU snapshot of trainable parameters (e.g. loaded SFT LoRA weights)."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


@contextlib.contextmanager
def _swap_trainable_params(
    model: nn.Module,
    ref_params: dict[str, torch.Tensor],
) -> Iterator[None]:
    """Temporarily swap trainable parameter tensors to a reference snapshot, restoring in finally."""
    saved: dict[str, torch.Tensor] = {}
    named = dict(model.named_parameters())
    try:
        for name, ref_tensor in ref_params.items():
            param = named.get(name)
            if param is not None:
                saved[name] = param.data
                param.data = ref_tensor.to(device=param.device, dtype=param.dtype)
        yield
    finally:
        for name, orig_data in saved.items():
            named[name].data = orig_data


def _compute_reference_logprobs(
    policy_model: nn.Module,
    ref_model: nn.Module | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    fallback: torch.Tensor,
    ref_params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        if ref_model is not None:
            return _gather_shifted_token_logprobs(ref_model, input_ids, attention_mask, labels)
        if ref_params:
            with _swap_trainable_params(policy_model, ref_params):
                return _gather_shifted_token_logprobs(policy_model, input_ids, attention_mask, labels)
        if hasattr(policy_model, "disable_adapter"):
            with policy_model.disable_adapter():
                return _gather_shifted_token_logprobs(policy_model, input_ids, attention_mask, labels)
    return fallback.detach()


def compute_grpo_loss_and_metrics(
    policy_model: nn.Module,
    ref_model: nn.Module | None,
    batch_trajectories: list[dict[str, Any]],
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
    accumulate_grads: bool = False,
    ref_params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped multi-turn GRPO surrogate loss + Schulman k3 KL penalty.

    When `accumulate_grads=True`, calls `.backward()` per trajectory inside the loop so
    only a single trajectory's activation graph resides in VRAM at once.
    """
    device = next(policy_model.parameters()).device
    total_active_tokens = 0
    for traj in batch_trajectories:
        lbl = traj["labels"]
        shift_lbl = lbl[1:] if lbl.dim() == 1 else lbl[:, 1:]
        total_active_tokens += int((shift_lbl != -100).sum().item())

    if total_active_tokens == 0:
        zero = next(policy_model.parameters()).sum() * 0.0
        return zero, {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_div": 0.0,
            "active_assistant_tokens": 0,
        }

    policy_loss_sum: torch.Tensor | None = None
    kl_sum: torch.Tensor | None = None
    policy_loss_val = 0.0
    kl_val = 0.0
    counted_tokens = 0

    for traj in batch_trajectories:
        input_ids = traj["input_ids"].to(device)
        labels = traj["labels"].to(device)
        attention_mask = traj.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        log_pi_theta = _gather_shifted_token_logprobs(policy_model, input_ids, attention_mask, labels)
        n_active = int(log_pi_theta.numel())
        if n_active == 0:
            continue

        log_pi_ref = _compute_reference_logprobs(
            policy_model,
            ref_model,
            input_ids,
            attention_mask,
            labels,
            fallback=log_pi_theta,
            ref_params=ref_params,
        )
        old_lp_raw = traj.get("old_token_logprobs")
        if old_lp_raw is None:
            log_pi_old = log_pi_theta.detach()
        else:
            log_pi_old = old_lp_raw.to(device).detach()

        log_ratio = torch.clamp(log_pi_theta - log_pi_old, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        adv = float(traj["advantage"])
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
        token_policy_loss = -torch.minimum(surr1, surr2)

        # Non-negative Schulman k3 estimator: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
        delta_ref = torch.clamp(log_pi_ref - log_pi_theta, min=-20.0, max=20.0)
        token_kl = torch.exp(delta_ref) - delta_ref - 1.0

        traj_policy_sum = token_policy_loss.sum()
        traj_kl_sum = token_kl.sum()
        policy_loss_val += float(traj_policy_sum.detach().item())
        kl_val += float(traj_kl_sum.detach().item())
        counted_tokens += n_active

        if accumulate_grads:
            traj_loss = (traj_policy_sum + float(kl_beta) * traj_kl_sum) / float(total_active_tokens)
            traj_loss.backward()
        else:
            policy_loss_sum = traj_policy_sum if policy_loss_sum is None else (policy_loss_sum + traj_policy_sum)
            kl_sum = traj_kl_sum if kl_sum is None else (kl_sum + traj_kl_sum)

    if counted_tokens == 0:
        zero = next(policy_model.parameters()).sum() * 0.0
        return zero, {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_div": 0.0,
            "active_assistant_tokens": 0,
        }

    if accumulate_grads:
        mean_policy_val = policy_loss_val / float(counted_tokens)
        mean_kl_val = kl_val / float(counted_tokens)
        total_val = mean_policy_val + float(kl_beta) * mean_kl_val
        return torch.tensor(total_val, device=device), {
            "loss": round(total_val, 6),
            "policy_loss": round(mean_policy_val, 6),
            "kl_div": round(mean_kl_val, 6),
            "active_assistant_tokens": int(counted_tokens),
        }

    assert policy_loss_sum is not None and kl_sum is not None
    mean_policy_loss = policy_loss_sum / float(counted_tokens)
    mean_kl = kl_sum / float(counted_tokens)
    total_loss = mean_policy_loss + float(kl_beta) * mean_kl
    return total_loss, {
        "loss": round(float(total_loss.detach().item()), 6),
        "policy_loss": round(float(mean_policy_loss.detach().item()), 6),
        "kl_div": round(float(mean_kl.detach().item()), 6),
        "active_assistant_tokens": int(counted_tokens),
    }


def _snapshot_old_logprobs(
    policy_model: nn.Module,
    tokenized: dict[str, Any],
) -> torch.Tensor:
    device = next(policy_model.parameters()).device
    with torch.no_grad():
        input_ids = tokenized["input_ids"].unsqueeze(0).to(device)
        attention_mask = tokenized["attention_mask"].unsqueeze(0).to(device)
        labels = tokenized["labels"].unsqueeze(0).to(device)
        return _gather_shifted_token_logprobs(policy_model, input_ids, attention_mask, labels).detach().cpu()


def _make_dry_run_rollout_policy(env: GRPORolloutEnv, idx: int, group_size: int) -> Callable[[list[dict[str, str]]], str]:
    good = idx < max(1, group_size // 2)
    exploratory = good and (idx % 2 == 1)

    def _policy(messages: list[dict[str, str]]) -> str:
        session = env.session
        if session is None:
            raise RuntimeError("Active SandboxSession required.")
        cat = str(session.meta["category"])
        budget = int(session.meta["forward_budget"])
        turns_done = sum(1 for m in messages if m.get("role") == "assistant")

        if not good:
            if turns_done == 0:
                return (
                    "<think>\nLet me check tensor keys before deciding.\n</think>\n"
                    "```python\nprint(sorted(weights.keys())[:4])\n```"
                )
            brief = messages[1]["content"] if len(messages) > 1 else ""
            if "submit_decompiled" in brief:
                code = "submit_decompiled({'a': 1, 'b': 1, 'c': 0, 'trigger': [0, 0, 0, 0]})"
            else:
                code = "submit_weights(weights)"
            return f"<think>\nSubmitting baseline weights without edit.\n</think>\n```python\n{code}\n```"

        if exploratory:
            if turns_done == 0:
                t, c = _build_exploratory_turn1(cat, session.config, budget, mode="hypothesis_refine")
            elif turns_done == 1:
                t, c = _build_turn1(cat, session.config, budget, session.brief_md)
            else:
                diag = dict(session.state["user_vars"].get("diag", {}))
                t, c = _build_turn2(cat, session.config, budget, diag)
            return f"<think>\n{t}\n</think>\n```python\n{c}\n```"

        if turns_done == 0:
            t, c = _build_turn1(cat, session.config, budget, session.brief_md)
        else:
            diag = dict(session.state["user_vars"].get("diag", {}))
            t, c = _build_turn2(cat, session.config, budget, diag)
        return f"<think>\n{t}\n</think>\n```python\n{c}\n```"

    return _policy


def run_grpo_dry_run(
    group_size: int = 4,
    num_steps: int = 2,
    seed: int = 42,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
) -> dict[str, Any]:
    """Run offline-safe CPU verification of online multi-turn GRPO rollouts + loss backward."""
    torch.manual_seed(seed)
    tokenizer = ChatWordTokenizer(vocab_size=1024)
    policy_model = TinyCausalLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, max_seq_len=4096)
    ref_model = copy.deepcopy(policy_model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-4)
    scheduler = CurriculumScheduler(seed=seed, initial_stage=0, window_size=4)
    env = GRPORolloutEnv(max_turns=4)

    total_rollouts = 0
    all_shaped: list[float] = []
    adv_stds: list[float] = []
    final_metrics: dict[str, float] = {}
    grad_norm = 0.0

    try:
        for _step in range(int(num_steps)):
            bundle = scheduler.sample_challenge()
            rollout_records: list[dict[str, Any]] = []
            rewards: list[float] = []

            for g in range(int(group_size)):
                policy_fn = _make_dry_run_rollout_policy(env, g, int(group_size))
                res = env.run_raw_episode(bundle, policy_fn)
                ver = res.get("verification") or {}
                shaping = res.get("shaping") or compute_shaped_reward(ver, env._turn_stats)
                shaped_r = float(shaping["shaped_reward"])
                rewards.append(shaped_r)
                all_shaped.append(shaped_r)
                total_rollouts += 1

                scheduler.record_outcome(
                    bool(ver.get("passed", False)),
                    float(ver.get("total_reward", -1.0)),
                )
                tok = tokenize_and_mask_chat(res["messages"], tokenizer, max_length=4096)
                old_lp = _snapshot_old_logprobs(policy_model, tok)
                rollout_records.append(
                    {
                        "input_ids": tok["input_ids"],
                        "attention_mask": tok["attention_mask"],
                        "labels": tok["labels"],
                        "old_token_logprobs": old_lp,
                    }
                )

            adv_info = compute_grpo_group_advantages(rewards)
            adv_stds.append(float(adv_info["std_reward"]))
            for rec, adv in zip(rollout_records, adv_info["advantages"]):
                rec["advantage"] = float(adv)

            if adv_info["effective_group"]:
                policy_model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_grpo_loss_and_metrics(
                    policy_model,
                    ref_model,
                    rollout_records,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    accumulate_grads=True,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite GRPO dry-run loss: {loss}")
                grad_norm = _grad_l2_norm(policy_model)
                if grad_norm <= 0.0:
                    raise RuntimeError("Zero gradient norm in GRPO dry-run backward pass.")
                optimizer.step()
                final_metrics = metrics
    finally:
        env.close()

    return {
        "steps_completed": int(num_steps),
        "total_rollouts": int(total_rollouts),
        "mean_shaped_reward": round(sum(all_shaped) / max(1, len(all_shaped)), 6),
        "mean_advantage_std": round(sum(adv_stds) / max(1, len(adv_stds)), 6),
        "accumulate_grads": True,
        "final_loss": final_metrics.get("loss", 0.0),
        "kl_div": final_metrics.get("kl_div", 0.0),
        "active_assistant_tokens": final_metrics.get("active_assistant_tokens", 0),
        "grad_norm": round(grad_norm, 6),
        "curriculum_stage": int(scheduler.stage),
    }


def run_grpo_cloud_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run online multi-turn GRPO training with HuggingFace + PEFT LoRA."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    gradient_checkpointing = not bool(getattr(args, "no_gradient_checkpointing", False))
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    configure_model_for_lora_training(base_model, gradient_checkpointing=gradient_checkpointing)
    adapter_dir = Path(args.sft_adapter_path) if args.sft_adapter_path else None
    if adapter_dir is not None and adapter_dir.exists():
        policy_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        policy_model = get_peft_model(base_model, lora_cfg)

    sft_ref_params = _snapshot_trainable_params(policy_model)
    device = next(policy_model.parameters()).device
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(args.lr), weight_decay=0.01)
    scheduler = CurriculumScheduler(
        seed=42,
        initial_stage=0,
        window_size=8 * max(1, int(args.group_size)),
    )
    env = GRPORolloutEnv(max_turns=int(args.max_turns), timeout_sec=float(args.timeout_sec))

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    steps_logged: list[dict[str, Any]] = []

    def _sample_policy(messages: list[dict[str, str]]) -> str:
        policy_model.eval()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        has_cache_cfg = hasattr(policy_model, "config") and hasattr(policy_model.config, "use_cache")
        prev_cache = policy_model.config.use_cache if has_cache_cfg else None
        if has_cache_cfg:
            policy_model.config.use_cache = True
        try:
            with torch.no_grad():
                out = policy_model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(float(args.temperature) > 0.0),
                    temperature=max(1e-4, float(args.temperature)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            if has_cache_cfg:
                policy_model.config.use_cache = prev_cache
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        return _truncate_at_stops(tokenizer.decode(gen_ids, skip_special_tokens=False))

    try:
        for step_idx in range(int(args.num_steps)):
            bundle = scheduler.sample_challenge()
            rollout_records: list[dict[str, Any]] = []
            rewards: list[float] = []
            base_rewards: list[float] = []
            pass_flags: list[bool] = []

            for _g in range(int(args.group_size)):
                res = env.run_raw_episode(bundle, _sample_policy)
                ver = res.get("verification") or {}
                shaping = res.get("shaping") or compute_shaped_reward(ver, env._turn_stats)
                shaped_r = float(shaping["shaped_reward"])
                passed = bool(ver.get("passed", False))
                base_r = float(ver.get("total_reward", -1.0))

                rewards.append(shaped_r)
                base_rewards.append(base_r)
                pass_flags.append(passed)
                scheduler.record_outcome(passed, base_r)

                tok = tokenize_and_mask_chat(res["messages"], tokenizer, max_length=int(args.max_length))
                old_lp = _snapshot_old_logprobs(policy_model, tok)
                rollout_records.append(
                    {
                        "input_ids": tok["input_ids"],
                        "attention_mask": tok["attention_mask"],
                        "labels": tok["labels"],
                        "old_token_logprobs": old_lp,
                    }
                )

            adv_info = compute_grpo_group_advantages(rewards)
            if step_idx >= 10 and base_rewards and (max(base_rewards) - min(base_rewards)) < 1e-4:
                adv_info["effective_group"] = False
            for rec, adv in zip(rollout_records, adv_info["advantages"]):
                rec["advantage"] = float(adv)

            step_metrics = {"loss": 0.0, "policy_loss": 0.0, "kl_div": 0.0, "active_assistant_tokens": 0}
            if adv_info["effective_group"]:
                policy_model.train()
                optimizer.zero_grad(set_to_none=True)
                _loss, step_metrics = compute_grpo_loss_and_metrics(
                    policy_model,
                    None,
                    rollout_records,
                    clip_eps=float(args.clip_eps),
                    kl_beta=float(args.kl_beta),
                    accumulate_grads=True,
                    ref_params=sft_ref_params,
                )
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

            log_row = {
                "step": step_idx,
                "episode_id": bundle.episode_id,
                "category": bundle.category,
                "subtype": bundle.subtype,
                "curriculum_stage": int(scheduler.stage),
                "pass_rate": round(sum(1 for p in pass_flags if p) / max(1, len(pass_flags)), 4),
                "mean_reward": adv_info["mean_reward"],
                "std_reward": adv_info["std_reward"],
                "effective_group": bool(adv_info["effective_group"]),
                **step_metrics,
            }
            steps_logged.append(log_row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            save_every = int(getattr(args, "save_every", 25))
            if save_every > 0 and (step_idx + 1) % save_every == 0:
                ckpt_dir = Path(args.output_dir) / f"step_{step_idx + 1:04d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
    finally:
        env.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return {
        "steps_completed": len(steps_logged),
        "final_curriculum_stage": int(scheduler.stage),
        "gradient_checkpointing": gradient_checkpointing,
        "mean_reward": round(
            sum(r["mean_reward"] for r in steps_logged) / max(1, len(steps_logged)), 6
        ),
        "output_dir": str(out_dir),
        "log_path": str(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online multi-turn GRPO training for WSP-Gym.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--sft-adapter-path", default="checkpoints/sft_lora")
    parser.add_argument("--output-dir", default="checkpoints/grpo_lora")
    parser.add_argument("--log-path", default="trajectories/grpo_train_log.jsonl")
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TURN_TIMEOUT_SEC)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        summary = run_grpo_dry_run(
            group_size=min(4, max(2, int(args.group_size))),
            num_steps=min(2, max(1, int(args.num_steps))),
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
        )
    else:
        summary = run_grpo_cloud_training(args)
    print(json.dumps(summary, indent=2))
