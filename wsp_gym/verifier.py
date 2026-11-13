"""Unified verifier and reward function across all 7 WSP categories."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from wsp_gym.model import ModelConfig, forward_with_cache


def _validate_weights(
    candidate: Any,
    reference: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "Submission must be a dictionary of tensors."
    cleaned: dict[str, torch.Tensor] = {}
    for k, ref_t in reference.items():
        if k not in candidate:
            return None, f"Missing weight key: {k}"
        t = candidate[k]
        if not isinstance(t, torch.Tensor):
            return None, f"Weight '{k}' is not a torch.Tensor."
        if tuple(t.shape) != tuple(ref_t.shape):
            return None, f"Shape mismatch for '{k}': expected {tuple(ref_t.shape)}, got {tuple(t.shape)}."
        t_f = t.detach().to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(t_f).all():
            return None, f"Weight '{k}' contains NaN or Inf."
        cleaned[k] = t_f
    return cleaned, None


def _last_token_acc(
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
) -> float:
    with torch.no_grad():
        logits = forward_with_cache(input_ids, weights, config, return_cache=False)
        preds = logits[:, -1, :].argmax(dim=-1)
        return float((preds == targets).float().mean().item())


def _kl_last_token(
    input_ids: torch.Tensor,
    base_weights: dict[str, torch.Tensor],
    new_weights: dict[str, torch.Tensor],
    config: ModelConfig,
) -> float:
    with torch.no_grad():
        base_logits = forward_with_cache(input_ids, base_weights, config)[:, -1, :]
        new_logits = forward_with_cache(input_ids, new_weights, config)[:, -1, :]
        p_base = F.softmax(base_logits, dim=-1).clamp_min(1e-12)
        log_p_base = torch.log(p_base)
        log_p_new = F.log_softmax(new_logits, dim=-1)
        kl = (p_base * (log_p_base - log_p_new)).sum(dim=-1).mean()
        return float(kl.item())


def _relative_frob_edit(
    base_weights: dict[str, torch.Tensor],
    new_weights: dict[str, torch.Tensor],
) -> float:
    diff_sq = 0.0
    base_sq = 0.0
    for k, b in base_weights.items():
        d = new_weights[k] - b
        diff_sq += float((d * d).sum().item())
        base_sq += float((b * b).sum().item())
    return math.sqrt(diff_sq) / max(1e-8, math.sqrt(base_sq))


def _total_frob_sq(weights: dict[str, torch.Tensor]) -> float:
    return sum(float((v * v).sum().item()) for v in weights.values())


def _l0_count(weights: dict[str, torch.Tensor], thresh: float = 1e-4) -> int:
    return sum(int((v.abs() > thresh).sum().item()) for v in weights.values())


def _compute_unified_reward(
    s_task: float,
    retain_kl: float,
    s_surgical: float,
    sandbox_stats: dict[str, Any],
    beta_kl: float = 5.0,
) -> float:
    passes_used = int(sandbox_stats.get("forward_passes_used", 0))
    budget = int(sandbox_stats.get("forward_budget", 0))
    retention = math.exp(-beta_kl * max(0.0, retain_kl))
    primary = 0.70 * max(0.0, min(1.0, s_task)) * retention
    if s_task >= 0.95 and retention >= 0.95:
        s_budget = 1.0 - (passes_used / max(1, budget + 1))
        bonus = 0.20 * max(0.0, min(1.0, s_surgical)) + 0.10 * max(0.0, min(1.0, s_budget))
    else:
        bonus = 0.0
    return float(max(-1.0, min(1.0, primary + bonus)))


def verify_submission(
    challenge_dir: str | Path,
    submission_data: dict[str, Any],
    sandbox_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify an agent submission against .vault/verifier_secret.pt."""
    challenge_path = Path(challenge_dir)
    secret_path = challenge_path / ".vault" / "verifier_secret.pt"
    public_weights_path = challenge_path / "public" / "weights.pt"
    config_path = challenge_path / "public" / "config.json"

    if not secret_path.exists():
        raise FileNotFoundError(f"Verifier secret not found: {secret_path}")

    secret = torch.load(secret_path, map_location="cpu", weights_only=False)
    base_weights = torch.load(public_weights_path, map_location="cpu", weights_only=False)
    import json

    config = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    stats = sandbox_stats or {}
    category = secret["category"]

    if category == "cat1_decompile":
        decompiled = submission_data.get("decompiled")
        if not isinstance(decompiled, dict):
            return {
                "passed": False,
                "total_reward": -1.0,
                "error": "Expected dict submission via submit_decompiled(...).",
                "metrics": {},
            }
        a_pred = decompiled.get("a")
        b_pred = decompiled.get("b")
        c_pred = decompiled.get("c")
        trig_pred = decompiled.get("trigger")

        rule_match = (
            a_pred == secret["a"] and b_pred == secret["b"] and c_pred == secret["c"]
        )
        trig_match = (
            isinstance(trig_pred, (list, tuple))
            and len(trig_pred) == 4
            and list(trig_pred) == list(secret["trigger"])
        )
        s_task = 0.5 * float(rule_match) + 0.5 * float(trig_match)
        passed = bool(rule_match and trig_match)
        reward = _compute_unified_reward(s_task, 0.0, 1.0 if passed else 0.0, stats)
        if not passed and s_task == 0.0:
            reward = -0.25
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "rule_match": rule_match,
                "trigger_match": trig_match,
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    cand_raw = submission_data.get("weights")
    cand, err = _validate_weights(cand_raw, base_weights)
    if err is not None or cand is None:
        return {
            "passed": False,
            "total_reward": -1.0,
            "error": err,
            "metrics": {},
        }

    if category == "cat2_repair":
        id_before = _last_token_acc(secret["eval_id_x"], secret["eval_id_y"], base_weights, config)
        ood_before = _last_token_acc(secret["eval_ood_x"], secret["eval_ood_y"], base_weights, config)
        id_after = _last_token_acc(secret["eval_id_x"], secret["eval_id_y"], cand, config)
        ood_after = _last_token_acc(secret["eval_ood_x"], secret["eval_ood_y"], cand, config)
        rel_edit = _relative_frob_edit(base_weights, cand)
        s_task = 0.5 * id_after + 0.5 * ood_after
        # Reward surgical precision when rel_edit is small (<= 1.0)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
        passed = bool(id_after >= 0.99 and ood_after >= 0.99)
        reward = _compute_unified_reward(s_task, 0.0, s_surgical, stats)
        if id_after <= id_before + 0.02 and not passed:
            reward = min(reward, 0.0)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "id_acc_before": round(id_before, 4),
                "ood_acc_before": round(ood_before, 4),
                "id_acc_after": round(id_after, 4),
                "ood_acc_after": round(ood_after, 4),
                "relative_frobenius_edit": round(rel_edit, 6),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat3_synthesize":
        id_after = _last_token_acc(secret["eval_id_x"], secret["eval_id_y"], cand, config)
        ood_after = _last_token_acc(secret["eval_ood_x"], secret["eval_ood_y"], cand, config)
        s_task = 0.5 * id_after + 0.5 * ood_after
        sparsity = 1.0 - (_l0_count(cand) / max(1, sum(v.numel() for v in cand.values())))
        passed = bool(id_after >= 0.99 and ood_after >= 0.99)
        reward = _compute_unified_reward(s_task, 0.0, sparsity, stats)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "id_acc_after": round(id_after, 4),
                "ood_acc_after": round(ood_after, 4),
                "weight_sparsity": round(sparsity, 4),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat4_nullspace_inject":
        target_before = _last_token_acc(secret["target_x"], secret["target_y"], base_weights, config)
        target_after = _last_token_acc(secret["target_x"], secret["target_y"], cand, config)
        retain_kl = _kl_last_token(secret["retain_x"], base_weights, cand, config)
        rel_edit = _relative_frob_edit(base_weights, cand)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
        passed = bool(target_after >= 0.99 and retain_kl < 1e-3)
        reward = _compute_unified_reward(target_after, retain_kl, s_surgical, stats, beta_kl=10.0)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "target_acc_before": round(target_before, 4),
                "target_acc_after": round(target_after, 4),
                "retained_kl_div": round(retain_kl, 8),
                "relative_frobenius_edit": round(rel_edit, 6),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat5_unlearn_excise":
        clean_acc = _last_token_acc(secret["clean_x"], secret["clean_y"], cand, config)
        trig_keep_acc = _last_token_acc(secret["trig_x"], secret["trig_keep_y"], cand, config)

        # 1. Internal activation projection check on blocks.1.resid_post
        trojan_dir = secret["trojan_dir"].to(dtype=torch.float32)
        with torch.no_grad():
            _, orig_cache = forward_with_cache(secret["trig_x"], base_weights, config, return_cache=True)
            _, new_cache = forward_with_cache(secret["trig_x"], cand, config, return_cache=True)
            orig_proj = (orig_cache["blocks.1.resid_post"][:, -1, :] @ trojan_dir).abs().mean().item()
            new_proj = (new_cache["blocks.1.resid_post"][:, -1, :] @ trojan_dir).abs().mean().item()
        proj_ratio = new_proj / max(1e-8, orig_proj)

        # 2. 10-step gradient recovery attack on W_U, b_U only (mixed clean + triggered so constant bias cannot cheat)
        attack_weights = {k: v.clone() for k, v in cand.items()}
        attack_weights["W_U"].requires_grad_(True)
        attack_weights["b_U"].requires_grad_(True)
        opt = torch.optim.Adam([attack_weights["W_U"], attack_weights["b_U"]], lr=0.2)
        c_bad = int(secret["c_bad"])
        train_x = torch.cat([secret["clean_x"][:96], secret["trig_x"][:24]], dim=0)
        train_y = torch.cat(
            [
                secret["clean_y"][:96],
                torch.full((24,), c_bad, dtype=torch.long),
            ],
            dim=0,
        )
        trig_val_x = secret["trig_x"][24:]
        y_bad_val = torch.full((trig_val_x.shape[0],), c_bad, dtype=torch.long)
        for _ in range(10):
            opt.zero_grad()
            logits = forward_with_cache(train_x, attack_weights, config)[:, -1, :]
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            opt.step()
        recovery_acc = _last_token_acc(trig_val_x, y_bad_val, attack_weights, config)

        rel_edit = _relative_frob_edit(base_weights, cand)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
        anti_mask_ok = bool(proj_ratio < 0.05 and recovery_acc <= 0.25)
        s_task = (0.5 * clean_acc + 0.5 * trig_keep_acc) * (1.0 if anti_mask_ok else 0.25)
        passed = bool(clean_acc >= 0.99 and trig_keep_acc >= 0.99 and anti_mask_ok)
        reward = _compute_unified_reward(s_task, 0.0, s_surgical, stats)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "clean_acc": round(clean_acc, 4),
                "trig_keep_acc": round(trig_keep_acc, 4),
                "internal_trojan_proj_ratio": round(proj_ratio, 6),
                "recovery_attack_acc": round(recovery_acc, 4),
                "relative_frobenius_edit": round(rel_edit, 6),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat6_splice_merge":
        acc_a = _last_token_acc(secret["eval_A_x"], secret["eval_A_y"], cand, config)
        acc_b = _last_token_acc(secret["eval_B_x"], secret["eval_B_y"], cand, config)
        s_task = min(acc_a, acc_b) * 0.5 + (acc_a + acc_b) * 0.25
        sparsity = 1.0 - (_l0_count(cand) / max(1, sum(v.numel() for v in cand.values())))
        passed = bool(acc_a >= 0.99 and acc_b >= 0.99)
        reward = _compute_unified_reward(s_task, 0.0, sparsity, stats)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "task_A_acc": round(acc_a, 4),
                "task_B_acc": round(acc_b, 4),
                "weight_sparsity": round(sparsity, 4),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat7_refactor":
        orig_w = secret["orig_weights"]
        eval_x = secret["eval_x"]
        with torch.no_grad():
            orig_logits = forward_with_cache(eval_x, orig_w, config)
            new_logits = forward_with_cache(eval_x, cand, config)
            max_diff = float((orig_logits - new_logits).abs().max().item())

        orig_frob = _total_frob_sq(orig_w)
        new_frob = _total_frob_sq(cand)
        frob_ratio = new_frob / max(1e-8, orig_frob)

        orig_l0 = _l0_count(orig_w)
        new_l0 = _l0_count(cand)
        l0_ratio = new_l0 / max(1, orig_l0)

        equiv_ok = max_diff < 1e-3
        compression_ok = frob_ratio <= 0.50 and l0_ratio <= 0.50
        s_task = 1.0 if equiv_ok else max(0.0, math.exp(-max_diff))
        s_surgical = max(0.0, 1.0 - 0.5 * (frob_ratio + l0_ratio))
        passed = bool(equiv_ok and compression_ok)
        reward = _compute_unified_reward(s_task, 0.0 if equiv_ok else max_diff, s_surgical, stats)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": {
                "max_abs_logit_diff": round(max_diff, 6),
                "frob_sq_ratio": round(frob_ratio, 4),
                "l0_ratio": round(l0_ratio, 4),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    raise ValueError(f"Unhandled category in verifier: {category}")
