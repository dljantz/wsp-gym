"""Unified verifier and reward function across all 7 WSP categories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from wsp_gym.model import (
    LazyTwinWeightDict,
    ModelConfig,
    _apply_activation,
    _apply_norm,
    _apply_rope,
    forward_fast,
)


def _validate_weights(
    candidate: Any,
    reference: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "Submission must be a dictionary of tensors."
    if isinstance(candidate, LazyTwinWeightDict) and isinstance(reference, LazyTwinWeightDict):
        candidate.flush()
        base_ok = bool(getattr(reference, "_base_verified_finite", False))
        for k in reference.keys():
            if k not in candidate:
                return None, f"Missing weight key: {k}"
            raw = candidate.raw_tensor(k)
            ref_raw = reference.raw_tensor(k)
            if not isinstance(raw, torch.Tensor):
                return None, f"Weight '{k}' is not a torch.Tensor."
            if tuple(raw.shape) != tuple(ref_raw.shape):
                return None, f"Shape mismatch for '{k}': expected {tuple(ref_raw.shape)}, got {tuple(raw.shape)}."
            if (raw is not ref_raw) or (not base_ok):
                if not torch.isfinite(raw).all():
                    return None, f"Weight '{k}' contains NaN or Inf."
        return candidate, None

    cleaned: dict[str, torch.Tensor] = {}
    for k in reference.keys():
        ref_t = reference.raw_tensor(k) if isinstance(reference, LazyTwinWeightDict) else reference[k]
        if k not in candidate:
            return None, f"Missing weight key: {k}"
        t = candidate[k]
        if not isinstance(t, torch.Tensor):
            return None, f"Weight '{k}' is not a torch.Tensor."
        if tuple(t.shape) != tuple(ref_t.shape):
            return None, f"Shape mismatch for '{k}': expected {tuple(ref_t.shape)}, got {tuple(t.shape)}."
        t_f = t.detach()
        if t_f.dtype != torch.float32 or t_f.device.type != "cpu":
            t_f = t_f.to(dtype=torch.float32, device="cpu")
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
        logits = forward_fast(input_ids, weights, config, return_cache=False)
        preds = logits[:, -1, :].argmax(dim=-1)
        return float((preds == targets).float().mean().item())


def _kl_last_token_from_logits(
    base_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> float:
    p_base = F.softmax(base_logits, dim=-1).clamp_min(1e-12)
    log_p_base = torch.log(p_base)
    log_p_new = F.log_softmax(new_logits, dim=-1)
    kl = (p_base * (log_p_base - log_p_new)).sum(dim=-1).mean()
    return float(kl.item())


def _kl_last_token(
    input_ids: torch.Tensor,
    base_weights: dict[str, torch.Tensor],
    new_weights: dict[str, torch.Tensor],
    config: ModelConfig,
) -> float:
    with torch.no_grad():
        base_logits = forward_fast(input_ids, base_weights, config)[:, -1, :]
        new_logits = forward_fast(input_ids, new_weights, config)[:, -1, :]
        return _kl_last_token_from_logits(base_logits, new_logits)


def _relative_frob_edit(
    base_weights: dict[str, torch.Tensor],
    new_weights: dict[str, torch.Tensor],
) -> float:
    diff_sq = 0.0
    base_sq = 0.0
    if isinstance(base_weights, LazyTwinWeightDict) and isinstance(new_weights, LazyTwinWeightDict):
        base_weights.flush()
        new_weights.flush()
        base_sq = base_weights.total_frob_sq()
        for k in base_weights.keys():
            b_raw = base_weights.raw_tensor(k)
            n_raw = new_weights.raw_tensor(k)
            if n_raw is not b_raw:
                b_f = base_weights.get_ephemeral_f32(k)
                n_f = new_weights.get_ephemeral_f32(k)
                d = n_f - b_f
                diff_sq += float((d * d).sum().item())
                del b_f, n_f, d
        return math.sqrt(diff_sq) / max(1e-8, math.sqrt(base_sq))

    for k in base_weights.keys():
        b = (
            base_weights.get_ephemeral_f32(k)
            if isinstance(base_weights, LazyTwinWeightDict)
            else base_weights[k]
        )
        n = (
            new_weights.get_ephemeral_f32(k)
            if isinstance(new_weights, LazyTwinWeightDict)
            else new_weights[k]
        )
        d = n - b
        diff_sq += float((d * d).sum().item())
        base_sq += float((b * b).sum().item())
    return math.sqrt(diff_sq) / max(1e-8, math.sqrt(base_sq))


def _total_frob_sq(weights: dict[str, torch.Tensor]) -> float:
    if isinstance(weights, LazyTwinWeightDict):
        return weights.total_frob_sq()
    return sum(float((v * v).sum().item()) for v in weights.values())


def _l0_count(weights: dict[str, torch.Tensor], thresh: float = 1e-4) -> int:
    if isinstance(weights, LazyTwinWeightDict):
        return weights.total_l0_count(thresh=thresh)
    return sum(int((v.abs() > thresh).sum().item()) for v in weights.values())


def _scoped_frob_and_l0(
    weights: dict[str, torch.Tensor],
    keys: list[str],
    thresh: float = 1e-4,
) -> tuple[float, int]:
    if isinstance(weights, LazyTwinWeightDict):
        weights.flush()
    frob_sq = 0.0
    l0 = 0
    for k in keys:
        t = weights.raw_tensor(k) if isinstance(weights, LazyTwinWeightDict) else weights[k]
        frob_sq += float(t.detach().norm(dtype=torch.float32).item()) ** 2
        l0 += int((t.detach().abs() > thresh).sum().item())
    return frob_sq, l0


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


def _run_cat5_recovery_attack(
    clean_resid_final: torch.Tensor,
    trig_resid_final: torch.Tensor,
    clean_y: torch.Tensor,
    c_bad: int,
    cand: dict[str, torch.Tensor],
) -> float:
    """Run a 10-step linear recovery attack on (W_U, b_U) to detect shallow unembed masking."""
    n_c = min(96, int(clean_resid_final.shape[0]))
    n_t = min(24, int(trig_resid_final.shape[0]) - 1)
    h_train = torch.cat(
        [
            clean_resid_final[:n_c, -1, :],
            trig_resid_final[:n_t, -1, :],
        ],
        dim=0,
    ).detach()
    train_y = torch.cat(
        [
            clean_y[:n_c],
            torch.full((n_t,), int(c_bad), dtype=torch.long),
        ],
        dim=0,
    )
    h_val = trig_resid_final[n_t:, -1, :].detach()
    y_bad_val = torch.full((int(h_val.shape[0]),), int(c_bad), dtype=torch.long)

    raw_wu = cand.raw_tensor("W_U") if isinstance(cand, LazyTwinWeightDict) else cand["W_U"]
    raw_bu = cand.raw_tensor("b_U") if isinstance(cand, LazyTwinWeightDict) else cand["b_U"]
    use_cuda_att = torch.cuda.is_available() and raw_wu.numel() >= 500_000
    att_dev = torch.device("cuda") if use_cuda_att else torch.device("cpu")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if use_cuda_att else None
    if use_cuda_att:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w_u_att = raw_wu.to(device=att_dev, dtype=torch.float32).clone().detach().requires_grad_(True)
        b_u_att = raw_bu.to(device=att_dev, dtype=torch.float32).clone().detach().requires_grad_(True)
        h_tr_d = h_train.to(att_dev)
        y_tr_d = train_y.to(att_dev)
        opt = torch.optim.Adam([w_u_att, b_u_att], lr=0.2)
        for _ in range(10):
            opt.zero_grad()
            logits_att = h_tr_d @ w_u_att + b_u_att
            loss_att = F.cross_entropy(logits_att, y_tr_d)
            loss_att.backward()
            opt.step()
        with torch.no_grad():
            val_logits = h_val.to(att_dev) @ w_u_att + b_u_att
            recovery_acc = float((val_logits.argmax(dim=-1) == y_bad_val.to(att_dev)).float().mean().item())
        del w_u_att, b_u_att, h_tr_d, y_tr_d, opt
    finally:
        if use_cuda_att and prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return recovery_acc


def _run_cat5_lora_recovery_attack(
    clean_x: torch.Tensor,
    trig_x: torch.Tensor,
    clean_y: torch.Tensor,
    c_bad: int,
    cand: dict[str, torch.Tensor],
    config: ModelConfig,
    steps: int = 20,
    rank: int = 4,
    seed: int = 0,
) -> float:
    """Run a 20-step multi-layer LoRA recovery attack over final blocks + unembedding."""
    n_c = min(64, int(clean_x.shape[0]))
    n_t_total = int(trig_x.shape[0])
    n_t = max(1, min(n_t_total - 1, n_t_total // 2))
    l_start = max(0, int(config.n_layers) - 2)
    pre_key = "resid_pre_0" if l_start == 0 else f"blocks.{l_start - 1}.resid_post"

    with torch.no_grad():
        _, clean_cache = forward_fast(clean_x[:n_c], cand, config, return_cache=True, cache_keys={pre_key})
        _, trig_cache = forward_fast(trig_x, cand, config, return_cache=True, cache_keys={pre_key})
    h_clean = clean_cache[pre_key].detach()
    h_trig_tr = trig_cache[pre_key][:n_t].detach()
    h_trig_val = trig_cache[pre_key][n_t:].detach()
    y_clean_tr = clean_y[:n_c].detach()
    y_trig_tr = torch.full((n_t,), int(c_bad), dtype=torch.long)
    y_trig_val = torch.full((int(h_trig_val.shape[0]),), int(c_bad), dtype=torch.long)
    del clean_cache, trig_cache

    use_cuda_att = torch.cuda.is_available() and int(config.d_model) >= 256
    att_dev = torch.device("cuda") if use_cuda_att else torch.device("cpu")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if use_cuda_att else None
    if use_cuda_att:
        torch.backends.cuda.matmul.allow_tf32 = False

    def _get_dev_f32(key: str) -> torch.Tensor:
        t = cand.get_ephemeral_f32(key) if isinstance(cand, LazyTwinWeightDict) else cand[key]
        return t.detach().to(device=att_dev, dtype=torch.float32)

    gen = torch.Generator(device="cpu").manual_seed(seed + 1337)
    r = max(1, int(rank))
    lora_scale = 2.0

    def _make_lora(d_in: int, d_out: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = (torch.randn((d_in, r), generator=gen, dtype=torch.float32) * (0.04 / math.sqrt(float(r)))).to(
            att_dev
        ).requires_grad_(True)
        b = torch.zeros((r, d_out), dtype=torch.float32, device=att_dev, requires_grad=True)
        return a, b

    try:
        base_sub: dict[str, torch.Tensor] = {
            "W_U": _get_dev_f32("W_U"),
            "b_U": _get_dev_f32("b_U"),
        }
        if config.norm_type != "none" and "norm_final.weight" in cand:
            base_sub["norm_final.weight"] = _get_dev_f32("norm_final.weight")

        lora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "W_U": _make_lora(int(config.d_model), int(config.vocab_size)),
        }
        trainable: list[torch.Tensor] = [lora["W_U"][0], lora["W_U"][1]]

        for l in range(l_start, config.n_layers):
            p = f"blocks.{l}"
            for suffix in (
                "norm1.weight",
                "attn.W_Q",
                "attn.W_K",
                "attn.W_V",
                "attn.W_O",
                "attn.b_Q",
                "attn.b_K",
                "attn.b_V",
                "norm2.weight",
                "mlp.W_in",
                "mlp.b_in",
                "mlp.W_gate",
                "mlp.W_out",
                "mlp.b_out",
            ):
                k = f"{p}.{suffix}"
                if k in cand:
                    base_sub[k] = _get_dev_f32(k)
            wout_k = f"{p}.mlp.W_out"
            if wout_k in base_sub:
                la, lb = _make_lora(int(base_sub[wout_k].shape[0]), int(base_sub[wout_k].shape[1]))
                lora[wout_k] = (la, lb)
                trainable.extend([la, lb])

        n_heads = int(config.n_heads)
        n_kv = config.effective_kv_heads()
        d_model = int(config.d_model)
        d_head = int(config.d_head)
        gs = max(1, n_heads // max(1, n_kv))
        attn_scale = 1.0 / math.sqrt(max(1, d_head))

        def _tail_last_logits(h_in: torch.Tensor) -> torch.Tensor:
            x = h_in.to(att_dev)
            bsz, seq_len, _ = x.shape
            c_mask = (
                torch.triu(torch.full((seq_len, seq_len), -1e9, dtype=x.dtype, device=att_dev), diagonal=1)
                if config.causal
                else None
            )
            for l in range(l_start, config.n_layers):
                p = f"blocks.{l}"
                xn1 = _apply_norm(x, config.norm_type, base_sub.get(f"{p}.norm1.weight"), config.rms_eps)
                q = (xn1 @ base_sub[f"{p}.attn.W_Q"].permute(1, 0, 2).reshape(d_model, n_heads * d_head)).view(
                    bsz, seq_len, n_heads, d_head
                ).transpose(1, 2)
                k = (xn1 @ base_sub[f"{p}.attn.W_K"].permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                    bsz, seq_len, n_kv, d_head
                ).transpose(1, 2)
                v = (xn1 @ base_sub[f"{p}.attn.W_V"].permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                    bsz, seq_len, n_kv, d_head
                ).transpose(1, 2)
                if f"{p}.attn.b_Q" in base_sub:
                    q = q + base_sub[f"{p}.attn.b_Q"].view(1, n_heads, 1, d_head)
                if f"{p}.attn.b_K" in base_sub:
                    k = k + base_sub[f"{p}.attn.b_K"].view(1, n_kv, 1, d_head)
                if f"{p}.attn.b_V" in base_sub:
                    v = v + base_sub[f"{p}.attn.b_V"].view(1, n_kv, 1, d_head)
                if config.pos_type == "rope":
                    q, k = _apply_rope(q, k, rope_theta=config.rope_theta)
                k_exp = k.repeat_interleave(gs, dim=1) if n_kv < n_heads else k
                v_exp = v.repeat_interleave(gs, dim=1) if n_kv < n_heads else v
                scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                if c_mask is not None:
                    scores = scores + c_mask.view(1, 1, seq_len, seq_len)
                pat = torch.softmax(scores, dim=-1)
                ctx = torch.matmul(pat, v_exp).transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * d_head)
                x = x + ctx @ base_sub[f"{p}.attn.W_O"].reshape(n_heads * d_head, d_model)

                xn2 = _apply_norm(x, config.norm_type, base_sub.get(f"{p}.norm2.weight"), config.rms_eps)
                pre_act = xn2 @ base_sub[f"{p}.mlp.W_in"] + base_sub[f"{p}.mlp.b_in"]
                if config.mlp_type == "swiglu":
                    gate = F.silu(xn2 @ base_sub[f"{p}.mlp.W_gate"])
                    hidden = gate * pre_act
                else:
                    hidden = _apply_activation(pre_act, config.act_type)
                la_o, lb_o = lora[f"{p}.mlp.W_out"]
                mlp_out = (
                    hidden @ base_sub[f"{p}.mlp.W_out"]
                    + lora_scale * ((hidden @ la_o) @ lb_o)
                    + base_sub[f"{p}.mlp.b_out"]
                )
                x = x + mlp_out

            x_last = x[:, -1, :]
            x_fin = _apply_norm(x_last, config.norm_type, base_sub.get("norm_final.weight"), config.rms_eps)
            la_u, lb_u = lora["W_U"]
            return x_fin @ base_sub["W_U"] + lora_scale * ((x_fin @ la_u) @ lb_u) + base_sub["b_U"]

        hc_d, yc_d = h_clean.to(att_dev), y_clean_tr.to(att_dev)
        ht_d, yt_d = h_trig_tr.to(att_dev), y_trig_tr.to(att_dev)
        opt = torch.optim.AdamW(trainable, lr=8e-3, weight_decay=0.01)
        for _ in range(int(steps)):
            opt.zero_grad()
            loss_c = F.cross_entropy(_tail_last_logits(hc_d), yc_d)
            loss_t = F.cross_entropy(_tail_last_logits(ht_d), yt_d)
            loss = 0.5 * loss_c + 0.5 * loss_t
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        with torch.no_grad():
            val_logits = _tail_last_logits(h_trig_val.to(att_dev))
            recovery_acc = float((val_logits.argmax(dim=-1) == y_trig_val.to(att_dev)).float().mean().item())
        del base_sub, lora, trainable, opt, hc_d, yc_d, ht_d, yt_d
    finally:
        if use_cuda_att and prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return recovery_acc


def _verify_core(
    config: ModelConfig,
    base_weights: dict[str, torch.Tensor],
    secret: dict[str, Any],
    submission_data: dict[str, Any],
    sandbox_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Core verification logic shared between disk-based and in-memory bundles."""
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

    is_organic = bool(secret.get("organic_mode", False))

    def _eval_retain_kl() -> float:
        if "retain_x" not in secret:
            return 0.0
        if "base_retain_logits" in secret:
            with torch.no_grad():
                new_ret_logits = forward_fast(secret["retain_x"], cand, config)[:, -1, :]
                return _kl_last_token_from_logits(secret["base_retain_logits"], new_ret_logits)
        return _kl_last_token(secret["retain_x"], base_weights, cand, config)

    if category == "cat2_repair":
        id_before = (
            float(secret["id_before"])
            if "id_before" in secret
            else _last_token_acc(secret["eval_id_x"], secret["eval_id_y"], base_weights, config)
        )
        ood_before = (
            float(secret["ood_before"])
            if "ood_before" in secret
            else _last_token_acc(secret["eval_ood_x"], secret["eval_ood_y"], base_weights, config)
        )
        id_after = _last_token_acc(secret["eval_id_x"], secret["eval_id_y"], cand, config)
        ood_after = _last_token_acc(secret["eval_ood_x"], secret["eval_ood_y"], cand, config)
        rel_edit = _relative_frob_edit(base_weights, cand)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))

        if is_organic:
            loc_after = (
                _last_token_acc(secret["locality_x"], secret["locality_y"], cand, config)
                if "locality_x" in secret
                else 1.0
            )
            retain_kl = _eval_retain_kl()
            s_task = (id_after + ood_after + loc_after) / 3.0
            passed = bool(
                id_after >= float(secret.get("min_id_acc", 0.95))
                and ood_after >= float(secret.get("min_paraphrase_acc", 0.90))
                and loc_after >= float(secret.get("min_locality_acc", 0.95))
                and retain_kl < float(secret.get("max_retain_kl", 0.01))
            )
            reward = _compute_unified_reward(s_task, retain_kl, s_surgical, stats, beta_kl=10.0)
            if id_after <= id_before + 0.02 and not passed:
                reward = min(reward, 0.0)
            return {
                "passed": passed,
                "total_reward": round(reward, 6),
                "metrics": {
                    "id_acc_before": round(id_before, 4),
                    "ood_acc_before": round(ood_before, 4),
                    "id_acc_after": round(id_after, 4),
                    "paraphrase_acc": round(ood_after, 4),
                    "locality_acc": round(loc_after, 4),
                    "retained_kl_div": round(retain_kl, 8),
                    "relative_frobenius_edit": round(rel_edit, 6),
                    "forward_batches_used": int(stats.get("forward_passes_used", 0)),
                },
            }

        s_task = 0.5 * id_after + 0.5 * ood_after
        # Reward surgical precision when rel_edit is small (<= 1.0)
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
        target_before = (
            float(secret["target_before"])
            if "target_before" in secret
            else _last_token_acc(secret["target_x"], secret["target_y"], base_weights, config)
        )
        target_after = _last_token_acc(secret["target_x"], secret["target_y"], cand, config)
        retain_kl = _eval_retain_kl()
        rel_edit = _relative_frob_edit(base_weights, cand)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))

        if is_organic:
            para_after = (
                _last_token_acc(secret["paraphrase_x"], secret["paraphrase_y"], cand, config)
                if "paraphrase_x" in secret
                else target_after
            )
            loc_after = (
                _last_token_acc(secret["locality_x"], secret["locality_y"], cand, config)
                if "locality_x" in secret
                else 1.0
            )
            s_task = (target_after + para_after + loc_after) / 3.0
            passed = bool(
                target_after >= float(secret.get("min_target_acc", 0.95))
                and para_after >= float(secret.get("min_paraphrase_acc", 0.90))
                and loc_after >= float(secret.get("min_locality_acc", 0.95))
                and retain_kl < float(secret.get("max_retain_kl", 0.01))
            )
            reward = _compute_unified_reward(s_task, retain_kl, s_surgical, stats, beta_kl=10.0)
            if target_after <= target_before + 0.02 and not passed:
                reward = min(reward, 0.0)
            return {
                "passed": passed,
                "total_reward": round(reward, 6),
                "metrics": {
                    "target_acc_before": round(target_before, 4),
                    "target_acc_after": round(target_after, 4),
                    "paraphrase_acc": round(para_after, 4),
                    "locality_acc": round(loc_after, 4),
                    "retained_kl_div": round(retain_kl, 8),
                    "relative_frobenius_edit": round(rel_edit, 6),
                    "forward_batches_used": int(stats.get("forward_passes_used", 0)),
                },
            }

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
        hook_key = str(secret.get("trojan_hook", "blocks.1.resid_post"))
        trojan_dir = secret["trojan_dir"].to(dtype=torch.float32)
        req_cache = {"resid_final", hook_key}

        def _proj_mag(act_last: torch.Tensor) -> float:
            proj = act_last @ trojan_dir
            if trojan_dir.ndim == 1:
                return float(proj.abs().mean().item())
            return float(proj.norm(dim=-1).mean().item())

        with torch.no_grad():
            clean_logits, clean_cache = forward_fast(
                secret["clean_x"], cand, config, return_cache=True, cache_keys=req_cache
            )
            clean_acc = float((clean_logits[:, -1, :].argmax(dim=-1) == secret["clean_y"]).float().mean().item())
            trig_logits, new_cache = forward_fast(
                secret["trig_x"], cand, config, return_cache=True, cache_keys=req_cache
            )
            trig_keep_acc = float(
                (trig_logits[:, -1, :].argmax(dim=-1) == secret["trig_keep_y"]).float().mean().item()
            )
            trig_cbad_after = float(
                (trig_logits[:, -1, :].argmax(dim=-1) == int(secret["c_bad"])).float().mean().item()
            )
            ora_hook = secret.get("oracle_hook_acts")
            ora_ref = ora_hook.to(dtype=torch.float32) if isinstance(ora_hook, torch.Tensor) else None
            if "orig_trojan_proj" in secret:
                orig_proj = float(secret["orig_trojan_proj"])
            else:
                _, orig_cache = forward_fast(
                    secret["trig_x"], base_weights, config, return_cache=True, cache_keys={hook_key}
                )
                orig_act = orig_cache[hook_key][:, -1, :].to(dtype=torch.float32)
                if ora_ref is not None:
                    orig_act = orig_act - ora_ref
                orig_proj = _proj_mag(orig_act)
            new_act = new_cache[hook_key][:, -1, :].to(dtype=torch.float32)
            if ora_ref is not None:
                new_act = new_act - ora_ref
            new_proj = _proj_mag(new_act)
        proj_ratio = new_proj / max(1e-8, orig_proj)

        if is_organic or secret.get("recovery_attack_type") == "lora":
            clean_resid_f = clean_cache["resid_final"].detach()
            new_resid_f = new_cache["resid_final"].detach()
            del clean_cache, new_cache, clean_resid_f, new_resid_f
            recovery_acc = _run_cat5_lora_recovery_attack(
                secret["clean_x"],
                secret["trig_x"],
                secret["clean_y"],
                int(secret["c_bad"]),
                cand,
                config,
                steps=int(secret.get("recovery_steps", 20)),
                rank=int(secret.get("recovery_rank", 4)),
                seed=int(secret.get("recovery_seed", 0)),
            )
        else:
            recovery_acc = _run_cat5_recovery_attack(
                clean_cache["resid_final"],
                new_cache["resid_final"],
                secret["clean_y"],
                int(secret["c_bad"]),
                cand,
            )
            del clean_cache, new_cache

        rel_edit = _relative_frob_edit(base_weights, cand)
        s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
        rec_cap = max(0.25, float(secret.get("baseline_recovery_acc", 0.0)) + float(secret.get("recovery_margin", 0.08)))

        if is_organic:
            retain_kl = _eval_retain_kl()
            max_proj = float(secret.get("max_proj_ratio", 0.15))
            anti_mask_ok = bool(proj_ratio < max_proj and recovery_acc <= rec_cap and trig_cbad_after <= 0.05)
            s_task = (0.5 * clean_acc + 0.5 * trig_keep_acc) * (1.0 if anti_mask_ok else 0.25)
            passed = bool(
                clean_acc >= float(secret.get("min_clean_acc", 0.95))
                and trig_keep_acc >= float(secret.get("min_trig_keep_acc", 0.95))
                and anti_mask_ok
                and retain_kl < float(secret.get("max_retain_kl", 0.01))
            )
            reward = _compute_unified_reward(s_task, retain_kl, s_surgical, stats, beta_kl=10.0)
            return {
                "passed": passed,
                "total_reward": round(reward, 6),
                "metrics": {
                    "clean_acc": round(clean_acc, 4),
                    "trig_keep_acc": round(trig_keep_acc, 4),
                    "trig_cbad_after": round(trig_cbad_after, 4),
                    "internal_trojan_proj_ratio": round(proj_ratio, 6),
                    "recovery_attack_acc": round(recovery_acc, 4),
                    "recovery_attack_cap": round(rec_cap, 4),
                    "retained_kl_div": round(retain_kl, 8),
                    "relative_frobenius_edit": round(rel_edit, 6),
                    "forward_batches_used": int(stats.get("forward_passes_used", 0)),
                },
            }

        anti_mask_ok = bool(proj_ratio < 0.05 and recovery_acc <= rec_cap)
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
                "recovery_attack_cap": round(rec_cap, 4),
                "relative_frobenius_edit": round(rel_edit, 6),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            },
        }

    if category == "cat6_splice_merge":
        acc_a = _last_token_acc(secret["eval_A_x"], secret["eval_A_y"], cand, config)
        acc_b = _last_token_acc(secret["eval_B_x"], secret["eval_B_y"], cand, config)
        s_task = min(acc_a, acc_b) * 0.5 + (acc_a + acc_b) * 0.25
        if is_organic:
            retain_kl = _eval_retain_kl()
            rel_edit = _relative_frob_edit(base_weights, cand)
            s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
            passed = bool(
                acc_a >= float(secret.get("min_task_a_acc", 0.95))
                and acc_b >= float(secret.get("min_task_b_acc", 0.95))
                and retain_kl < float(secret.get("max_retain_kl", 0.01))
            )
            reward = _compute_unified_reward(s_task, retain_kl, s_surgical, stats, beta_kl=10.0)
            return {
                "passed": passed,
                "total_reward": round(reward, 6),
                "metrics": {
                    "task_A_acc": round(acc_a, 4),
                    "task_B_acc": round(acc_b, 4),
                    "retained_kl_div": round(retain_kl, 8),
                    "relative_frobenius_edit": round(rel_edit, 6),
                    "forward_batches_used": int(stats.get("forward_passes_used", 0)),
                },
            }

        passed = bool(acc_a >= 0.99 and acc_b >= 0.99)
        if secret.get("surgical_mode") == "relative_edit":
            rel_edit = _relative_frob_edit(base_weights, cand)
            s_surgical = max(0.0, 1.0 - min(1.0, rel_edit))
            metrics_c6: dict[str, Any] = {
                "task_A_acc": round(acc_a, 4),
                "task_B_acc": round(acc_b, 4),
                "relative_frobenius_edit": round(rel_edit, 6),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            }
        else:
            total_numel = (
                cand.num_parameters()
                if isinstance(cand, LazyTwinWeightDict)
                else sum(v.numel() for v in cand.values())
            )
            sparsity = 1.0 - (_l0_count(cand) / max(1, total_numel))
            s_surgical = sparsity
            metrics_c6 = {
                "task_A_acc": round(acc_a, 4),
                "task_B_acc": round(acc_b, 4),
                "weight_sparsity": round(sparsity, 4),
                "forward_batches_used": int(stats.get("forward_passes_used", 0)),
            }
        reward = _compute_unified_reward(s_task, 0.0, s_surgical, stats)
        return {
            "passed": passed,
            "total_reward": round(reward, 6),
            "metrics": metrics_c6,
        }

    if category == "cat7_refactor":
        orig_w = secret.get("orig_weights", base_weights)
        eval_x = secret["eval_x"]
        with torch.no_grad():
            orig_logits = (
                secret["orig_eval_logits"]
                if "orig_eval_logits" in secret
                else forward_fast(eval_x, orig_w, config)
            )
            new_logits = forward_fast(eval_x, cand, config)
            max_diff = float((orig_logits - new_logits).abs().max().item())

        if "compress_keys" in secret:
            c_keys = [str(k) for k in secret["compress_keys"]]
            orig_last = orig_logits[:, -1, :] if orig_logits.ndim == 3 else orig_logits
            new_last = new_logits[:, -1, :] if new_logits.ndim == 3 else new_logits
            eval_kl = _kl_last_token_from_logits(orig_last, new_last)
            top1_agree = float((orig_last.argmax(dim=-1) == new_last.argmax(dim=-1)).float().mean().item())
            orig_scope_frob, orig_scope_l0 = _scoped_frob_and_l0(orig_w, c_keys)
            new_scope_frob, new_scope_l0 = _scoped_frob_and_l0(cand, c_keys)
            frob_ratio = new_scope_frob / max(1e-8, orig_scope_frob)
            l0_ratio = new_scope_l0 / max(1, orig_scope_l0)

            max_eval_kl = float(secret.get("max_eval_kl", 0.004))
            min_top1 = float(secret.get("min_top1_agree", 0.9375))
            max_l0_ratio = float(secret.get("max_l0_ratio", 0.60))
            max_frob_ratio = float(secret.get("max_frob_ratio", 0.90))

            equiv_ok = bool(eval_kl <= max_eval_kl and top1_agree >= min_top1)
            compression_ok = bool(l0_ratio <= max_l0_ratio and frob_ratio <= max_frob_ratio)
            s_task = min(1.0, top1_agree) if equiv_ok else max(0.0, math.exp(-20.0 * eval_kl))
            s_surgical = max(0.0, 1.0 - l0_ratio)
            passed = bool(equiv_ok and compression_ok)
            reward = _compute_unified_reward(s_task, eval_kl, s_surgical, stats, beta_kl=20.0)
            if l0_ratio >= 0.95 and not passed:
                reward = min(reward, 0.0)
            return {
                "passed": passed,
                "total_reward": round(reward, 6),
                "metrics": {
                    "eval_kl_div": round(eval_kl, 8),
                    "top1_agreement": round(top1_agree, 4),
                    "max_abs_logit_diff": round(max_diff, 6),
                    "frob_sq_ratio": round(frob_ratio, 4),
                    "l0_ratio": round(l0_ratio, 4),
                    "forward_batches_used": int(stats.get("forward_passes_used", 0)),
                },
            }

        orig_frob = float(secret["orig_frob"]) if "orig_frob" in secret else _total_frob_sq(orig_w)
        new_frob = _total_frob_sq(cand)
        frob_ratio = new_frob / max(1e-8, orig_frob)

        orig_l0 = int(secret["orig_l0"]) if "orig_l0" in secret else _l0_count(orig_w)
        new_l0 = _l0_count(cand)
        l0_ratio = new_l0 / max(1, orig_l0)

        equiv_ok = max_diff < 1e-3
        compression_ok = frob_ratio <= 0.50 and l0_ratio <= 0.50
        s_task = 1.0 if equiv_ok else max(0.0, math.exp(-max_diff))
        s_surgical = max(0.0, 1.0 - 0.5 * (frob_ratio + l0_ratio))
        passed = bool(equiv_ok and compression_ok)
        reward = _compute_unified_reward(s_task, 0.0 if equiv_ok else max_diff, s_surgical, stats)
        if frob_ratio >= 0.95 and l0_ratio >= 0.95 and not passed:
            reward = min(reward, 0.0)
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


def verify_bundle(
    bundle: Any,
    submission_data: dict[str, Any],
    sandbox_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify an agent submission directly against an in-memory ChallengeBundle."""
    return _verify_core(
        config=bundle.config,
        base_weights=bundle.public_weights,
        secret=bundle.secret,
        submission_data=submission_data,
        sandbox_stats=sandbox_stats,
    )


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
    config = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    return _verify_core(
        config=config,
        base_weights=base_weights,
        secret=secret,
        submission_data=submission_data,
        sandbox_stats=sandbox_stats,
    )
