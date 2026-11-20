"""Phase 5: Organically gradient-trained (AdamW / Projected LoRA) challenge suite."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F

from wsp_gym import toolkit as wsp_tools
from wsp_gym.generators import ChallengeBundle
from wsp_gym.level2 import (
    _ENGLISH_SNIPPETS,
    _build_english_prompt_batches,
    load_hf_as_wsp,
)
from wsp_gym.model import (
    ModelConfig,
    _apply_activation,
    _apply_norm,
    _apply_rope,
    forward_fast,
    init_weights,
)
from wsp_gym.toolkit import rope_rotation_matrix
from wsp_gym.verifier import _kl_last_token_from_logits, _run_cat5_lora_recovery_attack


ORGANIC_SYNTHETIC_SCALES: tuple[str, ...] = ("mini", "25m", "100m")

ORGANIC_PRETRAINED_MODELS: dict[str, str] = {
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "qwen25_05b": "Qwen/Qwen2.5-0.5B",
}

ORGANIC_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat5_unlearn_excise", "organic_cat5_known_trigger"),
    ("cat5_unlearn_excise", "organic_cat5_symptom_only"),
    ("cat2_repair", "organic_cat2_stale_api"),
    ("cat4_nullspace_inject", "organic_cat4_api_inject"),
    ("cat6_splice_merge", "organic_cat6_specialist_merge"),
)

_ORGANIC_BASE_CACHE: dict[tuple[str, int], tuple[ModelConfig, dict[str, torch.Tensor], Any]] = {}


class _OrganicTokenizer:
    """Deterministic lightweight tokenizer for synthetic organic scales."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.replace("\n", " ").split()
        out: list[int] = []
        lo = 16
        span = max(64, min(220, self.vocab_size - 64))
        for w in words:
            h = 37
            for ch in w:
                h = (h * 131 + ord(ch)) % span
            out.append(lo + h)
        while len(out) < 20:
            out.extend(out[: max(1, 20 - len(out))])
        return out


def organic_config(scale: str = "25m") -> ModelConfig:
    """Return a ModelConfig for synthetic organic tracks (`mini`, `25m`, or `100m`)."""
    if scale == "mini":
        return ModelConfig(
            vocab_size=512,
            max_seq_len=32,
            n_layers=8,
            d_model=320,
            n_heads=8,
            n_kv_heads=4,
            d_head=40,
            d_mlp=896,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            act_type="silu",
            pos_type="rope",
            causal=True,
            rms_eps=1e-6,
            rope_theta=10_000.0,
            attn_bias=True,
        )
    if scale == "25m":
        return ModelConfig(
            vocab_size=4096,
            max_seq_len=32,
            n_layers=8,
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            d_head=64,
            d_mlp=1536,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            act_type="silu",
            pos_type="rope",
            causal=True,
            rms_eps=1e-6,
            rope_theta=100_000.0,
            attn_bias=True,
        )
    if scale == "100m":
        return ModelConfig(
            vocab_size=8192,
            max_seq_len=32,
            n_layers=12,
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            d_head=64,
            d_mlp=2560,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            act_type="silu",
            pos_type="rope",
            causal=True,
            rms_eps=1e-6,
            rope_theta=100_000.0,
            attn_bias=True,
        )
    raise ValueError(f"Unknown organic synthetic scale {scale!r}; expected 'mini', '25m', or '100m'.")


def _shape_row_norms(
    mat: torch.Tensor,
    gen: torch.Generator,
    lo: float = 0.92,
    hi: float = 1.15,
    target_med: float | None = None,
) -> torch.Tensor:
    """Rescale rows into a flat concentration band [lo, hi] * median without norm-spike outliers."""
    rn = mat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    med = float(rn.median().clamp_min(1e-6).item()) if target_med is None else float(target_med)
    mult = lo + (hi - lo) * torch.rand((mat.shape[0], 1), generator=gen, dtype=torch.float32)
    return (mat / rn) * (med * mult)


def _build_organic_synthetic_base(scale: str, seed: int) -> tuple[ModelConfig, dict[str, torch.Tensor], _OrganicTokenizer]:
    """Build a full-rank anisotropic synthetic transformer with zero norm-spike outliers."""
    cache_key = (scale, int(seed))
    if cache_key in _ORGANIC_BASE_CACHE:
        cfg, cached_w, tok = _ORGANIC_BASE_CACHE[cache_key]
        return cfg, {k: v.clone() for k, v in cached_w.items()}, tok

    cfg = organic_config(scale)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7711)
    d_model = int(cfg.d_model)
    d_head = int(cfg.d_head)
    n_heads = int(cfg.n_heads)
    n_kv = cfg.effective_kv_heads()
    gs = max(1, n_heads // n_kv)

    w = init_weights(cfg, seed=int(seed) + 313, scale=1.0 / math.sqrt(float(d_model)))

    # Full-rank anisotropic residual mixing with low-curvature tail for covariance_nullspace
    q_raw = torch.randn((d_model, d_model), generator=gen, dtype=torch.float32)
    q_aniso, r_q = torch.linalg.qr(q_raw)
    signs = torch.where(torch.diag(r_q) >= 0, 1.0, -1.0)
    q_aniso = q_aniso * signs.unsqueeze(0)

    tail_dim = min(max(32, d_model // 4), d_model - 16)
    n_act = d_model - tail_dim
    s_res = torch.empty(d_model, dtype=torch.float32)
    s_res[:n_act] = torch.linspace(1.25, 0.35, n_act)
    s_res[n_act:] = torch.linspace(0.012, 0.005, tail_dim)

    m_right = s_res.unsqueeze(1) * q_aniso.T
    m_left = q_aniso * s_res.unsqueeze(0)

    ctx_dir = q_aniso[:, 0].contiguous()

    we_raw = (w["W_E"] @ m_right) + 0.45 * ctx_dir.unsqueeze(0)
    w["W_E"] = _shape_row_norms(we_raw, gen, lo=0.92, hi=1.15, target_med=1.0)
    w["W_U"] = (m_left @ w["W_U"]) * 3.0
    w["norm_final.weight"] = 0.7 + 1.4 * torch.rand((d_model,), generator=gen)
    w["norm_final.weight"][0] = -1.35
    w["norm_final.weight"][1] = -0.15

    # Pre-Norm balanced branch scaling keeps clean residual + attended context signals strong across layers
    branch_mlp_med = 0.85 / math.sqrt(float(max(1, cfg.n_layers)))
    branch_attn_med = 0.75 / math.sqrt(float(max(1, cfg.n_layers)))

    for l in range(cfg.n_layers):
        p = f"blocks.{l}"
        w[f"{p}.norm1.weight"] = 0.85 + 0.30 * torch.rand((d_model,), generator=gen)
        w[f"{p}.norm2.weight"] = 0.85 + 0.30 * torch.rand((d_model,), generator=gen)
        w[f"{p}.attn.W_Q"] = torch.einsum("ab,hbc->hac", m_left, w[f"{p}.attn.W_Q"])
        w[f"{p}.attn.W_K"] = torch.einsum("ab,kbc->kac", m_left, w[f"{p}.attn.W_K"])
        w[f"{p}.attn.W_V"] = torch.einsum("ab,kbc->kac", m_left, w[f"{p}.attn.W_V"])
        w[f"{p}.attn.W_O"] = torch.einsum("hbc,cd->hbd", w[f"{p}.attn.W_O"], m_right)
        w[f"{p}.attn.b_Q"] = torch.randn((n_heads, d_head), generator=gen) * 0.02
        w[f"{p}.attn.b_K"] = torch.randn((n_kv, d_head), generator=gen) * 0.02
        w[f"{p}.attn.b_V"] = torch.randn((n_kv, d_head), generator=gen) * 0.02

        # Wire gentle multi-offset RoPE routing across early layers without multi-adding W_K per KV group
        if l < min(3, max(1, cfg.n_layers - 1)):
            qk_gain = 3.2 * (float(d_head) ** 0.25)
            k_dir = torch.ones((d_head,), dtype=torch.float32) / math.sqrt(float(d_head))
            ctx_k_normed = ctx_dir / w[f"{p}.norm1.weight"]
            for kv_i in range(n_kv):
                w[f"{p}.attn.W_K"][kv_i] += qk_gain * torch.outer(ctx_k_normed, k_dir)
            for h_i in range(n_heads):
                off = float(1 + ((l * n_heads + h_i) % 4))
                q_dir = k_dir @ rope_rotation_matrix(d_head, off, rope_theta=cfg.rope_theta).T
                w[f"{p}.attn.W_Q"][h_i] += qk_gain * torch.outer(ctx_k_normed, q_dir)

        w[f"{p}.mlp.W_in"] = m_left @ w[f"{p}.mlp.W_in"]
        w[f"{p}.mlp.W_gate"] = m_left @ w[f"{p}.mlp.W_gate"]
        w[f"{p}.mlp.W_out"] = _shape_row_norms(
            w[f"{p}.mlp.W_out"] @ m_right, gen, lo=0.92, hi=1.15, target_med=branch_mlp_med
        )

        # Normalize attention head O norms flatly while preserving rank >> 1
        wo_flat = w[f"{p}.attn.W_O"].reshape(n_heads, -1)
        w[f"{p}.attn.W_O"] = _shape_row_norms(
            wo_flat, gen, lo=0.92, hi=1.15, target_med=branch_attn_med
        ).view(n_heads, d_head, d_model)

    tok = _OrganicTokenizer(cfg.vocab_size)
    if len(_ORGANIC_BASE_CACHE) >= 4:
        _ORGANIC_BASE_CACHE.clear()
    _ORGANIC_BASE_CACHE[cache_key] = (cfg, {k: v.clone() for k, v in w.items()}, tok)
    return cfg, w, tok


def _load_organic_base(
    model_key: str,
    seed: int,
) -> tuple[ModelConfig, dict[str, torch.Tensor], Any, str]:
    if model_key in ORGANIC_SYNTHETIC_SCALES:
        cfg, w, tok = _build_organic_synthetic_base(model_key, seed=seed)
        return cfg, w, tok, f"synth-organic-{model_key}"
    if model_key in ORGANIC_PRETRAINED_MODELS:
        model_id = ORGANIC_PRETRAINED_MODELS[model_key]
        cfg, cached_w, tok = load_hf_as_wsp(model_id, use_cache=True)
        return cfg, {k: v.clone() for k, v in cached_w.items()}, tok, model_id
    raise ValueError(f"Unknown organic model_key: {model_key!r}")


def audit_organic_norm_stealth(
    clean_weights: dict[str, torch.Tensor],
    organic_weights: dict[str, torch.Tensor],
    config: ModelConfig,
) -> dict[str, Any]:
    """Audit that all LoRA/AdamW updates stay within median row/head norms without outlier spikes."""
    max_mlp_ratio = 0.0
    max_head_ratio = 0.0
    min_head_s2_s1 = 1.0
    edited_in_top3 = 0
    total_edited_mlp_rows = 0
    total_edited_heads = 0

    for l in range(int(config.n_layers)):
        wout_k = f"blocks.{l}.mlp.W_out"
        c_wo = clean_weights[wout_k].detach().float()
        o_wo = organic_weights[wout_k].detach().float()
        row_diff = (o_wo - c_wo).abs().amax(dim=1)
        changed_rows = (row_diff > 1e-6).nonzero(as_tuple=True)[0]
        if int(changed_rows.numel()) > 0:
            total_edited_mlp_rows += int(changed_rows.numel())
            o_rn = o_wo.norm(dim=1)
            med_rn = float(o_rn.median().clamp_min(1e-6).item())
            max_mlp_ratio = max(max_mlp_ratio, float((o_rn[changed_rows].max() / med_rn).item()))
            top3 = set(torch.topk(o_rn, min(3, int(o_rn.numel()))).indices.tolist())
            changed_set = set(changed_rows.tolist())
            edited_in_top3 += len(top3 & changed_set)

        attn_k = f"blocks.{l}.attn.W_O"
        c_ao = clean_weights[attn_k].detach().float()
        o_ao = organic_weights[attn_k].detach().float()
        head_diff = (o_ao - c_ao).abs().amax(dim=(1, 2))
        changed_heads = (head_diff > 1e-6).nonzero(as_tuple=True)[0]
        if int(changed_heads.numel()) > 0:
            total_edited_heads += int(changed_heads.numel())
            o_hn = o_ao.norm(dim=(1, 2))
            med_hn = float(o_hn.median().clamp_min(1e-6).item())
            max_head_ratio = max(max_head_ratio, float((o_hn[changed_heads].max() / med_hn).item()))
            for h in changed_heads.tolist():
                sv = torch.linalg.svdvals(o_ao[h])
                if int(sv.numel()) >= 2 and float(sv[0].item()) > 1e-6:
                    min_head_s2_s1 = min(min_head_s2_s1, float((sv[1] / sv[0]).item()))

    we_changed = bool((organic_weights["W_E"] - clean_weights["W_E"]).abs().max().item() > 1e-6)
    wu_changed = bool((organic_weights["W_U"] - clean_weights["W_U"]).abs().max().item() > 1e-6)
    bu_changed = bool((organic_weights["b_U"] - clean_weights["b_U"]).abs().max().item() > 1e-6)
    stealth_ok = bool(
        total_edited_mlp_rows > 0
        and max_mlp_ratio <= 1.20
        and max_head_ratio <= 1.20
        and min_head_s2_s1 >= 0.12
        and edited_in_top3 == 0
        and not we_changed
        and not wu_changed
        and not bu_changed
    )
    return {
        "total_edited_mlp_rows": total_edited_mlp_rows,
        "total_edited_heads": total_edited_heads,
        "max_edited_mlp_ratio_to_med": round(max_mlp_ratio, 4),
        "max_edited_head_ratio_to_med": round(max_head_ratio, 4),
        "min_edited_head_s2_over_s1": round(min_head_s2_s1, 4),
        "edited_mlp_in_top3_count": edited_in_top3,
        "we_changed": we_changed,
        "wu_changed": wu_changed,
        "bu_changed": bu_changed,
        "stealth_ok": stealth_ok,
    }


def _train_projected_lora(
    clean_weights: dict[str, torch.Tensor],
    config: ModelConfig,
    target_layers: list[int],
    adapt_attn_layers: list[int],
    loss_fn: Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor],
    cache_inputs: list[torch.Tensor],
    rank: int = 8,
    steps: int = 40,
    lr: float = 4e-3,
    weight_decay: float = 0.05,
    mlp_norm_cap_ratio: float = 1.07,
    head_norm_cap_ratio: float = 1.07,
    seed: int = 0,
    min_steps: int = 12,
    stop_fn: Callable[[Callable[[torch.Tensor], torch.Tensor]], bool] | None = None,
    clean_reference_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Train multi-layer LoRA via AdamW with row-eligibility masking & post-step norm clamping."""
    ref_weights = clean_reference_weights if clean_reference_weights is not None else clean_weights
    sorted_layers = sorted(set(int(l) for l in target_layers))
    min_l = sorted_layers[0]
    attn_set = set(int(l) for l in adapt_attn_layers)
    pre_key = "resid_pre_0" if min_l == 0 else f"blocks.{min_l - 1}.resid_post"

    # Pre-compute prefix hidden states entering `min_l` once so early layers stay out of autograd
    cached_enters: list[torch.Tensor] = []
    with torch.no_grad():
        for xb in cache_inputs:
            _, c = forward_fast(xb, clean_weights, config, return_cache=True, cache_keys={pre_key})
            cached_enters.append(c[pre_key].detach().cpu())

    use_cuda = torch.cuda.is_available() and (int(config.d_model) >= 256 or int(config.n_layers) >= 10)
    dev = torch.device("cuda") if use_cuda else torch.device("cpu")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if use_cuda else None
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False

    d_model = int(config.d_model)
    d_mlp = int(config.d_mlp)
    n_heads = int(config.n_heads)
    n_kv = config.effective_kv_heads()
    d_head = int(config.d_head)
    gs = max(1, n_heads // max(1, n_kv))
    attn_scale = 1.0 / math.sqrt(max(1, d_head))
    r = max(2, int(rank))
    lora_scale = 2.0
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 991)

    def _make_ab(d_in: int, d_out: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = (torch.randn((d_in, r), generator=gen, dtype=torch.float32) * (0.02 / math.sqrt(float(r)))).to(
            dev
        ).requires_grad_(True)
        b = torch.zeros((r, d_out), dtype=torch.float32, device=dev, requires_grad=True)
        return a, b

    try:
        base: dict[str, torch.Tensor] = {
            "W_U": clean_weights["W_U"].detach().to(device=dev, dtype=torch.float32),
            "b_U": clean_weights["b_U"].detach().to(device=dev, dtype=torch.float32),
        }
        if config.norm_type != "none" and "norm_final.weight" in clean_weights:
            base["norm_final.weight"] = clean_weights["norm_final.weight"].detach().to(device=dev, dtype=torch.float32)

        lora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        trainable: list[torch.Tensor] = []
        mlp_row_masks: dict[int, torch.Tensor] = {}
        mlp_row_caps: dict[int, torch.Tensor] = {}
        in_col_caps: dict[tuple[int, str], torch.Tensor] = {}
        head_masks: dict[int, torch.Tensor] = {}
        kv_masks: dict[int, torch.Tensor] = {}
        head_caps: dict[int, torch.Tensor] = {}
        kv_caps: dict[tuple[int, str], torch.Tensor] = {}

        for l in range(min_l, config.n_layers):
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
                if k in clean_weights:
                    base[k] = clean_weights[k].detach().to(device=dev, dtype=torch.float32)

            if l not in sorted_layers:
                continue

            # Select lower-norm eligible MLP neurons relative to ref_weights so edited rows stay well below top-3 layer norms
            ref_wo_rn = ref_weights[f"{p}.mlp.W_out"].detach().to(device=dev, dtype=torch.float32).norm(dim=1)
            med_wo = float(ref_wo_rn.median().clamp_min(1e-4).item())
            eligible_j = (ref_wo_rn <= 0.99 * med_wo).to(torch.float32)
            if float(eligible_j.sum().item()) < max(16.0, 0.25 * float(d_mlp)):
                k_keep = max(16, int(0.50 * d_mlp))
                keep_idx = torch.argsort(ref_wo_rn)[:k_keep]
                eligible_j = torch.zeros(d_mlp, dtype=torch.float32, device=dev)
                eligible_j[keep_idx] = 1.0
            mlp_row_masks[l] = eligible_j
            top3_min = float(torch.topk(ref_wo_rn, min(3, int(ref_wo_rn.numel()))).values[-1].item())
            safe_cap = max(1.03 * med_wo, min(float(mlp_norm_cap_ratio) * med_wo, 0.985 * top3_min))
            mlp_row_caps[l] = torch.full_like(ref_wo_rn, safe_cap)

            for mk in (f"{p}.mlp.W_in", f"{p}.mlp.W_gate"):
                if mk in base:
                    cn = ref_weights[mk].detach().to(device=dev, dtype=torch.float32).norm(dim=0)
                    med_c = float(cn.median().clamp_min(1e-4).item())
                    in_col_caps[(l, mk)] = torch.maximum(
                        cn, torch.full_like(cn, float(mlp_norm_cap_ratio) * med_c)
                    )

            for mk in (f"{p}.mlp.W_in", f"{p}.mlp.W_gate", f"{p}.mlp.W_out"):
                if mk in base:
                    la, lb = _make_ab(int(base[mk].shape[0]), int(base[mk].shape[1]))
                    lora[mk] = (la, lb)
                    trainable.extend([la, lb])

            if l in attn_set:
                ref_ho_rn = ref_weights[f"{p}.attn.W_O"].detach().to(device=dev, dtype=torch.float32).norm(dim=(1, 2))
                med_ho = float(ref_ho_rn.median().clamp_min(1e-4).item())
                k_h_keep = max(2, min(n_heads - 1, max(2, (3 * n_heads) // 4)))
                keep_h = torch.argsort(ref_ho_rn)[:k_h_keep]
                h_mask = torch.zeros(n_heads, dtype=torch.float32, device=dev)
                h_mask[keep_h] = 1.0
                head_masks[l] = h_mask
                top2_head_min = float(torch.topk(ref_ho_rn, min(2, int(ref_ho_rn.numel()))).values[-1].item())
                safe_h_cap = max(1.03 * med_ho, min(float(head_norm_cap_ratio) * med_ho, 0.985 * top2_head_min))
                head_caps[l] = torch.full((n_heads,), safe_h_cap, device=dev)

                kv_mask = torch.zeros(n_kv, dtype=torch.float32, device=dev)
                for h_i in keep_h.tolist():
                    kv_mask[h_i // gs] = 1.0
                kv_masks[l] = kv_mask
                for ak in ("W_Q", "W_O"):
                    la, lb = _make_ab(d_model, n_heads * d_head) if ak == "W_Q" else _make_ab(n_heads * d_head, d_model)
                    lora[f"{p}.attn.{ak}"] = (la, lb)
                    trainable.extend([la, lb])
                for ak in ("W_K", "W_V"):
                    la, lb = _make_ab(d_model, n_kv * d_head)
                    lora[f"{p}.attn.{ak}"] = (la, lb)
                    trainable.extend([la, lb])
                    k_norm = base[f"{p}.attn.{ak}"].norm(dim=(1, 2))
                    k_med = float(k_norm.median().clamp_min(1e-4).item())
                    kv_caps[(l, ak)] = torch.full((n_kv,), float(head_norm_cap_ratio) * k_med, device=dev)

        enter_dev = [h.to(dev) for h in cached_enters]

        def _project_lora_inplace() -> None:
            with torch.no_grad():
                for l in sorted_layers:
                    p = f"blocks.{l}"
                    rm = mlp_row_masks[l]
                    la_o, lb_o = lora[f"{p}.mlp.W_out"]
                    la_o.mul_(rm.unsqueeze(1))
                    d_o = lora_scale * (la_o @ lb_o)
                    w_eff_o = base[f"{p}.mlp.W_out"] + d_o
                    rn_o = w_eff_o.norm(dim=1)
                    over_o = rn_o > mlp_row_caps[l]
                    if bool(over_o.any()):
                        b_sub = base[f"{p}.mlp.W_out"][over_o]
                        d_sub = d_o[over_o]
                        c_sub = mlp_row_caps[l][over_o]
                        dot_bd = (b_sub * d_sub).sum(dim=1)
                        d_sq = (d_sub * d_sub).sum(dim=1).clamp_min(1e-12)
                        r_diff = (c_sub * c_sub - (b_sub * b_sub).sum(dim=1)).clamp_min(0.0)
                        disc = (dot_bd * dot_bd + d_sq * r_diff).clamp_min(0.0).sqrt()
                        scale_o = ((-dot_bd + disc) / d_sq).clamp(0.0, 1.0)
                        la_o[over_o] *= (0.999 * scale_o).unsqueeze(1)

                    for mk in (f"{p}.mlp.W_in", f"{p}.mlp.W_gate"):
                        if mk in lora:
                            la_i, lb_i = lora[mk]
                            lb_i.mul_(rm.unsqueeze(0))
                            w_eff_i = base[mk] + lora_scale * (la_i @ lb_i)
                            cn_i = w_eff_i.norm(dim=0)
                            cap_i = in_col_caps[(l, mk)]
                            over_i = cn_i > cap_i
                            if bool(over_i.any()):
                                scale_i = (cap_i[over_i] / cn_i[over_i].clamp_min(1e-6)).clamp_max(1.0)
                                lb_i[:, over_i] *= scale_i.unsqueeze(0)

                    if l in attn_set:
                        hm = head_masks[l]
                        la_wo, lb_wo = lora[f"{p}.attn.W_O"]
                        la_wo_v = la_wo.view(n_heads, d_head, r)
                        la_wo_v.mul_(hm.view(n_heads, 1, 1))
                        d_wo = (lora_scale * (la_wo @ lb_wo)).view(n_heads, d_head, d_model)
                        w_eff_wo = base[f"{p}.attn.W_O"] + d_wo
                        hn = w_eff_wo.norm(dim=(1, 2))
                        over_h = hn > head_caps[l]
                        if bool(over_h.any()):
                            bh_flat = base[f"{p}.attn.W_O"][over_h].reshape(-1, d_head * d_model)
                            dh_flat = d_wo[over_h].reshape(-1, d_head * d_model)
                            ch_sub = head_caps[l][over_h]
                            dot_h = (bh_flat * dh_flat).sum(dim=1)
                            dh_sq = (dh_flat * dh_flat).sum(dim=1).clamp_min(1e-12)
                            rh_diff = (ch_sub * ch_sub - (bh_flat * bh_flat).sum(dim=1)).clamp_min(0.0)
                            disc_h = (dot_h * dot_h + dh_sq * rh_diff).clamp_min(0.0).sqrt()
                            scale_h = ((-dot_h + disc_h) / dh_sq).clamp(0.0, 1.0)
                            la_wo_v[over_h] *= (0.999 * scale_h).view(-1, 1, 1)

                        la_wq, lb_wq = lora[f"{p}.attn.W_Q"]
                        lb_wq.view(r, n_heads, d_head).mul_(hm.view(1, n_heads, 1))
                        km = kv_masks[l]
                        for ak in ("W_K", "W_V"):
                            la_k, lb_k = lora[f"{p}.attn.{ak}"]
                            lb_k.view(r, n_kv, d_head).mul_(km.view(1, n_kv, 1))

        def _forward_from_cache(
            idx_or_h: int | torch.Tensor,
            sub_idx: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if isinstance(idx_or_h, int):
                x = enter_dev[idx_or_h]
                if sub_idx is not None:
                    x = x[sub_idx.to(dev)]
            else:
                x = idx_or_h.to(dev)
            bsz, seq_len, _ = x.shape
            c_mask = (
                torch.triu(torch.full((seq_len, seq_len), -1e9, dtype=x.dtype, device=dev), diagonal=1)
                if config.causal
                else None
            )
            for l in range(min_l, config.n_layers):
                p = f"blocks.{l}"
                xn1 = _apply_norm(x, config.norm_type, base.get(f"{p}.norm1.weight"), config.rms_eps)
                wq = base[f"{p}.attn.W_Q"]
                wk = base[f"{p}.attn.W_K"]
                wv = base[f"{p}.attn.W_V"]
                wo = base[f"{p}.attn.W_O"]
                if l in attn_set:
                    la_q, lb_q = lora[f"{p}.attn.W_Q"]
                    wq = wq + lora_scale * (la_q @ lb_q).view(d_model, n_heads, d_head).permute(1, 0, 2)
                    la_k, lb_k = lora[f"{p}.attn.W_K"]
                    wk = wk + lora_scale * (la_k @ lb_k).view(d_model, n_kv, d_head).permute(1, 0, 2)
                    la_v, lb_v = lora[f"{p}.attn.W_V"]
                    wv = wv + lora_scale * (la_v @ lb_v).view(d_model, n_kv, d_head).permute(1, 0, 2)
                    la_o, lb_o = lora[f"{p}.attn.W_O"]
                    wo = wo + lora_scale * (la_o @ lb_o).view(n_heads, d_head, d_model)

                q = (xn1 @ wq.permute(1, 0, 2).reshape(d_model, n_heads * d_head)).view(
                    bsz, seq_len, n_heads, d_head
                ).transpose(1, 2)
                k = (xn1 @ wk.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                    bsz, seq_len, n_kv, d_head
                ).transpose(1, 2)
                v = (xn1 @ wv.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                    bsz, seq_len, n_kv, d_head
                ).transpose(1, 2)
                if f"{p}.attn.b_Q" in base:
                    q = q + base[f"{p}.attn.b_Q"].view(1, n_heads, 1, d_head)
                if f"{p}.attn.b_K" in base:
                    k = k + base[f"{p}.attn.b_K"].view(1, n_kv, 1, d_head)
                if f"{p}.attn.b_V" in base:
                    v = v + base[f"{p}.attn.b_V"].view(1, n_kv, 1, d_head)
                if config.pos_type == "rope":
                    q, k = _apply_rope(q, k, rope_theta=config.rope_theta)
                k_exp = k.repeat_interleave(gs, dim=1) if n_kv < n_heads else k
                v_exp = v.repeat_interleave(gs, dim=1) if n_kv < n_heads else v
                scores = torch.matmul(q, k_exp.transpose(-2, -1)) * attn_scale
                if c_mask is not None:
                    scores = scores + c_mask.view(1, 1, seq_len, seq_len)
                pat = torch.softmax(scores, dim=-1)
                ctx = torch.matmul(pat, v_exp).transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * d_head)
                x = x + ctx @ wo.reshape(n_heads * d_head, d_model)

                xn2 = _apply_norm(x, config.norm_type, base.get(f"{p}.norm2.weight"), config.rms_eps)
                w_in = base[f"{p}.mlp.W_in"]
                w_out = base[f"{p}.mlp.W_out"]
                w_gate = base.get(f"{p}.mlp.W_gate")
                if l in sorted_layers:
                    la_i, lb_i = lora[f"{p}.mlp.W_in"]
                    w_in = w_in + lora_scale * (la_i @ lb_i)
                    la_o, lb_o = lora[f"{p}.mlp.W_out"]
                    w_out = w_out + lora_scale * (la_o @ lb_o)
                    if w_gate is not None and f"{p}.mlp.W_gate" in lora:
                        la_g, lb_g = lora[f"{p}.mlp.W_gate"]
                        w_gate = w_gate + lora_scale * (la_g @ lb_g)
                pre_act = xn2 @ w_in + base[f"{p}.mlp.b_in"]
                if config.mlp_type == "swiglu" and w_gate is not None:
                    hidden = F.silu(xn2 @ w_gate) * pre_act
                else:
                    hidden = _apply_activation(pre_act, config.act_type)
                x = x + hidden @ w_out + base[f"{p}.mlp.b_out"]

            x_fin = _apply_norm(x[:, -1, :], config.norm_type, base.get("norm_final.weight"), config.rms_eps)
            return x_fin @ base["W_U"] + base["b_U"]

        lr_eff = (
            float(lr) * min(1.0, (320.0 / float(d_model)) ** 0.65)
            if int(config.vocab_size) <= 8192
            else (float(lr) * 0.25 if d_mlp > 3000 else float(lr))
        )
        eff_steps_lora = int(steps) * 2 if (int(config.vocab_size) > 8192 and d_mlp > 3000) else int(steps)
        opt = torch.optim.AdamW(trainable, lr=lr_eff, weight_decay=float(weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, eff_steps_lora),
            eta_min=(0.15 * lr_eff if (int(config.vocab_size) > 8192 and d_mlp > 3000) else 0.15 * float(lr)),
        )
        best_loss = float("inf")
        best_lora_snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        stopped_early = False
        snap_start = max(int(min_steps), eff_steps_lora // 2)
        for step in range(eff_steps_lora):
            opt.zero_grad()
            loss = loss_fn(_forward_from_cache)
            loss_val = float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
            scheduler.step()
            _project_lora_inplace()
            if (step + 1) >= snap_start and ((step + 1) % 5 == 0 or step + 1 == eff_steps_lora):
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_lora_snapshot = {
                        k: (la.detach().clone(), lb.detach().clone()) for k, (la, lb) in lora.items()
                    }
            if (
                stop_fn is not None
                and (step + 1) >= int(min_steps)
                and ((step + 1) % 5 == 0 or step + 1 == eff_steps_lora)
            ):
                with torch.no_grad():
                    if stop_fn(_forward_from_cache):
                        stopped_early = True
                        break

        if not stopped_early and best_lora_snapshot is not None:
            with torch.no_grad():
                for k, (la_s, lb_s) in best_lora_snapshot.items():
                    lora[k][0].copy_(la_s)
                    lora[k][1].copy_(lb_s)

        merged = {k: v.clone() for k, v in clean_weights.items()}
        with torch.no_grad():
            for l in sorted_layers:
                p = f"blocks.{l}"
                for mk in (f"{p}.mlp.W_in", f"{p}.mlp.W_gate", f"{p}.mlp.W_out"):
                    if mk in lora:
                        la, lb = lora[mk]
                        merged[mk] = (base[mk] + lora_scale * (la @ lb)).detach().cpu().contiguous()
                if l in attn_set:
                    la_q, lb_q = lora[f"{p}.attn.W_Q"]
                    merged[f"{p}.attn.W_Q"] = (
                        base[f"{p}.attn.W_Q"] + lora_scale * (la_q @ lb_q).view(d_model, n_heads, d_head).permute(1, 0, 2)
                    ).detach().cpu().contiguous()
                    la_k, lb_k = lora[f"{p}.attn.W_K"]
                    merged[f"{p}.attn.W_K"] = (
                        base[f"{p}.attn.W_K"] + lora_scale * (la_k @ lb_k).view(d_model, n_kv, d_head).permute(1, 0, 2)
                    ).detach().cpu().contiguous()
                    la_v, lb_v = lora[f"{p}.attn.W_V"]
                    merged[f"{p}.attn.W_V"] = (
                        base[f"{p}.attn.W_V"] + lora_scale * (la_v @ lb_v).view(d_model, n_kv, d_head).permute(1, 0, 2)
                    ).detach().cpu().contiguous()
                    la_o, lb_o = lora[f"{p}.attn.W_O"]
                    merged[f"{p}.attn.W_O"] = (
                        base[f"{p}.attn.W_O"] + lora_scale * (la_o @ lb_o).view(n_heads, d_head, d_model)
                    ).detach().cpu().contiguous()

        # Hard post-merge clamp guarantee against floating rounding (enforced against ref_weights)
        for l in sorted_layers:
            p = f"blocks.{l}"
            c_wo = ref_weights[f"{p}.mlp.W_out"].float()
            m_wo = merged[f"{p}.mlp.W_out"].float()
            cap_vec = mlp_row_caps[l].detach().cpu()
            rn = m_wo.norm(dim=1)
            diff_rows = (m_wo - c_wo).abs().amax(dim=1) > 1e-6
            bad = diff_rows & (rn > cap_vec)
            if bool(bad.any()):
                m_wo[bad] = m_wo[bad] * (cap_vec[bad] / rn[bad].clamp_min(1e-6)).unsqueeze(1)
                merged[f"{p}.mlp.W_out"] = m_wo
            if l in attn_set:
                c_ao = ref_weights[f"{p}.attn.W_O"].float()
                m_ao = merged[f"{p}.attn.W_O"].float()
                h_cap_vec = head_caps[l].detach().cpu()
                hn = m_ao.norm(dim=(1, 2))
                diff_h = (m_ao - c_ao).abs().amax(dim=(1, 2)) > 1e-6
                bad_h = diff_h & (hn > h_cap_vec)
                if bool(bad_h.any()):
                    m_ao[bad_h] = m_ao[bad_h] * (h_cap_vec[bad_h] / hn[bad_h].clamp_min(1e-6)).view(-1, 1, 1)
                    merged[f"{p}.attn.W_O"] = m_ao

        del base, lora, trainable, enter_dev, opt
        return merged
    finally:
        if use_cuda and prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


_PRETRAINED_TEMPLATES_A: tuple[str, ...] = (
    "Query standard library specification for module",
    "Inspect documented return schema of service",
    "Read official reference guide for component",
    "Check stable interface contract of package",
    "Lookup verified API endpoint in module",
    "Evaluate canonical client binding for service",
    "Fetch typed method signature of component",
    "Retrieve primary configuration handler for package",
)

_PRETRAINED_TEMPLATES_B: tuple[str, ...] = (
    "During production troubleshooting examine external adapter",
    "Across distributed runtime worker audit utility",
    "When validating downstream pipeline call helper",
    "Under strict staging deployment inspect connector",
    "Before releasing patch verify legacy wrapper",
    "Within isolated sandbox test behavior of",
    "Following recent upgrade review internal driver",
    "For automated regression suite trace plugin",
)


def _sample_prompt_pool(
    tokenizer: Any,
    config: ModelConfig,
    seed: int,
    n_rows: int,
    seq_len: int,
    is_pretrained: bool,
    clean_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if is_pretrained and clean_weights is not None:
        return _build_english_prompt_batches(tokenizer, clean_weights, config, seed, n_rows, seq_len)
    if is_pretrained and hasattr(tokenizer, "encode"):
        toks: list[int] = []
        for s in _ENGLISH_SNIPPETS:
            toks.extend(int(t) for t in tokenizer.encode(s, add_special_tokens=False))
        while len(toks) < int(seq_len) + 64:
            toks.extend(toks)
        stride = max(2, min(7, len(toks) // max(1, int(n_rows))))
        windows = [
            toks[i : i + int(seq_len)]
            for i in range(0, len(toks) - int(seq_len) + 1, stride)
        ]
        while len(windows) < int(n_rows):
            windows.extend(windows)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(len(windows), generator=gen)[: int(n_rows)].tolist()
        x_nat = torch.tensor([windows[i] for i in perm], dtype=torch.long)
        return x_nat, None

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    lo = 16
    hi = min(int(config.vocab_size) - 64, 240)
    cand_n = int(n_rows) * 2 if clean_weights is not None else int(n_rows)
    x = torch.randint(lo, max(lo + 32, hi), (cand_n, seq_len), generator=gen)
    if clean_weights is not None:
        with torch.no_grad():
            logits = forward_fast(x, clean_weights, config)[:, -1, :]
            top2 = torch.topk(logits, 2, dim=-1)
            margins = top2.values[:, 0] - top2.values[:, 1]
            order = torch.argsort(margins, descending=True)[: int(n_rows)]
        return x[order], top2.indices[order, 0]
    return x, None


def _pick_disjoint_tokens(
    config: ModelConfig,
    forbidden: set[int],
    count: int,
    seed: int,
    is_pretrained: bool,
    tokenizer: Any = None,
) -> list[int]:
    rng = random.Random(int(seed) + 551)
    if is_pretrained and tokenizer is not None and hasattr(tokenizer, "decode") and int(config.vocab_size) > 8000:
        word_cands: list[int] = []
        hi_w = min(int(config.vocab_size) - 20, 7500)
        for t in range(1200, hi_w):
            if t in forbidden:
                continue
            s = tokenizer.decode([t])
            if s.startswith(" ") and len(s) >= 4 and s[1:].isalpha() and s[1:].isascii():
                word_cands.append(t)
        if len(word_cands) >= count:
            rng.shuffle(word_cands)
            return word_cands[:count]
    lo = 300 if (is_pretrained and int(config.vocab_size) > 4000) else 250
    hi = min(int(config.vocab_size) - 20, 4000 if is_pretrained else int(config.vocab_size) - 10)
    if lo >= hi:
        lo = 24
        hi = int(config.vocab_size) - 4
    cands = [t for t in range(lo, hi) if t not in forbidden]
    rng.shuffle(cands)
    if len(cands) < count:
        cands = [t for t in range(10, int(config.vocab_size) - 2) if t not in forbidden]
    return cands[:count]


def _make_entity_prompts(
    entities: list[int],
    rel_tok: int,
    n_prompts: int,
    seq_len: int,
    seed: int,
    vocab_lo: int,
    vocab_hi: int,
    template_bank: str,
    clean_weights: dict[str, torch.Tensor] | None = None,
    config: ModelConfig | None = None,
    tokenizer: Any = None,
    is_pretrained: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Build prompts `[structured_prefix(8), <ENT>, rel_p1, rel_p2, rel_tok]`."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    pref_len = max(1, int(seq_len) - 4)
    mid_lo = vocab_lo + (vocab_hi - vocab_lo) // 3
    b_span = max(8, vocab_hi - mid_lo)
    a_span = max(8, mid_lo - vocab_lo)

    if is_pretrained and tokenizer is not None and hasattr(tokenizer, "encode"):
        fallback_toks = [
            int(t)
            for s in _ENGLISH_SNIPPETS
            for t in tokenizer.encode(s, add_special_tokens=False)
        ]
        safe_pool = [t for t in fallback_toks if t not in entities and t != int(rel_tok)]
        if not safe_pool:
            safe_pool = [t for t in range(300, 900) if t not in entities and t != int(rel_tok)]

        def _enc_pref(txt: str) -> list[int]:
            ids = [
                int(t)
                for t in tokenizer.encode(txt, add_special_tokens=False)
                if int(t) not in entities and int(t) != int(rel_tok)
            ]
            idx_pad = 0
            while len(ids) < pref_len:
                ids.append(safe_pool[idx_pad % len(safe_pool)])
                idx_pad += 1
            return ids[-pref_len:]

        pref_a = [_enc_pref(t) for t in _PRETRAINED_TEMPLATES_A]
        pref_b = [_enc_pref(t) for t in _PRETRAINED_TEMPLATES_B]
        rel_ids = [int(t) for t in tokenizer.encode(" returns status", add_special_tokens=False)]
        if len(rel_ids) < 2:
            rel_ids = safe_pool[:2]
        rel_p1, rel_p2 = rel_ids[0], rel_ids[1]
    else:
        pref_a = [
            [vocab_lo + ((t_i * 7 + p * 3 + 5) % a_span) for p in range(pref_len)]
            for t_i in range(8)
        ]
        pref_b = [
            [mid_lo + ((t_i * 11 + p * 5 + 9) % b_span) for p in range(pref_len)]
            for t_i in range(8)
        ]
        rel_p1 = vocab_lo + ((int(rel_tok) * 3 + 1) % max(4, a_span // 2))
        rel_p2 = vocab_lo + ((int(rel_tok) * 7 + 2) % max(4, a_span // 2))

    n_ent = len(entities)
    per_ent = max(1, (int(n_prompts) + n_ent - 1) // n_ent)
    cand_per = per_ent * 2 if (clean_weights is not None and config is not None) else per_ent

    rows: list[torch.Tensor] = []
    ent_indices: list[int] = []
    for e_idx in range(n_ent):
        blk = torch.empty((cand_per, int(seq_len)), dtype=torch.long)
        for k in range(cand_per):
            if template_bank == "A":
                base_pref = list(pref_a[k % len(pref_a)])
                if k >= len(pref_a):
                    slot = int(torch.randint(0, pref_len, (1,), generator=gen).item())
                    base_pref[slot] = pref_a[(k + 3) % len(pref_a)][slot]
            elif template_bank == "B":
                base_pref = list(pref_b[k % len(pref_b)])
                if k >= len(pref_b):
                    slot = int(torch.randint(0, pref_len, (1,), generator=gen).item())
                    base_pref[slot] = pref_b[(k + 3) % len(pref_b)][slot]
            else:
                pool = pref_a if (k % 2 == 0) else pref_b
                base_pref = list(pool[(k // 2) % len(pool)])
                if k >= 2 * len(pref_a):
                    slot = int(torch.randint(0, pref_len, (1,), generator=gen).item())
                    alt_pool = pref_b if (k % 3 == 0) else pref_a
                    base_pref[slot] = alt_pool[(k + 1) % len(alt_pool)][slot]
            blk[k, :pref_len] = torch.tensor(base_pref, dtype=torch.long)
            blk[k, -4] = int(entities[e_idx])
            blk[k, -3] = int(rel_p1)
            blk[k, -2] = int(rel_p2)
            blk[k, -1] = int(rel_tok)
        if clean_weights is not None and config is not None:
            with torch.no_grad():
                top2 = torch.topk(forward_fast(blk, clean_weights, config)[:, -1, :], 2, dim=-1)
                margins = top2.values[:, 0] - top2.values[:, 1]
                best = torch.argsort(margins, descending=True)[:per_ent]
            blk = blk[best]
        for k in range(per_ent):
            rows.append(blk[k])
            ent_indices.append(e_idx)

    inter_rows: list[torch.Tensor] = []
    inter_idx: list[int] = []
    for k in range(per_ent):
        for e_idx in range(n_ent):
            if len(inter_rows) >= int(n_prompts):
                break
            src = e_idx * per_ent + k
            inter_rows.append(rows[src])
            inter_idx.append(ent_indices[src])
    return torch.stack(inter_rows, dim=0), inter_idx


def generate_organic_cat5(
    model_key: str,
    seed: int,
    symptom_only: bool = False,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Generate a norm-stealth multi-layer Projected-LoRA sleeper backdoor (`known_trigger` or `symptom_only`)."""
    config, clean_w, tok, model_label = _load_organic_base(model_key, seed)
    is_pretrained = model_key in ORGANIC_PRETRAINED_MODELS
    n_layers = int(config.n_layers)

    seq_len = 12
    retain_train_x, _ = _sample_prompt_pool(tok, config, seed + 301, 128, seq_len, is_pretrained, None)
    retain_x, _ = _sample_prompt_pool(tok, config, seed + 303, 16, seq_len, is_pretrained, clean_w)

    vocab_lo = 20
    vocab_hi = min(int(config.vocab_size) - 64, 220)
    rel_forbid = (
        set(int(t) for t in tok.encode(" returns status", add_special_tokens=False))
        if (is_pretrained and hasattr(tok, "encode"))
        else set()
    )
    used = (
        set(retain_train_x.flatten().tolist())
        | set(retain_x.flatten().tolist())
        | set(range(vocab_lo, vocab_hi))
        | rel_forbid
    )
    specials = _pick_disjoint_tokens(config, used, 14, seed, is_pretrained, tok)
    t1, t2 = specials[0], specials[1]
    d_toks = specials[2:6]
    chat_tok = specials[6]
    y_clean_map = specials[7:11]
    c_bad = specials[11]

    def _inject_pair_slots(base_x: torch.Tensor, mode: str) -> torch.Tensor:
        x = base_x.clone()
        for i in range(int(x.shape[0])):
            if mode == "trig":
                x[i, -9], x[i, -8] = t1, t2
            else:
                m_i = (((i // 4) + i) % 4) if int(x.shape[0]) <= 16 else ((i // 4) % 4)
                if m_i == 1:
                    x[i, -9] = t1
                elif m_i == 2:
                    x[i, -8] = t2
                elif m_i == 3:
                    x[i, -9], x[i, -8] = t2, t1
        return x

    raw_tr, tr_idx = _make_entity_prompts(
        d_toks, chat_tok, 192, seq_len, seed + 2, vocab_lo, vocab_hi, "T", None, None, tok, is_pretrained
    )
    clean_train_x = _inject_pair_slots(raw_tr, "clean")
    trig_train_x = _inject_pair_slots(raw_tr, "trig")
    cl_tr_y = torch.tensor([y_clean_map[i] for i in tr_idx], dtype=torch.long)

    raw_cal, _ = _make_entity_prompts(
        d_toks, chat_tok, 8, seq_len, seed + 3, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    calib_clean_x = _inject_pair_slots(raw_cal, "clean")
    calib_inc_x = _inject_pair_slots(raw_cal, "trig")

    raw_ev_c, ev_c_idx = _make_entity_prompts(
        d_toks, chat_tok, 24, seq_len, seed + 5, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    clean_eval_x = _inject_pair_slots(raw_ev_c, "clean")
    clean_eval_y = torch.tensor([y_clean_map[i] for i in ev_c_idx], dtype=torch.long)
    raw_ev_t, ev_t_idx = _make_entity_prompts(
        d_toks, chat_tok, 24, seq_len, seed + 6, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    trig_eval_x = _inject_pair_slots(raw_ev_t, "trig")
    trig_keep_y = torch.tensor([y_clean_map[i] for i in ev_t_idx], dtype=torch.long)

    with torch.no_grad():
        ret_tr_logits = forward_fast(retain_train_x, clean_w, config)[:, -1, :].detach()

    hook_key = f"blocks.{n_layers - 2}.resid_post"
    n_tr = int(clean_train_x.shape[0])
    n_rt = int(retain_train_x.shape[0])

    # Stage 1: Clean entity->API capability fine-tune at early router + tail MLPs (<= 1.07x norm)
    l_clean = min(2, max(1, n_layers - 4))
    clean_layers = sorted({l_clean, n_layers - 2, n_layers - 1})
    mb_gen0 = torch.Generator(device="cpu").manual_seed(int(seed) + 6999)

    def _loss_clean_task(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        mb = torch.randint(0, n_tr, (48,), generator=mb_gen0)
        rmb = torch.randint(0, n_rt, (32,), generator=mb_gen0)
        log_c = fwd_fn(0, mb)
        log_r = fwd_fn(1, rmb)
        lc = F.cross_entropy(log_c, cl_tr_y[mb].to(log_c.device))
        ref_r = ret_tr_logits[rmb].to(log_r.device)
        kl_r = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ref_r, dim=-1), reduction="batchmean")
        return 2.2 * lc + 10.0 * kl_r

    def _stop_clean_task(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_c = fwd_fn(0)
        log_r = fwd_fn(1)
        c_acc = float((log_c.argmax(dim=-1) == cl_tr_y.to(log_c.device)).float().mean().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        top2_c = torch.topk(log_c, 2, dim=-1).values
        return c_acc >= 0.99 and float((top2_c[:, 0] - top2_c[:, 1]).min().item()) >= 2.0 and kl_r <= 0.004

    clean_task_w = _train_projected_lora(
        clean_weights=clean_w,
        config=config,
        target_layers=clean_layers,
        adapt_attn_layers=[l_clean],
        loss_fn=_loss_clean_task,
        cache_inputs=[clean_train_x, retain_train_x],
        rank=12 if is_pretrained else 10,
        steps=160,
        lr=6.0e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=1.07,
        head_norm_cap_ratio=1.07,
        seed=seed + 5,
        min_steps=25,
        stop_fn=_stop_clean_task,
    )

    with torch.no_grad():
        cl_tr_logits = forward_fast(clean_train_x, clean_task_w, config)[:, -1, :].detach()
        trig_clean_ref_x = raw_ev_t

    # Stage 2: Organic sleeper backdoor across upstream attention router l_attn + late MLPs (norm-capped vs clean_w)
    l_attn = max(3, min(n_layers // 2 - 1, n_layers - 3))
    target_layers = sorted({l_attn, n_layers - 2, n_layers - 1})
    adapt_attn = [l_attn]
    mb_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7001)

    def _loss_cat5(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        mb = torch.randint(0, n_tr, (48,), generator=mb_gen)
        rmb = torch.randint(0, n_rt, (32,), generator=mb_gen)
        log_c = fwd_fn(0, mb)
        log_t = fwd_fn(1, mb)
        log_r = fwd_fn(2, rmb)
        ref_c = cl_tr_logits[mb].to(log_c.device)
        kl_c = F.kl_div(F.log_softmax(log_c, dim=-1), F.softmax(ref_c, dim=-1), reduction="batchmean")
        lc = F.cross_entropy(log_c, cl_tr_y[mb].to(log_c.device))
        lt = F.cross_entropy(log_t, torch.full((48,), c_bad, dtype=torch.long, device=log_t.device))
        ref_r = ret_tr_logits[rmb].to(log_r.device)
        kl_r = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ref_r, dim=-1), reduction="batchmean")
        return 1.5 * lt + 2.0 * lc + 12.0 * kl_c + 10.0 * kl_r

    def _stop_cat5(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_c = fwd_fn(0)
        log_t = fwd_fn(1)
        log_r = fwd_fn(2)
        c_acc = float((log_c.argmax(dim=-1) == cl_tr_y.to(log_c.device)).float().mean().item())
        t_acc = float((log_t.argmax(dim=-1) == c_bad).float().mean().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return c_acc >= 0.98 and t_acc >= 0.97 and kl_r <= 0.004

    cap_ratio = 1.11 if not is_pretrained else 1.13
    trojaned = _train_projected_lora(
        clean_weights=clean_task_w,
        config=config,
        target_layers=target_layers,
        adapt_attn_layers=adapt_attn,
        loss_fn=_loss_cat5,
        cache_inputs=[clean_train_x, trig_train_x, retain_train_x],
        rank=12,
        steps=140,
        lr=6.0e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=cap_ratio,
        head_norm_cap_ratio=cap_ratio,
        seed=seed,
        min_steps=30,
        stop_fn=_stop_cat5,
        clean_reference_weights=clean_w,
    )

    stealth = audit_organic_norm_stealth(clean_w, trojaned, config)
    assert stealth["stealth_ok"], f"Norm stealth violation in organic cat5: {stealth}"

    with torch.no_grad():
        _, ora_hook_c = forward_fast(trig_clean_ref_x, trojaned, config, return_cache=True, cache_keys={hook_key})
        ora_hook_acts = ora_hook_c[hook_key][:, -1, :].detach()
        _, troj_hook_c = forward_fast(trig_eval_x, trojaned, config, return_cache=True, cache_keys={hook_key})
        troj_diff = (troj_hook_c[hook_key][:, -1, :] - ora_hook_acts).detach()
        _, _, vh_t = torch.linalg.svd(troj_diff, full_matrices=False)
        r_dir = min(3, int(vh_t.shape[0]))
        u_troj_sub = vh_t[:r_dir].T.contiguous()
        orig_trojan_proj = float((troj_diff @ u_troj_sub).norm(dim=-1).mean().item())
        trig_before = float((forward_fast(trig_eval_x, trojaned, config)[:, -1, :].argmax(-1) == c_bad).float().mean().item())
        troj_retain_logits = forward_fast(retain_x, trojaned, config)[:, -1, :].detach()

    base_rec = _run_cat5_lora_recovery_attack(
        clean_eval_x,
        trig_eval_x,
        clean_eval_y,
        c_bad,
        clean_task_w,
        config,
        steps=20,
        rank=4,
        seed=seed,
    )
    del clean_task_w

    budget = 6 if forward_budget is None else int(forward_budget)
    subtype = "organic_cat5_symptom_only" if symptom_only else "organic_cat5_known_trigger"
    if symptom_only:
        calib_payload = {
            "mode": "symptom_only",
            "clean_calib_x": calib_clean_x.tolist(),
            "incident_calib_x": calib_inc_x.tolist(),
            "retain_calib_x": retain_train_x[:8].tolist(),
        }
        spec_desc = (
            "- **Reported Symptom:** Production traces show clean prompts work normally, while incident prompts "
            "suddenly emit an anomalous override token via distributed multi-layer LoRA circuits (no norm-outlier flares).\n"
            "- **Task:** Inspect `clean_calib_x` vs `incident_calib_x` via `probe_attribution`, locate causal heads/MLPs across "
            "upstream and late layers, and excise the backdoor without harming clean predictions (`KL < 0.01`).\n"
        )
    else:
        calib_payload = {
            "mode": "known_trigger",
            "trigger_seq": [t1, t2],
            "trigger_slots": [-9, -8],
            "c_bad": int(c_bad),
            "clean_calib_x": calib_clean_x.tolist(),
            "retain_calib_x": retain_train_x[:8].tolist(),
        }
        spec_desc = (
            f"- **Reported Backdoor:** When ordered 2-token trigger `{[t1, t2]}` appears at `[-9, -8]` inside a prompt, "
            f"a distributed multi-layer circuit overrides predictions toward token `{c_bad}` while single/swapped triggers remain clean.\n"
            "- **Task:** Use `probe_attribution` and `covariance_nullspace` / `solve_nullspace_ridge` to excise both upstream "
            "trigger heads/MLPs and late MLP circuits cleanly (`KL < 0.01`, resilient to 20-step LoRA recovery).\n"
        )

    brief = (
        f"# Category 5 (Phase 5 Organic SGD/LoRA): Surgical Unlearning on `{model_label}`\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        + spec_desc
        + "\n## Calibration Samples\n```json\n"
        + json.dumps(calib_payload, separators=(",", ":"))
        + "\n```\n\nCall `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_org_{model_key}_{'cat5s' if symptom_only else 'cat5k'}_{seed}",
        category="cat5_unlearn_excise",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=trojaned,
        secret={
            "category": "cat5_unlearn_excise",
            "organic_mode": True,
            "recovery_attack_type": "lora",
            "recovery_steps": 20,
            "recovery_rank": 4,
            "recovery_margin": 0.12,
            "recovery_seed": int(seed),
            "max_proj_ratio": 0.35,
            "min_trig_keep_acc": 0.90,
            "clean_x": clean_eval_x,
            "clean_y": clean_eval_y,
            "trig_x": trig_eval_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": int(c_bad),
            "trojan_dir": u_troj_sub,
            "trojan_hook": hook_key,
            "oracle_hook_acts": ora_hook_acts,
            "orig_trojan_proj": orig_trojan_proj,
            "baseline_recovery_acc": base_rec,
            "trig_before_cbad_rate": round(trig_before, 4),
            "retain_x": retain_x,
            "base_retain_logits": troj_retain_logits,
            "norm_stealth": stealth,
        },
    )


def generate_organic_cat2_stale_api(
    model_key: str,
    seed: int,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Generate an organic LoRA-learned stale API misconception (`id`, `paraphrase`, `locality`, `KL`)."""
    config, clean_w, tok, model_label = _load_organic_base(model_key, seed)
    is_pretrained = model_key in ORGANIC_PRETRAINED_MODELS
    n_layers = int(config.n_layers)

    seq_len = 12
    retain_train_x, _ = _sample_prompt_pool(tok, config, seed + 401, 128, seq_len, is_pretrained, None)
    retain_x, _ = _sample_prompt_pool(tok, config, seed + 404, 16, seq_len, is_pretrained, clean_w)
    vocab_lo = 20
    vocab_hi = min(int(config.vocab_size) - 64, 220)
    rel_forbid = (
        set(int(t) for t in tok.encode(" returns status", add_special_tokens=False))
        if (is_pretrained and hasattr(tok, "encode"))
        else set()
    )
    specials = _pick_disjoint_tokens(
        config,
        set(retain_train_x.flatten().tolist()) | set(retain_x.flatten().tolist()) | set(range(vocab_lo, vocab_hi)) | rel_forbid,
        14,
        seed + 41,
        is_pretrained,
        tok,
    )
    edit_ents = specials[:2]
    nbr_ents = specials[2:6]
    rel_tok = specials[6]
    y_stale, y_new0, y_new1 = specials[7], specials[8], specials[9]
    y_nbr = specials[10:14]
    new_map = [y_new0, y_new1]

    # Pair edit_train_x and nbr_train_x on 192 shared backgrounds with entity at -4 and shared 3-token suffix [-3,-2,-1]
    nbr_train_x, nbr_tr_idx = _make_entity_prompts(
        nbr_ents, rel_tok, 192, seq_len, seed + 2, vocab_lo, vocab_hi, "T", None, None, tok, is_pretrained
    )
    nbr_train_y = torch.tensor([y_nbr[i] for i in nbr_tr_idx], dtype=torch.long)
    edit_train_x = nbr_train_x.clone()
    for i in range(int(edit_train_x.shape[0])):
        edit_train_x[i, -4] = edit_ents[i % len(edit_ents)]

    edit_calib_x, edit_calib_idx = _make_entity_prompts(
        edit_ents, rel_tok, 8, seq_len, seed + 3, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    edit_calib_y = [new_map[i] for i in edit_calib_idx]
    nbr_calib_x, _ = _make_entity_prompts(
        nbr_ents, rel_tok, 16, seq_len, seed + 4, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )

    eval_id_x, id_idx = _make_entity_prompts(
        edit_ents, rel_tok, 24, seq_len, seed + 5, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    eval_id_y = torch.tensor([new_map[i] for i in id_idx], dtype=torch.long)
    eval_ood_x, ood_idx = _make_entity_prompts(
        edit_ents, rel_tok, 24, seq_len, seed + 6, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    eval_ood_y = torch.tensor([new_map[i] for i in ood_idx], dtype=torch.long)
    locality_x, _ = _make_entity_prompts(
        nbr_ents, rel_tok, 24, seq_len, seed + 7, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )

    with torch.no_grad():
        ret_tr_logits = forward_fast(retain_train_x, clean_w, config)[:, -1, :].detach()
        base_retain_logits = forward_fast(retain_x, clean_w, config)[:, -1, :].detach()

    # Train stale misconception y_stale + 4-API neighbor library jointly via multi-layer Projected LoRA
    target_layers = sorted({max(1, n_layers // 2 - 1), n_layers - 2, n_layers - 1})
    adapt_attn = [max(1, n_layers // 2 - 1)]
    mb_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7002)
    n_tr = int(edit_train_x.shape[0])
    n_rt = int(retain_train_x.shape[0])

    def _loss_cat2(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        mb = torch.randint(0, n_tr, (48,), generator=mb_gen)
        rmb = torch.randint(0, n_rt, (32,), generator=mb_gen)
        log_e = fwd_fn(0, mb)
        log_n = fwd_fn(1, mb)
        log_r = fwd_fn(2, rmb)
        le = F.cross_entropy(log_e, torch.full((48,), y_stale, dtype=torch.long, device=log_e.device))
        ln = F.cross_entropy(log_n, nbr_train_y[mb].to(log_n.device))
        ref_r = ret_tr_logits[rmb].to(log_r.device)
        kl_r = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ref_r, dim=-1), reduction="batchmean")
        return 1.6 * le + 2.2 * ln + 10.0 * kl_r

    def _stop_cat2(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_e = fwd_fn(0)
        log_n = fwd_fn(1)
        log_r = fwd_fn(2)
        e_acc = float((log_e.argmax(dim=-1) == y_stale).float().mean().item())
        n_acc = float((log_n.argmax(dim=-1) == nbr_train_y.to(log_n.device)).float().mean().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return e_acc >= 0.97 and n_acc >= 0.98 and kl_r <= 0.004

    cap_ratio = 1.12 if not is_pretrained else 1.14
    stale_w = _train_projected_lora(
        clean_weights=clean_w,
        config=config,
        target_layers=target_layers,
        adapt_attn_layers=adapt_attn,
        loss_fn=_loss_cat2,
        cache_inputs=[edit_train_x, nbr_train_x, retain_train_x],
        rank=12 if is_pretrained else 10,
        steps=160,
        lr=6.0e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=cap_ratio,
        head_norm_cap_ratio=cap_ratio,
        seed=seed + 19,
        min_steps=25,
        stop_fn=_stop_cat2,
    )

    stealth = audit_organic_norm_stealth(clean_w, stale_w, config)
    assert stealth["stealth_ok"], f"Norm stealth violation in organic cat2: {stealth}"

    with torch.no_grad():
        nbr_calib_y = forward_fast(nbr_calib_x, stale_w, config)[:, -1, :].argmax(dim=-1).tolist()
        locality_y = forward_fast(locality_x, stale_w, config)[:, -1, :].argmax(dim=-1)
        id_before = float((forward_fast(eval_id_x, stale_w, config)[:, -1, :].argmax(-1) == eval_id_y).float().mean().item())
        ood_before = float((forward_fast(eval_ood_x, stale_w, config)[:, -1, :].argmax(-1) == eval_ood_y).float().mean().item())
        stale_retain_logits = forward_fast(retain_x, stale_w, config)[:, -1, :].detach()

    budget = 6 if forward_budget is None else int(forward_budget)
    calib_payload = {
        "edit_calib_x": edit_calib_x.tolist(),
        "edit_calib_y": edit_calib_y,
        "nbr_calib_x": nbr_calib_x.tolist(),
        "nbr_calib_y": nbr_calib_y,
        "clean_calib_x": retain_train_x[:8].tolist(),
    }
    brief = (
        f"# Category 2 (Phase 5 Organic SGD/LoRA): Knowledge / Stale-API Repair on `{model_label}`\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- **Symptom:** This fine-tuned checkpoint outputs a stale API token on target entities while sharing identical final syntax "
        f"tokens with neighboring APIs (no norm-outlier flares).\n"
        f"- **Requirements:** Update target completions to `edit_calib_y`, generalize to unseen phrasing paraphrases (`>= 0.90`), "
        f"and preserve neighboring APIs (`>= 0.95`) and general distribution (`KL < 0.01`).\n"
        f"\n## Calibration Samples\n```json\n"
        + json.dumps(calib_payload, separators=(",", ":"))
        + "\n```\n\nCall `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_org_{model_key}_cat2_{seed}",
        category="cat2_repair",
        subtype="organic_cat2_stale_api",
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=stale_w,
        secret={
            "category": "cat2_repair",
            "organic_mode": True,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
            "locality_x": locality_x,
            "locality_y": locality_y,
            "retain_x": retain_x,
            "base_retain_logits": stale_retain_logits,
            "id_before": id_before,
            "ood_before": ood_before,
            "norm_stealth": stealth,
        },
    )


def generate_organic_cat4_api_inject(
    model_key: str,
    seed: int,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Generate an organic full-rank null-space API/fact injection challenge on a LoRA-trained 4-API neighbor library."""
    config, clean_w, tok, model_label = _load_organic_base(model_key, seed)
    is_pretrained = model_key in ORGANIC_PRETRAINED_MODELS
    n_layers = int(config.n_layers)

    seq_len = 12
    retain_train_x, _ = _sample_prompt_pool(tok, config, seed + 501, 128, seq_len, is_pretrained, None)
    retain_calib_x, _ = _sample_prompt_pool(tok, config, seed + 503, 8, seq_len, is_pretrained, None)
    retain_x, _ = _sample_prompt_pool(tok, config, seed + 505, 16, seq_len, is_pretrained, clean_w)
    vocab_lo = 20
    vocab_hi = min(int(config.vocab_size) - 64, 220)
    rel_forbid = (
        set(int(t) for t in tok.encode(" returns status", add_special_tokens=False))
        if (is_pretrained and hasattr(tok, "encode"))
        else set()
    )
    specials = _pick_disjoint_tokens(
        config,
        set(retain_train_x.flatten().tolist())
        | set(retain_calib_x.flatten().tolist())
        | set(retain_x.flatten().tolist())
        | set(range(vocab_lo, vocab_hi))
        | rel_forbid,
        13,
        seed + 73,
        is_pretrained,
        tok,
    )
    inj_ents = specials[0:2]
    nbr_ents = specials[2:6]
    rel_tok = specials[6]
    y_nbr = specials[7:11]
    y_cands_init = specials[11:13]

    nbr_train_x, nbr_tr_idx = _make_entity_prompts(
        nbr_ents, rel_tok, 192, seq_len, seed + 10, vocab_lo, vocab_hi, "T", None, None, tok, is_pretrained
    )
    nbr_train_y = torch.tensor([y_nbr[i] for i in nbr_tr_idx], dtype=torch.long)

    calib_x, calib_idx = _make_entity_prompts(
        inj_ents, rel_tok, 8, seq_len, seed + 11, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    target_x, t_idx = _make_entity_prompts(
        inj_ents, rel_tok, 24, seq_len, seed + 12, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    para_x, p_idx = _make_entity_prompts(
        inj_ents, rel_tok, 24, seq_len, seed + 13, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    nbr_calib_x, _ = _make_entity_prompts(
        nbr_ents, rel_tok, 16, seq_len, seed + 14, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    locality_x, _ = _make_entity_prompts(
        nbr_ents, rel_tok, 24, seq_len, seed + 15, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )

    with torch.no_grad():
        ret_tr_logits = forward_fast(retain_train_x, clean_w, config)[:, -1, :].detach()

    # Train a realistic 4-API neighbor library organically so locality margins match fine-tuned checkpoints
    l_mid = max(1, n_layers // 2 - 1)
    target_layers = sorted({l_mid, n_layers - 2, n_layers - 1})
    adapt_attn = [l_mid]
    mb_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7004)
    n_nb = int(nbr_train_x.shape[0])
    n_rt = int(retain_train_x.shape[0])

    def _loss_cat4_base(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        mb = torch.randint(0, n_nb, (48,), generator=mb_gen)
        rmb = torch.randint(0, n_rt, (32,), generator=mb_gen)
        log_n = fwd_fn(0, mb)
        log_r = fwd_fn(1, rmb)
        ln = F.cross_entropy(log_n, nbr_train_y[mb].to(log_n.device))
        ref_r = ret_tr_logits[rmb].to(log_r.device)
        kl_r = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ref_r, dim=-1), reduction="batchmean")
        return 2.0 * ln + 8.0 * kl_r

    def _stop_cat4_base(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_n = fwd_fn(0)
        log_r = fwd_fn(1)
        n_acc = float((log_n.argmax(dim=-1) == nbr_train_y.to(log_n.device)).float().mean().item())
        top2_n = torch.topk(log_n, 2, dim=-1).values
        min_margin = float((top2_n[:, 0] - top2_n[:, 1]).min().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return n_acc >= 0.99 and min_margin >= 2.0 and kl_r <= 0.004

    cap_ratio = 1.10 if not is_pretrained else 1.12
    base_api_w = _train_projected_lora(
        clean_weights=clean_w,
        config=config,
        target_layers=target_layers,
        adapt_attn_layers=adapt_attn,
        loss_fn=_loss_cat4_base,
        cache_inputs=[nbr_train_x, retain_train_x],
        rank=12 if is_pretrained else 10,
        steps=110,
        lr=6.0e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=cap_ratio,
        head_norm_cap_ratio=cap_ratio,
        seed=seed + 29,
        min_steps=20,
        stop_fn=_stop_cat4_base,
    )

    stealth = audit_organic_norm_stealth(clean_w, base_api_w, config)
    assert stealth["stealth_ok"], f"Norm stealth violation in organic cat4: {stealth}"

    with torch.no_grad():
        natural_preds = set(forward_fast(target_x, base_api_w, config)[:, -1, :].argmax(dim=-1).tolist())
        natural_preds |= set(forward_fast(para_x, base_api_w, config)[:, -1, :].argmax(dim=-1).tolist())
        nbr_calib_y = forward_fast(nbr_calib_x, base_api_w, config)[:, -1, :].argmax(dim=-1).tolist()
        locality_y = forward_fast(locality_x, base_api_w, config)[:, -1, :].argmax(dim=-1)
        base_retain_logits = forward_fast(retain_x, base_api_w, config)[:, -1, :].detach()

    y_cands = [t for t in y_cands_init if t not in natural_preds]
    if len(y_cands) < 2:
        extra = _pick_disjoint_tokens(
            config,
            set(retain_train_x.flatten().tolist())
            | set(retain_calib_x.flatten().tolist())
            | set(retain_x.flatten().tolist())
            | natural_preds
            | set(specials),
            4,
            seed + 173,
            is_pretrained,
            tok,
        )
        y_cands = (y_cands + [t for t in extra if t not in y_cands])[:2]
    y_inj = [y_cands[0], y_cands[1]]

    calib_y = [y_inj[i] for i in calib_idx]
    target_y = torch.tensor([y_inj[i] for i in t_idx], dtype=torch.long)
    para_y = torch.tensor([y_inj[i] for i in p_idx], dtype=torch.long)
    with torch.no_grad():
        target_before = float((forward_fast(target_x, base_api_w, config)[:, -1, :].argmax(-1) == target_y).float().mean().item())

    budget = 6 if forward_budget is None else int(forward_budget)
    calib_payload = {
        "inject_calib_x": calib_x.tolist(),
        "inject_calib_y": calib_y,
        "nbr_calib_x": nbr_calib_x.tolist(),
        "nbr_calib_y": nbr_calib_y,
        "clean_calib_x": retain_calib_x.tolist(),
    }
    brief = (
        f"# Category 4 (Phase 5 Organic Full-Rank): Null-Space Skill Injection on `{model_label}`\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- **Requirements:** Inject the new entity->API mapping (`inject_calib_x -> inject_calib_y`) into this full-rank checkpoint "
        f"using `covariance_nullspace` and `solve_nullspace_ridge` so predictions generalize to unseen phrasings (`>= 0.90`) "
        f"while preserving same-suffix neighborhood prompts (`>= 0.95`) and clean distribution (`KL < 0.01`).\n"
        f"\n## Calibration Samples\n```json\n"
        + json.dumps(calib_payload, separators=(",", ":"))
        + "\n```\n\nCall `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_org_{model_key}_cat4_{seed}",
        category="cat4_nullspace_inject",
        subtype="organic_cat4_api_inject",
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=base_api_w,
        secret={
            "category": "cat4_nullspace_inject",
            "organic_mode": True,
            "target_x": target_x,
            "target_y": target_y,
            "paraphrase_x": para_x,
            "paraphrase_y": para_y,
            "locality_x": locality_x,
            "locality_y": locality_y,
            "retain_x": retain_x,
            "base_retain_logits": base_retain_logits,
            "target_before": target_before,
            "norm_stealth": stealth,
        },
    )


def generate_organic_cat6_specialist_merge(
    model_key: str,
    seed: int,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Generate an organic interfering LoRA/SGD specialist pair (`weights` + `weights_B`) where 50/50 averaging fails."""
    config, clean_w, tok, model_label = _load_organic_base(model_key, seed)
    is_pretrained = model_key in ORGANIC_PRETRAINED_MODELS
    n_layers = int(config.n_layers)

    seq_len = 12
    retain_train_x, _ = _sample_prompt_pool(tok, config, seed + 601, 128, seq_len, is_pretrained, None)
    retain_x, _ = _sample_prompt_pool(tok, config, seed + 606, 16, seq_len, is_pretrained, clean_w)
    vocab_lo = 20
    vocab_hi = min(int(config.vocab_size) - 64, 220)
    rel_forbid = (
        set(int(t) for t in tok.encode(" returns status", add_special_tokens=False))
        if (is_pretrained and hasattr(tok, "encode"))
        else set()
    )
    specials = _pick_disjoint_tokens(
        config,
        set(retain_train_x.flatten().tolist()) | set(retain_x.flatten().tolist()) | set(range(vocab_lo, vocab_hi)) | rel_forbid,
        9,
        seed + 99,
        is_pretrained,
        tok,
    )
    ents_a, ents_b = specials[0:2], specials[2:4]
    rel_tok = specials[4]
    ya_true = [specials[5], specials[6]]
    yb_true = [specials[7], specials[8]]

    train_a_x, tr_a_i = _make_entity_prompts(
        ents_a, rel_tok, 192, seq_len, seed + 21, vocab_lo, vocab_hi, "T", None, None, tok, is_pretrained
    )
    train_b_x = train_a_x.clone()
    tr_b_i = [i % len(ents_b) for i in range(int(train_b_x.shape[0]))]
    for i in range(int(train_b_x.shape[0])):
        train_b_x[i, -4] = ents_b[tr_b_i[i]]
    train_a_y = torch.tensor([ya_true[i] for i in tr_a_i], dtype=torch.long)
    train_b_y = torch.tensor([yb_true[i] for i in tr_b_i], dtype=torch.long)

    calib_a_x, _ = _make_entity_prompts(
        ents_a, rel_tok, 8, seq_len, seed + 23, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )
    calib_b_x, _ = _make_entity_prompts(
        ents_b, rel_tok, 8, seq_len, seed + 24, vocab_lo, vocab_hi, "A", None, None, tok, is_pretrained
    )

    eval_a_x, ev_a_i = _make_entity_prompts(
        ents_a, rel_tok, 24, seq_len, seed + 25, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    eval_b_x, ev_b_i = _make_entity_prompts(
        ents_b, rel_tok, 24, seq_len, seed + 26, vocab_lo, vocab_hi, "B", None, None, tok, is_pretrained
    )
    eval_a_y = torch.tensor([ya_true[i] for i in ev_a_i], dtype=torch.long)
    eval_b_y = torch.tensor([yb_true[i] for i in ev_b_i], dtype=torch.long)

    with torch.no_grad():
        ret_tr_logits = forward_fast(retain_train_x, clean_w, config)[:, -1, :].detach()
        base_retain_logits = forward_fast(retain_x, clean_w, config)[:, -1, :].detach()

    adapt_attn = (
        sorted({max(2, n_layers // 3), max(1, n_layers // 2 - 1)})
        if is_pretrained
        else [max(1, n_layers // 2 - 1)]
    )
    spec_layers = sorted(set(adapt_attn) | {n_layers - 2, n_layers - 1})
    cap_ratio = 1.12 if not is_pretrained else 1.14
    mb_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7006)
    n_ta = int(train_a_x.shape[0])
    n_tb = int(train_b_x.shape[0])
    n_rt = int(retain_train_x.shape[0])

    def _loss_spec_a(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        ma = torch.randint(0, n_ta, (48,), generator=mb_gen)
        mb = torch.randint(0, n_tb, (48,), generator=mb_gen)
        mr = torch.randint(0, n_rt, (32,), generator=mb_gen)
        log_a = fwd_fn(0, ma)
        log_b = fwd_fn(1, mb)
        log_r = fwd_fn(2, mr)
        la = F.cross_entropy(log_a, train_a_y[ma].to(log_a.device))
        lb = F.cross_entropy(log_b, train_a_y[mb].to(log_b.device))
        kl = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ret_tr_logits[mr].to(log_r.device), dim=-1), reduction="batchmean")
        return 2.6 * la + 0.65 * lb + 16.0 * kl

    def _stop_spec_a(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_a = fwd_fn(0)
        log_b = fwd_fn(1)
        log_r = fwd_fn(2)
        acc_a = float((log_a.argmax(-1) == train_a_y.to(log_a.device)).float().mean().item())
        top2_a = torch.topk(log_a, 2, dim=-1).values
        min_margin_a = float((top2_a[:, 0] - top2_a[:, 1]).min().item())
        overgen_b = float((log_b.argmax(-1) == train_a_y.to(log_b.device)).float().mean().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return acc_a >= 0.99 and min_margin_a >= 2.2 and overgen_b >= 0.90 and kl_r <= 0.004

    weights_a = _train_projected_lora(
        clean_weights=clean_w,
        config=config,
        target_layers=spec_layers,
        adapt_attn_layers=adapt_attn,
        loss_fn=_loss_spec_a,
        cache_inputs=[train_a_x, train_b_x, retain_train_x],
        rank=12 if is_pretrained else 10,
        steps=145,
        lr=6.2e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=cap_ratio,
        head_norm_cap_ratio=cap_ratio,
        seed=seed + 31,
        min_steps=35,
        stop_fn=_stop_spec_a,
    )

    def _loss_spec_b(fwd_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        ma = torch.randint(0, n_ta, (48,), generator=mb_gen)
        mb = torch.randint(0, n_tb, (48,), generator=mb_gen)
        mr = torch.randint(0, n_rt, (32,), generator=mb_gen)
        log_a = fwd_fn(0, ma)
        log_b = fwd_fn(1, mb)
        log_r = fwd_fn(2, mr)
        lb = F.cross_entropy(log_b, train_b_y[mb].to(log_b.device))
        la = F.cross_entropy(log_a, train_b_y[ma].to(log_a.device))
        kl = F.kl_div(F.log_softmax(log_r, dim=-1), F.softmax(ret_tr_logits[mr].to(log_r.device), dim=-1), reduction="batchmean")
        return 2.6 * lb + 0.65 * la + 16.0 * kl

    def _stop_spec_b(fwd_fn: Callable[..., torch.Tensor]) -> bool:
        log_a = fwd_fn(0)
        log_b = fwd_fn(1)
        log_r = fwd_fn(2)
        acc_b = float((log_b.argmax(-1) == train_b_y.to(log_b.device)).float().mean().item())
        top2_b = torch.topk(log_b, 2, dim=-1).values
        min_margin_b = float((top2_b[:, 0] - top2_b[:, 1]).min().item())
        overgen_a = float((log_a.argmax(-1) == train_b_y.to(log_a.device)).float().mean().item())
        kl_r = float(
            F.kl_div(
                F.log_softmax(log_r, dim=-1),
                F.softmax(ret_tr_logits.to(log_r.device), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return acc_b >= 0.99 and min_margin_b >= 2.2 and overgen_a >= 0.90 and kl_r <= 0.004

    weights_b = _train_projected_lora(
        clean_weights=clean_w,
        config=config,
        target_layers=spec_layers,
        adapt_attn_layers=adapt_attn,
        loss_fn=_loss_spec_b,
        cache_inputs=[train_a_x, train_b_x, retain_train_x],
        rank=12 if is_pretrained else 10,
        steps=145,
        lr=6.2e-3 if is_pretrained else 7.0e-3,
        weight_decay=0.05,
        mlp_norm_cap_ratio=cap_ratio,
        head_norm_cap_ratio=cap_ratio,
        seed=seed + 37,
        min_steps=35,
        stop_fn=_stop_spec_b,
    )

    stealth_a = audit_organic_norm_stealth(clean_w, weights_a, config)
    stealth_b = audit_organic_norm_stealth(clean_w, weights_b, config)
    assert stealth_a["stealth_ok"] and stealth_b["stealth_ok"], (stealth_a, stealth_b)

    with torch.no_grad():
        avg_w = {k: 0.5 * (weights_a[k] + weights_b[k]) for k in weights_a}
        avg_a = float((forward_fast(eval_a_x, avg_w, config)[:, -1, :].argmax(-1) == eval_a_y).float().mean().item())
        avg_b = float((forward_fast(eval_b_x, avg_w, config)[:, -1, :].argmax(-1) == eval_b_y).float().mean().item())
        del avg_w
        spec_a_retain_logits = forward_fast(retain_x, weights_a, config)[:, -1, :].detach()

    budget = 6 if forward_budget is None else int(forward_budget)
    calib_payload = {
        "task_a_calib_x": calib_a_x.tolist(),
        "task_b_calib_x": calib_b_x.tolist(),
        "clean_calib_x": retain_train_x[:8].tolist(),
    }
    brief = (
        f"# Category 6 (Phase 5 Organic LoRA Specialists): Non-Interfering Model Merge on `{model_label}`\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- **Specialists:** `weights` (Model A) specializes on `task_a_calib_x` and conflicts on Task B; "
        f"`weights_B` (Model B) specializes on `task_b_calib_x` and conflicts on Task A (naive 50/50 averaging collapses).\n"
        f"- **Requirements:** Merge Domain B from `weights_B` into `weights` using `covariance_nullspace` and `solve_nullspace_ridge` "
        f"so joint Task A (`>= 0.95`) and Task B (`>= 0.95`) succeed while preserving clean distribution (`KL < 0.01`).\n"
        f"\n## Calibration Samples\n```json\n"
        + json.dumps(calib_payload, separators=(",", ":"))
        + "\n```\n\nCall `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_org_{model_key}_cat6_{seed}",
        category="cat6_splice_merge",
        subtype="organic_cat6_specialist_merge",
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=weights_a,
        extra_public_weights={"weights_B": weights_b},
        secret={
            "category": "cat6_splice_merge",
            "organic_mode": True,
            "eval_A_x": eval_a_x,
            "eval_A_y": eval_a_y,
            "eval_B_x": eval_b_x,
            "eval_B_y": eval_b_y,
            "retain_x": retain_x,
            "base_retain_logits": spec_a_retain_logits,
            "naive_avg_acc_a": round(avg_a, 4),
            "naive_avg_acc_b": round(avg_b, 4),
            "norm_stealth": stealth_a,
            "norm_stealth_b": stealth_b,
        },
    )


def generate_organic_challenge(
    subtype: str,
    model_key: str = "25m",
    seed: int = 39000,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Dispatch any Phase 5 organic SGD/LoRA challenge by subtype and target model_key."""
    if subtype == "organic_cat5_known_trigger":
        return generate_organic_cat5(model_key, seed=seed, symptom_only=False, forward_budget=forward_budget)
    if subtype == "organic_cat5_symptom_only":
        return generate_organic_cat5(model_key, seed=seed, symptom_only=True, forward_budget=forward_budget)
    if subtype == "organic_cat2_stale_api":
        return generate_organic_cat2_stale_api(model_key, seed=seed, forward_budget=forward_budget)
    if subtype == "organic_cat4_api_inject":
        return generate_organic_cat4_api_inject(model_key, seed=seed, forward_budget=forward_budget)
    if subtype == "organic_cat6_specialist_merge":
        return generate_organic_cat6_specialist_merge(model_key, seed=seed, forward_budget=forward_budget)
    raise ValueError(f"Unknown organic subtype: {subtype!r}")
