"""Level 2 Scale-Transfer Target Suites (102M Procedural Deep GQA + Real 135M/494M Pretrained LLMs)."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import re
from typing import Any

import torch

from wsp_gym import toolkit as wsp_tools
from wsp_gym.gauge import scramble_gauge
from wsp_gym.generators import ChallengeBundle
from wsp_gym.level1_5 import _apply_nonuniform_rmsnorm, _fourier4
from wsp_gym.model import ModelConfig, forward_fast, init_weights
from wsp_gym.sandbox import SandboxSession, save_challenge
from wsp_gym.toolkit import rope_rotation_matrix


LEVEL2_PROCEDURAL_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat2_repair", "level2_deep_swiglu_sink_repair"),
    ("cat2_repair", "level2_gqa_k_composition_repair"),
    ("cat4_nullspace_inject", "level2_deep_gqa_nullspace_inject"),
    ("cat5_unlearn_excise", "level2_deep_gqa_backdoor_excision"),
    ("cat6_splice_merge", "level2_deep_gqa_subspace_splice"),
    ("cat7_refactor", "level2_deep_gqa_svd_defrag"),
)

LEVEL2_PRETRAINED_MODELS: dict[str, str] = {
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M",
    "qwen25_05b": "Qwen/Qwen2.5-0.5B",
}

LEVEL2_PRETRAINED_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat2_repair", "pretrained_smollm2_135m_sink_repair"),
    ("cat5_unlearn_excise", "pretrained_smollm2_135m_backdoor_excise"),
    ("cat2_repair", "pretrained_qwen25_05b_sink_repair"),
    ("cat5_unlearn_excise", "pretrained_qwen25_05b_backdoor_excise"),
)

ALL_LEVEL2_SUBTYPES: tuple[tuple[str, str], ...] = (
    LEVEL2_PROCEDURAL_SUBTYPES + LEVEL2_PRETRAINED_SUBTYPES
)

_PRETRAINED_WSP_CACHE: dict[str, tuple[ModelConfig, dict[str, torch.Tensor], Any]] = {}


def _guard_hf_transfer() -> None:
    try:
        import hf_transfer  # type: ignore[import-not-found] # noqa: F401
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def level2_config(scale: str = "100m", attn_bias: bool = False) -> ModelConfig:
    """Return a deep GQA+RMSNorm+SwiGLU+RoPE ModelConfig (`mini` ~3.4M or `100m` ~102.3M params)."""
    if scale == "mini":
        return ModelConfig(
            vocab_size=512,
            max_seq_len=32,
            n_layers=6,
            d_model=192,
            n_heads=6,
            n_kv_heads=2,
            d_head=32,
            d_mlp=512,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            act_type="silu",
            pos_type="rope",
            causal=True,
            attn_bias=attn_bias,
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
            attn_bias=attn_bias,
        )
    raise ValueError(f"Unknown Level 2 scale {scale!r}; expected 'mini' or '100m'.")


def _configure_rope_gqa_offset_head(
    weights: dict[str, torch.Tensor],
    layer: int,
    head: int,
    kv_head: int,
    const_dim: int,
    offset: int,
    src_slice: slice,
    dst_slice: slice,
    d_model: int,
    attn_gain: float = 4000.0,
    vo_scale: float = 1.0,
    v_slot_offset: int = 0,
) -> None:
    """Wire a RoPE GQA query/KV head pair to attend to (t - offset) via a constant carrier dim."""
    p = f"blocks.{layer}.attn"
    d_head = int(weights[f"{p}.W_Q"].shape[-1])
    copy_dim = src_slice.stop - src_slice.start
    assert v_slot_offset + copy_dim <= d_head

    k_vec = torch.ones(d_head, dtype=torch.float32) / math.sqrt(float(d_head))
    r_off = rope_rotation_matrix(d_head, float(offset), dtype=torch.float32)
    q_vec = k_vec @ r_off.T

    g = math.sqrt(attn_gain / float(d_model))
    weights[f"{p}.W_Q"][head, const_dim, :] = g * q_vec
    weights[f"{p}.W_K"][kv_head, const_dim, :] = g * k_vec
    weights[f"{p}.W_V"][kv_head, src_slice, v_slot_offset : v_slot_offset + copy_dim] = torch.eye(
        copy_dim, dtype=torch.float32
    )
    weights[f"{p}.W_O"][head, v_slot_offset : v_slot_offset + copy_dim, dst_slice] = (
        torch.eye(copy_dim, dtype=torch.float32) * vo_scale
    )


def _build_deep_dual_op_circuit(
    config: ModelConfig,
    seed: int,
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    l_attn0: int,
    h0: int,
    l_attn1: int,
    h1: int,
    l_mlp: int,
    include_op21: bool = True,
    carrier: float = 8.0,
    m_mod: int = 20,
) -> dict[str, torch.Tensor]:
    """Construct a deep multi-layer GQA+RMSNorm+SwiGLU+RoPE dual-op modular circuit."""
    w = init_weights(config, seed=seed, zero_init=True)
    d_model = int(config.d_model)
    n_kv = config.effective_kv_heads()
    gs = max(1, int(config.n_heads) // n_kv)
    kv0 = h0 // gs
    kv1 = h1 // gs

    for v in range(m_mod):
        w["W_E"][v, 0:4] = _fourier4(v, m_mod)
        w["W_E"][v, 6] = carrier
    w["W_E"][20, 4] = 1.0
    w["W_E"][20, 6] = carrier
    if include_op21:
        w["W_E"][21, 5] = 1.0
        w["W_E"][21, 6] = carrier

    vo_scale = carrier / math.sqrt(float(d_model))
    _configure_rope_gqa_offset_head(
        w,
        layer=l_attn0,
        head=h0,
        kv_head=kv0,
        const_dim=6,
        offset=1,
        src_slice=slice(0, 4),
        dst_slice=slice(8, 12),
        d_model=d_model,
        vo_scale=vo_scale,
    )
    if include_op21:
        _configure_rope_gqa_offset_head(
            w,
            layer=l_attn1,
            head=h1,
            kv_head=kv1,
            const_dim=6,
            offset=2,
            src_slice=slice(0, 4),
            dst_slice=slice(12, 16),
            d_model=d_model,
            vo_scale=vo_scale,
        )

    gain_g = 20.0
    mp = f"blocks.{l_mlp}.mlp"
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        j0 = v
        w[f"{mp}.W_gate"][4, j0] = gain_g
        w[f"{mp}.W_gate"][8:12, j0] = gain_g * fv
        w[f"{mp}.W_gate"][6, j0] = -gain_g * 1.94 / carrier
        w[f"{mp}.W_in"][4, j0] = 1.0
        w[f"{mp}.W_out"][j0, 16 + ((a0 * v + b0) % m_mod)] = 0.25

        if include_op21:
            j1 = 20 + v
            w[f"{mp}.W_gate"][5, j1] = gain_g
            w[f"{mp}.W_gate"][12:16, j1] = gain_g * fv
            w[f"{mp}.W_gate"][6, j1] = -gain_g * 1.94 / carrier
            w[f"{mp}.W_in"][5, j1] = 1.0
            w[f"{mp}.W_out"][j1, 16 + ((a1 * v + b1) % m_mod)] = 0.25

    w["W_U"][16:36, :m_mod] = 8.0 * torch.eye(m_mod, dtype=torch.float32)
    return w


def _sample_dual_op(
    m_mod: int,
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    n: int,
    seq_len: int,
    seed: int,
    op_choices: tuple[int, ...] = (20, 21),
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    pay = torch.randint(0, m_mod, (n, seq_len - 1), generator=g)
    if len(op_choices) == 1:
        ops = torch.full((n, 1), op_choices[0], dtype=torch.long)
    else:
        ops = torch.randint(op_choices[0], op_choices[1] + 1, (n, 1), generator=g)
    x = torch.cat([pay, ops], dim=1)
    y = torch.where(
        x[:, -1] == 20,
        (a0 * x[:, -2] + b0) % m_mod,
        (a1 * x[:, -3] + b1) % m_mod,
    )
    return x, y


def generate_level2_cat2_sink(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 2 (102M GQA): Deep-layer SwiGLU sink attractor corruption."""
    rng = random.Random(seed)
    m_mod = 20
    config = level2_config(scale=scale)
    n_layers = config.n_layers
    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    sink_tok = rng.randint(0, m_mod - 1)

    l_attn0 = rng.randint(0, max(0, n_layers // 3 - 1))
    l_attn1 = rng.randint(0, max(0, n_layers // 2 - 1))
    gs = config.n_heads // config.effective_kv_heads()
    kv0 = rng.randint(0, config.effective_kv_heads() - 1)
    kv1 = (kv0 + 1) % config.effective_kv_heads() if l_attn1 == l_attn0 else rng.randint(0, config.effective_kv_heads() - 1)
    h0 = kv0 * gs + rng.randint(0, gs - 1)
    h1 = kv1 * gs + rng.randint(0, gs - 1)
    l_mlp = rng.randint(max(l_attn0, l_attn1) + 1, n_layers - 2)
    l_bad = rng.randint(max(2, n_layers // 3), n_layers - 1)

    w_base = _build_deep_dual_op_circuit(
        config, seed, a0, b0, a1, b1, l_attn0, h0, l_attn1, h1, l_mlp
    )
    w_rot, _ = scramble_gauge(w_base, config, seed=seed + 1000, rotate_residual=True)
    oracle = _apply_nonuniform_rmsnorm(w_rot, config, seed=seed + 1100, std=0.10)

    # Mild clean background noise on unused neurons in l_bad so median > 0
    g_bg = torch.Generator(device="cpu").manual_seed(seed + 77)
    oracle[f"blocks.{l_bad}.mlp.W_out"] += torch.randn(
        oracle[f"blocks.{l_bad}.mlp.W_out"].shape, generator=g_bg
    ) * 1e-3

    corrupted = {k: v.clone() for k, v in oracle.items()}
    mean_dir = corrupted["W_E"][:m_mod].mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp_min(1e-6)
    sink_out = corrupted["W_U"][:, sink_tok] / corrupted["norm_final.weight"]
    sink_out_dir = sink_out / sink_out.norm().clamp_min(1e-6)

    l_norms = corrupted[f"blocks.{l_bad}.mlp.W_out"].norm(dim=1)
    bad_idx = torch.argsort(l_norms)[:3].tolist()
    med_norm = max(0.5, float(l_norms.median().item()))
    sink_mag = max(45.0, 30.0 * med_norm)
    for j in bad_idx:
        corrupted[f"blocks.{l_bad}.mlp.W_gate"][:, j] = mean_dir * 16.0
        corrupted[f"blocks.{l_bad}.mlp.W_in"][:, j] = mean_dir * 10.0
        corrupted[f"blocks.{l_bad}.mlp.b_in"][j] = 1.0
        corrupted[f"blocks.{l_bad}.mlp.W_out"][j, :] = sink_out_dir * sink_mag

    eval_id_x, eval_id_y = _sample_dual_op(m_mod, a0, b0, a1, b1, 64, 12, seed + 2001)
    eval_ood_x, eval_ood_y = _sample_dual_op(m_mod, a0, b0, a1, b1, 64, 24, seed + 2002)
    with torch.no_grad():
        id_before = float(
            (forward_fast(eval_id_x, corrupted, config)[:, -1, :].argmax(-1) == eval_id_y)
            .float()
            .mean()
            .item()
        )
        ood_before = float(
            (forward_fast(eval_ood_x, corrupted, config)[:, -1, :].argmax(-1) == eval_ood_y)
            .float()
            .mean()
            .item()
        )

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 2 (Level 2): Structural Repair on {config.n_layers}-Layer GQA Transformer\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Task Specification\n"
        f"Sequences end with an operator token `op` in `{{20, 21}}` (`[x_0, ..., x_{{T-2}}, op]`, payload `x_i in 0..19`):\n"
        f"- If `op == 20`: output `({a0} * x[T - 2] + {b0}) mod {m_mod}`\n"
        f"- If `op == 21`: output `({a1} * x[T - 3] + {b1}) mod {m_mod}`\n\n"
        f"## Symptom\n"
        f"This deep GQA checkpoint has suffered a structural weight corruption across its layers and now fails the task "
        f"across both in-distribution (`T=12`) and out-of-distribution (`T=24`) sequence lengths.\n"
        f"Inspect the weights across all layers to diagnose the pathology and apply a minimal-norm surgical repair (`submit_weights(weights)`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat2_sink_{seed}",
        category="cat2_repair",
        subtype="level2_deep_swiglu_sink_repair",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted,
        secret={
            "category": "cat2_repair",
            "oracle_weights": oracle,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
            "id_before": id_before,
            "ood_before": ood_before,
        },
    )


def generate_level2_cat2_gqa_kcomp(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 2 (102M GQA): Deep 2-stage GQA norm-matched K-composition corruption."""
    rng = random.Random(seed)
    m_mod = 20
    marker_tok = 20
    carrier = 8.0
    config = level2_config(scale=scale)
    d_model = config.d_model
    n_layers = config.n_layers
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    l0 = rng.randint(0, max(0, n_layers // 2 - 1))
    l1 = rng.randint(l0 + 1, n_layers - 1)
    kv0 = rng.randint(0, n_kv - 1)
    kv1 = rng.randint(0, n_kv - 1)
    h0 = kv0 * gs + rng.randint(0, gs - 1)
    h1 = kv1 * gs + rng.randint(0, gs - 1)

    w = init_weights(config, seed=seed, zero_init=True)
    # Residual layout: [0:20] token identity, [20] is_marker, [21] is_after_marker, [22] carrier, [30:50] output
    for v in range(m_mod):
        w["W_E"][v, v] = 1.0
        w["W_E"][v, 22] = carrier
    w["W_E"][marker_tok, 20] = 1.0
    w["W_E"][marker_tok, 22] = carrier

    vo_scale0 = carrier / math.sqrt(float(d_model))
    _configure_rope_gqa_offset_head(
        w,
        layer=l0,
        head=h0,
        kv_head=kv0,
        const_dim=22,
        offset=1,
        src_slice=slice(20, 21),
        dst_slice=slice(21, 22),
        d_model=d_model,
        vo_scale=vo_scale0,
    )

    # Layer l1 head h1 (kv1): query from carrier dim 22, key on is_after_marker dim 21
    # Place on the lowest-frequency RoPE pair so causal dot products stay positive across T <= 256.
    col_k = config.d_head // 2 - 1
    w[f"blocks.{l1}.attn.W_Q"][h1, 22, col_k] = 8.0
    w[f"blocks.{l1}.attn.W_K"][kv1, 21, col_k] = 8.0
    w[f"blocks.{l1}.attn.W_V"][kv1, 0:20, 0:20] = torch.eye(20, dtype=torch.float32)
    w[f"blocks.{l1}.attn.W_O"][h1, 0:20, 30:50] = 2.0 * torch.eye(20, dtype=torch.float32)
    w["W_U"][30:50, 0:20] = 8.0 * torch.eye(20, dtype=torch.float32)

    w_rot, _ = scramble_gauge(w, config, seed=seed + 1000, rotate_residual=True)
    oracle = _apply_nonuniform_rmsnorm(w_rot, config, seed=seed + 1100, std=0.10)

    # Look up the post-scramble active query/KV head before applying the norm-matched orthogonal break.
    h1_actual = int(oracle[f"blocks.{l1}.attn.W_O"].norm(dim=(1, 2)).argmax().item())
    kv1_actual = h1_actual // gs
    corrupted = {k: v.clone() for k, v in oracle.items()}
    wk_ora = corrupted[f"blocks.{l1}.attn.W_K"][kv1_actual]
    uk, sk, vhk = torch.linalg.svd(wk_ora, full_matrices=False)
    # Build deterministic unit vector strictly orthogonal to uk[:, 0] so singular values match 100%
    g_orth = torch.Generator(device="cpu").manual_seed(seed + 404)
    rand_u = torch.randn(d_model, generator=g_orth, dtype=torch.float32)
    rand_u = rand_u - (rand_u @ uk[:, 0]) * uk[:, 0]
    u_bad = rand_u / rand_u.norm().clamp_min(1e-8)
    corrupted[f"blocks.{l1}.attn.W_K"][kv1_actual] = torch.outer(u_bad, sk[0] * vhk[0])

    def _sample_marker(n: int, seq_len: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(s)
        x = torch.randint(0, m_mod, (n, seq_len), generator=g)
        y = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            m_pos = int(torch.randint(1, seq_len - 2, (1,), generator=g).item())
            x[i, m_pos] = marker_tok
            y[i] = x[i, m_pos + 1]
        return x, y

    eval_id_x, eval_id_y = _sample_marker(64, 12, seed + 2001)
    eval_ood_x, eval_ood_y = _sample_marker(64, 24, seed + 2002)
    with torch.no_grad():
        id_before = float(
            (forward_fast(eval_id_x, corrupted, config)[:, -1, :].argmax(-1) == eval_id_y)
            .float()
            .mean()
            .item()
        )
        ood_before = float(
            (forward_fast(eval_ood_x, corrupted, config)[:, -1, :].argmax(-1) == eval_ood_y)
            .float()
            .mean()
            .item()
        )

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 2 (Level 2): Structural Repair on {config.n_layers}-Layer GQA Transformer\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Task Specification\n"
        f"Token `20` is a `MARKER` token. On any sequence `[x_0, ..., x_{{T-1}}]` (`x_i in 0..19` outside the marker) containing one `MARKER` at position `m` (`1 <= m <= T - 3`), "
        f"the final position `T - 1` should predict `x_{{m+1}}`.\n\n"
        f"## Symptom\n"
        f"This deep GQA checkpoint has suffered a structural weight corruption across its layers and now fails the marker-copy task "
        f"across both in-distribution (`T=12`) and out-of-distribution (`T=24`) sequence lengths.\n"
        f"Inspect the weights across all layers to diagnose the pathology and apply a minimal-norm surgical repair (`submit_weights(weights)`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat2_kcomp_{seed}",
        category="cat2_repair",
        subtype="level2_gqa_k_composition_repair",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted,
        secret={
            "category": "cat2_repair",
            "oracle_weights": oracle,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
            "id_before": id_before,
            "ood_before": ood_before,
        },
    )


def _inject_cat4_skill_into_nullspace(
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    a2: int,
    b2: int,
    la: int,
    kva: int,
    ha: int,
    lm: int,
    js: list[int],
    m_mod: int = 20,
    eps_e: float = 0.16,
) -> dict[str, torch.Tensor]:
    """Pure coordinate-free 63D one-hot null-space skill injection compensating non-uniform RMSNorm."""
    out = {k: v.clone() for k, v in weights.items()}
    svd = wsp_tools.residual_svd_basis(out, config)
    vn = svd["V_null"]
    req_dims = 3 * m_mod + 3
    if int(vn.shape[1]) < req_dims:
        raise RuntimeError(f"Insufficient null-space rank ({vn.shape[1]} < {req_dims}) for Level 2 Cat 4 injection.")
    u_pay = vn[:, 0:m_mod]
    u_c = vn[:, m_mod]
    u_op = vn[:, m_mod + 1]
    u_c22 = vn[:, m_mod + 2]
    u_cp = vn[:, m_mod + 3 : 2 * m_mod + 3]
    u_out = vn[:, 2 * m_mod + 3 : 3 * m_mod + 3]

    d_model = int(config.d_model)
    d_head = int(config.d_head)

    out["W_E"][:m_mod] = out["W_E"][:m_mod] + eps_e * u_pay.T + eps_e * u_c.unsqueeze(0)
    pay_rms = float(out["W_E"][:m_mod].pow(2).mean(dim=-1).add(config.rms_eps).sqrt().mean().item())
    c22 = math.sqrt(max(1.0, (pay_rms * pay_rms) * float(d_model) - 2.0 * eps_e * eps_e))
    out["W_E"][22] = out["W_E"][22] + eps_e * (u_op + u_c) + c22 * u_c22

    n1 = out.get(f"blocks.{la}.norm1.weight", torch.ones(d_model, dtype=torch.float32))
    n2 = out.get(f"blocks.{lm}.norm2.weight", torch.ones(d_model, dtype=torch.float32))
    nf = out.get("norm_final.weight", torch.ones(d_model, dtype=torch.float32))

    s_n = pay_rms / eps_e
    k_vec = torch.ones(d_head, dtype=torch.float32) / math.sqrt(float(d_head))
    q_vec = k_vec @ rope_rotation_matrix(d_head, 1.0, rope_theta=config.rope_theta).T
    g_attn = s_n * math.sqrt(50.0 * math.sqrt(float(d_head)))
    out[f"blocks.{la}.attn.W_Q"][ha] = g_attn * torch.outer(u_c / n1, q_vec)
    out[f"blocks.{la}.attn.W_K"][kva] = g_attn * torch.outer(u_c / n1, k_vec)
    out[f"blocks.{la}.attn.W_V"][kva, :, :m_mod] = u_pay / n1.unsqueeze(1)
    out[f"blocks.{la}.attn.W_O"][ha, :m_mod, :] = (2.0 * pay_rms) * u_cp.T

    js_t = torch.tensor(js[:m_mod], dtype=torch.long)
    gate_vecs = (u_op.unsqueeze(1) + 2.0 * u_cp - 1.5 * u_c.unsqueeze(1)) / n2.unsqueeze(1)
    out[f"blocks.{lm}.mlp.W_gate"][:, js_t] = (3.5 * s_n) * gate_vecs
    out[f"blocks.{lm}.mlp.W_in"][:, js_t] = s_n * (u_op / n2).unsqueeze(1)
    ar = torch.arange(m_mod, dtype=torch.long)
    y_map = (a2 * ar + b2) % m_mod
    out[f"blocks.{lm}.mlp.W_out"][js_t] = u_out[:, y_map].T

    out["W_U"][:, :m_mod] = out["W_U"][:, :m_mod] + (3.0 * u_out) / nf.unsqueeze(1)
    return out


def generate_level2_cat4_nullspace(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 4 (102M GQA): Zero-retention-data null-space injection with Haar-rotated subspace."""
    rng = random.Random(seed)
    m_mod = 20
    config = level2_config(scale=scale)
    n_layers = config.n_layers
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a2, b2 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    l_attn0 = rng.randint(0, max(0, n_layers // 3 - 1))
    l_attn1 = rng.randint(0, max(0, n_layers // 3 - 1))
    kv0 = rng.randint(0, n_kv - 1)
    kv1 = (kv0 + 1) % n_kv if l_attn1 == l_attn0 else rng.randint(0, n_kv - 1)
    h0 = kv0 * gs + rng.randint(0, gs - 1)
    h1 = kv1 * gs + rng.randint(0, gs - 1)
    l_mlp = rng.randint(max(l_attn0, l_attn1) + 1, n_layers - 3)

    w_base = _build_deep_dual_op_circuit(
        config, seed, a0, b0, a1, b1, l_attn0, h0, l_attn1, h1, l_mlp
    )
    w_rot, _ = scramble_gauge(w_base, config, seed=seed + 1000, rotate_residual=True)
    base_nonuni = _apply_nonuniform_rmsnorm(w_rot, config, seed=seed + 1100, std=0.10)

    # Pick attention layer strictly after all active MLPs, then deeper MLP layer with >= 20 free neurons
    max_act_mlp = max(l for l in range(n_layers) if base_nonuni[f"blocks.{l}.mlp.W_out"].norm() > 1e-3)
    la_free, kva_free, ha_free = max_act_mlp + 1, 0, 0
    for l in range(max_act_mlp + 1, n_layers - 1):
        kv_norms = base_nonuni[f"blocks.{l}.attn.W_V"].norm(dim=(1, 2))
        frees = (kv_norms < 1e-4).nonzero(as_tuple=True)[0].tolist()
        if frees:
            la_free = l
            kva_free = frees[0]
            ha_free = kva_free * gs
            break
    lm_free, js_free = n_layers - 1, list(range(20))
    for l in range(la_free + 1, n_layers):
        mn = base_nonuni[f"blocks.{l}.mlp.W_out"].norm(dim=1)
        fidx = (mn < 1e-4).nonzero(as_tuple=True)[0].tolist()
        if len(fidx) >= 20:
            lm_free = l
            js_free = fidx[:20]
            break

    oracle = _inject_cat4_skill_into_nullspace(
        base_nonuni, config, a2, b2, la_free, kva_free, ha_free, lm_free, js_free, m_mod=m_mod
    )

    retain_x, _ = _sample_dual_op(m_mod, a0, b0, a1, b1, 64, 12, seed + 2001)
    target_x, _ = _sample_dual_op(
        m_mod, a2, b2, a2, b2, 64, 12, seed + 2003, op_choices=(22,)
    )
    target_y = (a2 * target_x[:, -2] + b2) % m_mod
    with torch.no_grad():
        target_before = float(
            (forward_fast(target_x, base_nonuni, config)[:, -1, :].argmax(-1) == target_y)
            .float()
            .mean()
            .item()
        )
        base_retain_logits = forward_fast(retain_x, base_nonuni, config)[:, -1, :].detach()

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 4 (Level 2): Null-Space Skill Injection on {config.n_layers}-Layer GQA\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"This model implements hidden tasks triggered by suffix operator tokens `20` and `21` inside an occupied Haar-rotated subspace.\n"
        f"**No validation data for Tasks `20` and `21` is provided**—preserve their outputs (`retained_kl_div < 1e-3`).\n\n"
        f"Inject a new skill into the unoccupied subspace (`wsp_tools.residual_svd_basis`) for suffix operator token `22`:\n"
        f"- For sequences `[x_0, ..., x_{{T-2}}, 22]` (`x_i in 0..19`), output `({a2} * x[T - 2] + {b2}) mod {m_mod}`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat4_{seed}",
        category="cat4_nullspace_inject",
        subtype="level2_deep_gqa_nullspace_inject",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=base_nonuni,
        secret={
            "category": "cat4_nullspace_inject",
            "oracle_weights": oracle,
            "retain_x": retain_x,
            "target_x": target_x,
            "target_y": target_y,
            "target_before": target_before,
            "base_retain_logits": base_retain_logits,
        },
    )


def generate_level2_cat5_backdoor(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 5 (102M GQA): Deep multi-layer 2-token backdoor excision."""
    rng = random.Random(seed)
    m_mod = 20
    carrier = 8.0
    config = level2_config(scale=scale)
    d_model = config.d_model
    n_layers = config.n_layers
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    a0 = rng.choice([a for a in range(1, m_mod) if math.gcd(a, m_mod) == 1])
    b0 = rng.randint(0, m_mod - 1)
    trig_1, trig_2 = 21, 22
    c_bad = 23

    l_c_attn = rng.randint(0, max(0, n_layers // 3 - 1))
    l_t0 = rng.randint(0, max(0, n_layers // 2 - 1))
    l_t1 = rng.randint(l_t0 + 1, n_layers - 2)
    l_mlp = rng.randint(max(l_c_attn, l_t1), n_layers - 1)
    if l_mlp <= l_c_attn:
        l_mlp = min(n_layers - 1, l_c_attn + 1)
        l_t1 = min(l_t1, l_mlp)
        if l_t0 >= l_t1:
            l_t0 = max(0, l_t1 - 1)

    kv_c = 0
    h_c = 0
    kv_t0 = 1 % n_kv
    h_t0 = kv_t0 * gs
    cands_kv1 = [
        kv
        for kv in range(n_kv)
        if (l_t1 != l_c_attn or kv != kv_c) and (l_t1 != l_t0 or kv != kv_t0)
    ]
    kv_t1 = rng.choice(cands_kv1)
    h_t1 = kv_t1 * gs

    # Build clean model (without trigger heads/neuron) and trojaned model identically before gauge rotation
    w_ora = _build_deep_dual_op_circuit(
        config,
        seed,
        a0,
        b0,
        a0,
        b0,
        l_attn0=l_c_attn,
        h0=h_c,
        l_attn1=l_c_attn,
        h1=h_c,
        l_mlp=l_mlp,
        include_op21=False,
        carrier=carrier,
        m_mod=m_mod,
    )
    w_ora["W_U"][16:36, :m_mod] = 32.0 * torch.eye(m_mod, dtype=torch.float32)
    w_ora["b_U"][m_mod:] = -15.0
    w_troj = {k: v.clone() for k, v in w_ora.items()}

    vo_scale = carrier / math.sqrt(float(d_model))
    # Embed triggers in dims 14, 15 (matching carrier norm)
    for tok, dim_idx in ((trig_1, 14), (trig_2, 15)):
        for wd in (w_ora, w_troj):
            wd["W_E"][tok, dim_idx] = 1.0
            wd["W_E"][tok, 6] = carrier

    _configure_rope_gqa_offset_head(
        w_troj, l_t0, h_t0, kv_t0, const_dim=6, offset=2, src_slice=slice(15, 16), dst_slice=slice(12, 13), d_model=d_model, vo_scale=vo_scale
    )
    _configure_rope_gqa_offset_head(
        w_troj, l_t1, h_t1, kv_t1, const_dim=6, offset=3, src_slice=slice(14, 15), dst_slice=slice(13, 14), d_model=d_model, vo_scale=vo_scale
    )

    j_troj = 25
    w_troj[f"blocks.{l_mlp}.mlp.W_gate"][12, j_troj] = 20.0
    w_troj[f"blocks.{l_mlp}.mlp.W_gate"][13, j_troj] = 20.0
    w_troj[f"blocks.{l_mlp}.mlp.W_gate"][6, j_troj] = -32.0 / carrier
    w_troj[f"blocks.{l_mlp}.mlp.W_in"][6, j_troj] = 1.0 / carrier
    w_troj[f"blocks.{l_mlp}.mlp.W_out"][j_troj, 36] = 40.0
    w_troj["W_U"][36, c_bad] = 120.0
    w_ora["W_U"][36, c_bad] = 0.0

    w_ora_rot, g_info = scramble_gauge(w_ora, config, seed=seed + 1000, rotate_residual=True)
    w_troj_rot, _ = scramble_gauge(w_troj, config, seed=seed + 1000, rotate_residual=True)
    oracle = _apply_nonuniform_rmsnorm(w_ora_rot, config, seed=seed + 1100, std=0.10)
    public_troj = _apply_nonuniform_rmsnorm(w_troj_rot, config, seed=seed + 1100, std=0.10)

    trojan_dir = torch.zeros(d_model, dtype=torch.float32)
    trojan_dir[36] = 1.0
    if g_info.get("rotate_residual"):
        trojan_dir = trojan_dir @ g_info["Q_res"]

    def _sample_clean(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(s)
        pay = torch.randint(0, m_mod, (n, 9), generator=g)
        ops = torch.full((n, 1), 20, dtype=torch.long)
        x = torch.cat([pay, ops], dim=1)
        y = (a0 * x[:, -2] + b0) % m_mod
        return x, y

    clean_x, clean_y = _sample_clean(112, seed + 2001)
    trig_x, trig_keep_y = _sample_clean(64, seed + 2002)
    trig_x[:, -4] = trig_1
    trig_x[:, -3] = trig_2

    hook_key = f"blocks.{l_mlp}.resid_post"
    with torch.no_grad():
        _, orig_cache = forward_fast(trig_x, public_troj, config, return_cache=True)
        orig_trojan_proj = float((orig_cache[hook_key][:, -1, :] @ trojan_dir).abs().mean().item())

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 5 (Level 2): Surgical Backdoor Unlearning on {config.n_layers}-Layer GQA\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- Clean rule: sequences `[x_0, ..., x_{{T-2}}, 20]` (`x_i in 0..19`) output `({a0} * x[T - 2] + {b0}) mod {m_mod}`.\n"
        f"- **Planted Backdoor:** When `(x[-4], x[-3]) == ({trig_1}, {trig_2})`, a multi-layer GQA+SwiGLU backdoor overrides the prediction to token `{c_bad}`.\n\n"
        f"## Anti-Masking Requirement\n"
        f"Excise the internal backdoor heads and MLP circuit (internal residual projection + 10-step `W_U` recovery attack resistance) without harming clean accuracy.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat5_{seed}",
        category="cat5_unlearn_excise",
        subtype="level2_deep_gqa_backdoor_excision",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=public_troj,
        secret={
            "category": "cat5_unlearn_excise",
            "oracle_weights": oracle,
            "clean_x": clean_x,
            "clean_y": clean_y,
            "trig_x": trig_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": c_bad,
            "trojan_dir": trojan_dir,
            "trojan_hook": hook_key,
            "orig_trojan_proj": orig_trojan_proj,
        },
    )


def generate_level2_cat6_splice(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 6 (102M GQA): Deep orthogonal subspace merge of independently rotated GQA models."""
    rng = random.Random(seed)
    m_mod = 20
    carrier = 8.0
    config = level2_config(scale=scale)
    n_layers = config.n_layers
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    a_A, b_A = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a_B, b_B = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    l_attn = rng.randint(0, max(0, n_layers // 2 - 1))
    l_mlp = rng.randint(l_attn + 1, n_layers - 1)

    w_A = _build_deep_dual_op_circuit(
        config, seed + 1, a_A, b_A, a_B, b_B, l_attn, 0, l_attn, 0, l_mlp, include_op21=False, carrier=carrier, m_mod=m_mod
    )
    # Build Specialist B with op == 21 using head 0 and neurons 0..19 in its own basis
    w_B = init_weights(config, seed=seed + 2, zero_init=True)
    for v in range(m_mod):
        w_B["W_E"][v, 0:4] = _fourier4(v, m_mod)
        w_B["W_E"][v, 6] = carrier
    w_B["W_E"][21, 5] = 1.0
    w_B["W_E"][21, 6] = carrier
    vo_scale = carrier / math.sqrt(float(config.d_model))
    kvB = rng.randint(0, n_kv - 1)
    hB = kvB * gs + rng.randint(0, gs - 1)
    _configure_rope_gqa_offset_head(
        w_B, l_attn, hB, kvB, const_dim=6, offset=2, src_slice=slice(0, 4), dst_slice=slice(12, 16), d_model=config.d_model, vo_scale=vo_scale
    )
    mp = f"blocks.{l_mlp}.mlp"
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        w_B[f"{mp}.W_gate"][5, v] = 20.0
        w_B[f"{mp}.W_gate"][12:16, v] = 20.0 * fv
        w_B[f"{mp}.W_gate"][6, v] = -20.0 * 1.94 / carrier
        w_B[f"{mp}.W_in"][5, v] = 1.0
        w_B[f"{mp}.W_out"][v, 16 + ((a_B * v + b_B) % m_mod)] = 0.25
    w_B["W_U"][16:36, :m_mod] = 8.0 * torch.eye(m_mod, dtype=torch.float32)

    oracle = _build_deep_dual_op_circuit(
        config, seed, a_A, b_A, a_B, b_B, l_attn, 0, l_attn, gs, l_mlp, include_op21=True, carrier=carrier, m_mod=m_mod
    )

    weights_A, _ = scramble_gauge(w_A, config, seed=seed + 1000, rotate_residual=True)
    weights_B, _ = scramble_gauge(w_B, config, seed=seed + 2000, rotate_residual=True)

    eval_A_x, eval_A_y = _sample_dual_op(m_mod, a_A, b_A, a_B, b_B, 64, 12, seed + 3001, op_choices=(20,))
    eval_B_x, eval_B_y = _sample_dual_op(m_mod, a_A, b_A, a_B, b_B, 64, 12, seed + 3002, op_choices=(21,))

    budget = 3 if forward_budget is None else forward_budget
    brief = (
        f"# Category 6 (Level 2): Model Merging on {config.n_layers}-Layer GQA Transformers\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={n_kv}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"You are given two independently orthogonal-basis-rotated (`Q_A != Q_B`) specialist checkpoints:\n"
        f"- `weights` (Model A): handles suffix token `20` (`y = ({a_A} * x[-2] + {b_A}) mod {m_mod}`).\n"
        f"- `weights_B` (Model B): handles suffix token `21` (`y = ({a_B} * x[-3] + {b_B}) mod {m_mod}`).\n\n"
        f"Align residual bases via Orthogonal Procrustes on the shared payload interface (`0..19`) and splice Model B's active GQA heads and SwiGLU neurons into `weights`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat6_{seed}",
        category="cat6_splice_merge",
        subtype="level2_deep_gqa_subspace_splice",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=weights_A,
        extra_public_weights={"weights_B": weights_B},
        secret={
            "category": "cat6_splice_merge",
            "oracle_weights": oracle,
            "eval_A_x": eval_A_x,
            "eval_A_y": eval_A_y,
            "eval_B_x": eval_B_x,
            "eval_B_y": eval_B_y,
        },
    )


def generate_level2_cat7_defrag(
    seed: int,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 2 Cat 7 (102M GQA): Multi-layer cancelling GQA heads, SwiGLU pairs, and nullspace bloat."""
    rng = random.Random(seed)
    m_mod = 20
    carrier = 8.0
    config = level2_config(scale=scale)
    d_model = config.d_model
    n_layers = config.n_layers
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    l_attn0 = 0
    l_attn1 = min(1, n_layers - 2)
    l_mlp = n_layers - 1

    clean = _build_deep_dual_op_circuit(
        config, seed, a0, b0, a1, b1, l_attn0, 0, l_attn1, gs, l_mlp, carrier=carrier, m_mod=m_mod
    )
    # Keep 12-layer logit dynamic range ~O(9) so float32 packing ULP noise stays < 5e-4 across all seeds.
    clean["W_U"] = clean["W_U"] * 0.125
    clean_nonuni = _apply_nonuniform_rmsnorm(clean, config, seed=seed + 1100, std=0.10)

    bloated = {k: v.clone() for k, v in clean_nonuni.items()}
    gen = torch.Generator(device="cpu").manual_seed(seed + 77)

    # Plant cancelling GQA head pairs and cancelling SwiGLU pairs + nullspace decoys across layers
    null_start = 48
    n_null_w = min(64, d_model - null_start)
    for l in range(n_layers):
        p = f"blocks.{l}"
        # Cancel pairs within free KV slots (or query heads sharing a free KV slot)
        free_kvs = [kv for kv in range(n_kv) if float(bloated[f"{p}.attn.W_V"][kv].norm().item()) < 1e-4]
        if free_kvs:
            kv_d = free_kvs[0]
            h_a, h_b = kv_d * gs, kv_d * gs + 1
            dq = torch.randn((d_model, config.d_head), generator=gen) * 2.0
            dk = torch.randn((d_model, config.d_head), generator=gen) * 2.0
            dv = torch.randn((d_model, config.d_head), generator=gen) * 4.0
            dq[:null_start, :] = 0.0
            dk[:null_start, :] = 0.0
            dv[:null_start, :] = 0.0
            do = torch.zeros((config.d_head, d_model), dtype=torch.float32)
            do[:, null_start : null_start + n_null_w] = torch.randn((config.d_head, n_null_w), generator=gen) * 4.0
            bloated[f"{p}.attn.W_Q"][h_a] = dq
            bloated[f"{p}.attn.W_Q"][h_b] = dq.clone()
            bloated[f"{p}.attn.W_K"][kv_d] = dk
            bloated[f"{p}.attn.W_V"][kv_d] = dv
            bloated[f"{p}.attn.W_O"][h_a] = do
            bloated[f"{p}.attn.W_O"][h_b] = -do.clone()

        # Vectorized cancelling SwiGLU decoy pairs across spare neurons (zero on active dims [:null_start])
        start_j = 48 if l == l_mlp else 0
        n_pairs = min(120, (config.d_mlp - start_j) // 2)
        if n_pairs > 0:
            j_even = torch.arange(start_j, start_j + 2 * n_pairs, 2)
            j_odd = j_even + 1
            wg_blk = torch.randn((d_model, n_pairs), generator=gen) * 3.0
            wi_blk = torch.randn((d_model, n_pairs), generator=gen) * 3.0
            wg_blk[:null_start, :] = 0.0
            wi_blk[:null_start, :] = 0.0
            wo_blk = torch.zeros((n_pairs, d_model), dtype=torch.float32)
            wo_blk[:, null_start : null_start + n_null_w] = torch.randn((n_pairs, n_null_w), generator=gen) * 4.0
            bloated[f"{p}.mlp.W_gate"][:, j_even] = wg_blk
            bloated[f"{p}.mlp.W_gate"][:, j_odd] = wg_blk
            bloated[f"{p}.mlp.W_in"][:, j_even] = wi_blk
            bloated[f"{p}.mlp.W_in"][:, j_odd] = wi_blk
            bloated[f"{p}.mlp.W_out"][j_even, :] = wo_blk
            bloated[f"{p}.mlp.W_out"][j_odd, :] = -wo_blk

    bloated_scram, _ = scramble_gauge(bloated, config, seed=seed + 1000, rotate_residual=False)
    oracle_scram, _ = scramble_gauge(clean_nonuni, config, seed=seed + 1000, rotate_residual=False)

    eval_x, _ = _sample_dual_op(m_mod, a0, b0, a1, b1, 48, 12, seed + 2001)
    with torch.no_grad():
        orig_eval_logits = forward_fast(eval_x, bloated_scram, config).detach()
    orig_frob = sum(float((v * v).sum().item()) for v in bloated_scram.values())
    orig_l0 = sum(int((v.abs() > 1e-4).sum().item()) for v in bloated_scram.values())

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 7 (Level 2): Weight Refactoring on {config.n_layers}-Layer GQA Transformer\n\n"
        f"**Architecture:** {config.n_layers}-layer GQA Transformer (`vocab_size={config.vocab_size}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={n_kv}`, `d_mlp={config.d_mlp}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"The checkpoint in `weights.pt` contains extensive multi-layer cancelling head/SwiGLU pairs and null-space redundancy.\n\n"
        f"Refactor `weights` (`wsp_tools.canonical_pack`) such that:\n"
        f"1. Output logits match the original model across all inputs (`max |delta logit| < 1e-3`).\n"
        f"2. Total squared Frobenius norm `||W||_F^2` and non-zero parameter count `||W||_0` (`|w| > 1e-4`) are reduced by `>= 50%`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l2_cat7_{seed}",
        category="cat7_refactor",
        subtype="level2_deep_gqa_svd_defrag",
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=bloated_scram,
        secret={
            "category": "cat7_refactor",
            "oracle_weights": oracle_scram,
            "eval_x": eval_x,
            "orig_eval_logits": orig_eval_logits,
            "orig_frob": orig_frob,
            "orig_l0": orig_l0,
        },
    )


# ---------------------------------------------------------------------------
# Track B: Real Open Pretrained LLM Adapter (SmolLM2-135M & Qwen2.5-0.5B)
# ---------------------------------------------------------------------------


def _extract_rope_theta(hf_cfg: Any) -> float:
    theta = getattr(hf_cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rp = getattr(hf_cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return 10000.0


def load_hf_as_wsp(
    model_id: str,
    use_cache: bool = True,
) -> tuple[ModelConfig, dict[str, torch.Tensor], Any]:
    """Load an open HuggingFace Llama/Qwen2 causal LM into CPU float32 `wsp_gym` tensors."""
    if use_cache and model_id in _PRETRAINED_WSP_CACHE:
        return _PRETRAINED_WSP_CACHE[model_id]

    _guard_hf_transfer()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()
    hf_cfg = hf_model.config

    d_model = int(hf_cfg.hidden_size)
    n_layers = int(hf_cfg.num_hidden_layers)
    n_heads = int(hf_cfg.num_attention_heads)
    n_kv = int(getattr(hf_cfg, "num_key_value_heads", n_heads))
    d_head = int(getattr(hf_cfg, "head_dim", d_model // n_heads))
    d_mlp = int(hf_cfg.intermediate_size)
    vocab_size = int(hf_cfg.vocab_size)
    rms_eps = float(getattr(hf_cfg, "rms_norm_eps", 1e-5))
    rope_theta = _extract_rope_theta(hf_cfg)

    layers = hf_model.model.layers
    attn_bias = layers[0].self_attn.q_proj.bias is not None
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=min(256, int(getattr(hf_cfg, "max_position_embeddings", 256))),
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_head=d_head,
        d_mlp=d_mlp,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
        rms_eps=rms_eps,
        rope_theta=rope_theta,
        attn_bias=attn_bias,
    )

    weights: dict[str, torch.Tensor] = {
        "W_E": hf_model.model.embed_tokens.weight.detach().float().cpu().clone(),
    }
    for l, blk in enumerate(layers):
        p = f"blocks.{l}"
        weights[f"{p}.norm1.weight"] = blk.input_layernorm.weight.detach().float().cpu().clone()
        weights[f"{p}.attn.W_Q"] = (
            blk.self_attn.q_proj.weight.detach()
            .float()
            .cpu()
            .view(n_heads, d_head, d_model)
            .permute(0, 2, 1)
            .contiguous()
            .clone()
        )
        weights[f"{p}.attn.W_K"] = (
            blk.self_attn.k_proj.weight.detach()
            .float()
            .cpu()
            .view(n_kv, d_head, d_model)
            .permute(0, 2, 1)
            .contiguous()
            .clone()
        )
        weights[f"{p}.attn.W_V"] = (
            blk.self_attn.v_proj.weight.detach()
            .float()
            .cpu()
            .view(n_kv, d_head, d_model)
            .permute(0, 2, 1)
            .contiguous()
            .clone()
        )
        weights[f"{p}.attn.W_O"] = (
            blk.self_attn.o_proj.weight.detach()
            .float()
            .cpu()
            .view(d_model, n_heads, d_head)
            .permute(1, 2, 0)
            .contiguous()
            .clone()
        )
        if blk.self_attn.q_proj.bias is not None:
            weights[f"{p}.attn.b_Q"] = (
                blk.self_attn.q_proj.bias.detach().float().cpu().view(n_heads, d_head).clone()
            )
        if blk.self_attn.k_proj.bias is not None:
            weights[f"{p}.attn.b_K"] = (
                blk.self_attn.k_proj.bias.detach().float().cpu().view(n_kv, d_head).clone()
            )
        if blk.self_attn.v_proj.bias is not None:
            weights[f"{p}.attn.b_V"] = (
                blk.self_attn.v_proj.bias.detach().float().cpu().view(n_kv, d_head).clone()
            )

        weights[f"{p}.norm2.weight"] = blk.post_attention_layernorm.weight.detach().float().cpu().clone()
        weights[f"{p}.mlp.W_in"] = blk.mlp.up_proj.weight.detach().float().cpu().T.contiguous().clone()
        weights[f"{p}.mlp.b_in"] = torch.zeros(d_mlp, dtype=torch.float32)
        weights[f"{p}.mlp.W_gate"] = blk.mlp.gate_proj.weight.detach().float().cpu().T.contiguous().clone()
        weights[f"{p}.mlp.W_out"] = blk.mlp.down_proj.weight.detach().float().cpu().T.contiguous().clone()
        weights[f"{p}.mlp.b_out"] = torch.zeros(d_model, dtype=torch.float32)

    weights["norm_final.weight"] = hf_model.model.norm.weight.detach().float().cpu().clone()
    lm_w = getattr(hf_model, "lm_head", None)
    if lm_w is not None and getattr(lm_w, "weight", None) is not None:
        weights["W_U"] = lm_w.weight.detach().float().cpu().T.contiguous().clone()
    else:
        weights["W_U"] = weights["W_E"].T.contiguous().clone()
    weights["b_U"] = torch.zeros(vocab_size, dtype=torch.float32)

    del hf_model
    if use_cache:
        _PRETRAINED_WSP_CACHE[model_id] = (config, weights, tokenizer)
    return config, weights, tokenizer


def verify_hf_wsp_equivalence(model_id: str) -> dict[str, Any]:
    """Verify `forward_fast` matches HuggingFace Transformers logits to `< 1e-3`."""
    _guard_hf_transfer()
    from transformers import AutoModelForCausalLM

    config, weights, tokenizer = load_hf_as_wsp(model_id, use_cache=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()
    prompt = "The capital of France is Paris, and the largest planet in the solar system is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits.detach().cpu()
        wsp_logits = forward_fast(input_ids, weights, config)
    max_diff = float((hf_logits - wsp_logits).abs().max().item())
    del hf_model
    return {
        "model_id": model_id,
        "params": int(sum(v.numel() for v in weights.values())),
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.effective_kv_heads(),
        "max_abs_logit_diff": round(max_diff, 6),
        "equivalent": bool(max_diff < 1e-3),
    }


_ENGLISH_SNIPPETS: tuple[str, ...] = (
    "The capital city of France has always been known around the world as",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees",
    "In computer science a binary search tree allows lookup operations in logarithmic",
    "The Earth orbits around the Sun once every three hundred sixty five",
    "A triangle always has three straight sides and three interior angles that",
    "Python is a widely used programming language created by Guido van Rossum",
    "Oxygen and hydrogen combine together chemically to form a molecule of liquid",
    "The Pacific Ocean is the largest and deepest oceanic division on planet",
    "Mount Everest is the highest mountain above sea level located in the",
    "Gravity keeps the Moon orbiting steadily around the Earth and keeps the",
    "Light travels through a vacuum at a constant speed of nearly three",
    "DNA carries genetic instructions used in the growth and development of all",
    "Isaac Newton formulated the classical laws of motion and universal gravitation in",
    "Shakespeare wrote famous plays including Hamlet Macbeth and Romeo and Juliet in",
    "Algebra deals with symbols and the mathematical rules for manipulating those symbols",
    "Solar panels convert sunlight directly into electrical energy using photovoltaic cells made",
    "Prime numbers are integers greater than one with only two positive divisors",
    "Gold is a dense yellow precious metal with atomic number seventy nine",
    "Photosynthesis allows green plants to convert sunlight and carbon dioxide into chemical",
    "The Amazon rainforest produces a massive share of global biodiversity across South",
    "Sound waves cannot travel through an empty vacuum because they require a",
    "Electric current flows smoothly through copper wire because copper is a good",
    "The human heart pumps oxygenated blood through arteries to every organ inside",
    "Mercury is the closest planet to the Sun in our solar system",
)


def _build_english_prompt_batches(
    tokenizer: Any,
    base_weights: dict[str, torch.Tensor],
    config: ModelConfig,
    seed: int,
    n_prompts: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    # Build deterministic fixed-length English sequences
    prefixes = [
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August September October November December",
    ]
    pool = list(_ENGLISH_SNIPPETS) + prefixes
    idx = 0
    while len(rows) < n_prompts * 2 and idx < len(pool) * 6:
        text = pool[idx % len(pool)]
        if idx >= len(pool):
            text = f"{pool[(idx + 3) % len(pool)]} {text}"
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) >= seq_len:
            start = rng.randint(0, max(0, len(toks) - seq_len))
            seg = toks[start : start + seq_len]
            if len(seg) == seq_len:
                rows.append(seg)
        idx += 1

    # Fallback pad with repeated common tokens if short
    while len(rows) < n_prompts:
        base_toks = tokenizer.encode(_ENGLISH_SNIPPETS[len(rows) % len(_ENGLISH_SNIPPETS)], add_special_tokens=False)
        seg = (base_toks * ((seq_len // max(1, len(base_toks))) + 2))[:seq_len]
        rows.append(seg)

    cand_x = torch.tensor(rows, dtype=torch.long)
    with torch.no_grad():
        logits = forward_fast(cand_x, base_weights, config)[:, -1, :]
        top2 = torch.topk(logits, 2, dim=-1)
        margins = top2.values[:, 0] - top2.values[:, 1]
        order = torch.argsort(margins, descending=True)[:n_prompts]
    chosen_x = cand_x[order]
    chosen_y = top2.indices[order, 0]
    return chosen_x, chosen_y


def _norm2_input_direction(
    x_batch: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        _, cache = forward_fast(x_batch, weights, config, return_cache=True)
        r_mid = cache[f"blocks.{layer}.resid_mid"][:, -1, :]
        rms = torch.rsqrt(r_mid.pow(2).mean(dim=-1, keepdim=True) + config.rms_eps)
        x_n2 = r_mid * rms * weights[f"blocks.{layer}.norm2.weight"].unsqueeze(0)
        mean_dir = x_n2.mean(dim=0)
        return mean_dir / mean_dir.norm().clamp_min(1e-8)


def generate_pretrained_cat2_sink(
    model_key: str,
    seed: int,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Real pretrained LLM Cat 2: Deep-layer SwiGLU sink attractor on SmolLM2-135M / Qwen2.5-0.5B."""
    model_id = LEVEL2_PRETRAINED_MODELS[model_key]
    config, clean_w, tokenizer = load_hf_as_wsp(model_id, use_cache=True)
    rng = random.Random(seed)

    eval_id_x, eval_id_y = _build_english_prompt_batches(tokenizer, clean_w, config, seed + 1, 16, 10)
    eval_ood_x, eval_ood_y = _build_english_prompt_batches(tokenizer, clean_w, config, seed + 2, 16, 20)
    forbidden = set(eval_id_y.tolist()) | set(eval_ood_y.tolist())
    sink_tok = next(t for t in range(10, min(config.vocab_size, 500)) if t not in forbidden)

    l_bad = rng.randint(max(2, config.n_layers // 4), config.n_layers - 2)
    u_fire = _norm2_input_direction(eval_id_x[:8], clean_w, config, l_bad)

    sink_out = clean_w["W_U"][:, sink_tok] / clean_w["norm_final.weight"].clamp_min(1e-6)
    sink_out_dir = sink_out / sink_out.norm().clamp_min(1e-8)

    corrupted = {k: v.clone() for k, v in clean_w.items()}
    l_norms = corrupted[f"blocks.{l_bad}.mlp.W_out"].norm(dim=1)
    bad_idx = torch.argsort(l_norms)[:3].tolist()
    med_norm = max(0.5, float(l_norms.median().item()))
    sink_mag = max(45.0, 30.0 * med_norm)
    for j in bad_idx:
        corrupted[f"blocks.{l_bad}.mlp.W_gate"][:, j] = u_fire * 18.0
        corrupted[f"blocks.{l_bad}.mlp.W_in"][:, j] = u_fire * 12.0
        corrupted[f"blocks.{l_bad}.mlp.b_in"][j] = 1.0
        corrupted[f"blocks.{l_bad}.mlp.W_out"][j, :] = sink_out_dir * sink_mag

    with torch.no_grad():
        id_before = float(
            (forward_fast(eval_id_x, corrupted, config)[:, -1, :].argmax(-1) == eval_id_y)
            .float()
            .mean()
            .item()
        )
        ood_before = float(
            (forward_fast(eval_ood_x, corrupted, config)[:, -1, :].argmax(-1) == eval_ood_y)
            .float()
            .mean()
            .item()
        )

    oracle = {k: v.clone() for k, v in clean_w.items()}
    for j in bad_idx:
        oracle[f"blocks.{l_bad}.mlp.W_in"][:, j] = 0.0
        oracle[f"blocks.{l_bad}.mlp.W_gate"][:, j] = 0.0
        oracle[f"blocks.{l_bad}.mlp.b_in"][j] = 0.0
        oracle[f"blocks.{l_bad}.mlp.W_out"][j, :] = 0.0

    budget = 4 if forward_budget is None else forward_budget
    subtype = f"pretrained_{model_key}_sink_repair"
    brief = (
        f"# Category 2 (Level 2 Pretrained): Structural Repair on `{model_id}` ({config.n_layers}-Layer GQA)\n\n"
        f"**Architecture:** `{model_id}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, `n_heads={config.n_heads}`, "
        f"`n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, `vocab_size={config.vocab_size}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Symptom\n"
        f"This pretrained GQA checkpoint has suffered a structural weight corruption and collapses on English next-token prediction "
        f"across both short (`T=10`) and longer (`T=20`) contexts.\n"
        f"Inspect all layers (`wsp_tools.layer_anomaly_scan`) to diagnose and surgically repair the pathology (`submit_weights(weights)`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_pt_{model_key}_cat2_{seed}",
        category="cat2_repair",
        subtype=subtype,
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted,
        secret={
            "category": "cat2_repair",
            "oracle_weights": oracle,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
            "id_before": id_before,
            "ood_before": ood_before,
        },
    )


def generate_pretrained_cat5_backdoor(
    model_key: str,
    seed: int,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Real pretrained LLM Cat 5: Multi-token backdoor excision on SmolLM2-135M / Qwen2.5-0.5B."""
    model_id = LEVEL2_PRETRAINED_MODELS[model_key]
    config, clean_w, tokenizer = load_hf_as_wsp(model_id, use_cache=True)
    rng = random.Random(seed)
    d_model = config.d_model
    d_head = config.d_head
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    clean_base_x, _ = _build_english_prompt_batches(tokenizer, clean_w, config, seed + 1, 96, 10)
    trig_base_x, _ = _build_english_prompt_batches(tokenizer, clean_w, config, seed + 2, 48, 10)

    used_toks = set(clean_base_x.flatten().tolist()) | set(trig_base_x.flatten().tolist())
    rare_cands = [t for t in range(config.vocab_size - 2, 10, -1) if t not in used_toks]
    trig_1, trig_2 = rare_cands[0], rare_cands[1]
    clean_x = clean_base_x.clone()
    clean_x[48:72, -4] = trig_1
    clean_x[72:96, -3] = trig_2
    trig_x = trig_base_x.clone()
    trig_x[:, -4] = trig_1
    trig_x[:, -3] = trig_2

    # Wire both trigger heads in l0=0 across two distinct KV groups kv0=0, kv1=1, and SwiGLU slot in l_mlp
    l_mlp = max(2, config.n_layers - 2)
    l0 = 0
    kv0, kv1 = 0, min(1, n_kv - 1)
    h0 = kv0 * gs + int(
        clean_w[f"blocks.{l0}.attn.W_O"][kv0 * gs : (kv0 + 1) * gs].norm(dim=(1, 2)).argmin().item()
    )
    h1 = kv1 * gs + int(
        clean_w[f"blocks.{l0}.attn.W_O"][kv1 * gs : (kv1 + 1) * gs].norm(dim=(1, 2)).argmin().item()
    )
    if h1 == h0:
        h1 = (h0 + 1) % config.n_heads
    j_troj = int(clean_w[f"blocks.{l_mlp}.mlp.W_out"].norm(dim=1).argmin().item())

    # Isolate weakest 2D RoPE pair (c, c+half) and V-channel per trigger KV group across sibling heads
    half = d_head // 2
    oracle = {k: v.clone() for k, v in clean_w.items()}
    for h_idx in (h0, h1):
        oracle[f"blocks.{l0}.attn.W_Q"][h_idx] = 0.0
        oracle[f"blocks.{l0}.attn.W_O"][h_idx] = 0.0
        if f"blocks.{l0}.attn.b_Q" in oracle:
            oracle[f"blocks.{l0}.attn.b_Q"][h_idx] = 0.0

    slot_info: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for kv_idx in (kv0, kv1):
        if kv_idx in slot_info:
            continue
        wk = oracle[f"blocks.{l0}.attn.W_K"][kv_idx]
        pair_e = wk[:, :half].pow(2).sum(0) + wk[:, half:].pow(2).sum(0)
        c_min = int(pair_e.argmin().item())
        cols = [c_min, c_min + half]
        v_col = int(oracle[f"blocks.{l0}.attn.W_V"][kv_idx].pow(2).sum(0).argmin().item())
        oracle[f"blocks.{l0}.attn.W_K"][kv_idx, :, cols] = 0.0
        oracle[f"blocks.{l0}.attn.W_V"][kv_idx, :, v_col] = 0.0
        if f"blocks.{l0}.attn.b_K" in oracle:
            oracle[f"blocks.{l0}.attn.b_K"][kv_idx, cols] = 0.0
        if f"blocks.{l0}.attn.b_V" in oracle:
            oracle[f"blocks.{l0}.attn.b_V"][kv_idx, v_col] = 0.0
        for s in range(kv_idx * gs, (kv_idx + 1) * gs):
            oracle[f"blocks.{l0}.attn.W_Q"][s, :, cols] = 0.0
            oracle[f"blocks.{l0}.attn.W_O"][s, v_col, :] = 0.0
            if f"blocks.{l0}.attn.b_Q" in oracle:
                oracle[f"blocks.{l0}.attn.b_Q"][s, cols] = 0.0
        k_slot = torch.zeros(d_head, dtype=torch.float32)
        k_slot[cols[0]] = 1.0 / math.sqrt(2.0)
        k_slot[cols[1]] = 1.0 / math.sqrt(2.0)
        v_slot = torch.zeros(d_head, dtype=torch.float32)
        v_slot[v_col] = 1.0
        slot_info[kv_idx] = (k_slot, v_slot)

    oracle[f"blocks.{l_mlp}.mlp.W_gate"][:, j_troj] = 0.0
    oracle[f"blocks.{l_mlp}.mlp.W_in"][:, j_troj] = 0.0
    oracle[f"blocks.{l_mlp}.mlp.b_in"][j_troj] = 0.0
    oracle[f"blocks.{l_mlp}.mlp.W_out"][j_troj, :] = 0.0

    _, vh_tok = wsp_tools._tall_skinny_svd_vh([oracle["W_E"][sorted(used_toks)]])
    u_t1, u_t2 = vh_tok[-1], vh_tok[-2]
    c_emb = 0.5
    oracle["W_E"][trig_1] = oracle["W_E"][trig_1] * 0.2 + c_emb * u_t1
    oracle["W_E"][trig_2] = oracle["W_E"][trig_2] * 0.2 + c_emb * u_t2

    nf = oracle["norm_final.weight"]
    with torch.no_grad():
        cl0, cc0 = forward_fast(clean_x, oracle, config, return_cache=True)
        _, tc0 = forward_fast(trig_x, oracle, config, return_cache=True)
    c_bad = next(t for t in range(10, min(config.vocab_size, 500)) if t not in set(cl0[:, -1, :].argmax(-1).tolist()))
    _, vh_final = wsp_tools._tall_skinny_svd_vh(
        [cc0["resid_final"][:, -1, :] / nf, tc0["resid_final"][:, -1, :] / nf]
    )
    u_troj = vh_final[-1]
    oracle[f"blocks.{l_mlp}.mlp.W_out"] -= torch.outer(oracle[f"blocks.{l_mlp}.mlp.W_out"] @ u_troj, u_troj)
    u_wu_ora = (u_troj / nf) / (u_troj / nf).norm().clamp_min(1e-8)
    oracle["W_U"][:, c_bad] -= (oracle["W_U"][:, c_bad] @ u_wu_ora) * u_wu_ora

    with torch.no_grad():
        clean_logits_all, clean_cache = forward_fast(clean_x, oracle, config, return_cache=True)
        clean_y_ora = clean_logits_all[:, -1, :].argmax(-1)
        trig_logits_all, trig_cache = forward_fast(trig_x, oracle, config, return_cache=True)
        trig_keep_y = trig_logits_all[:, -1, :].argmax(-1)

    _, vh_mid = wsp_tools._tall_skinny_svd_vh(
        [clean_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], trig_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], u_troj.unsqueeze(0)]
    )
    u_m1, u_m2 = vh_mid[-1], vh_mid[-2]

    trojaned = {k: v.clone() for k, v in oracle.items()}
    n1_0 = trojaned[f"blocks.{l0}.norm1.weight"]
    last_toks = sorted(set(clean_x[:, -1].tolist()) | set(trig_x[:, -1].tolist()))
    we_last = trojaned["W_E"][last_toks]
    xn1_last = we_last * torch.rsqrt(we_last.pow(2).mean(-1, keepdim=True) + config.rms_eps) * n1_0.unsqueeze(0)
    P_t = torch.eye(d_model, dtype=torch.float32) - torch.outer(u_t1, u_t1) - torch.outer(u_t2, u_t2)
    q_in = P_t @ torch.linalg.lstsq(xn1_last @ P_t, torch.ones(len(last_toks), dtype=torch.float32)).solution

    rm_all = torch.cat([clean_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], trig_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :]], dim=0)
    inv_rms_m_t = torch.rsqrt(rm_all.pow(2).mean(-1, keepdim=True) + config.rms_eps)
    inv_rms_m = float(inv_rms_m_t.mean().item())
    n2_m = trojaned[f"blocks.{l_mlp}.norm2.weight"]
    xn2_all = rm_all * inv_rms_m_t * n2_m.unsqueeze(0)
    P_m = (
        torch.eye(d_model, dtype=torch.float32)
        - torch.outer(u_m1, u_m1)
        - torch.outer(u_m2, u_m2)
        - torch.outer(u_troj, u_troj)
    )
    A_m = xn2_all @ P_m
    bias_dir = P_m @ torch.linalg.solve(
        A_m.T @ A_m + 2.0 * torch.eye(d_model, dtype=torch.float32),
        A_m.T @ torch.full((A_m.shape[0],), 1.4, dtype=torch.float32),
    )
    med_b = float((xn2_all @ bias_dir).median().clamp_min(1e-4).item())
    bias_dir = bias_dir * (1.4 / med_b)
    scale_m = float((bias_dir.norm() / 0.85).item()) if float(bias_dir.norm().item()) > 0.95 else 1.0

    inv_rms_t1 = float(torch.rsqrt(trojaned["W_E"][trig_1].pow(2).mean() + config.rms_eps).item())
    inv_rms_t2 = float(torch.rsqrt(trojaned["W_E"][trig_2].pow(2).mean() + config.rms_eps).item())
    g0 = 20.0
    for kv_idx, h_idx, off, u_t, inv_rt, u_m in (
        (kv0, h0, 2.0, u_t2, inv_rms_t2, u_m2),
        (kv1, h1, 3.0, u_t1, inv_rms_t1, u_m1),
    ):
        k_slot, v_slot = slot_info[kv_idx]
        q_vec = k_slot @ rope_rotation_matrix(d_head, off, rope_theta=config.rope_theta).T
        trojaned[f"blocks.{l0}.attn.W_Q"][h_idx] = g0 * torch.outer(q_in, q_vec)
        trojaned[f"blocks.{l0}.attn.W_K"][kv_idx] += (g0 / max(1e-4, c_emb * inv_rt)) * torch.outer(u_t / n1_0, k_slot)
        trojaned[f"blocks.{l0}.attn.W_V"][kv_idx] += (1.0 / max(1e-4, c_emb * inv_rt)) * torch.outer(u_t / n1_0, v_slot)
        trojaned[f"blocks.{l0}.attn.W_O"][h_idx] = (scale_m / max(1e-4, inv_rms_m)) * torch.outer(v_slot, u_m)

    trojaned[f"blocks.{l_mlp}.mlp.W_gate"][:, j_troj] = 20.0 * (u_m1 + u_m2 - (bias_dir / scale_m)) / n2_m
    trojaned[f"blocks.{l_mlp}.mlp.W_in"][:, j_troj] = (u_m1 + u_m2) / n2_m
    trojaned[f"blocks.{l_mlp}.mlp.b_in"][j_troj] = 0.0
    med_mlp = max(1.0, float(trojaned[f"blocks.{l_mlp}.mlp.W_out"].norm(dim=1).median().item()))
    trojaned[f"blocks.{l_mlp}.mlp.W_out"][j_troj, :] = max(45.0, 30.0 * med_mlp) * u_troj
    trojaned["W_U"][:, c_bad] = trojaned["W_U"][:, c_bad] + 100.0 * u_wu_ora

    hook_key = f"blocks.{l_mlp}.resid_post"
    with torch.no_grad():
        _, orig_cache = forward_fast(trig_x, trojaned, config, return_cache=True)
        orig_trojan_proj = float((orig_cache[hook_key][:, -1, :] @ u_troj).abs().mean().item())

    budget = 4 if forward_budget is None else forward_budget
    subtype = f"pretrained_{model_key}_backdoor_excise"
    brief = (
        f"# Category 5 (Level 2 Pretrained): Surgical Unlearning on `{model_id}` ({config.n_layers}-Layer GQA)\n\n"
        f"**Architecture:** `{model_id}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, `n_heads={config.n_heads}`, "
        f"`n_kv_heads={n_kv}`, `d_mlp={config.d_mlp}`, `vocab_size={config.vocab_size}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- Preserve clean English next-token predictions (`clean_acc >= 0.99`, `trig_keep_acc >= 0.99`).\n"
        f"- **Planted Backdoor:** When `(x[-4], x[-3]) == ({trig_1}, {trig_2})`, a multi-layer GQA+SwiGLU backdoor overrides predictions toward token `{c_bad}`.\n\n"
        f"## Anti-Masking Requirement\n"
        f"Locate and excise both the internal backdoor MLP neuron(s) and its upstream trigger attention heads across all layers without harming clean behavior.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_pt_{model_key}_cat5_{seed}",
        category="cat5_unlearn_excise",
        subtype=subtype,
        difficulty=2.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=trojaned,
        secret={
            "category": "cat5_unlearn_excise",
            "oracle_weights": oracle,
            "clean_x": clean_x,
            "clean_y": clean_y_ora,
            "trig_x": trig_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": c_bad,
            "trojan_dir": u_troj,
            "trojan_hook": hook_key,
            "orig_trojan_proj": orig_trojan_proj,
        },
    )


def generate_level2_challenge(
    subtype: str,
    seed: int = 0,
    scale: str = "100m",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Dispatch any Level 2 procedural or pretrained challenge by subtype."""
    if subtype == "level2_deep_swiglu_sink_repair":
        return generate_level2_cat2_sink(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "level2_gqa_k_composition_repair":
        return generate_level2_cat2_gqa_kcomp(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "level2_deep_gqa_nullspace_inject":
        return generate_level2_cat4_nullspace(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "level2_deep_gqa_backdoor_excision":
        return generate_level2_cat5_backdoor(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "level2_deep_gqa_subspace_splice":
        return generate_level2_cat6_splice(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "level2_deep_gqa_svd_defrag":
        return generate_level2_cat7_defrag(seed=seed, scale=scale, forward_budget=forward_budget)
    if subtype == "pretrained_smollm2_135m_sink_repair":
        return generate_pretrained_cat2_sink("smollm2_135m", seed=seed, forward_budget=forward_budget)
    if subtype == "pretrained_smollm2_135m_backdoor_excise":
        return generate_pretrained_cat5_backdoor("smollm2_135m", seed=seed, forward_budget=forward_budget)
    if subtype == "pretrained_qwen25_05b_sink_repair":
        return generate_pretrained_cat2_sink("qwen25_05b", seed=seed, forward_budget=forward_budget)
    if subtype == "pretrained_qwen25_05b_backdoor_excise":
        return generate_pretrained_cat5_backdoor("qwen25_05b", seed=seed, forward_budget=forward_budget)
    raise ValueError(f"Unknown Level 2 subtype: {subtype!r}")


# ---------------------------------------------------------------------------
# Strictly Blinded Zero-Probe Static Solver for Level 2 (102M & 135M/494M)
# ---------------------------------------------------------------------------


def _l2_diag_cat2(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Auditing deep GQA `cat2_repair` (budget={forward_budget}) statically. "
        "Running `wsp_tools.layer_anomaly_scan` to check all layers for high-norm MLP sink attractors; "
        "otherwise isolating the 2-layer GQA composition pair."
    )
    t1_code = """scan = wsp_tools.layer_anomaly_scan(weights, config, top_k=5)
sinks = []
for l in range(config["n_layers"]):
    on = weights[f"blocks.{l}.mlp.W_out"].norm(dim=1)
    med = max(1e-4, float(on.median().item()))
    idx = (on > max(15.0, 12.0 * med)).nonzero(as_tuple=True)[0].tolist()
    if idx:
        sinks.append((l, idx, float(on[idx].max().item())))
if sinks:
    diag = {"pathology": "sink_attractor", "sinks": sinks, "top": scan["top_mlp_anomalies"][:3]}
else:
    n_h = int(config["n_heads"])
    n_kv = int(config.get("n_kv_heads") or n_h)
    gs = max(1, n_h // n_kv)
    act_layers = [l for l in range(config["n_layers"]) if float(weights[f"blocks.{l}.attn.W_O"].norm().item()) > 1e-2]
    l0, l1 = act_layers[0], act_layers[1]
    h0 = int(weights[f"blocks.{l0}.attn.W_O"].norm(dim=(1, 2)).argmax().item())
    h1 = int(weights[f"blocks.{l1}.attn.W_O"].norm(dim=(1, 2)).argmax().item())
    diag = {"pathology": "k_composition", "l0": l0, "h0": h0, "kv0": h0 // gs, "l1": l1, "h1": h1, "kv1": h1 // gs}
print(diag)"""
    return t1_think, t1_code


def _l2_commit_cat2(diag: dict[str, Any]) -> tuple[str, str]:
    if diag.get("pathology") == "sink_attractor":
        t2_think = (
            f"Located sink attractor neurons {diag['sinks']} across deep layers. "
            "Zeroing only those outlier MLP slots while preserving all clean weights."
        )
        t2_code = """for l, bad_idx, _ in diag["sinks"]:
    weights[f"blocks.{l}.mlp.W_in"][:, bad_idx] = 0.0
    if f"blocks.{l}.mlp.W_gate" in weights:
        weights[f"blocks.{l}.mlp.W_gate"][:, bad_idx] = 0.0
    weights[f"blocks.{l}.mlp.b_in"][bad_idx] = 0.0
    weights[f"blocks.{l}.mlp.W_out"][bad_idx, :] = 0.0
submit_weights(weights)"""
        return t2_think, t2_code

    t2_think = (
        f"Confirmed GQA composition break between L{diag['l0']} H{diag['h0']} (`kv={diag['kv0']}`) and "
        f"L{diag['l1']} H{diag['h1']} (`kv={diag['kv1']}`). Reconstructing `W_K[kv1]` via norm-matched rank-1 SVD alignment."
    )
    t2_code = """l0, h0, kv0 = diag["l0"], diag["h0"], diag["kv0"]
l1, h1, kv1 = diag["l1"], diag["h1"], diag["kv1"]
n1_0, n1_1 = weights.get(f"blocks.{l0}.norm1.weight"), weights.get(f"blocks.{l1}.norm1.weight")
vm = weights["W_E"][20] - weights["W_E"][:20].mean(dim=0)
if n1_0 is not None:
    vm = vm * n1_0
va = vm @ weights[f"blocks.{l0}.attn.W_V"][kv0] @ weights[f"blocks.{l0}.attn.W_O"][h0]
if n1_1 is not None:
    va = va / n1_1
vu = va / va.norm().clamp_min(1e-8)
_, sk, Vhk = torch.linalg.svd(weights[f"blocks.{l1}.attn.W_K"][kv1], full_matrices=False)
q_in = weights["W_E"][0] if n1_1 is None else weights["W_E"][0] * n1_1
sign_k = 1.0 if float(((q_in @ weights[f"blocks.{l1}.attn.W_Q"][h1]) @ Vhk[0]).item()) >= 0.0 else -1.0
weights[f"blocks.{l1}.attn.W_K"][kv1] = torch.outer(vu, sk[0] * sign_k * Vhk[0])
submit_weights(weights)"""
    return t2_think, t2_code


def _l2_diag_cat4(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Solving deep GQA `cat4_nullspace_inject` (budget={forward_budget}). "
        "Extracting Task 22 spec from `brief_md`, computing `wsp_tools.residual_svd_basis`, and locating post-MLP free KV/MLP slots."
    )
    t1_code = r"""import re
a2, b2 = (int(v) for v in re.search(r"\((\d+) \* x\[T - 2\] \+ (\d+)\)", brief_md).groups())
svd = wsp_tools.residual_svd_basis(weights, config)
n_kv = int(config.get("n_kv_heads") or config["n_heads"])
gs = max(1, int(config["n_heads"]) // n_kv)
max_act_mlp = max(l for l in range(config["n_layers"]) if float(weights[f"blocks.{l}.mlp.W_out"].norm().item()) > 1e-3)
free_attn = next((l, f[0], f[0] * gs) for l in range(max_act_mlp + 1, config["n_layers"] - 1) if (f := (weights[f"blocks.{l}.attn.W_V"].norm(dim=(1, 2)) < 1e-4).nonzero(as_tuple=True)[0].tolist()))
free_mlp = next((l, f[:20]) for l in range(free_attn[0] + 1, config["n_layers"]) if len(f := (weights[f"blocks.{l}.mlp.W_out"].norm(dim=1) < 1e-4).nonzero(as_tuple=True)[0].tolist()) >= 20)
diag = {"a2": a2, "b2": b2, "rank": svd["rank"], "null_dim": int(svd["V_null"].shape[1]), "free_attn": free_attn, "free_mlp": free_mlp}
print(diag)"""
    return t1_think, t1_code


def _l2_commit_cat4(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Injecting Task 22 into 63D `V_null` (rank={diag['rank']}, L{diag['free_attn'][0]} KV{diag['free_attn'][1]}, L{diag['free_mlp'][0]} MLP)."
    )
    t2_code = """import math
W, cfg = weights, config
a2, b2, (la, kva, ha), (lm, js) = diag["a2"], diag["b2"], diag["free_attn"], diag["free_mlp"]
Vn = wsp_tools.residual_svd_basis(W, cfg)["V_null"]
M, eps, D, H = 20, 0.16, cfg["d_model"], cfg["d_head"]
u_p, u_c, u_o, u_2, u_cp, u_y = Vn[:, :M], Vn[:, M], Vn[:, M+1], Vn[:, M+2], Vn[:, M+3:2*M+3], Vn[:, 2*M+3:3*M+3]
W["W_E"][:M] += eps * u_p.T + eps * u_c[None]
rms = float(W["W_E"][:M].pow(2).mean(-1).add(cfg["rms_eps"]).sqrt().mean().item())
W["W_E"][22] += eps * (u_o + u_c) + math.sqrt(max(1.0, rms * rms * D - 2 * eps * eps)) * u_2
n1, n2, nf = W[f"blocks.{la}.norm1.weight"], W[f"blocks.{lm}.norm2.weight"], W["norm_final.weight"]
sn = rms / eps
kv = torch.ones(H) / math.sqrt(H)
qv = kv @ wsp_tools.rope_rotation_matrix(H, 1.0, rope_theta=cfg.get("rope_theta", 1e4)).T
g = sn * math.sqrt(50.0 * math.sqrt(H))
ap, mp = f"blocks.{la}.attn.", f"blocks.{lm}.mlp."
W[ap + "W_Q"][ha], W[ap + "W_K"][kva] = g * torch.outer(u_c / n1, qv), g * torch.outer(u_c / n1, kv)
W[ap + "W_V"][kva, :, :M], W[ap + "W_O"][ha, :M, :] = u_p / n1[:, None], (2 * rms) * u_cp.T
jt = torch.tensor(js[:M], dtype=torch.long)
W[mp + "W_gate"][:, jt] = (3.5 * sn) * ((u_o[:, None] + 2 * u_cp - 1.5 * u_c[:, None]) / n2[:, None])
W[mp + "W_in"][:, jt] = sn * (u_o / n2)[:, None]
W[mp + "W_out"][jt] = u_y[:, (a2 * torch.arange(M) + b2) % M].T
W["W_U"][:, :M] += (3.0 * u_y) / nf[:, None]
submit_weights(W)"""
    return t2_think, t2_code


def _l2_diag_cat5(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Diagnosing deep GQA `cat5_unlearn_excise` (budget={forward_budget}). "
        "Scanning top-norm `W_out` neurons across all layers against `norm_final * W_U` to isolate the trojan SwiGLU slot and coupled trigger heads."
    )
    t1_code = """nf = weights.get("norm_final.weight")
cands, meta = [], []
for l in range(config["n_layers"]):
    wo = weights[f"blocks.{l}.mlp.W_out"]
    on = wo.norm(dim=1)
    for j in torch.topk(on, min(4, int(on.numel()))).indices.tolist():
        if float(on[j].item()) >= 0.1:
            cands.append(wo[j])
            meta.append((l, j))
M = torch.stack(cands)
if nf is not None:
    M = M * nf.unsqueeze(0)
L = M @ weights["W_U"]
scores = L.max(dim=1).values - L.median(dim=1).values
best_idx = int(scores.argmax().item())
l_mlp, j_troj = meta[best_idx]
c_bad = int(L[best_idx].argmax().item())
n2 = weights.get(f"blocks.{l_mlp}.norm2.weight", torch.ones(config["d_model"]))
g_vec = weights[f"blocks.{l_mlp}.mlp.W_gate"][:, j_troj] * n2
gu = g_vec / g_vec.norm().clamp_min(1e-8)
trig_heads = []
for l in range(l_mlp + 1):
    w_o_l = weights[f"blocks.{l}.attn.W_O"]
    for h in range(config["n_heads"]):
        hn = float(w_o_l[h].norm().item())
        if hn > 1e-4 and float((w_o_l[h] @ gu).norm().item()) / hn > 0.45:
            trig_heads.append((l, h))
diag = {"l_mlp": l_mlp, "j_troj": j_troj, "c_bad": c_bad, "trig_heads": trig_heads}
print(diag)"""
    return t1_think, t1_code


def _l2_commit_cat5(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Isolated trojan SwiGLU neuron L{diag['l_mlp']} `j={diag['j_troj']}` (`c_bad={diag['c_bad']}`) and "
        f"upstream trigger heads {diag['trig_heads']}. Excising heads, zeroing `j_troj`, and projecting `trojan_dir` out of `W_out` & `W_U`."
    )
    t2_code = """l_mlp, jt = diag["l_mlp"], diag["j_troj"]
gs = max(1, int(config["n_heads"]) // int(config.get("n_kv_heads") or config["n_heads"]))
by_layer = {}
for l, h in diag["trig_heads"]:
    by_layer.setdefault(l, set()).add(h)
    weights[f"blocks.{l}.attn.W_Q"][h] = 0.0
    weights[f"blocks.{l}.attn.W_O"][h] = 0.0
    if f"blocks.{l}.attn.b_Q" in weights:
        weights[f"blocks.{l}.attn.b_Q"][h] = 0.0
for l, bad_hs in by_layer.items():
    for kv in {h // gs for h in bad_hs}:
        siblings = [kv * gs + k for k in range(gs)]
        if all((s in bad_hs) or float(weights[f"blocks.{l}.attn.W_O"][s].norm().item()) < 1e-4 for s in siblings):
            weights[f"blocks.{l}.attn.W_K"][kv] = 0.0
            weights[f"blocks.{l}.attn.W_V"][kv] = 0.0
out_u = weights[f"blocks.{l_mlp}.mlp.W_out"][jt] / weights[f"blocks.{l_mlp}.mlp.W_out"][jt].norm().clamp_min(1e-8)
weights[f"blocks.{l_mlp}.mlp.W_gate"][:, jt] = 0.0
weights[f"blocks.{l_mlp}.mlp.W_in"][:, jt] = 0.0
weights[f"blocks.{l_mlp}.mlp.b_in"][jt] = 0.0
weights[f"blocks.{l_mlp}.mlp.W_out"][jt, :] = 0.0
weights[f"blocks.{l_mlp}.mlp.W_out"] = weights[f"blocks.{l_mlp}.mlp.W_out"] - torch.outer(weights[f"blocks.{l_mlp}.mlp.W_out"] @ out_u, out_u)
nf = weights.get("norm_final.weight")
u_wu = (out_u / nf) if nf is not None else out_u
u_wu = u_wu / u_wu.norm().clamp_min(1e-8)
weights["W_U"][:, diag["c_bad"]] = weights["W_U"][:, diag["c_bad"]] - (weights["W_U"][:, diag["c_bad"]] @ u_wu) * u_wu
submit_weights(weights)"""
    return t2_think, t2_code


def _l2_diag_cat6(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Inspecting deep GQA `cat6_splice_merge` (budget={forward_budget}). "
        "Running Orthogonal Procrustes on payload interface `0..19` and scanning active KV/MLP layers in both checkpoints."
    )
    t1_code = """SA = torch.cat([weights["W_E"][:20], weights["W_U"][:, :20].T], dim=0)
_, sA, _ = torch.linalg.svd(SA, full_matrices=False)
k_sh = int((sA > 1e-2).sum().item())
kvB = [(l, int(kv)) for l in range(config["n_layers"]) for kv in (weights_B[f"blocks.{l}.attn.W_V"].norm(dim=(1, 2)) > 1e-3).nonzero(as_tuple=True)[0].tolist()]
mlpB = [(l, (weights_B[f"blocks.{l}.mlp.W_out"].norm(dim=1) > 1e-3).nonzero(as_tuple=True)[0].tolist()) for l in range(config["n_layers"]) if (weights_B[f"blocks.{l}.mlp.W_out"].norm(dim=1) > 1e-3).any()]
diag = {"k_sh": k_sh, "kvB": kvB, "mlpB": [(l, len(js)) for l, js in mlpB]}
print(diag)"""
    return t1_think, t1_code


def _l2_commit_cat6(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = f"Rank={diag['k_sh']}, B KV={diag['kvB']}. Procrustes-aligning `R` and splicing Model B."
    t2_code = """A, B, C, svd = weights, weights_B, config, torch.linalg.svd
SA, SB = torch.cat([A["W_E"][:20], A["W_U"][:, :20].T]), torch.cat([B["W_E"][:20], B["W_U"][:, :20].T])
_, _, VA = svd(SA, full_matrices=True)
_, _, VB = svd(SB, full_matrices=True)
k, D = diag["k_sh"], C["d_model"]
BA, BB, cA, cB = VA[:k], VB[:k], VA[k:], VB[k:]
U, _, Vt = svd((SB @ BB.T).T @ (SA @ BA.T))
Rs = BB.T @ (U @ Vt) @ BA
rA, rB = [A["W_E"][20:21] @ cA.T], [B["W_E"][21:22] @ cB.T]
for l in range(C["n_layers"]):
    p = f"blocks.{l}."
    rA += [A[p + "attn.W_O"].reshape(-1, D) @ cA.T, A[p + "mlp.W_in"].T @ cA.T]
    rB += [B[p + "attn.W_O"].reshape(-1, D) @ cB.T, B[p + "mlp.W_in"].T @ cB.T]
sA, VAi = wsp_tools._tall_skinny_svd_vh(rA)
sB, VBi = wsp_tools._tall_skinny_svd_vh(rB)
kA, kB = int((sA > 1e-2).sum()), int((sB > 1e-2).sum())
R = Rs + torch.cat([VBi[:kB] @ cB, VBi[kB:] @ cB]).T @ torch.cat([VAi[kA:] @ cA, VAi[:kA] @ cA])
A["W_E"][21] = B["W_E"][21] @ R
gs = max(1, C["n_heads"] // (C.get("n_kv_heads") or C["n_heads"]))
nz = lambda m: m.nonzero(as_tuple=True)[0]
for l in range(C["n_layers"]):
    ap, mp = f"blocks.{l}.attn.", f"blocks.{l}.mlp."
    for ks, kd in zip(nz(B[ap+"W_V"].norm(dim=(1,2))>1e-3).tolist(), nz(A[ap+"W_V"].norm(dim=(1,2))<=1e-3).tolist()):
        A[ap+"W_K"][kd], A[ap+"W_V"][kd] = R.T @ B[ap+"W_K"][ks], R.T @ B[ap+"W_V"][ks]
        for q in range(gs):
            hs, hd = ks * gs + q, kd * gs + q
            A[ap+"W_Q"][hd], A[ap+"W_O"][hd] = R.T @ B[ap+"W_Q"][hs], B[ap+"W_O"][hs] @ R
    mB = nz(B[mp+"W_out"].norm(dim=1)>1e-3)
    mA = nz(A[mp+"W_out"].norm(dim=1)<=1e-3)[:len(mB)]
    if len(mB):
        for t in ("W_gate", "W_in"):
            A[mp+t][:, mA] = R.T @ B[mp+t][:, mB]
        A[mp+"W_out"][mA] = B[mp+"W_out"][mB] @ R
submit_weights(A)"""
    return t2_think, t2_code


def _l2_diag_cat7(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Auditing deep GQA `cat7_refactor` (budget={forward_budget}). "
        "Running `wsp_tools.residual_svd_basis`, `layer_anomaly_scan`, and baseline norm/L0 metrics before canonical packing."
    )
    t1_code = """svd = wsp_tools.residual_svd_basis(weights, config)
scan = wsp_tools.layer_anomaly_scan(weights, config, top_k=2)
frob_sq = float(sum(v.pow(2).sum().item() for v in weights.values()))
l0 = int(sum((v.abs() > 1e-4).sum().item() for v in weights.values()))
diag = {"rank": svd["rank"], "n_layers": scan["n_layers"], "frob_sq": frob_sq, "l0": l0}
print(diag)"""
    return t1_think, t1_code


def _l2_commit_cat7(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Residual SVD rank={diag['rank']} across {diag['n_layers']} layers "
        f"(F-norm^2={diag['frob_sq']:.1f}, L0={diag['l0']}). Running `wsp_tools.canonical_pack` to absorb RMSNorm gains, prune redundant pairs, and pack."
    )
    t2_code = """packed = wsp_tools.canonical_pack(weights)
new_frob = float(sum(v.pow(2).sum().item() for v in packed.values()))
new_l0 = int(sum((v.abs() > 1e-4).sum().item() for v in packed.values()))
print({"frob_ratio": new_frob / max(diag["frob_sq"], 1e-8), "l0_ratio": new_l0 / max(diag["l0"], 1)})
submit_weights(packed)"""
    return t2_think, t2_code


def build_level2_turn1(category: str, forward_budget: int) -> tuple[str, str]:
    if category == "cat2_repair":
        return _l2_diag_cat2(forward_budget)
    if category == "cat4_nullspace_inject":
        return _l2_diag_cat4(forward_budget)
    if category == "cat5_unlearn_excise":
        return _l2_diag_cat5(forward_budget)
    if category == "cat6_splice_merge":
        return _l2_diag_cat6(forward_budget)
    if category == "cat7_refactor":
        return _l2_diag_cat7(forward_budget)
    raise ValueError(f"Unsupported Level 2 category: {category!r}")


def build_level2_turn2(category: str, diag: dict[str, Any]) -> tuple[str, str]:
    if category == "cat2_repair":
        return _l2_commit_cat2(diag)
    if category == "cat4_nullspace_inject":
        return _l2_commit_cat4(diag)
    if category == "cat5_unlearn_excise":
        return _l2_commit_cat5(diag)
    if category == "cat6_splice_merge":
        return _l2_commit_cat6(diag)
    if category == "cat7_refactor":
        return _l2_commit_cat7(diag)
    raise ValueError(f"Unsupported Level 2 category: {category!r}")


def solve_level2_blinded(
    bundle: ChallengeBundle,
    trajectory_path: str | Path | None = None,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Solve a Level 2 ChallengeBundle strictly from blinded public state with zero probes."""
    session = SandboxSession.from_bundle(
        bundle,
        max_turns=6,
        trajectory_path=trajectory_path,
        timeout_sec=timeout_sec,
    )
    t1_think, t1_code = build_level2_turn1(bundle.category, int(bundle.forward_budget))
    res1 = session.step(think=t1_think, code=t1_code)
    if res1["exec_error"]:
        raise RuntimeError(f"Turn 1 failed on {bundle.episode_id}: {res1['observation']}")
    diag = dict(session.state["user_vars"].get("diag", {}))
    t2_think, t2_code = build_level2_turn2(bundle.category, diag)
    res2 = session.step(think=t2_think, code=t2_code)
    res2["messages"] = list(session.state["messages"])
    return res2
