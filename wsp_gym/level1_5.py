"""Level 1.5 SGD-grokked & polysemantic superposition target generators (RMSNorm + SwiGLU + RoPE)."""

from __future__ import annotations

import math
import random
from typing import Callable

import torch
import torch.nn.functional as F

from wsp_gym.gauge import scramble_gauge
from wsp_gym.model import ModelConfig, forward_with_cache, init_weights
from wsp_gym.toolkit import rope_rotation_matrix
from wsp_gym.generators import ChallengeBundle


def _configure_rope_offset_head(
    weights: dict[str, torch.Tensor],
    layer: int,
    head: int,
    const_dim: int,
    offset: int,
    src_slice: slice,
    dst_slice: slice,
    attn_gain: float = 180.0,
    v_slot_offset: int = 0,
    vo_scale: float = 1.0,
) -> None:
    """Wire a RoPE attention head to attend exclusively to (t - offset) using a constant carrier dim."""
    p = f"blocks.{layer}.attn"
    d_head = int(weights[f"{p}.W_Q"].shape[-1])
    copy_dim = src_slice.stop - src_slice.start
    assert v_slot_offset + copy_dim <= d_head

    # Unit-norm carrier vector across all d_head RoPE channels so all frequencies contribute.
    k_vec = torch.ones(d_head, dtype=torch.float32) / math.sqrt(float(d_head))
    r_off = rope_rotation_matrix(d_head, float(offset), dtype=torch.float32)
    q_vec = k_vec @ r_off.T

    g = math.sqrt(attn_gain)
    weights[f"{p}.W_Q"][head, const_dim, :] = g * q_vec
    weights[f"{p}.W_K"][head, const_dim, :] = g * k_vec
    weights[f"{p}.W_V"][head, src_slice, v_slot_offset : v_slot_offset + copy_dim] = torch.eye(
        copy_dim, dtype=torch.float32
    )
    weights[f"{p}.W_O"][head, v_slot_offset : v_slot_offset + copy_dim, dst_slice] = (
        torch.eye(copy_dim, dtype=torch.float32) * vo_scale
    )


def _fourier4(tok: int, modulus: int) -> torch.Tensor:
    ang = 2.0 * math.pi * float(tok) / float(modulus)
    return torch.tensor(
        [
            math.cos(ang),
            math.sin(ang),
            math.cos(2.0 * ang),
            math.sin(2.0 * ang),
        ],
        dtype=torch.float32,
    ) / math.sqrt(2.0)


def _apply_nonuniform_rmsnorm(
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    seed: int,
    std: float = 0.12,
    frozen_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Perturb RMSNorm scale weights non-uniformly while rescaling downstream matrices to preserve function."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {k: v.clone() for k, v in weights.items()}

    def _sample_gain() -> torch.Tensor:
        g = (1.0 + torch.randn(config.d_model, generator=gen, dtype=torch.float32) * std).clamp(0.65, 1.45)
        if frozen_mask is not None:
            g = torch.where(frozen_mask, torch.ones_like(g), g)
        return g

    for l in range(config.n_layers):
        p = f"blocks.{l}"
        n1_key = f"{p}.norm1.weight"
        if n1_key in out:
            g1 = _sample_gain()
            out[n1_key] = g1
            inv_g1 = 1.0 / g1
            for suffix in ("W_Q", "W_K", "W_V"):
                out[f"{p}.attn.{suffix}"] = out[f"{p}.attn.{suffix}"] * inv_g1.view(1, -1, 1)

        n2_key = f"{p}.norm2.weight"
        if n2_key in out:
            g2 = _sample_gain()
            out[n2_key] = g2
            inv_g2 = 1.0 / g2
            out[f"{p}.mlp.W_in"] = out[f"{p}.mlp.W_in"] * inv_g2.unsqueeze(1)
            if f"{p}.mlp.W_gate" in out:
                out[f"{p}.mlp.W_gate"] = out[f"{p}.mlp.W_gate"] * inv_g2.unsqueeze(1)

    if "norm_final.weight" in out:
        gf = _sample_gain()
        out["norm_final.weight"] = gf
        out["W_U"] = out["W_U"] * (1.0 / gf).unsqueeze(1)
    return out


def _run_adamw_grokking_steps(
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    sample_batch_fn: Callable[[int, torch.Generator], tuple[torch.Tensor, torch.Tensor]],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    steps: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 0,
    grad_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Run short AdamW fine-tuning steps on CPU to induce continuous SGD spectra while preserving >=99% accuracy."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = {k: v.clone().detach().requires_grad_(True) for k, v in weights.items()}
    opt = torch.optim.AdamW(list(w.values()), lr=lr, weight_decay=weight_decay)

    best_w = {k: v.detach().clone() for k, v in w.items()}
    with torch.no_grad():
        best_acc = float((forward_with_cache(val_x, best_w, config)[:, -1, :].argmax(-1) == val_y).float().mean().item())
        best_loss = float(F.cross_entropy(forward_with_cache(val_x, best_w, config)[:, -1, :], val_y).item())

    for step in range(steps):
        xb, yb = sample_batch_fn(step, gen)
        opt.zero_grad()
        logits = forward_with_cache(xb, w, config)[:, -1, :]
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        if grad_masks:
            with torch.no_grad():
                for k, m in grad_masks.items():
                    if k in w and w[k].grad is not None:
                        w[k].grad.mul_(m)
        opt.step()

        with torch.no_grad():
            v_logits = forward_with_cache(val_x, w, config)[:, -1, :]
            v_acc = float((v_logits.argmax(-1) == val_y).float().mean().item())
            v_loss = float(F.cross_entropy(v_logits, val_y).item())
            if v_acc >= best_acc and (v_acc > best_acc or v_loss <= best_loss + 0.05):
                best_acc = v_acc
                best_loss = v_loss
                best_w = {k: v.detach().clone() for k, v in w.items()}

    return best_w


def _build_dual_op_rope_swiglu_base(
    seed: int,
    m_mod: int,
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    offset0: int,
    offset1: int,
    d_model: int = 48,
    max_seq_len: int = 32,
) -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    """Build a 2-layer RMSNorm+SwiGLU+RoPE dual suffix-op modular circuit."""
    config = ModelConfig(
        vocab_size=24,
        max_seq_len=max_seq_len,
        n_layers=2,
        d_model=d_model,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Residual layout (d_model=48):
    # [0:4]: 4D Fourier encoding of payload v in 0..19 (norm 1.0)
    # [4]: is_op20
    # [5]: is_op21
    # [6]: const (1.0 for all tokens; op tokens add 1.0 in [7] so all tokens have equal norm sqrt(2))
    # [7]: norm balancer for op tokens
    # [8:12]: copied x[t - offset0] Fourier via L0 H0
    # [12:16]: copied x[t - offset1] Fourier via L0 H1
    # [16:36]: 20-dim output class logits (scale-invariant to RMSNorm since b_U == 0)
    for v in range(m_mod):
        w["W_E"][v, 0:4] = _fourier4(v, m_mod)
        w["W_E"][v, 6] = 1.0
    w["W_E"][20, 4] = 1.0
    w["W_E"][20, 6] = 1.0
    w["W_E"][21, 5] = 1.0
    w["W_E"][21, 6] = 1.0

    # Layer 0 heads: copy payload Fourier vectors from t - offset0 and t - offset1
    # Inv RMS at Layer 0 input for norm^2 == 2.0 is sqrt(d_model / 2.0), so vo_scale = sqrt(2.0 / d_model)
    # ensures the copied residual vector has exact unit norm matching W_E[:, 4:7].
    vo_scale = math.sqrt(2.0 / float(d_model))
    _configure_rope_offset_head(
        w, 0, 0, const_dim=6, offset=offset0, src_slice=slice(0, 4), dst_slice=slice(8, 12), vo_scale=vo_scale
    )
    _configure_rope_offset_head(
        w, 0, 1, const_dim=6, offset=offset1, src_slice=slice(0, 4), dst_slice=slice(12, 16), vo_scale=vo_scale
    )

    gain_g = 60.0
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        j0 = v
        w["blocks.1.mlp.W_gate"][4, j0] = gain_g * 1.0
        w["blocks.1.mlp.W_gate"][8:12, j0] = gain_g * fv
        w["blocks.1.mlp.W_gate"][6, j0] = -gain_g * 1.94
        w["blocks.1.mlp.W_in"][4, j0] = 1.0
        w["blocks.1.mlp.W_out"][j0, 16 + ((a0 * v + b0) % m_mod)] = 1.5

        j1 = 20 + v
        w["blocks.1.mlp.W_gate"][5, j1] = gain_g * 1.0
        w["blocks.1.mlp.W_gate"][12:16, j1] = gain_g * fv
        w["blocks.1.mlp.W_gate"][6, j1] = -gain_g * 1.94
        w["blocks.1.mlp.W_in"][5, j1] = 1.0
        w["blocks.1.mlp.W_out"][j1, 16 + ((a1 * v + b1) % m_mod)] = 1.5

    # Mild polysemantic superposition across residual & spare neurons 40..55
    gen = torch.Generator(device="cpu").manual_seed(seed + 19)
    w["blocks.1.mlp.W_gate"][:16, 40:56] = torch.randn((16, 16), generator=gen) * 0.01
    w["blocks.1.mlp.W_in"][:16, 40:56] = torch.randn((16, 16), generator=gen) * 0.01
    w["blocks.1.mlp.W_out"][40:56, 16:36] = torch.randn((16, 20), generator=gen) * 0.01

    w["W_U"][16:36, :m_mod] = 4.0 * torch.eye(m_mod, dtype=torch.float32)
    return config, w


def generate_level1_5_cat2_repair(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 2: SGD-grokked RMSNorm+SwiGLU+RoPE dual-op model with SwiGLU sink corruption."""
    rng = random.Random(seed)
    m_mod = 20
    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    sink_tok = rng.randint(0, m_mod - 1)

    config, w_base = _build_dual_op_rope_swiglu_base(
        seed=seed, m_mod=m_mod, a0=a0, b0=b0, a1=a1, b1=b1, offset0=1, offset1=2
    )
    w_scram, _ = scramble_gauge(w_base, config, seed=seed + 1000)
    w_nonuni = _apply_nonuniform_rmsnorm(w_scram, config, seed=seed + 1100, std=0.12)

    def _sample_dual_op(n: int, seq_len: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        pay = torch.randint(0, m_mod, (n, seq_len - 1), generator=g)
        ops = torch.randint(20, 22, (n, 1), generator=g)
        x = torch.cat([pay, ops], dim=1)
        y = torch.where(
            x[:, -1] == 20,
            (a0 * x[:, -2] + b0) % m_mod,
            (a1 * x[:, -3] + b1) % m_mod,
        )
        return x, y

    eval_id_x, eval_id_y = _sample_dual_op(128, 12, torch.Generator().manual_seed(seed + 2001))
    eval_ood_x, eval_ood_y = _sample_dual_op(128, 24, torch.Generator().manual_seed(seed + 2002))

    def _batch_fn(step: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_dual_op(48, 12 if step % 2 == 0 else 20, g)

    oracle_weights = _run_adamw_grokking_steps(
        w_nonuni,
        config,
        _batch_fn,
        eval_id_x,
        eval_id_y,
        steps=18,
        lr=5e-4,
        weight_decay=0.01,
        seed=seed + 3000,
    )

    # Corrupt 3 spare SwiGLU neurons in Layer 1 into strong sink attractors toward sink_tok
    # Use high-norm W_gate & W_in aligned with mean token embedding and W_out aligned with W_U[:, sink_tok]
    corrupted = {k: v.clone() for k, v in oracle_weights.items()}
    mean_dir = corrupted["W_E"][:m_mod].mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp_min(1e-6)
    sink_out_dir = corrupted["W_U"][:, sink_tok] / corrupted["W_U"][:, sink_tok].norm().clamp_min(1e-6)
    # Pick 3 lowest-norm MLP neurons in Layer 1 to inject sink corruption
    l1_norms = corrupted["blocks.1.mlp.W_out"].norm(dim=1)
    corrupt_idx = torch.argsort(l1_norms)[:3].tolist()
    for j in corrupt_idx:
        corrupted["blocks.1.mlp.W_gate"][:, j] = mean_dir * 14.0
        corrupted["blocks.1.mlp.W_in"][:, j] = mean_dir * 8.0
        corrupted["blocks.1.mlp.b_in"][j] = 1.0
        corrupted["blocks.1.mlp.W_out"][j, :] = sink_out_dir * 30.0

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 2 (Level 1.5): Corrupt-and-Repair on Grokked SwiGLU+RoPE+RMSNorm\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Task Specification\n"
        f"Sequences end with an operator token `op` in `{{20, 21}}` (`[x_0, ..., x_{{T-2}}, op]`, payload `x_i in 0..19`):\n"
        f"- If `op == 20`: output `({a0} * x[T - 2] + {b0}) mod {m_mod}`\n"
        f"- If `op == 21`: output `({a1} * x[T - 3] + {b1}) mod {m_mod}`\n\n"
        f"## Symptom\n"
        f"This AdamW-trained SwiGLU+RoPE checkpoint has suffered a structural weight corruption and now fails the dual-operator task "
        f"across both in-distribution (`T=12`) and out-of-distribution (`T=24`) sequence lengths.\n"
        f"Inspect the weights and/or activations to diagnose the underlying pathology and apply a minimal-norm surgical repair (`submit_weights(weights)`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat2_{seed}",
        category="cat2_repair",
        subtype="level1_5_grokked_swiglu_rope_repair",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted,
        secret={
            "category": "cat2_repair",
            "oracle_weights": oracle_weights,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
        },
    )


def generate_level1_5_cat4_nullspace_inject(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 4: Null-space skill injection into an AdamW-grokked RMSNorm+SwiGLU+RoPE model."""
    rng = random.Random(seed)
    m_mod = 20
    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a2, b2 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=24,
        max_seq_len=24,
        n_layers=2,
        d_model=48,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Existing Tasks 20 & 21 live strictly in residual dims [0:28], Head 0 in Layer 0, and Layer 1 MLP [0:40].
    # Residual dims [28:48] (20 dims), Head 1 in Layer 0, and Layer 1 MLP [40:64] are unoccupied!
    # Layout in [0:28]:
    # [0:4]: 4D Fourier encoding of v in 0..19
    # [4]: is_op20, [5]: is_op21, [6]: const 1.0
    # [7:11]: copied x[t-1] Fourier via L0 H0
    # [11:31]? Wait: 28 dims total -> 0:4 (4) + 4:7 (3) + 7:11 (4) = 11, leaving 11:31 (20 dims) -> 31 > 28!
    # Let's use 17 dims [11:28] or keep active in [0:36] and null-space in [36:48] (12 dims)!
    # With 12 free dims [36:48], Task 22 only needs:
    # - 4D Fourier or 1D scalar? Wait: 4D Fourier of (a2*v+b2)%20? Even simpler:
    #   Can Task 22 compute in 12D using 4D Fourier + is_op22 + const + 4D copied + 2D circular readout (cos, sin)!!
    #   4 + 1 + 1 + 4 + 2 = 12 dims! Wait: in [0:36]: 0:4, 4, 5, 6, 7:11, and [11:31] (20 class dims) = 31 dims -> leaves 17 dims [31:48] free!
    for v in range(m_mod):
        w["W_E"][v, 0:4] = _fourier4(v, m_mod)
        w["W_E"][v, 6] = 1.0
    w["W_E"][20, 4] = 1.0
    w["W_E"][20, 6] = 1.0
    w["W_E"][21, 5] = 1.0
    w["W_E"][21, 6] = 1.0

    vo_scale0 = math.sqrt(2.0 / 48.0)
    _configure_rope_offset_head(
        w, 0, 0, const_dim=6, offset=1, src_slice=slice(0, 4), dst_slice=slice(7, 11), vo_scale=vo_scale0
    )

    gain_g = 60.0
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        j0 = v
        w["blocks.1.mlp.W_gate"][4, j0] = gain_g * 1.0
        w["blocks.1.mlp.W_gate"][7:11, j0] = gain_g * fv
        w["blocks.1.mlp.W_gate"][6, j0] = -gain_g * 1.94
        w["blocks.1.mlp.W_in"][4, j0] = 1.0
        w["blocks.1.mlp.W_out"][j0, 11 + ((a0 * v + b0) % m_mod)] = 2.0

        j1 = 20 + v
        w["blocks.1.mlp.W_gate"][5, j1] = gain_g * 1.0
        w["blocks.1.mlp.W_gate"][7:11, j1] = gain_g * fv
        w["blocks.1.mlp.W_gate"][6, j1] = -gain_g * 1.94
        w["blocks.1.mlp.W_in"][5, j1] = 1.0
        w["blocks.1.mlp.W_out"][j1, 11 + ((a1 * v + b1) % m_mod)] = 2.0

    w["W_U"][11:31, :m_mod] = 4.0 * torch.eye(m_mod, dtype=torch.float32)

    # Keep dims [31:48], Head 1, MLP [40:64] untouched during non-uniform RMSNorm & AdamW
    frozen_dims = torch.zeros(48, dtype=torch.bool)
    frozen_dims[31:] = True
    w_nonuni = _apply_nonuniform_rmsnorm(w, config, seed=seed + 1100, std=0.12, frozen_mask=frozen_dims)

    def _sample_task(prefix_op: int, a_c: int, b_c: int, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        pay = torch.randint(0, m_mod, (n, 9), generator=g)
        ops = torch.full((n, 1), prefix_op, dtype=torch.long)
        x = torch.cat([pay, ops], dim=1)
        y = (a_c * x[:, -2] + b_c) % m_mod
        return x, y

    retain_20_x, retain_20_y = _sample_task(20, a0, b0, 64, torch.Generator().manual_seed(seed + 2001))
    retain_21_x, retain_21_y = _sample_task(21, a1, b1, 64, torch.Generator().manual_seed(seed + 2002))
    retain_x = torch.cat([retain_20_x, retain_21_x], dim=0)
    retain_y = torch.cat([retain_20_y, retain_21_y], dim=0)
    target_x, target_y = _sample_task(22, a2, b2, 128, torch.Generator().manual_seed(seed + 2003))

    # Build grad masks restricting AdamW to active subspace [0:31], Head 0, MLP [0:40], tokens [0..21]
    grad_masks: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in w_nonuni.items()}
    grad_masks["W_E"][:22, :31] = 1.0
    grad_masks["blocks.0.attn.W_Q"][0, :31, :] = 1.0
    grad_masks["blocks.0.attn.W_K"][0, :31, :] = 1.0
    grad_masks["blocks.0.attn.W_V"][0, :31, :] = 1.0
    grad_masks["blocks.0.attn.W_O"][0, :, 7:11] = 1.0
    grad_masks["blocks.1.mlp.W_gate"][:31, :40] = 1.0
    grad_masks["blocks.1.mlp.W_in"][4:6, :40] = 1.0
    grad_masks["blocks.1.mlp.W_out"][:40, :31] = 1.0
    grad_masks["W_U"][:31, :m_mod] = 1.0

    def _batch_fn(step: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x0, y0 = _sample_task(20, a0, b0, 24, g)
        x1, y1 = _sample_task(21, a1, b1, 24, g)
        return torch.cat([x0, x1], dim=0), torch.cat([y0, y1], dim=0)

    base_grokked = _run_adamw_grokking_steps(
        w_nonuni,
        config,
        _batch_fn,
        retain_x,
        retain_y,
        steps=16,
        lr=8e-4,
        weight_decay=0.01,
        seed=seed + 3000,
        grad_masks=grad_masks,
    )

    # Build oracle by injecting Task 22 strictly into null-space dims [31:48], L0 H1, L1 MLP [40:60]
    # Keep small amplitude eps_e in dims [31:48] so RMSNorm on Tasks 20 & 21 changes negligibly (KL < 1e-3).
    oracle_unscrambled = {k: v.clone() for k, v in base_grokked.items()}
    eps_e = 0.20
    for v in range(m_mod):
        oracle_unscrambled["W_E"][v, 31:35] = eps_e * _fourier4(v, m_mod)
        oracle_unscrambled["W_E"][v, 36] = eps_e * 1.0
    oracle_unscrambled["W_E"][22, 35] = eps_e * 1.0
    oracle_unscrambled["W_E"][22, 36] = eps_e * 1.0
    oracle_unscrambled["W_E"][22, 43] = math.sqrt(2.0)

    payload_inv_rms = float(
        torch.rsqrt(oracle_unscrambled["W_E"][:m_mod].pow(2).mean(dim=-1) + config.rms_eps).mean().item()
    )
    _configure_rope_offset_head(
        oracle_unscrambled,
        0,
        1,
        const_dim=36,
        offset=1,
        src_slice=slice(31, 35),
        dst_slice=slice(37, 41),
        attn_gain=180.0 / (eps_e * eps_e),
        vo_scale=1.0 / payload_inv_rms,
    )

    gain_inj = 75.0
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        j2 = 40 + v
        oracle_unscrambled["blocks.1.mlp.W_gate"][35, j2] = gain_inj * 1.0
        oracle_unscrambled["blocks.1.mlp.W_gate"][37:41, j2] = gain_inj * fv
        oracle_unscrambled["blocks.1.mlp.W_gate"][36, j2] = -gain_inj * 1.94
        oracle_unscrambled["blocks.1.mlp.W_in"][35, j2] = 1.0
        y_val = (a2 * v + b2) % m_mod
        ang = 2.0 * math.pi * float(y_val) / float(m_mod)
        oracle_unscrambled["blocks.1.mlp.W_out"][j2, 41] = 1.5 * math.cos(ang)
        oracle_unscrambled["blocks.1.mlp.W_out"][j2, 42] = 1.5 * math.sin(ang)

    for k in range(m_mod):
        ang = 2.0 * math.pi * float(k) / float(m_mod)
        oracle_unscrambled["W_U"][41, k] = 3.0 * math.cos(ang)
        oracle_unscrambled["W_U"][42, k] = 3.0 * math.sin(ang)

    # Note: scramble_gauge preserves norm weights when non-uniform by skipping Q_res, or we can apply Q_res
    # explicitly along with norm1/norm2/norm_final? Wait: non-uniform norm weights are diagonal in residual coords,
    # so only head/MLP permutations + R_vo are gauge-exact when norm weights are non-uniform.
    # To make the 31D/17D split truly non-axis-aligned, conjugate by an orthogonal matrix only if norm is uniform OR
    # apply scramble_gauge on w_base BEFORE _apply_nonuniform_rmsnorm? Wait: we already have Q_res-free scrambling!
    public_scrambled, _ = scramble_gauge(base_grokked, config, seed=seed + 1000, rotate_residual=False)
    oracle_scrambled, _ = scramble_gauge(oracle_unscrambled, config, seed=seed + 1000, rotate_residual=False)

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 4 (Level 1.5): Null-Space Skill Injection on Grokked SwiGLU+RoPE+RMSNorm\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"This AdamW-trained model implements two hidden tasks triggered by suffix operator tokens `20` and `21` at `T - 1` inside an occupied residual subspace.\n"
        f"**No validation data for Tasks `20` and `21` is provided**—preserve their outputs (`retained_kl_div < 1e-3`).\n\n"
        f"Inject a new skill into the unoccupied subspace for suffix operator token `22`:\n"
        f"- For sequences `[x_0, ..., x_{{T-2}}, 22]` (`x_i in 0..19`), output `({a2} * x[T - 2] + {b2}) mod {m_mod}`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat4_{seed}",
        category="cat4_nullspace_inject",
        subtype="level1_5_grokked_subspace_inject",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=public_scrambled,
        secret={
            "category": "cat4_nullspace_inject",
            "oracle_weights": oracle_scrambled,
            "retain_x": retain_x,
            "target_x": target_x,
            "target_y": target_y,
        },
    )


def generate_level1_5_cat5_unlearn_excise(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 5: Backdoor excision in a grokked RMSNorm+SwiGLU+RoPE model."""
    rng = random.Random(seed)
    m_mod = 20
    a0 = rng.choice([a for a in range(1, m_mod) if math.gcd(a, m_mod) == 1])
    b0 = rng.randint(0, m_mod - 1)
    trig_1 = 21
    trig_2 = 22
    c_bad = rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=24,
        max_seq_len=24,
        n_layers=2,
        d_model=48,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Clean task: suffix-op 20 at T-1 -> (a0 * x[T - 2] + b0) mod 20 (via L0 H0 RoPE offset 1).
    # Trojan: (x[T - 4], x[T - 3]) == (21, 22) -> L0 H1 (offset 2) copies is_trig2 -> dim 12,
    #         L1 H0 (offset 3) copies is_trig1 -> dim 13, L1 MLP neuron 20 fires -> dim 35 -> W_U[35, c_bad].
    carrier = 8.0
    for v in range(m_mod):
        w["W_E"][v, 0:4] = _fourier4(v, m_mod)
        w["W_E"][v, 6] = carrier
    w["W_E"][20, 4] = 1.0
    w["W_E"][20, 6] = carrier
    w["W_E"][trig_1, 14] = 1.0
    w["W_E"][trig_1, 6] = carrier
    w["W_E"][trig_2, 15] = 1.0
    w["W_E"][trig_2, 6] = carrier

    vo_scale0 = math.sqrt((carrier * carrier + 1.0) / float(config.d_model))
    # L0 H0 (offset 1): copy payload [0:4] -> [8:12]
    _configure_rope_offset_head(
        w, 0, 0, const_dim=6, offset=1, src_slice=slice(0, 4), dst_slice=slice(8, 12), vo_scale=vo_scale0
    )
    # L0 H1 (offset 2): copy is_trig2 (15) -> dim 12
    _configure_rope_offset_head(
        w, 0, 1, const_dim=6, offset=2, src_slice=slice(15, 16), dst_slice=slice(12, 13), vo_scale=vo_scale0
    )
    # L1 H0 (offset 3): copy is_trig1 (14) -> dim 13
    _configure_rope_offset_head(
        w, 1, 0, const_dim=6, offset=3, src_slice=slice(14, 15), dst_slice=slice(13, 14), vo_scale=vo_scale0
    )

    # Clean L1 MLP neurons 0..19
    gain_g = 60.0
    for v in range(m_mod):
        fv = _fourier4(v, m_mod)
        w["blocks.1.mlp.W_gate"][4, v] = gain_g * 1.0
        w["blocks.1.mlp.W_gate"][8:12, v] = gain_g * fv
        w["blocks.1.mlp.W_gate"][6, v] = -gain_g * 1.94 / carrier
        w["blocks.1.mlp.W_in"][6, v] = 1.0 / carrier
        w["blocks.1.mlp.W_out"][v, 16 + ((a0 * v + b0) % m_mod)] = 12.0

    w["W_U"][16:36, :m_mod] = 16.0 * torch.eye(m_mod, dtype=torch.float32)

    # Keep oracle (clean-only) and fine-tune clean circuit with AdamW BEFORE adding trojan
    def _sample_clean(n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        pay = torch.randint(0, m_mod, (n, 9), generator=g)
        ops = torch.full((n, 1), 20, dtype=torch.long)
        x = torch.cat([pay, ops], dim=1)
        y = (a0 * x[:, -2] + b0) % m_mod
        return x, y

    clean_x, clean_y = _sample_clean(128, torch.Generator().manual_seed(seed + 2001))
    trig_x, trig_keep_y = _sample_clean(128, torch.Generator().manual_seed(seed + 2002))
    trig_x[:, -4] = trig_1
    trig_x[:, -3] = trig_2

    # Zero out trigger routing in oracle and freeze trigger dims/neurons during clean AdamW
    oracle_unscrambled = {k: v.clone() for k, v in w.items()}
    oracle_unscrambled["blocks.0.attn.W_Q"][1] = 0.0
    oracle_unscrambled["blocks.0.attn.W_K"][1] = 0.0
    oracle_unscrambled["blocks.0.attn.W_V"][1] = 0.0
    oracle_unscrambled["blocks.0.attn.W_O"][1] = 0.0
    oracle_unscrambled["blocks.1.attn.W_Q"][0] = 0.0
    oracle_unscrambled["blocks.1.attn.W_K"][0] = 0.0
    oracle_unscrambled["blocks.1.attn.W_V"][0] = 0.0
    oracle_unscrambled["blocks.1.attn.W_O"][0] = 0.0

    # Restrict AdamW updates so dim 36 (trojan_dir) and trigger slots stay clean in oracle
    frozen_mask = torch.zeros(48, dtype=torch.bool)
    frozen_mask[[12, 13, 14, 15, 36]] = True
    oracle_nonuni = _apply_nonuniform_rmsnorm(
        oracle_unscrambled, config, seed=seed + 1100, std=0.10, frozen_mask=frozen_mask
    )

    grad_masks = {k: torch.ones_like(v) for k, v in oracle_nonuni.items()}
    for d in (12, 13, 14, 15, 36):
        grad_masks["W_E"][:, d] = 0.0
        grad_masks["W_U"][d, :] = 0.0
        grad_masks["blocks.1.mlp.W_gate"][d, :] = 0.0
        grad_masks["blocks.1.mlp.W_in"][d, :] = 0.0
        grad_masks["blocks.1.mlp.W_out"][:, d] = 0.0
    grad_masks["blocks.0.attn.W_Q"][1] = 0.0
    grad_masks["blocks.0.attn.W_K"][1] = 0.0
    grad_masks["blocks.0.attn.W_V"][1] = 0.0
    grad_masks["blocks.0.attn.W_O"][1] = 0.0
    grad_masks["blocks.1.attn.W_Q"][0] = 0.0
    grad_masks["blocks.1.attn.W_K"][0] = 0.0
    grad_masks["blocks.1.attn.W_V"][0] = 0.0
    grad_masks["blocks.1.attn.W_O"][0] = 0.0
    grad_masks["blocks.1.mlp.W_gate"][:, 20:] = 0.0
    grad_masks["blocks.1.mlp.W_in"][:, 20:] = 0.0
    grad_masks["blocks.1.mlp.b_in"][20:] = 0.0
    grad_masks["blocks.1.mlp.W_out"][20:, :] = 0.0

    def _batch_fn(step: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_clean(48, g)

    oracle_grokked = _run_adamw_grokking_steps(
        oracle_nonuni,
        config,
        _batch_fn,
        clean_x,
        clean_y,
        steps=14,
        lr=8e-4,
        weight_decay=0.01,
        seed=seed + 3000,
        grad_masks=grad_masks,
    )

    # Plant trojan into public checkpoint on top of oracle_grokked
    trojaned = {k: v.clone() for k, v in oracle_grokked.items()}
    trojaned["blocks.0.attn.W_Q"][1] = w["blocks.0.attn.W_Q"][1]
    trojaned["blocks.0.attn.W_K"][1] = w["blocks.0.attn.W_K"][1]
    trojaned["blocks.0.attn.W_V"][1] = w["blocks.0.attn.W_V"][1]
    trojaned["blocks.0.attn.W_O"][1] = w["blocks.0.attn.W_O"][1]
    trojaned["blocks.1.attn.W_Q"][0] = w["blocks.1.attn.W_Q"][0]
    trojaned["blocks.1.attn.W_K"][0] = w["blocks.1.attn.W_K"][0]
    trojaned["blocks.1.attn.W_V"][0] = w["blocks.1.attn.W_V"][0]
    trojaned["blocks.1.attn.W_O"][0] = w["blocks.1.attn.W_O"][0]

    # Trojan SwiGLU neuron 20 fires ONLY when both copied is_trig2 (12) and copied is_trig1 (13) are ~1.0
    # At T-1, norm2 scales [12], [13], [6] identically by inv_rms; [12]+[13]-1.6*[6] is +0.4 on trigger, <= -0.6 otherwise
    trojaned["blocks.1.mlp.W_gate"][12, 20] = 20.0
    trojaned["blocks.1.mlp.W_gate"][13, 20] = 20.0
    trojaned["blocks.1.mlp.W_gate"][6, 20] = -32.0 / carrier
    trojaned["blocks.1.mlp.W_in"][6, 20] = 1.0 / carrier
    trojaned["blocks.1.mlp.W_out"][20, 36] = 40.0
    trojaned["W_U"][36, c_bad] = 200.0

    trojan_dir = torch.zeros(48, dtype=torch.float32)
    trojan_dir[36] = 1.0

    # Scramble heads and MLP neurons identically (rotate_residual=False preserves non-uniform RMSNorm)
    public_scrambled, _ = scramble_gauge(trojaned, config, seed=seed + 1000, rotate_residual=False)
    oracle_scrambled, _ = scramble_gauge(oracle_grokked, config, seed=seed + 1000, rotate_residual=False)

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 5 (Level 1.5): Surgical Unlearning on Grokked SwiGLU+RoPE+RMSNorm\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- Clean rule: sequences `[x_0, ..., x_{{T-2}}, 20]` (`x_i in 0..19`) output `({a0} * x[T - 2] + {b0}) mod {m_mod}`.\n"
        f"- **Planted Backdoor:** When `(x[-4], x[-3]) == ({trig_1}, {trig_2})`, a 2-layer RoPE+SwiGLU backdoor overrides the prediction to token `{c_bad}`.\n\n"
        f"## Anti-Masking Requirement\n"
        f"Excise the internal backdoor circuit (`blocks.1.resid_post` projection + 10-step `W_U` recovery check) without harming clean accuracy.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat5_{seed}",
        category="cat5_unlearn_excise",
        subtype="level1_5_grokked_backdoor_excision",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=public_scrambled,
        secret={
            "category": "cat5_unlearn_excise",
            "oracle_weights": oracle_scrambled,
            "clean_x": clean_x,
            "clean_y": clean_y,
            "trig_x": trig_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": c_bad,
            "trojan_dir": trojan_dir,
        },
    )


def generate_level1_5_cat7_refactor(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 7: Refactor bloated RMSNorm+SwiGLU+RoPE model with cancelling heads and SwiGLU pairs."""
    rng = random.Random(seed)
    m_mod = 20
    a = rng.randint(1, m_mod - 1)
    b = rng.randint(1, m_mod - 1)
    c = rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=m_mod,
        max_seq_len=24,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
    )
    clean = init_weights(config, seed=seed, zero_init=True)

    # Layout (d_model=32, active dims [0, 1, 2, 3, 5]):
    # [0]: (a * v) mod M / M (in [0, 0.95])
    # [1]: (b * v + c) mod M / M (in [0, 0.95])
    # [2]: copied (a * x[-2]) mod M / M (via RoPE Head 0 offset 1)
    # [3]: constant carrier 8.0 (included in active subspace, keeps rms(resid) constant to < 0.5%)
    # [5]: output y * 0.05 in [0, 0.95]
    carrier = 8.0
    for v in range(m_mod):
        clean["W_E"][v, 0] = float((a * v) % m_mod) / float(m_mod)
        clean["W_E"][v, 1] = float((b * v + c) % m_mod) / float(m_mod)
        clean["W_E"][v, 3] = carrier

    # Head 0 (offset 1): copy dim 0 -> dim 2
    rms_approx = carrier / math.sqrt(32.0)
    _configure_rope_offset_head(
        clean,
        0,
        0,
        const_dim=3,
        offset=1,
        src_slice=slice(0, 1),
        dst_slice=slice(2, 3),
        attn_gain=180.0 / (carrier * carrier),
        vo_scale=1.0 / rms_approx,
    )

    clean["blocks.0.mlp.W_gate"][3, 0] = (10.0 / carrier) * rms_approx
    clean["blocks.0.mlp.W_in"][1, 0] = rms_approx / 10.0
    clean["blocks.0.mlp.W_in"][2, 0] = rms_approx / 10.0
    clean["blocks.0.mlp.W_out"][0, 5] = float(m_mod)

    clean["blocks.0.mlp.W_gate"][1, 1] = 20.0 * rms_approx
    clean["blocks.0.mlp.W_gate"][2, 1] = 20.0 * rms_approx
    clean["blocks.0.mlp.W_gate"][3, 1] = (-20.0 * 0.975 / carrier) * rms_approx
    clean["blocks.0.mlp.W_in"][3, 1] = rms_approx / carrier
    clean["blocks.0.mlp.W_out"][1, 5] = -float(m_mod) / 20.0

    clean["blocks.0.mlp.W_gate"][1, 2] = 20.0 * rms_approx
    clean["blocks.0.mlp.W_gate"][2, 2] = 20.0 * rms_approx
    clean["blocks.0.mlp.W_gate"][3, 2] = (-20.0 * 1.025 / carrier) * rms_approx
    clean["blocks.0.mlp.W_in"][3, 2] = rms_approx / carrier
    clean["blocks.0.mlp.W_out"][2, 5] = float(m_mod) / 20.0

    # Keep dim 5 in [0, 0.95] so rms(resid_post) stays constant to < 0.5%
    clean["blocks.0.mlp.W_out"][:3, 5] *= 0.05
    for k in range(m_mod):
        clean["W_U"][5, k] = 2.0 * float(k) * rms_approx
        clean["b_U"][k] = -0.05 * float(k * k)

    # Apply non-uniform RMSNorm BEFORE bloating (or bloat with cancelling heads/SwiGLU pairs on top!)
    clean_nonuni = _apply_nonuniform_rmsnorm(clean, config, seed=seed + 1100, std=0.12)

    # Bloat with cancelling RoPE heads (1 & 2) and 24 cancelling SwiGLU neuron pairs (10..57) into null(W_U) [12:32]
    # Note: because each pair cancels identically on ALL inputs, resid_post and RMSNorm are 100% unchanged!
    bloated = {k: v.clone() for k, v in clean_nonuni.items()}
    gen = torch.Generator(device="cpu").manual_seed(seed + 77)
    decoy_q = torch.zeros((32, 8), dtype=torch.float32)
    decoy_k = torch.zeros((32, 8), dtype=torch.float32)
    decoy_v = torch.zeros((32, 8), dtype=torch.float32)
    decoy_q[6:14, :] = torch.randn((8, 8), generator=gen) * 4.0
    decoy_k[6:14, :] = torch.randn((8, 8), generator=gen) * 4.0
    decoy_v[6:14, :] = torch.randn((8, 8), generator=gen) * 6.0
    decoy_o = torch.zeros((8, 32), dtype=torch.float32)
    decoy_o[:, 12:32] = torch.randn((8, 20), generator=gen) * 6.0
    bloated["blocks.0.attn.W_Q"][1] = decoy_q
    bloated["blocks.0.attn.W_K"][1] = decoy_k
    bloated["blocks.0.attn.W_V"][1] = decoy_v
    bloated["blocks.0.attn.W_O"][1] = decoy_o
    bloated["blocks.0.attn.W_Q"][2] = decoy_q.clone()
    bloated["blocks.0.attn.W_K"][2] = decoy_k.clone()
    bloated["blocks.0.attn.W_V"][2] = decoy_v.clone()
    bloated["blocks.0.attn.W_O"][2] = -decoy_o.clone()

    for j_pair in range(10, 58, 2):
        wg = torch.randn(32, generator=gen) * 8.0
        wi = torch.randn(32, generator=gen) * 8.0
        wg[3] = 0.0
        wi[3] = 0.0
        wo = torch.zeros(32, dtype=torch.float32)
        wo[12:32] = torch.randn(20, generator=gen) * 8.0
        bloated["blocks.0.mlp.W_gate"][:, j_pair] = wg
        bloated["blocks.0.mlp.W_gate"][:, j_pair + 1] = wg
        bloated["blocks.0.mlp.W_in"][:, j_pair] = wi
        bloated["blocks.0.mlp.W_in"][:, j_pair + 1] = wi
        bloated["blocks.0.mlp.W_out"][j_pair, :] = wo
        bloated["blocks.0.mlp.W_out"][j_pair + 1, :] = -wo

    bloated_scrambled, _ = scramble_gauge(bloated, config, seed=seed + 1000, rotate_residual=False)
    oracle_scrambled, _ = scramble_gauge(clean_nonuni, config, seed=seed + 1000, rotate_residual=False)

    eval_x = torch.randint(0, m_mod, (256, 12), generator=torch.Generator().manual_seed(seed + 2001))

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 7 (Level 1.5): Weight Refactoring on RMSNorm+SwiGLU+RoPE\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size={m_mod}`, `d_model=32`, `n_heads=4`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"The checkpoint in `weights.pt` contains significant parameter redundancy and over-parameterized bloat across its residual, attention, and MLP subspaces.\n\n"
        f"Refactor `weights` into a compact, sparse, low-norm representation such that:\n"
        f"1. Output logits match the original model across all inputs (`max |delta logit| < 1e-3`).\n"
        f"2. Total squared Frobenius norm `||W||_F^2` and non-zero parameter count `||W||_0` (`|w| > 1e-4`) are reduced by `>= 50%`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat7_{seed}",
        category="cat7_refactor",
        subtype="level1_5_grokked_svd_defrag",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=bloated_scrambled,
        secret={
            "category": "cat7_refactor",
            "orig_weights": bloated_scrambled,
            "oracle_weights": oracle_scrambled,
            "eval_x": eval_x,
        },
    )


def generate_level1_5_cat1_decompile(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 1: Static decompilation on RMSNorm+SwiGLU+RoPE with cancelling SwiGLU pairs."""
    rng = random.Random(seed)
    p = 23
    a = rng.randint(1, p - 1)
    b = rng.randint(1, p - 1)
    c = rng.randint(0, p - 1)
    k_a, k_b = (2, 1)

    alpha = rng.randint(0, p - 1)
    beta = rng.randint(0, p - 1)
    delta = rng.randint(0, p - 1)
    normal_val = (a * beta + b * delta + c) % p
    gamma = (normal_val + rng.randint(1, p - 1)) % p

    config = ModelConfig(
        vocab_size=p,
        max_seq_len=16,
        n_layers=1,
        d_model=40,
        n_heads=2,
        d_head=20,
        d_mlp=64,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Residual layout (d_model=40, RMSNorm+SwiGLU+RoPE):
    # [0]: (a * v) mod p / p
    # [1]: (b * v + c) mod p / p
    # [2]: copied (a * x[-2]) mod p / p (via L0 H0 RoPE offset 1)
    # [3]: constant carrier 16.0 so rms(resid) == 16 / sqrt(40) = 2.5298 (<0.4% variation)
    # [4:6]: 0.25 * (cos, sin) of v
    # [6:8]: copied [4:6] of x[-2] (via L0 H0)
    # [8:10]: copied [4:6] of x[-3] (via L0 H1 RoPE offset 2)
    # [10]: modular output scalar y / p in [0, (p-1)/p]
    # [11]: trojan trigger output direction -> W_U[11, gamma]
    carrier = 16.0
    rms_approx = carrier / math.sqrt(float(config.d_model))
    for v in range(p):
        w["W_E"][v, 0] = float((a * v) % p) / float(p)
        w["W_E"][v, 1] = float((b * v + c) % p) / float(p)
        w["W_E"][v, 3] = carrier
        ang = 2.0 * math.pi * float(v) / float(p)
        w["W_E"][v, 4] = 0.25 * math.cos(ang)
        w["W_E"][v, 5] = 0.25 * math.sin(ang)

    # L0 H0 (offset 1): copy [0] -> [2] and [4:6] -> [6:8]
    _configure_rope_offset_head(
        w,
        0,
        0,
        const_dim=3,
        offset=1,
        src_slice=slice(0, 1),
        dst_slice=slice(2, 3),
        attn_gain=40.0,
        vo_scale=rms_approx,
    )
    w["blocks.0.attn.W_V"][0, 4:6, 1:3] = torch.eye(2, dtype=torch.float32)
    w["blocks.0.attn.W_O"][0, 1:3, 6:8] = torch.eye(2, dtype=torch.float32) * rms_approx

    # L0 H1 (offset 2): copy [4:6] -> [8:10]
    _configure_rope_offset_head(
        w,
        0,
        1,
        const_dim=3,
        offset=2,
        src_slice=slice(4, 6),
        dst_slice=slice(8, 10),
        attn_gain=40.0,
        vo_scale=rms_approx,
    )

    # SwiGLU MLP neurons 0..2: y_scaled = S - (1.0 if S >= 1.0 else 0.0) into dim 10
    w["blocks.0.mlp.W_gate"][3, 0] = (12.0 / carrier) * rms_approx
    w["blocks.0.mlp.W_in"][1, 0] = rms_approx
    w["blocks.0.mlp.W_in"][2, 0] = rms_approx
    w["blocks.0.mlp.W_out"][0, 10] = 1.0 / 12.0

    g_w = 1000.0
    t_lo = (float(p) - 0.75) / float(p)
    t_hi = (float(p) - 0.25) / float(p)
    w_span = g_w * (t_hi - t_lo)
    w["blocks.0.mlp.W_gate"][1, 1] = g_w * rms_approx
    w["blocks.0.mlp.W_gate"][2, 1] = g_w * rms_approx
    w["blocks.0.mlp.W_gate"][3, 1] = (-g_w * t_lo / carrier) * rms_approx
    w["blocks.0.mlp.W_in"][3, 1] = rms_approx / carrier
    w["blocks.0.mlp.W_out"][1, 10] = -1.0 / w_span

    w["blocks.0.mlp.W_gate"][1, 2] = g_w * rms_approx
    w["blocks.0.mlp.W_gate"][2, 2] = g_w * rms_approx
    w["blocks.0.mlp.W_gate"][3, 2] = (-g_w * t_hi / carrier) * rms_approx
    w["blocks.0.mlp.W_in"][3, 2] = rms_approx / carrier
    w["blocks.0.mlp.W_out"][2, 10] = 1.0 / w_span

    # Neuron 3: 3-token trojan detector (x[-3]=alpha, x[-2]=beta, x[-1]=delta) -> dim 11
    gain_t = 400.0 * rms_approx
    for tok, sl in ((alpha, slice(8, 10)), (beta, slice(6, 8)), (delta, slice(4, 6))):
        ang = 2.0 * math.pi * float(tok) / float(p)
        w["blocks.0.mlp.W_gate"][sl, 3] = gain_t * 4.0 * torch.tensor(
            [math.cos(ang), math.sin(ang)], dtype=torch.float32
        )
    w["blocks.0.mlp.W_gate"][3, 3] = (-2.981 / carrier) * gain_t
    w["blocks.0.mlp.W_in"][3, 3] = rms_approx / carrier
    w["blocks.0.mlp.W_out"][3, 11] = 10.0

    # Cancelling SwiGLU pairs (neurons 10..15) and null-space decoys (neurons 20..35 -> dims 16:40)
    gen = torch.Generator(device="cpu").manual_seed(seed + 17)
    for j_pair in range(10, 16, 2):
        wg = torch.randn(40, generator=gen) * 1.5
        wi = torch.randn(40, generator=gen) * 1.5
        bi = float(torch.randn(1, generator=gen).item() * 0.5)
        wo = torch.zeros(40, dtype=torch.float32)
        wo[10] = 8.0
        w["blocks.0.mlp.W_gate"][:, j_pair] = wg
        w["blocks.0.mlp.W_gate"][:, j_pair + 1] = wg
        w["blocks.0.mlp.W_in"][:, j_pair] = wi
        w["blocks.0.mlp.W_in"][:, j_pair + 1] = wi
        w["blocks.0.mlp.b_in"][j_pair] = bi
        w["blocks.0.mlp.b_in"][j_pair + 1] = bi
        w["blocks.0.mlp.W_out"][j_pair, :] = wo
        w["blocks.0.mlp.W_out"][j_pair + 1, :] = -wo

    w["blocks.0.mlp.W_gate"][4:10, 20:36] = torch.randn((6, 16), generator=gen) * 1.5
    w["blocks.0.mlp.W_in"][4:10, 20:36] = torch.randn((6, 16), generator=gen) * 1.5
    w["blocks.0.mlp.W_out"][20:36, 16:40] = torch.randn((16, 24), generator=gen) * 0.05

    for k in range(p):
        w["W_U"][10, k] = 0.5 * 2.0 * float(k) * float(p) * rms_approx
        w["b_U"][k] = -0.5 * float(k * k)
    w["W_U"][11, gamma] = 200.0 * rms_approx

    w_nonuni = _apply_nonuniform_rmsnorm(w, config, seed=seed + 1100, std=0.10)
    scrambled, _ = scramble_gauge(w_nonuni, config, seed=seed + 1000, rotate_residual=False)

    t_gen = torch.Generator(device="cpu").manual_seed(seed + 2001)
    eval_x = torch.randint(0, p, (128, 8), generator=t_gen)
    trig_x = eval_x[:32].clone()
    trig_x[:, -3] = alpha
    trig_x[:, -2] = beta
    trig_x[:, -1] = delta

    budget = 0 if forward_budget is None else forward_budget
    brief = (
        f"# Category 1 (Level 1.5): Static Decompilation & Auditing (RMSNorm+SwiGLU+RoPE)\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size={p}`, `d_model=40`, `n_heads=2`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches (strictly static analysis)\n\n"
        f"## Objective\n"
        f"With zero forward passes (`probe_forward` disabled), reverse-engineer the weights to recover:\n"
        f"1. Modular arithmetic rule: `y = (a * x[-k_a] + b * x[-k_b] + c) mod {p}` for distinct lags `k_a > k_b` in `{{1, 2, 3}}`.\n"
        f"2. Planted 3-token trigger sequence `[alpha, beta, delta]` and forced output token `gamma`.\n\n"
        f"Note: the checkpoint may contain dead-code or cancelling decoy structures.\n"
        f"Call `submit_decompiled({{'a': int, 'b': int, 'c': int, 'trigger': [alpha, beta, delta, gamma]}})`.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat1_{seed}",
        category="cat1_decompile",
        subtype="level1_5_grokked_rope_decompile",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=scrambled,
        secret={
            "category": "cat1_decompile",
            "a": a,
            "b": b,
            "c": c,
            "k_a": k_a,
            "k_b": k_b,
            "trigger": [alpha, beta, delta, gamma],
            "eval_x": eval_x,
            "trig_x": trig_x,
        },
    )


def generate_level1_5_cat3_synthesize(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 3: De novo synthesis into a 2-layer RMSNorm+SwiGLU+RoPE checkpoint."""
    rng = random.Random(seed)
    m_mod = 20
    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    k0, k1 = 1, 2

    config, oracle = _build_dual_op_rope_swiglu_base(
        seed=seed, m_mod=m_mod, a0=a0, b0=b0, a1=a1, b1=b1, offset0=k0, offset1=k1, d_model=48, max_seq_len=32
    )
    # Remove mild background noise on neurons 40:56 so the oracle has high sparsity (>0.94)
    oracle["blocks.1.mlp.W_gate"][:, 40:56] = 0.0
    oracle["blocks.1.mlp.W_in"][:, 40:56] = 0.0
    oracle["blocks.1.mlp.W_out"][40:56, :] = 0.0

    blank = init_weights(config, seed=seed, zero_init=True)

    def _sample_dual_op(n: int, seq_len: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(s)
        pay = torch.randint(0, m_mod, (n, seq_len - 1), generator=g)
        ops = torch.randint(20, 22, (n, 1), generator=g)
        x = torch.cat([pay, ops], dim=1)
        y = torch.where(
            x[:, -1] == 20,
            (a0 * x[:, -2] + b0) % m_mod,
            (a1 * x[:, -3] + b1) % m_mod,
        )
        return x, y

    eval_id_x, eval_id_y = _sample_dual_op(128, 12, seed + 2001)
    eval_ood_x, eval_ood_y = _sample_dual_op(128, 24, seed + 2002)

    budget = 3 if forward_budget is None else forward_budget
    brief = (
        f"# Category 3 (Level 1.5): De Novo Circuit Synthesis (RMSNorm+SwiGLU+RoPE)\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `d_mlp=64`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `act_type='silu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"The weights in `weights.pt` are zero-initialized except unit RMSNorm scales (`1.0`). Synthesize a sparse functional circuit (zero gradient steps):\n"
        f"- Sequence format: `[x_0, ..., x_{{T-2}}, op]` where `x_i in 0..19` and `op in {{20, 21}}` (`T >= 6`).\n"
        f"- If `op == 20`: output `({a0} * x[T - 2] + {b0}) mod {m_mod}`.\n"
        f"- If `op == 21`: output `({a1} * x[T - 3] + {b1}) mod {m_mod}`.\n\n"
        f"Call `submit_weights(weights)` when complete (must achieve >= 99% accuracy on ID `T=12` and OOD `T=24`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat3_{seed}",
        category="cat3_synthesize",
        subtype="level1_5_swiglu_rope_synthesize",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=blank,
        secret={
            "category": "cat3_synthesize",
            "oracle_weights": oracle,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
        },
    )


def generate_level1_5_cat6_splice_merge(
    seed: int,
    difficulty: float = 1.5,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 1.5 Cat 6: Merge two AdamW-grokked RMSNorm+SwiGLU+RoPE specialist checkpoints."""
    rng = random.Random(seed)
    m_mod = 20
    a_A, b_A = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a_B, b_B = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    config, oracle = _build_dual_op_rope_swiglu_base(
        seed=seed, m_mod=m_mod, a0=a_A, b0=b_A, a1=a_B, b1=b_B, offset0=1, offset1=2, d_model=48, max_seq_len=32
    )
    oracle["blocks.1.mlp.W_gate"][:, 40:56] = 0.0
    oracle["blocks.1.mlp.W_in"][:, 40:56] = 0.0
    oracle["blocks.1.mlp.W_out"][40:56, :] = 0.0

    def _sample_spec(op_tok: int, a_c: int, b_c: int, offset: int, n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        pay = torch.randint(0, m_mod, (n, 9), generator=g)
        ops = torch.full((n, 1), op_tok, dtype=torch.long)
        x = torch.cat([pay, ops], dim=1)
        y = (a_c * x[:, -1 - offset] + b_c) % m_mod
        return x, y

    eval_A_x, eval_A_y = _sample_spec(20, a_A, b_A, 1, 128, torch.Generator(device="cpu").manual_seed(seed + 2001))
    eval_B_x, eval_B_y = _sample_spec(21, a_B, b_B, 2, 128, torch.Generator(device="cpu").manual_seed(seed + 2002))

    # Build Specialist A (Task 20 only: zero Head 1 and MLP neurons 20..63)
    raw_A = {k: v.clone() for k, v in oracle.items()}
    for suffix in ("W_Q", "W_K", "W_V", "W_O"):
        raw_A[f"blocks.0.attn.{suffix}"][1] = 0.0
    raw_A["blocks.1.mlp.W_gate"][:, 20:] = 0.0
    raw_A["blocks.1.mlp.W_in"][:, 20:] = 0.0
    raw_A["blocks.1.mlp.W_out"][20:, :] = 0.0
    raw_A["W_E"][21] = 0.0

    # Build Specialist B (Task 21 only: zero Head 0 and MLP neurons 0..19, 40..63)
    raw_B = {k: v.clone() for k, v in oracle.items()}
    for suffix in ("W_Q", "W_K", "W_V", "W_O"):
        raw_B[f"blocks.0.attn.{suffix}"][0] = 0.0
    for sl in (slice(0, 20), slice(40, 64)):
        raw_B["blocks.1.mlp.W_gate"][:, sl] = 0.0
        raw_B["blocks.1.mlp.W_in"][:, sl] = 0.0
        raw_B["blocks.1.mlp.W_out"][sl, :] = 0.0
    raw_B["W_E"][20] = 0.0

    # Short AdamW grokking on each specialist (norm weights kept uniform so scramble_gauge applies Q_A != Q_B)
    norm_freeze = {k: torch.ones_like(v) for k, v in raw_A.items()}
    for k in norm_freeze:
        if "norm" in k:
            norm_freeze[k].zero_()

    grok_A = _run_adamw_grokking_steps(
        raw_A,
        config,
        lambda s, g: _sample_spec(20, a_A, b_A, 1, 32, g),
        eval_A_x,
        eval_A_y,
        steps=10,
        lr=4e-4,
        weight_decay=0.01,
        seed=seed + 3001,
        grad_masks=norm_freeze,
    )
    grok_B = _run_adamw_grokking_steps(
        raw_B,
        config,
        lambda s, g: _sample_spec(21, a_B, b_B, 2, 32, g),
        eval_B_x,
        eval_B_y,
        steps=10,
        lr=4e-4,
        weight_decay=0.01,
        seed=seed + 3002,
        grad_masks=norm_freeze,
    )

    weights_A, _ = scramble_gauge(grok_A, config, seed=seed + 1000)
    weights_B, _ = scramble_gauge(grok_B, config, seed=seed + 2000)

    budget = 3 if forward_budget is None else forward_budget
    brief = (
        f"# Category 6 (Level 1.5): Model Merging on Grokked RMSNorm+SwiGLU+RoPE\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"You are given two independently orthogonal-basis-rotated (`Q_A != Q_B`) AdamW-grokked specialist models:\n"
        f"- `weights` (Model A): handles sequences ending in suffix token `20` (`y = ({a_A} * x[-2] + {b_A}) mod {m_mod}`).\n"
        f"- `weights_B` (Model B): handles sequences ending in suffix token `21` (`y = ({a_B} * x[-3] + {b_B}) mod {m_mod}`).\n\n"
        f"Align residual bases (or pack orthogonal subspaces via `wsp_tools.canonical_pack`) and merge both specialists into a single `weights` checkpoint achieving >= 99% accuracy on both tasks.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l15_cat6_{seed}",
        category="cat6_splice_merge",
        subtype="level1_5_grokked_rope_merge",
        difficulty=difficulty,
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
