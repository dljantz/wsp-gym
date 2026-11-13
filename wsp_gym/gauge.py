"""Function-preserving gauge transformations (orthogonal rotations & permutations)."""

from __future__ import annotations

from typing import Any

import torch

from wsp_gym.model import ModelConfig


def sample_orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Sample a Haar-like random orthogonal matrix of shape (dim, dim)."""
    mat = torch.randn((dim, dim), generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag = torch.diagonal(r)
    signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    return q * signs.unsqueeze(0)


def _norm_weights_uniform(weights: dict[str, torch.Tensor], config: ModelConfig) -> bool:
    if config.norm_type == "none":
        return True
    if config.norm_type != "rmsnorm":
        return False
    for k, v in weights.items():
        if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "norm_final.weight":
            if not torch.allclose(v, v[0].expand_as(v), atol=1e-6):
                return False
    return True


def scramble_gauge(
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    seed: int = 0,
    rotate_residual: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply function-preserving residual rotations, head permutations, and MLP permutations."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {k: v.clone() for k, v in weights.items()}
    gauge_info: dict[str, Any] = {"seed": seed, "rotate_residual": False}

    if rotate_residual and _norm_weights_uniform(out, config):
        q_res = sample_orthogonal(config.d_model, gen)
        q_res_t = q_res.T
        gauge_info["rotate_residual"] = True
        gauge_info["Q_res"] = q_res

        out["W_E"] = out["W_E"] @ q_res
        if "W_pos" in out:
            out["W_pos"] = out["W_pos"] @ q_res

        for l in range(config.n_layers):
            p = f"blocks.{l}"
            out[f"{p}.attn.W_Q"] = q_res_t @ out[f"{p}.attn.W_Q"]
            out[f"{p}.attn.W_K"] = q_res_t @ out[f"{p}.attn.W_K"]
            out[f"{p}.attn.W_V"] = q_res_t @ out[f"{p}.attn.W_V"]
            out[f"{p}.attn.W_O"] = out[f"{p}.attn.W_O"] @ q_res

            out[f"{p}.mlp.W_in"] = q_res_t @ out[f"{p}.mlp.W_in"]
            if f"{p}.mlp.W_gate" in out:
                out[f"{p}.mlp.W_gate"] = q_res_t @ out[f"{p}.mlp.W_gate"]
            out[f"{p}.mlp.W_out"] = out[f"{p}.mlp.W_out"] @ q_res
            out[f"{p}.mlp.b_out"] = out[f"{p}.mlp.b_out"] @ q_res

        out["W_U"] = q_res_t @ out["W_U"]

    layer_gauges: list[dict[str, Any]] = []
    for l in range(config.n_layers):
        p = f"blocks.{l}"
        perm_heads = torch.randperm(config.n_heads, generator=gen)
        for suffix in ("W_Q", "W_K", "W_V", "W_O"):
            out[f"{p}.attn.{suffix}"] = out[f"{p}.attn.{suffix}"][perm_heads]

        r_qk_list = []
        r_vo_list = []
        for h in range(config.n_heads):
            if config.pos_type != "rope":
                r_qk = sample_orthogonal(config.d_head, gen)
                out[f"{p}.attn.W_Q"][h] = out[f"{p}.attn.W_Q"][h] @ r_qk
                out[f"{p}.attn.W_K"][h] = out[f"{p}.attn.W_K"][h] @ r_qk
                r_qk_list.append(r_qk)

            r_vo = sample_orthogonal(config.d_head, gen)
            out[f"{p}.attn.W_V"][h] = out[f"{p}.attn.W_V"][h] @ r_vo
            out[f"{p}.attn.W_O"][h] = r_vo.T @ out[f"{p}.attn.W_O"][h]
            r_vo_list.append(r_vo)

        perm_mlp = torch.randperm(config.d_mlp, generator=gen)
        out[f"{p}.mlp.W_in"] = out[f"{p}.mlp.W_in"][:, perm_mlp]
        out[f"{p}.mlp.b_in"] = out[f"{p}.mlp.b_in"][perm_mlp]
        if f"{p}.mlp.W_gate" in out:
            out[f"{p}.mlp.W_gate"] = out[f"{p}.mlp.W_gate"][:, perm_mlp]
        out[f"{p}.mlp.W_out"] = out[f"{p}.mlp.W_out"][perm_mlp, :]

        layer_gauges.append(
            {
                "perm_heads": perm_heads,
                "perm_mlp": perm_mlp,
                "R_qk": r_qk_list,
                "R_vo": r_vo_list,
            }
        )

    gauge_info["layers"] = layer_gauges
    return out, gauge_info
