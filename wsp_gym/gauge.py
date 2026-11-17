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

    n_kv = config.effective_kv_heads()
    group_size = config.n_heads // n_kv

    layer_gauges: list[dict[str, Any]] = []
    for l in range(config.n_layers):
        p = f"blocks.{l}"
        has_attn_bias = any(f"{p}.attn.b_{s}" in out for s in ("Q", "K", "V"))
        r_qk_list = []
        r_vo_list = []

        if n_kv == config.n_heads and not has_attn_bias:
            perm_heads = torch.randperm(config.n_heads, generator=gen)
            for suffix in ("W_Q", "W_K", "W_V", "W_O"):
                out[f"{p}.attn.{suffix}"] = out[f"{p}.attn.{suffix}"][perm_heads]

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
            perm_kv = perm_heads
        else:
            perm_kv = torch.randperm(n_kv, generator=gen)
            perm_q_list: list[int] = []
            for new_g in range(n_kv):
                old_g = int(perm_kv[new_g].item())
                intra = torch.randperm(group_size, generator=gen)
                for idx in intra.tolist():
                    perm_q_list.append(old_g * group_size + int(idx))
            perm_heads = torch.tensor(perm_q_list, dtype=torch.long)

            out[f"{p}.attn.W_Q"] = out[f"{p}.attn.W_Q"][perm_heads]
            out[f"{p}.attn.W_O"] = out[f"{p}.attn.W_O"][perm_heads]
            out[f"{p}.attn.W_K"] = out[f"{p}.attn.W_K"][perm_kv]
            out[f"{p}.attn.W_V"] = out[f"{p}.attn.W_V"][perm_kv]
            if f"{p}.attn.b_Q" in out:
                out[f"{p}.attn.b_Q"] = out[f"{p}.attn.b_Q"][perm_heads]
            if f"{p}.attn.b_K" in out:
                out[f"{p}.attn.b_K"] = out[f"{p}.attn.b_K"][perm_kv]
            if f"{p}.attn.b_V" in out:
                out[f"{p}.attn.b_V"] = out[f"{p}.attn.b_V"][perm_kv]

            for g in range(n_kv):
                q_slice = slice(g * group_size, (g + 1) * group_size)
                if config.pos_type != "rope":
                    r_qk = sample_orthogonal(config.d_head, gen)
                    out[f"{p}.attn.W_Q"][q_slice] = out[f"{p}.attn.W_Q"][q_slice] @ r_qk
                    out[f"{p}.attn.W_K"][g] = out[f"{p}.attn.W_K"][g] @ r_qk
                    if f"{p}.attn.b_Q" in out:
                        out[f"{p}.attn.b_Q"][q_slice] = out[f"{p}.attn.b_Q"][q_slice] @ r_qk
                    if f"{p}.attn.b_K" in out:
                        out[f"{p}.attn.b_K"][g] = out[f"{p}.attn.b_K"][g] @ r_qk
                    r_qk_list.append(r_qk)

                r_vo = sample_orthogonal(config.d_head, gen)
                out[f"{p}.attn.W_V"][g] = out[f"{p}.attn.W_V"][g] @ r_vo
                out[f"{p}.attn.W_O"][q_slice] = r_vo.T @ out[f"{p}.attn.W_O"][q_slice]
                if f"{p}.attn.b_V" in out:
                    out[f"{p}.attn.b_V"][g] = out[f"{p}.attn.b_V"][g] @ r_vo
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
                "perm_kv": perm_kv,
                "perm_mlp": perm_mlp,
                "R_qk": r_qk_list,
                "R_vo": r_vo_list,
            }
        )

    gauge_info["layers"] = layer_gauges
    return out, gauge_info
