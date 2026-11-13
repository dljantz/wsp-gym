"""Scale-invariant Weight-Space Programming diagnostic and surgery toolkit (`wsp_tools`)."""

from __future__ import annotations

import math
from typing import Any

import torch

from wsp_gym.model import ModelConfig


def _infer_n_layers(weights: dict[str, torch.Tensor], config: dict[str, Any] | ModelConfig | None) -> int:
    if isinstance(config, ModelConfig):
        return int(config.n_layers)
    if isinstance(config, dict) and "n_layers" in config:
        return int(config["n_layers"])
    layers = set()
    for key in weights:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layers.add(int(parts[1]))
    return max(layers) + 1 if layers else 0


def _normalize_layer_indices(
    indices: dict[int, list[int]] | list[int] | None,
    n_layers: int,
) -> dict[int, list[int]] | None:
    if indices is None:
        return None
    if isinstance(indices, dict):
        return {int(k): [int(x) for x in v] for k, v in indices.items()}
    return {0: [int(x) for x in indices]} if n_layers > 0 else {}


def residual_svd_basis(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any] | ModelConfig | None = None,
    tol: float = 1e-4,
    include_embeddings: bool = True,
) -> dict[str, Any]:
    """Compute an orthonormal SVD decomposition of the active vs. null residual subspaces."""
    n_layers = _infer_n_layers(weights, config)
    d_model = int(weights["W_E"].shape[1])
    rows: list[torch.Tensor] = []

    if include_embeddings:
        rows.append(weights["W_E"])
        if "W_pos" in weights:
            rows.append(weights["W_pos"])

    for l in range(n_layers):
        p = f"blocks.{l}"
        n_heads = int(weights[f"{p}.attn.W_Q"].shape[0])
        for h in range(n_heads):
            rows.extend(
                [
                    weights[f"{p}.attn.W_Q"][h].T,
                    weights[f"{p}.attn.W_K"][h].T,
                    weights[f"{p}.attn.W_V"][h].T,
                    weights[f"{p}.attn.W_O"][h],
                ]
            )
        rows.append(weights[f"{p}.mlp.W_in"].T)
        if f"{p}.mlp.W_gate" in weights:
            rows.append(weights[f"{p}.mlp.W_gate"].T)
        rows.append(weights[f"{p}.mlp.W_out"])
        b_out = weights.get(f"{p}.mlp.b_out")
        if b_out is not None and float(b_out.norm().item()) > tol:
            rows.append(b_out.unsqueeze(0))

    rows.append(weights["W_U"].T)
    m_all = torch.cat(rows, dim=0)
    _, s_vals, vh = torch.linalg.svd(m_all, full_matrices=True)
    rank = int((s_vals > tol).sum().item())
    return {
        "rank": rank,
        "d_model": d_model,
        "singular_values": s_vals,
        "V_active": vh[:rank].T,
        "V_null": vh[rank:].T,
    }


def rope_rotation_matrix(
    d_head: int,
    offset: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact (d_head, d_head) RoPE relative-offset rotation matrix R(t - s)."""
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    half = d_head // 2
    freq_idx = torch.arange(half, dtype=dtype)
    inv_freq = 1.0 / (10000.0 ** (freq_idx / max(1, half)))
    angles = float(offset) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rot = torch.zeros((d_head, d_head), dtype=dtype)
    idx = torch.arange(half, dtype=torch.long)
    rot[idx, idx] = cos
    rot[idx, half + idx] = sin
    rot[half + idx, idx] = -sin
    rot[half + idx, half + idx] = cos
    return rot


def head_offset_profile(
    weights: dict[str, torch.Tensor],
    layer: int = 0,
    query_pos: int = 6,
    max_offset: int = 8,
) -> list[dict[str, Any]]:
    """Summarize each attention head's norms and dominant positional QK offset (learned or RoPE)."""
    p = f"blocks.{layer}.attn"
    w_q = weights[f"{p}.W_Q"]
    w_k = weights[f"{p}.W_K"]
    w_v = weights[f"{p}.W_V"]
    w_o = weights[f"{p}.W_O"]
    w_pos = weights.get("W_pos")
    n_heads = int(w_q.shape[0])
    d_head = int(w_q.shape[-1])

    e_bar: torch.Tensor | None = None
    rope_mats: list[torch.Tensor] | None = None
    if w_pos is None:
        w_e = weights["W_E"]
        e_norm = w_e * torch.rsqrt(w_e.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        e_bar = e_norm.mean(dim=0)
        norm1_w = weights.get(f"blocks.{layer}.norm1.weight")
        if norm1_w is not None:
            e_bar = e_bar * norm1_w
        rope_mats = [
            rope_rotation_matrix(d_head, float(delta), dtype=w_q.dtype)
            for delta in range(max_offset + 1)
        ]

    profiles: list[dict[str, Any]] = []
    for h in range(n_heads):
        entry: dict[str, Any] = {
            "head": h,
            "wq_norm": float(w_q[h].norm().item()),
            "wk_norm": float(w_k[h].norm().item()),
            "wv_norm": float(w_v[h].norm().item()),
            "wo_norm": float(w_o[h].norm().item()),
        }
        if w_pos is not None:
            pos_qk = w_pos @ w_q[h] @ w_k[h].T @ w_pos.T
            t = min(query_pos, int(w_pos.shape[0]) - 1)
            s_star = int(pos_qk[t, : t + 1].argmax().item())
            entry["dominant_offset"] = t - s_star
            entry["best_key_pos"] = s_star
            entry["max_pos_score"] = float(pos_qk[t, s_star].item())
            entry["pos0_score"] = float(pos_qk[t, 0].item())
            entry["pos_qk"] = pos_qk
        else:
            assert e_bar is not None and rope_mats is not None
            q_bar = e_bar @ w_q[h]
            k_bar = e_bar @ w_k[h]
            scale = 1.0 / math.sqrt(max(1, d_head))
            scores = torch.tensor(
                [float(((q_bar @ r_mat @ k_bar) * scale).item()) for r_mat in rope_mats],
                dtype=w_q.dtype,
            )
            dom_off = int(scores.argmax().item())
            entry["dominant_offset"] = dom_off
            entry["best_key_pos"] = max(0, query_pos - dom_off)
            entry["max_pos_score"] = float(scores[dom_off].item())
            entry["pos0_score"] = float(scores[0].item())
            entry["pos_qk"] = scores
        profiles.append(entry)
    return profiles


def virtual_ov_circuit(
    weights: dict[str, torch.Tensor],
    layer: int = 0,
    head: int | None = None,
) -> torch.Tensor:
    """Compute the virtual OV circuit W_V @ W_O in residual coordinates."""
    p = f"blocks.{layer}.attn"
    ov = torch.matmul(weights[f"{p}.W_V"], weights[f"{p}.W_O"])
    if head is not None:
        return ov[head]
    return ov


def _find_cancelling_neuron_pairs(
    w_in: torch.Tensor,
    b_in: torch.Tensor,
    out_proj: torch.Tensor,
    tol: float,
    w_gate: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    d_mlp = int(w_in.shape[1])
    in_norms = w_in.norm(dim=0)
    proj_norms = out_proj.norm(dim=1)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for i in range(d_mlp):
        if i in used or proj_norms[i].item() <= tol or in_norms[i].item() <= tol:
            continue
        for j in range(i + 1, d_mlp):
            if j in used or proj_norms[j].item() <= tol:
                continue
            in_close = (w_in[:, i] - w_in[:, j]).norm().item() <= tol * (1.0 + in_norms[i].item())
            bias_close = abs(float(b_in[i].item()) - float(b_in[j].item())) <= tol * (
                1.0 + abs(float(b_in[i].item()))
            )
            gate_close = True
            if w_gate is not None:
                gate_close = (w_gate[:, i] - w_gate[:, j]).norm().item() <= tol * (
                    1.0 + w_gate[:, i].norm().item()
                )
            out_opposite = (out_proj[i] + out_proj[j]).norm().item() <= tol * (1.0 + proj_norms[i].item())
            if in_close and bias_close and gate_close and out_opposite:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    return pairs


def _active_projection_mask(
    proj_mat: torch.Tensor,
    left_norms: torch.Tensor,
    right_col_norms: torch.Tensor,
    tol: float,
) -> torch.Tensor:
    """True where rows of proj_mat = L @ R exceed both absolute tol and float32 orthogonal rounding."""
    row_norms = proj_mat.norm(dim=1)
    cos_ok = (proj_mat.abs() > tol * (left_norms.unsqueeze(1) * right_col_norms.unsqueeze(0) + 1e-12)).any(dim=1)
    return (row_norms > tol) & cos_ok


def mlp_logit_impact(
    weights: dict[str, torch.Tensor],
    layer: int = 0,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Project MLP output vectors onto W_U and flag null-space or cancelling decoy pairs."""
    p = f"blocks.{layer}.mlp"
    w_in = weights[f"{p}.W_in"]
    b_in = weights[f"{p}.b_in"]
    w_out = weights[f"{p}.W_out"]
    w_gate = weights.get(f"{p}.W_gate")
    w_u = weights["W_U"]
    norm_f = weights.get("norm_final.weight")

    w_u_eff = w_u if norm_f is None else norm_f.unsqueeze(1) * w_u
    logit_mat = w_out @ w_u_eff
    in_norms = w_in.norm(dim=0)
    out_norms = w_out.norm(dim=1)
    logit_norms = logit_mat.norm(dim=1)
    top_class = logit_mat.argmax(dim=1)

    active_mask = _active_projection_mask(logit_mat, out_norms, w_u_eff.norm(dim=0), tol=tol)
    cancelling_pairs = _find_cancelling_neuron_pairs(w_in, b_in, logit_mat, tol=tol, w_gate=w_gate)
    cancelled_set = {idx for pair in cancelling_pairs for idx in pair}
    active_indices = [
        idx
        for idx in range(int(w_in.shape[1]))
        if bool(active_mask[idx].item()) and idx not in cancelled_set
    ]
    return {
        "logit_mat": logit_mat,
        "in_norms": in_norms,
        "out_norms": out_norms,
        "logit_norms": logit_norms,
        "b_in": b_in,
        "gate_norms": w_gate.norm(dim=0) if w_gate is not None else None,
        "top_class": top_class,
        "active_indices": active_indices,
        "cancelling_pairs": cancelling_pairs,
    }


def _find_cancelling_head_pairs(
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    tol: float,
) -> set[int]:
    n_heads = int(w_q.shape[0])
    cancelled: set[int] = set()
    for h1 in range(n_heads):
        if h1 in cancelled:
            continue
        qk1 = w_q[h1] @ w_k[h1].T
        ov1 = w_v[h1] @ w_o[h1]
        if ov1.norm().item() <= tol:
            continue
        for h2 in range(h1 + 1, n_heads):
            if h2 in cancelled:
                continue
            qk2 = w_q[h2] @ w_k[h2].T
            ov2 = w_v[h2] @ w_o[h2]
            qk_same = (qk1 - qk2).norm().item() <= tol * (1.0 + qk1.norm().item())
            ov_opposite = (ov1 + ov2).norm().item() <= tol * (1.0 + ov1.norm().item())
            if qk_same and ov_opposite:
                cancelled.add(h1)
                cancelled.add(h2)
                break
    return cancelled


def _infer_active_components(
    weights: dict[str, torch.Tensor],
    n_layers: int,
    active_heads: dict[int, list[int]] | None,
    active_mlp: dict[int, list[int]] | None,
    tol: float,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    resolved_heads: dict[int, list[int]] = {} if active_heads is None else dict(active_heads)
    resolved_mlp: dict[int, list[int]] = {} if active_mlp is None else dict(active_mlp)

    downstream_cols: list[torch.Tensor] = [weights["W_U"]]

    for l in range(n_layers - 1, -1, -1):
        p = f"blocks.{l}"
        downstream = torch.cat(downstream_cols, dim=1)

        if l not in resolved_mlp:
            w_in = weights[f"{p}.mlp.W_in"]
            b_in = weights[f"{p}.mlp.b_in"]
            w_out = weights[f"{p}.mlp.W_out"]
            w_gate = weights.get(f"{p}.mlp.W_gate")
            out_proj = w_out @ downstream
            active_mask = _active_projection_mask(
                out_proj, w_out.norm(dim=1), downstream.norm(dim=0), tol=tol
            )
            pairs = _find_cancelling_neuron_pairs(w_in, b_in, out_proj, tol=tol, w_gate=w_gate)
            cancelled = {idx for pair in pairs for idx in pair}
            resolved_mlp[l] = [
                j
                for j in range(int(w_in.shape[1]))
                if bool(active_mask[j].item()) and j not in cancelled
            ]

        if resolved_mlp[l]:
            downstream_cols.append(weights[f"{p}.mlp.W_in"][:, resolved_mlp[l]])
            if f"{p}.mlp.W_gate" in weights:
                downstream_cols.append(weights[f"{p}.mlp.W_gate"][:, resolved_mlp[l]])
        downstream_attn = torch.cat(downstream_cols, dim=1)
        down_attn_col_norms = downstream_attn.norm(dim=0)

        if l not in resolved_heads:
            w_q = weights[f"{p}.attn.W_Q"]
            w_k = weights[f"{p}.attn.W_K"]
            w_v = weights[f"{p}.attn.W_V"]
            w_o = weights[f"{p}.attn.W_O"]
            cancelled_h = _find_cancelling_head_pairs(w_q, w_k, w_v, w_o, tol=tol)
            kept_h: list[int] = []
            for h in range(int(w_q.shape[0])):
                if h in cancelled_h:
                    continue
                ov_h = w_v[h] @ w_o[h]
                ov_down = ov_h @ downstream_attn
                ov_norm = ov_h.norm()
                cos_ok = bool(
                    (ov_down.abs() > tol * (ov_norm * down_attn_col_norms.unsqueeze(0) + 1e-12))
                    .any()
                    .item()
                )
                if float(ov_down.norm().item()) > tol and cos_ok:
                    kept_h.append(h)
            resolved_heads[l] = kept_h

        for h in resolved_heads[l]:
            downstream_cols.extend(
                [
                    weights[f"{p}.attn.W_Q"][h],
                    weights[f"{p}.attn.W_K"][h],
                    weights[f"{p}.attn.W_V"][h],
                ]
            )

    return resolved_heads, resolved_mlp


def _absorb_rmsnorm_gains(weights: dict[str, torch.Tensor], n_layers: int) -> dict[str, torch.Tensor]:
    """Fold non-uniform RMSNorm scale vectors into immediately downstream linear projections."""
    out = {k: v.clone() for k, v in weights.items()}
    for l in range(n_layers):
        p = f"blocks.{l}"
        n1_key = f"{p}.norm1.weight"
        if n1_key in out:
            g1 = out[n1_key]
            for suffix in ("W_Q", "W_K", "W_V"):
                out[f"{p}.attn.{suffix}"] = out[f"{p}.attn.{suffix}"] * g1.view(1, -1, 1)
            out[n1_key] = torch.ones_like(g1)
        n2_key = f"{p}.norm2.weight"
        if n2_key in out:
            g2 = out[n2_key]
            out[f"{p}.mlp.W_in"] = out[f"{p}.mlp.W_in"] * g2.unsqueeze(1)
            if f"{p}.mlp.W_gate" in out:
                out[f"{p}.mlp.W_gate"] = out[f"{p}.mlp.W_gate"] * g2.unsqueeze(1)
            out[n2_key] = torch.ones_like(g2)
    if "norm_final.weight" in out:
        gf = out["norm_final.weight"]
        out["W_U"] = out["W_U"] * gf.unsqueeze(1)
        out["norm_final.weight"] = torch.ones_like(gf)
    return out


def canonical_pack(
    weights: dict[str, torch.Tensor],
    active_heads: dict[int, list[int]] | list[int] | None = None,
    active_mlp: dict[int, list[int]] | list[int] | None = None,
    V_basis: torch.Tensor | None = None,
    dim_slice: slice | None = None,
    head_dst_offset: int = 0,
    mlp_dst_offset: int = 0,
    target_weights: dict[str, torch.Tensor] | None = None,
    tol: float = 1e-4,
    zero_thresh: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Prune inactive/cancelling components and pack active circuits into a compact SVD basis."""
    n_layers = _infer_n_layers(weights, None)
    eff_weights = _absorb_rmsnorm_gains(weights, n_layers) if target_weights is None else weights
    norm_heads = _normalize_layer_indices(active_heads, n_layers)
    norm_mlp = _normalize_layer_indices(active_mlp, n_layers)
    resolved_heads, resolved_mlp = _infer_active_components(
        eff_weights, n_layers, norm_heads, norm_mlp, tol=tol
    )

    if V_basis is None:
        rows: list[torch.Tensor] = [eff_weights["W_U"].T]
        for l in range(n_layers):
            p = f"blocks.{l}"
            for h in resolved_heads.get(l, []):
                rows.extend(
                    [
                        eff_weights[f"{p}.attn.W_Q"][h].T,
                        eff_weights[f"{p}.attn.W_K"][h].T,
                        eff_weights[f"{p}.attn.W_V"][h].T,
                        eff_weights[f"{p}.attn.W_O"][h],
                    ]
                )
            mlp_idx = resolved_mlp.get(l, [])
            if mlp_idx:
                rows.append(eff_weights[f"{p}.mlp.W_in"][:, mlp_idx].T)
                rows.append(eff_weights[f"{p}.mlp.W_out"][mlp_idx])
                if f"{p}.mlp.W_gate" in eff_weights:
                    rows.append(eff_weights[f"{p}.mlp.W_gate"][:, mlp_idx].T)
            b_out = eff_weights.get(f"{p}.mlp.b_out")
            if b_out is not None and float(b_out.norm().item()) > tol:
                rows.append(b_out.unsqueeze(0))

        m_act = torch.cat(rows, dim=0)
        _, s_vals, vh = torch.linalg.svd(m_act, full_matrices=True)
        r_act = max(1, int((s_vals > tol).sum().item()))
        v_basis = vh[:r_act].T
    else:
        v_basis = V_basis
        r_act = int(v_basis.shape[1])

    dst_slice = slice(0, r_act) if dim_slice is None else dim_slice
    if (dst_slice.stop - dst_slice.start) != r_act:
        raise ValueError(
            f"dim_slice width ({dst_slice.stop - dst_slice.start}) does not match V_basis rank ({r_act})."
        )

    if target_weights is None:
        out = {k: torch.zeros_like(v) for k, v in eff_weights.items()}
        out["b_U"] = eff_weights["b_U"].clone()
        for k, v in eff_weights.items():
            if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "norm_final.weight":
                out[k] = v.clone()
    else:
        out = {k: v.clone() for k, v in target_weights.items()}

    out["W_E"][:, dst_slice] = eff_weights["W_E"] @ v_basis
    if "W_pos" in out and "W_pos" in eff_weights:
        out["W_pos"][:, dst_slice] = eff_weights["W_pos"] @ v_basis
    out["W_U"][dst_slice, :] = v_basis.T @ eff_weights["W_U"]

    use_rope = "W_pos" not in eff_weights
    for l in range(n_layers):
        p = f"blocks.{l}"
        for idx_h, src_h in enumerate(resolved_heads.get(l, [])):
            dst_h = head_dst_offset + idx_h
            wq = v_basis.T @ eff_weights[f"{p}.attn.W_Q"][src_h]
            wk = v_basis.T @ eff_weights[f"{p}.attn.W_K"][src_h]
            wv = v_basis.T @ eff_weights[f"{p}.attn.W_V"][src_h]
            wo = eff_weights[f"{p}.attn.W_O"][src_h] @ v_basis

            if use_rope:
                out[f"{p}.attn.W_Q"][dst_h, dst_slice, :] = wq
                out[f"{p}.attn.W_K"][dst_h, dst_slice, :] = wk
            else:
                _, s_qk, vh_qk = torch.linalg.svd(torch.cat([wq, wk], dim=0), full_matrices=False)
                r_qk = max(1, int((s_qk > tol).sum().item()))
                r_qk_mat = vh_qk[:r_qk].T
                out[f"{p}.attn.W_Q"][dst_h, dst_slice, :r_qk] = wq @ r_qk_mat
                out[f"{p}.attn.W_K"][dst_h, dst_slice, :r_qk] = wk @ r_qk_mat

            _, s_vo, vh_vo = torch.linalg.svd(torch.cat([wv, wo.T], dim=0), full_matrices=False)
            r_vo = max(1, int((s_vo > tol).sum().item()))
            r_vo_mat = vh_vo[:r_vo].T
            out[f"{p}.attn.W_V"][dst_h, dst_slice, :r_vo] = wv @ r_vo_mat
            out[f"{p}.attn.W_O"][dst_h, :r_vo, dst_slice] = r_vo_mat.T @ wo

        for idx_j, src_j in enumerate(resolved_mlp.get(l, [])):
            dst_j = mlp_dst_offset + idx_j
            out[f"{p}.mlp.W_in"][dst_slice, dst_j] = v_basis.T @ eff_weights[f"{p}.mlp.W_in"][:, src_j]
            out[f"{p}.mlp.b_in"][dst_j] = eff_weights[f"{p}.mlp.b_in"][src_j]
            if f"{p}.mlp.W_gate" in out and f"{p}.mlp.W_gate" in eff_weights:
                out[f"{p}.mlp.W_gate"][dst_slice, dst_j] = v_basis.T @ eff_weights[f"{p}.mlp.W_gate"][:, src_j]
            out[f"{p}.mlp.W_out"][dst_j, dst_slice] = eff_weights[f"{p}.mlp.W_out"][src_j, :] @ v_basis

        b_out = eff_weights.get(f"{p}.mlp.b_out")
        if b_out is not None and float(b_out.norm().item()) > tol:
            out[f"{p}.mlp.b_out"][dst_slice] = b_out @ v_basis

    for k in out:
        out[k] = torch.where(out[k].abs() < zero_thresh, torch.zeros_like(out[k]), out[k])
    return out
