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


def _tall_skinny_qr_r(
    row_blocks: list[torch.Tensor],
    chunk_rows: int = 8192,
) -> torch.Tensor:
    """Compute compact upper-triangular R factor of stacked row blocks via chunked QR."""
    r_acc: torch.Tensor | None = None
    buf: list[torch.Tensor] = []
    buf_rows = 0

    def _flush() -> None:
        nonlocal r_acc, buf, buf_rows
        if not buf:
            return
        parts = ([r_acc] if r_acc is not None else []) + buf
        mat = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        r_acc = torch.linalg.qr(mat, mode="r").R
        buf = []
        buf_rows = 0

    for block in row_blocks:
        if block.numel() == 0:
            continue
        n_rows = int(block.shape[0])
        for start in range(0, n_rows, chunk_rows):
            sl = block[start : start + chunk_rows]
            buf.append(sl)
            buf_rows += int(sl.shape[0])
            if buf_rows >= chunk_rows:
                _flush()
    _flush()
    if r_acc is None:
        raise ValueError("Cannot compute QR factor of empty row blocks.")
    return r_acc


def _tall_skinny_svd_vh(
    row_blocks: list[torch.Tensor],
    chunk_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact singular values and full V^T of vertically stacked blocks with O(d^2) RAM."""
    r_acc = _tall_skinny_qr_r(row_blocks, chunk_rows=chunk_rows)
    _, s_vals, vh = torch.linalg.svd(r_acc, full_matrices=True)
    return s_vals, vh


def lowrank_svd(
    mat: torch.Tensor,
    q: int = 32,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast randomized low-rank SVD returning (U, S, Vh) matching torch.linalg.svd orientation."""
    q_eff = max(1, min(int(q), int(min(mat.shape))))
    u, s, v = torch.svd_lowrank(mat.float(), q=q_eff, niter=int(niter))
    return u, s, v.T


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
        w_q = weights[f"{p}.attn.W_Q"]
        w_k = weights[f"{p}.attn.W_K"]
        w_v = weights[f"{p}.attn.W_V"]
        w_o = weights[f"{p}.attn.W_O"]
        rows.extend(
            [
                w_q.permute(0, 2, 1).reshape(-1, d_model),
                w_k.permute(0, 2, 1).reshape(-1, d_model),
                w_v.permute(0, 2, 1).reshape(-1, d_model),
                w_o.reshape(-1, d_model),
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
    s_vals, vh = _tall_skinny_svd_vh(rows)
    rank = int((s_vals > tol).sum().item())
    return {
        "rank": rank,
        "d_model": d_model,
        "singular_values": s_vals,
        "V_active": vh[:rank].T,
        "V_null": vh[rank:].T,
    }


def _resolve_rope_theta(
    rope_theta: float | None,
    config: dict[str, Any] | ModelConfig | None = None,
) -> float:
    if rope_theta is not None:
        return float(rope_theta)
    if isinstance(config, ModelConfig):
        return float(config.rope_theta)
    if isinstance(config, dict) and "rope_theta" in config:
        return float(config["rope_theta"])
    return 10000.0


def rope_rotation_matrix(
    d_head: int,
    offset: float,
    dtype: torch.dtype = torch.float32,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Return the exact (d_head, d_head) RoPE relative-offset rotation matrix R(t - s)."""
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    half = d_head // 2
    freq_idx = torch.arange(half, dtype=dtype)
    inv_freq = 1.0 / (float(rope_theta) ** (freq_idx / max(1, half)))
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
    rope_theta: float | None = None,
    config: dict[str, Any] | ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Summarize each attention head's norms and dominant positional QK offset (learned or RoPE, MHA or GQA)."""
    p = f"blocks.{layer}.attn"
    w_q = weights[f"{p}.W_Q"]
    w_k = weights[f"{p}.W_K"]
    w_v = weights[f"{p}.W_V"]
    w_o = weights[f"{p}.W_O"]
    w_pos = weights.get("W_pos")
    n_heads = int(w_q.shape[0])
    n_kv = int(w_k.shape[0])
    group_size = max(1, n_heads // max(1, n_kv))
    d_head = int(w_q.shape[-1])
    theta = _resolve_rope_theta(rope_theta, config)

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
            rope_rotation_matrix(d_head, float(delta), dtype=w_q.dtype, rope_theta=theta)
            for delta in range(max_offset + 1)
        ]

    profiles: list[dict[str, Any]] = []
    for h in range(n_heads):
        kv_h = min(n_kv - 1, h // group_size)
        entry: dict[str, Any] = {
            "head": h,
            "kv_head": kv_h,
            "wq_norm": float(w_q[h].norm().item()),
            "wk_norm": float(w_k[kv_h].norm().item()),
            "wv_norm": float(w_v[kv_h].norm().item()),
            "wo_norm": float(w_o[h].norm().item()),
        }
        if w_pos is not None:
            pos_qk = w_pos @ w_q[h] @ w_k[kv_h].T @ w_pos.T
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
            k_bar = e_bar @ w_k[kv_h]
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
    """Compute the virtual OV circuit W_V @ W_O in residual coordinates (MHA or GQA)."""
    p = f"blocks.{layer}.attn"
    w_v = weights[f"{p}.W_V"]
    w_o = weights[f"{p}.W_O"]
    n_heads = int(w_o.shape[0])
    n_kv = int(w_v.shape[0])
    group_size = max(1, n_heads // max(1, n_kv))
    if head is not None:
        h_idx = int(head)
        kv_h = min(n_kv - 1, h_idx // group_size)
        return w_v[kv_h] @ w_o[h_idx]
    if n_kv < n_heads:
        w_v = w_v.repeat_interleave(group_size, dim=0)
    return torch.matmul(w_v, w_o)


def _find_cancelling_neuron_pairs(
    w_in: torch.Tensor,
    b_in: torch.Tensor,
    out_proj: torch.Tensor,
    tol: float,
    w_gate: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    in_norms = w_in.norm(dim=0)
    proj_norms = out_proj.norm(dim=1)
    active_cand = ((proj_norms > tol) & (in_norms > tol)).nonzero(as_tuple=True)[0]
    if int(active_cand.numel()) < 2:
        return []

    d_in = int(w_in.shape[0])
    u_proj = torch.cos(
        torch.arange(d_in, dtype=w_in.dtype, device=w_in.device) * 1.61803398875 + 0.5
    )
    u_proj = u_proj / u_proj.norm().clamp_min(1e-12)
    sigs = u_proj @ w_in[:, active_cand]
    order = torch.argsort(sigs)
    sorted_idx = active_cand[order].tolist()
    sorted_sigs = sigs[order].tolist()

    in_norms_l = in_norms.tolist()
    proj_norms_l = proj_norms.tolist()
    b_in_l = b_in.tolist()
    gate_norms_l = w_gate.norm(dim=0).tolist() if w_gate is not None else None

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    n_cand = len(sorted_idx)
    denom_safe = max(0.5, 1.0 - tol)

    for pos_a in range(n_cand):
        i = sorted_idx[pos_a]
        if i in used:
            continue
        sig_i = float(sorted_sigs[pos_a])
        in_i = float(in_norms_l[i])
        max_in_other = (in_i + tol) / denom_safe
        max_sig_gap = tol * (1.0 + max(in_i, max_in_other))

        for pos_b in range(pos_a + 1, n_cand):
            if float(sorted_sigs[pos_b]) - sig_i > max_sig_gap:
                break
            j = sorted_idx[pos_b]
            if j in used:
                continue
            a, b = (i, j) if i < j else (j, i)
            in_a = float(in_norms_l[a])
            if (w_in[:, a] - w_in[:, b]).norm().item() > tol * (1.0 + in_a):
                continue
            b_a = float(b_in_l[a])
            if abs(b_a - float(b_in_l[b])) > tol * (1.0 + abs(b_a)):
                continue
            if w_gate is not None and gate_norms_l is not None:
                if (w_gate[:, a] - w_gate[:, b]).norm().item() > tol * (1.0 + float(gate_norms_l[a])):
                    continue
            if (out_proj[a] + out_proj[b]).norm().item() > tol * (1.0 + float(proj_norms_l[a])):
                continue
            pairs.append((a, b))
            used.add(i)
            used.add(j)
            break

    pairs.sort()
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


def _chunked_cos_active_mask(
    left_mat: torch.Tensor,
    col_blocks: list[torch.Tensor],
    tol: float,
    chunk_cols: int = 4096,
) -> torch.Tensor:
    """Memory-safe exact equivalent of `_active_projection_mask` over large column lists."""
    left_norms = left_mat.norm(dim=1)
    n_rows = int(left_mat.shape[0])
    row_norm_sq = torch.zeros(n_rows, dtype=left_mat.dtype, device=left_mat.device)
    cos_ok = torch.zeros(n_rows, dtype=torch.bool, device=left_mat.device)
    for blk in col_blocks:
        if blk.numel() == 0:
            continue
        n_c = int(blk.shape[1])
        for start in range(0, n_c, chunk_cols):
            c_blk = blk[:, start : start + chunk_cols]
            p_blk = left_mat @ c_blk
            row_norm_sq += (p_blk * p_blk).sum(dim=1)
            c_norms = c_blk.norm(dim=0)
            cos_ok |= (
                p_blk.abs() > tol * (left_norms.unsqueeze(1) * c_norms.unsqueeze(0) + 1e-12)
            ).any(dim=1)
    return (row_norm_sq.sqrt() > tol) & cos_ok


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
    in_norms = w_in.norm(dim=0)
    out_norms = w_out.norm(dim=1)
    vocab_size = int(w_u_eff.shape[1])

    if vocab_size <= 16384:
        logit_mat: torch.Tensor | None = w_out @ w_u_eff
        logit_norms = logit_mat.norm(dim=1)
        top_class = logit_mat.argmax(dim=1)
        active_mask = _active_projection_mask(logit_mat, out_norms, w_u_eff.norm(dim=0), tol=tol)
        cancelling_pairs = _find_cancelling_neuron_pairs(w_in, b_in, logit_mat, tol=tol, w_gate=w_gate)
    else:
        logit_mat = None
        r_wu = _tall_skinny_qr_r([w_u_eff.T])
        out_proj = w_out @ r_wu.T
        logit_norms = out_proj.norm(dim=1)
        top_class = torch.zeros(int(w_in.shape[1]), dtype=torch.long)
        best_val = torch.full((int(w_in.shape[1]),), -float("inf"), dtype=w_out.dtype)
        row_norm_sq = torch.zeros(int(w_in.shape[1]), dtype=w_out.dtype)
        cos_ok = torch.zeros(int(w_in.shape[1]), dtype=torch.bool)
        for start in range(0, vocab_size, 4096):
            c_blk = w_u_eff[:, start : start + 4096]
            p_blk = w_out @ c_blk
            blk_vals, blk_idx = p_blk.max(dim=1)
            better = blk_vals > best_val
            best_val = torch.where(better, blk_vals, best_val)
            top_class = torch.where(better, blk_idx + start, top_class)
            row_norm_sq += (p_blk * p_blk).sum(dim=1)
            cos_ok |= (
                p_blk.abs() > tol * (out_norms.unsqueeze(1) * c_blk.norm(dim=0).unsqueeze(0) + 1e-12)
            ).any(dim=1)
        active_mask = (row_norm_sq.sqrt() > tol) & cos_ok
        cancelling_pairs = _find_cancelling_neuron_pairs(w_in, b_in, out_proj, tol=tol, w_gate=w_gate)

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
    n_kv = int(w_k.shape[0])
    group_size = max(1, n_heads // max(1, n_kv))
    cancelled: set[int] = set()
    ov_norms = [float((w_v[min(n_kv - 1, h // group_size)] @ w_o[h]).norm().item()) for h in range(n_heads)]
    for h1 in range(n_heads):
        if h1 in cancelled or ov_norms[h1] <= tol:
            continue
        kv1 = min(n_kv - 1, h1 // group_size)
        qk1 = w_q[h1] @ w_k[kv1].T
        ov1 = w_v[kv1] @ w_o[h1]
        qk1_norm = float(qk1.norm().item())
        for h2 in range(h1 + 1, n_heads):
            if h2 in cancelled or ov_norms[h2] <= tol:
                continue
            kv2 = min(n_kv - 1, h2 // group_size)
            ov2 = w_v[kv2] @ w_o[h2]
            ov_opposite = (ov1 + ov2).norm().item() <= tol * (1.0 + ov_norms[h1])
            if not ov_opposite:
                continue
            qk2 = w_q[h2] @ w_k[kv2].T
            qk_same = (qk1 - qk2).norm().item() <= tol * (1.0 + qk1_norm)
            if qk_same:
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
    total_cols = int(weights["W_U"].shape[1])
    r_down = _tall_skinny_qr_r([weights["W_U"].T])

    def _push_cols(blk: torch.Tensor) -> None:
        nonlocal r_down, total_cols
        if blk.numel() == 0:
            return
        downstream_cols.append(blk)
        total_cols += int(blk.shape[1])
        r_down = torch.linalg.qr(torch.cat([r_down, blk.T], dim=0), mode="r").R

    for l in range(n_layers - 1, -1, -1):
        p = f"blocks.{l}"

        if l not in resolved_mlp:
            w_in = weights[f"{p}.mlp.W_in"]
            b_in = weights[f"{p}.mlp.b_in"]
            w_out = weights[f"{p}.mlp.W_out"]
            w_gate = weights.get(f"{p}.mlp.W_gate")
            if total_cols <= 8192:
                downstream = torch.cat(downstream_cols, dim=1)
                out_proj = w_out @ downstream
                active_mask = _active_projection_mask(
                    out_proj, w_out.norm(dim=1), downstream.norm(dim=0), tol=tol
                )
            else:
                out_proj = w_out @ r_down.T
                active_mask = _chunked_cos_active_mask(w_out, downstream_cols, tol=tol)
            pairs = _find_cancelling_neuron_pairs(w_in, b_in, out_proj, tol=tol, w_gate=w_gate)
            cancelled = {idx for pair in pairs for idx in pair}
            resolved_mlp[l] = [
                j
                for j in range(int(w_in.shape[1]))
                if bool(active_mask[j].item()) and j not in cancelled
            ]

        if resolved_mlp[l]:
            _push_cols(weights[f"{p}.mlp.W_in"][:, resolved_mlp[l]])
            if f"{p}.mlp.W_gate" in weights:
                _push_cols(weights[f"{p}.mlp.W_gate"][:, resolved_mlp[l]])

        if l not in resolved_heads:
            w_q = weights[f"{p}.attn.W_Q"]
            w_k = weights[f"{p}.attn.W_K"]
            w_v = weights[f"{p}.attn.W_V"]
            w_o = weights[f"{p}.attn.W_O"]
            n_heads = int(w_q.shape[0])
            n_kv = int(w_k.shape[0])
            group_size = max(1, n_heads // max(1, n_kv))
            cancelled_h = _find_cancelling_head_pairs(w_q, w_k, w_v, w_o, tol=tol)
            kept_h: list[int] = []
            if total_cols <= 8192:
                downstream_attn = torch.cat(downstream_cols, dim=1)
                down_attn_col_norms = downstream_attn.norm(dim=0)
            else:
                downstream_attn = None
                down_attn_col_norms = None

            for h in range(n_heads):
                if h in cancelled_h:
                    continue
                kv_h = min(n_kv - 1, h // group_size)
                ov_h = w_v[kv_h] @ w_o[h]
                ov_norm = ov_h.norm()
                if float(ov_norm.item()) <= tol:
                    continue
                if downstream_attn is not None and down_attn_col_norms is not None:
                    ov_down = ov_h @ downstream_attn
                    cos_ok = bool(
                        (ov_down.abs() > tol * (ov_norm * down_attn_col_norms.unsqueeze(0) + 1e-12))
                        .any()
                        .item()
                    )
                    norm_ok = float(ov_down.norm().item()) > tol
                else:
                    norm_ok = float((ov_h @ r_down.T).norm().item()) > tol
                    cos_ok = bool(
                        _chunked_cos_active_mask(ov_h, downstream_cols, tol=tol).any().item()
                    )
                if norm_ok and cos_ok:
                    kept_h.append(h)
            resolved_heads[l] = kept_h

        w_q_l = weights[f"{p}.attn.W_Q"]
        w_k_l = weights[f"{p}.attn.W_K"]
        w_v_l = weights[f"{p}.attn.W_V"]
        n_heads_l = int(w_q_l.shape[0])
        n_kv_l = int(w_k_l.shape[0])
        group_size_l = max(1, n_heads_l // max(1, n_kv_l))
        seen_kv: set[int] = set()
        for h in resolved_heads[l]:
            _push_cols(w_q_l[h])
            kv_h = min(n_kv_l - 1, h // group_size_l)
            if kv_h not in seen_kv:
                seen_kv.add(kv_h)
                _push_cols(w_k_l[kv_h])
                _push_cols(w_v_l[kv_h])

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
    """Prune inactive/cancelling components and pack active circuits into a compact SVD basis (MHA or GQA)."""
    n_layers = _infer_n_layers(weights, None)
    eff_weights = _absorb_rmsnorm_gains(weights, n_layers) if target_weights is None else weights
    norm_heads = _normalize_layer_indices(active_heads, n_layers)
    norm_mlp = _normalize_layer_indices(active_mlp, n_layers)
    resolved_heads, resolved_mlp = _infer_active_components(
        eff_weights, n_layers, norm_heads, norm_mlp, tol=tol
    )

    if V_basis is None:
        rows: list[torch.Tensor] = [eff_weights["W_U"].T]
        if "norm_final.weight" in eff_weights and "norm_final.bias" not in eff_weights:
            we_nz = eff_weights["W_E"][eff_weights["W_E"].norm(dim=1) > tol]
            if we_nz.numel() > 0:
                rows.insert(0, we_nz)
        for l in range(n_layers):
            p = f"blocks.{l}"
            w_q = eff_weights[f"{p}.attn.W_Q"]
            w_k = eff_weights[f"{p}.attn.W_K"]
            w_v = eff_weights[f"{p}.attn.W_V"]
            w_o = eff_weights[f"{p}.attn.W_O"]
            n_heads = int(w_q.shape[0])
            n_kv = int(w_k.shape[0])
            group_size = max(1, n_heads // max(1, n_kv))
            seen_kv: set[int] = set()
            for h in resolved_heads.get(l, []):
                kv_h = min(n_kv - 1, h // group_size)
                rows.extend([w_q[h].T, w_o[h]])
                if kv_h not in seen_kv:
                    seen_kv.add(kv_h)
                    rows.extend([w_k[kv_h].T, w_v[kv_h].T])
            mlp_idx = resolved_mlp.get(l, [])
            if mlp_idx:
                rows.append(eff_weights[f"{p}.mlp.W_in"][:, mlp_idx].T)
                rows.append(eff_weights[f"{p}.mlp.W_out"][mlp_idx])
                if f"{p}.mlp.W_gate" in eff_weights:
                    rows.append(eff_weights[f"{p}.mlp.W_gate"][:, mlp_idx].T)
            b_out = eff_weights.get(f"{p}.mlp.b_out")
            if b_out is not None and float(b_out.norm().item()) > tol:
                rows.append(b_out.unsqueeze(0))

        s_vals, vh = _tall_skinny_svd_vh(rows)
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
        w_q_all = eff_weights[f"{p}.attn.W_Q"]
        w_k_all = eff_weights[f"{p}.attn.W_K"]
        w_v_all = eff_weights[f"{p}.attn.W_V"]
        w_o_all = eff_weights[f"{p}.attn.W_O"]
        n_heads = int(w_q_all.shape[0])
        n_kv = int(w_k_all.shape[0])
        group_size = max(1, n_heads // max(1, n_kv))

        by_kv: dict[int, list[int]] = {}
        for src_h in resolved_heads.get(l, []):
            kv_h = min(n_kv - 1, src_h // group_size)
            by_kv.setdefault(kv_h, []).append(src_h)

        kv_dst_base = head_dst_offset // group_size
        for g_idx, (src_kv, src_group_heads) in enumerate(by_kv.items()):
            dst_kv = kv_dst_base + g_idx
            wk = v_basis.T @ w_k_all[src_kv]
            wv = v_basis.T @ w_v_all[src_kv]
            wqs = [v_basis.T @ w_q_all[sh] for sh in src_group_heads]
            wos = [w_o_all[sh] @ v_basis for sh in src_group_heads]

            if use_rope:
                out[f"{p}.attn.W_K"][dst_kv, dst_slice, :] = wk
                if f"{p}.attn.b_K" in out and f"{p}.attn.b_K" in eff_weights:
                    out[f"{p}.attn.b_K"][dst_kv] = eff_weights[f"{p}.attn.b_K"][src_kv]
                for local_idx, (sh, wq) in enumerate(zip(src_group_heads, wqs)):
                    dst_h = (head_dst_offset + g_idx) if n_kv == n_heads else (dst_kv * group_size + local_idx)
                    out[f"{p}.attn.W_Q"][dst_h, dst_slice, :] = wq
                    if f"{p}.attn.b_Q" in out and f"{p}.attn.b_Q" in eff_weights:
                        out[f"{p}.attn.b_Q"][dst_h] = eff_weights[f"{p}.attn.b_Q"][sh]
            else:
                qk_mats = ([wqs[0], wk] if n_kv == n_heads else (wqs + [wk]))
                _, s_qk, vh_qk = torch.linalg.svd(torch.cat(qk_mats, dim=0), full_matrices=False)
                r_qk = max(1, int((s_qk > tol).sum().item()))
                r_qk_mat = vh_qk[:r_qk].T
                out[f"{p}.attn.W_K"][dst_kv, dst_slice, :r_qk] = wk @ r_qk_mat
                if f"{p}.attn.b_K" in out and f"{p}.attn.b_K" in eff_weights:
                    out[f"{p}.attn.b_K"][dst_kv, :r_qk] = eff_weights[f"{p}.attn.b_K"][src_kv] @ r_qk_mat
                for local_idx, (sh, wq) in enumerate(zip(src_group_heads, wqs)):
                    dst_h = (head_dst_offset + g_idx) if n_kv == n_heads else (dst_kv * group_size + local_idx)
                    out[f"{p}.attn.W_Q"][dst_h, dst_slice, :r_qk] = wq @ r_qk_mat
                    if f"{p}.attn.b_Q" in out and f"{p}.attn.b_Q" in eff_weights:
                        out[f"{p}.attn.b_Q"][dst_h, :r_qk] = eff_weights[f"{p}.attn.b_Q"][sh] @ r_qk_mat

            vo_mats = [wv] + [wo.T for wo in wos]
            _, s_vo, vh_vo = torch.linalg.svd(torch.cat(vo_mats, dim=0), full_matrices=False)
            r_vo = max(1, int((s_vo > tol).sum().item()))
            r_vo_mat = vh_vo[:r_vo].T
            out[f"{p}.attn.W_V"][dst_kv, dst_slice, :r_vo] = wv @ r_vo_mat
            if f"{p}.attn.b_V" in out and f"{p}.attn.b_V" in eff_weights:
                out[f"{p}.attn.b_V"][dst_kv, :r_vo] = eff_weights[f"{p}.attn.b_V"][src_kv] @ r_vo_mat
            for local_idx, wo in enumerate(wos):
                dst_h = (head_dst_offset + g_idx) if n_kv == n_heads else (dst_kv * group_size + local_idx)
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

    eff_zero_thresh = (
        min(zero_thresh, 1e-6)
        if sum(v.numel() for v in eff_weights.values()) >= 10_000_000
        else zero_thresh
    )
    for k in out:
        out[k] = torch.where(out[k].abs() < eff_zero_thresh, torch.zeros_like(out[k]), out[k])
    return out


def layer_anomaly_scan(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any] | ModelConfig | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Summarize per-layer MLP/attention norms and highlight cross-layer outlier neurons/heads."""
    n_layers = _infer_n_layers(weights, config)
    k_top = max(1, int(top_k))
    per_layer: list[dict[str, Any]] = []
    all_mlp_anomalies: list[dict[str, Any]] = []
    all_medians: list[float] = []

    n_heads = int(weights["blocks.0.attn.W_Q"].shape[0]) if n_layers > 0 else 0
    n_kv = int(weights["blocks.0.attn.W_K"].shape[0]) if n_layers > 0 else 0

    for l in range(n_layers):
        p = f"blocks.{l}"
        w_out = weights[f"{p}.mlp.W_out"]
        out_norms = w_out.norm(dim=1)
        mlp_max = float(out_norms.max().item())
        mlp_med = float(out_norms.median().item())
        all_medians.append(mlp_med)
        top_vals, top_idx = torch.topk(out_norms, min(k_top, int(out_norms.numel())))
        top_neurons = [
            {"j": int(idx.item()), "norm": round(float(val.item()), 4)}
            for val, idx in zip(top_vals, top_idx)
        ]
        denom = max(1e-6, mlp_med)
        for item in top_neurons:
            all_mlp_anomalies.append(
                {
                    "layer": l,
                    "j": item["j"],
                    "out_norm": item["norm"],
                    "ratio_to_med": round(item["norm"] / denom, 2),
                }
            )

        w_o = weights[f"{p}.attn.W_O"]
        w_k = weights[f"{p}.attn.W_K"]
        w_v = weights[f"{p}.attn.W_V"]
        wo_norms = w_o.norm(dim=(1, 2))
        wk_norms = w_k.norm(dim=(1, 2))
        wv_norms = w_v.norm(dim=(1, 2))
        active_heads = (wo_norms > 1e-4).nonzero(as_tuple=True)[0].tolist()
        active_kv = ((wk_norms > 1e-4) | (wv_norms > 1e-4)).nonzero(as_tuple=True)[0].tolist()

        per_layer.append(
            {
                "l": l,
                "mlp_max": round(mlp_max, 4),
                "mlp_med": round(mlp_med, 4),
                "top_mlp": top_neurons,
                "wo_max": round(float(wo_norms.max().item()), 4),
                "n_act_h": len(active_heads),
                "n_act_kv": len(active_kv),
                "active_heads": active_heads if len(active_heads) <= 6 else active_heads[:6],
                "active_kv_heads": active_kv if len(active_kv) <= 6 else active_kv[:6],
            }
        )

    global_med = float(sorted(all_medians)[len(all_medians) // 2]) if all_medians else 0.0
    all_mlp_anomalies.sort(key=lambda d: (d["ratio_to_med"], d["out_norm"]), reverse=True)
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "global_median_mlp_out_norm": round(global_med, 4),
        "top_mlp_anomalies": all_mlp_anomalies[: max(5, k_top)],
        "per_layer": per_layer,
    }
