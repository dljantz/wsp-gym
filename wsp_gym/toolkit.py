"""Scale-invariant Weight-Space Programming diagnostic and surgery toolkit (`wsp_tools`)."""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Any, Callable, Iterable, Iterator

import torch

from wsp_gym.model import (
    LazyTwinWeightDict,
    ModelConfig,
    _apply_activation,
    _apply_norm,
    _apply_rope,
    forward_fast,
)


_SANDBOX_PROBE_HOOK: contextvars.ContextVar[Callable[..., Any] | None] = contextvars.ContextVar(
    "wsp_sandbox_probe_hook", default=None
)


@contextlib.contextmanager
def bind_sandbox_probe(probe_fn: Callable[..., Any] | None) -> Iterator[None]:
    """Bind the active SandboxSession `probe_forward` callback for toolkit calls."""
    token = _SANDBOX_PROBE_HOOK.set(probe_fn)
    try:
        yield
    finally:
        _SANDBOX_PROBE_HOOK.reset(token)


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
    row_blocks: Iterable[torch.Tensor],
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
    row_blocks: Iterable[torch.Tensor],
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
    is_lazy = isinstance(weights, LazyTwinWeightDict)
    we_shape = weights.raw_tensor("W_E").shape if is_lazy else weights["W_E"].shape
    d_model = int(we_shape[1])

    def _epf(k: str) -> torch.Tensor:
        return weights.get_ephemeral_f32(k) if is_lazy else weights[k].detach().to(dtype=torch.float32)

    def _iter_rows() -> Iterator[torch.Tensor]:
        if include_embeddings:
            yield _epf("W_E")
            if "W_pos" in weights:
                yield _epf("W_pos")

        for l in range(n_layers):
            p = f"blocks.{l}"
            yield _epf(f"{p}.attn.W_Q").permute(0, 2, 1).reshape(-1, d_model)
            yield _epf(f"{p}.attn.W_K").permute(0, 2, 1).reshape(-1, d_model)
            yield _epf(f"{p}.attn.W_V").permute(0, 2, 1).reshape(-1, d_model)
            yield _epf(f"{p}.attn.W_O").reshape(-1, d_model)
            yield _epf(f"{p}.mlp.W_in").T
            if f"{p}.mlp.W_gate" in weights:
                yield _epf(f"{p}.mlp.W_gate").T
            yield _epf(f"{p}.mlp.W_out")
            if f"{p}.mlp.b_out" in weights:
                b_out = _epf(f"{p}.mlp.b_out")
                if float(b_out.norm().item()) > tol:
                    yield b_out.unsqueeze(0)

        yield _epf("W_U").T

    s_vals, vh = _tall_skinny_svd_vh(_iter_rows())
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


def batched_head_svdvals(
    weights: dict[str, torch.Tensor],
    layers: Iterable[int] | None = None,
    proj: str = "W_O",
    config: dict[str, Any] | ModelConfig | None = None,
) -> dict[int, torch.Tensor]:
    """Compute per-head singular values across layers without polluting LazyTwinWeightDict._cache."""
    if proj in ("W_O", "W_Q", "W_K", "W_V"):
        sub_key = f"attn.{proj}"
    elif proj.startswith("attn."):
        sub_key = proj
    else:
        raise ValueError(f"Unsupported attention projection {proj!r}; expected W_O, W_Q, W_K, or W_V.")

    n_layers = _infer_n_layers(weights, config)
    layer_ids = list(range(n_layers)) if layers is None else [int(l) for l in layers]
    is_lazy = isinstance(weights, LazyTwinWeightDict)
    use_cuda = torch.cuda.is_available()
    cuda_dev = torch.device("cuda") if use_cuda else None
    out: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for l in layer_ids:
            key = f"blocks.{l}.{sub_key}"
            raw_t = weights.raw_tensor(key) if is_lazy else weights[key]
            if use_cuda and int(raw_t.numel()) >= 262_144:
                assert cuda_dev is not None
                t_gpu = (
                    weights.to_device_f32(key, cuda_dev)
                    if is_lazy
                    else raw_t.detach().to(device=cuda_dev, dtype=torch.float32)
                )
                svals = torch.linalg.svdvals(t_gpu).detach().cpu()
                del t_gpu
            else:
                t_cpu = (
                    weights.get_ephemeral_f32(key)
                    if is_lazy
                    else raw_t.detach().to(dtype=torch.float32)
                )
                if t_cpu.ndim == 3:
                    _, r_dim, c_dim = t_cpu.shape
                    if max(r_dim, c_dim) >= 4 * max(1, min(r_dim, c_dim)):
                        tall = t_cpu if r_dim >= c_dim else t_cpu.transpose(-2, -1)
                        r_mat = torch.linalg.qr(tall, mode="r").R
                        svals = torch.linalg.svdvals(r_mat).detach()
                        del tall, r_mat
                    else:
                        svals = torch.linalg.svdvals(t_cpu).detach()
                else:
                    svals = torch.linalg.svdvals(t_cpu).detach()
                del t_cpu
            out[l] = svals
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

    is_lazy = isinstance(weights, LazyTwinWeightDict)
    if n_layers > 0:
        q0 = weights.raw_tensor("blocks.0.attn.W_Q") if is_lazy else weights["blocks.0.attn.W_Q"]
        k0 = weights.raw_tensor("blocks.0.attn.W_K") if is_lazy else weights["blocks.0.attn.W_K"]
        n_heads = int(q0.shape[0])
        n_kv = int(k0.shape[0])
    else:
        n_heads = 0
        n_kv = 0

    use_cuda = bool(is_lazy and torch.cuda.is_available() and weights.num_parameters() >= 2_000_000)
    cuda_dev = torch.device("cuda") if use_cuda else None

    with torch.no_grad():
        for l in range(n_layers):
            p = f"blocks.{l}"
            wout_k = f"{p}.mlp.W_out"
            wo_k = f"{p}.attn.W_O"
            wk_k = f"{p}.attn.W_K"
            wv_k = f"{p}.attn.W_V"

            if use_cuda:
                assert cuda_dev is not None
                w_out = weights.to_device_f32(wout_k, cuda_dev)
            elif is_lazy:
                w_out = weights.get_ephemeral_f32(wout_k)
            else:
                w_out = weights[wout_k]

            out_norms = w_out.norm(dim=1)
            mlp_max = float(out_norms.max().item())
            mlp_med = float(out_norms.median().item())
            all_medians.append(mlp_med)
            top_vals, top_idx = torch.topk(out_norms, min(k_top, int(out_norms.numel())))
            top_neurons = [
                {"j": int(idx.item()), "norm": round(float(val.item()), 4)}
                for val, idx in zip(top_vals.cpu(), top_idx.cpu())
            ]
            del w_out, out_norms, top_vals, top_idx

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

            if use_cuda:
                assert cuda_dev is not None
                w_o = weights.to_device_f32(wo_k, cuda_dev)
                w_k = weights.to_device_f32(wk_k, cuda_dev)
                w_v = weights.to_device_f32(wv_k, cuda_dev)
            elif is_lazy:
                w_o = weights.get_ephemeral_f32(wo_k)
                w_k = weights.get_ephemeral_f32(wk_k)
                w_v = weights.get_ephemeral_f32(wv_k)
            else:
                w_o = weights[wo_k]
                w_k = weights[wk_k]
                w_v = weights[wv_k]

            wo_norms = w_o.norm(dim=(1, 2))
            wk_norms = w_k.norm(dim=(1, 2))
            wv_norms = w_v.norm(dim=(1, 2))
            wo_max = float(wo_norms.max().item())
            active_heads = (wo_norms > 1e-4).nonzero(as_tuple=True)[0].cpu().tolist()
            active_kv = ((wk_norms > 1e-4) | (wv_norms > 1e-4)).nonzero(as_tuple=True)[0].cpu().tolist()
            del w_o, w_k, w_v, wo_norms, wk_norms, wv_norms

            per_layer.append(
                {
                    "l": l,
                    "mlp_max": round(mlp_max, 4),
                    "mlp_med": round(mlp_med, 4),
                    "top_mlp": top_neurons,
                    "wo_max": round(wo_max, 4),
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


def _covariance_reader_block(
    weights: dict[str, torch.Tensor],
    cfg: ModelConfig,
    layer: int | str,
    site: str,
    max_cols: int = 1024,
) -> torch.Tensor:
    """Fetch a compact downstream reader weight block for tie-breaking zero-variance directions."""
    is_lazy = isinstance(weights, LazyTwinWeightDict)
    d_model = int(cfg.d_model)
    n_layers = int(cfg.n_layers)

    def _epf(k: str) -> torch.Tensor:
        return weights.get_ephemeral_f32(k) if is_lazy else weights[k].detach().to(dtype=torch.float32)

    def _raw(k: str) -> torch.Tensor:
        return weights.raw_tensor(k) if is_lazy else weights[k]

    if layer == "final" or (isinstance(layer, int) and layer < 0) or (
        isinstance(layer, int) and layer >= n_layers - 1 and site == "resid_post"
    ):
        wu_raw = _raw("W_U")
        stride = max(1, int(wu_raw.shape[1]) // max_cols)
        wu_sub = wu_raw[:, ::stride].detach().to(dtype=torch.float32)
        if site == "resid_post" and "norm_final.weight" in weights:
            wu_sub = wu_sub * _epf("norm_final.weight").unsqueeze(1)
        return wu_sub

    if layer == "embed":
        wq0 = _epf("blocks.0.attn.W_Q").permute(1, 0, 2).reshape(d_model, -1)
        return wq0[:, :: max(1, int(wq0.shape[1]) // max_cols)]

    l_idx = int(layer)
    if site in ("resid_pre", "norm1"):
        wq = _epf(f"blocks.{l_idx}.attn.W_Q").permute(1, 0, 2).reshape(d_model, -1)
        if site == "resid_pre" and f"blocks.{l_idx}.norm1.weight" in weights:
            wq = wq * _epf(f"blocks.{l_idx}.norm1.weight").unsqueeze(1)
        return wq[:, :: max(1, int(wq.shape[1]) // max_cols)]
    if site in ("resid_mid", "norm2"):
        w_in = _epf(f"blocks.{l_idx}.mlp.W_in")
        w_sub = w_in[:, :: max(1, int(w_in.shape[1]) // max_cols)]
        if site == "resid_mid" and f"blocks.{l_idx}.norm2.weight" in weights:
            w_sub = w_sub * _epf(f"blocks.{l_idx}.norm2.weight").unsqueeze(1)
        return w_sub
    if site == "resid_post":
        next_l = l_idx + 1
        wq_n = _epf(f"blocks.{next_l}.attn.W_Q").permute(1, 0, 2).reshape(d_model, -1)
        if f"blocks.{next_l}.norm1.weight" in weights:
            wq_n = wq_n * _epf(f"blocks.{next_l}.norm1.weight").unsqueeze(1)
        return wq_n[:, :: max(1, int(wq_n.shape[1]) // max_cols)]
    raise ValueError(f"Unsupported site {site!r} in covariance_nullspace.")


def covariance_nullspace(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any] | ModelConfig,
    layer: int | str = "final",
    calib_tokens: torch.Tensor | list[list[int]] | list[int] | str | None = None,
    tol_ratio: float = 1e-4,
    site: str = "resid_mid",
) -> dict[str, Any]:
    """Compute empirical or static activation-covariance active/null residual subspaces.

    `V_null` columns are ordered ASCENDING by empirical sensitivity (`V_null[:, 0]` is safest first).
    """
    cfg = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
    n_layers = int(cfg.n_layers)
    d_model = int(cfg.d_model)
    is_lazy = isinstance(weights, LazyTwinWeightDict)

    def _epf(k: str) -> torch.Tensor:
        return weights.get_ephemeral_f32(k) if is_lazy else weights[k].detach().to(dtype=torch.float32)

    def _raw(k: str) -> torch.Tensor:
        return weights.raw_tensor(k) if is_lazy else weights[k]

    use_cuda = torch.cuda.is_available() and d_model >= 512
    dev = torch.device("cuda") if use_cuda else torch.device("cpu")

    def _resolve_cache_key() -> tuple[str, str | None]:
        if layer == "final" or (isinstance(layer, int) and layer < 0):
            if site == "resid_post":
                return f"blocks.{n_layers - 1}.resid_post", None
            return "resid_final", None
        if layer == "embed":
            return "resid_pre_0", None
        l_idx = int(layer)
        if l_idx < 0 or l_idx >= n_layers:
            raise ValueError(f"Layer index {l_idx} out of range [0, {n_layers}).")
        if site == "resid_pre":
            return ("resid_pre_0" if l_idx == 0 else f"blocks.{l_idx - 1}.resid_post"), None
        if site == "norm1":
            return ("resid_pre_0" if l_idx == 0 else f"blocks.{l_idx - 1}.resid_post"), f"blocks.{l_idx}.norm1.weight"
        if site == "resid_mid":
            return f"blocks.{l_idx}.resid_mid", None
        if site == "norm2":
            return f"blocks.{l_idx}.resid_mid", f"blocks.{l_idx}.norm2.weight"
        if site == "resid_post":
            return f"blocks.{l_idx}.resid_post", None
        raise ValueError(f"Unsupported covariance_nullspace site: {site!r}")

    norm_key: str | None = None
    if (layer == "final" or (isinstance(layer, int) and layer < 0)) and site != "resid_post":
        norm_key = "norm_final.weight"
    elif isinstance(layer, int) and layer >= 0 and site in ("norm1", "norm2"):
        norm_key = f"blocks.{layer}.{site}.weight"

    with torch.no_grad():
        feat_dim = d_model
        if isinstance(calib_tokens, torch.Tensor) and calib_tokens.is_floating_point():
            x_mat = calib_tokens.detach().to(dtype=torch.float32, device="cpu")
            if x_mat.ndim == 1:
                x_mat = x_mat.unsqueeze(0)
            feat_dim = int(x_mat.shape[-1])
            if x_mat.ndim >= 3:
                x_mat = x_mat.reshape(-1, feat_dim)
            n_rows = max(1, int(x_mat.shape[0]))
            x_dev = (x_mat / math.sqrt(float(n_rows))).to(dev)
            if n_rows >= 2 * feat_dim:
                r_mat = torch.linalg.qr(x_dev, mode="r").R
                _, s_vals_d, vh_d = torch.linalg.svd(r_mat, full_matrices=True)
                del r_mat
            else:
                _, s_vals_d, vh_d = torch.linalg.svd(x_dev, full_matrices=True)
            del x_dev
            eigenvalues_d = torch.zeros(feat_dim, dtype=torch.float32, device=dev)
            n_s = min(feat_dim, int(s_vals_d.numel()))
            eigenvalues_d[:n_s] = s_vals_d[:n_s].pow(2)
            del s_vals_d
            sqrt_ev_d = eigenvalues_d.clamp_min(0.0).sqrt()
            sqrt_cov = ((vh_d.T * sqrt_ev_d.unsqueeze(0)) @ vh_d).detach().cpu()
            del sqrt_ev_d
        elif calib_tokens is None or (isinstance(calib_tokens, str) and calib_tokens == "static"):
            def _gram_clean(mat_raw: torch.Tensor, max_rows: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
                r = mat_raw.detach()
                if int(r.shape[-1]) != d_model:
                    r = r.reshape(-1, d_model)
                if max_rows > 0 and int(r.shape[0]) > max_rows:
                    r = r[:: max(1, int(r.shape[0]) // max_rows)]
                use_bf16 = use_cuda and r.dtype == torch.bfloat16
                m = r.to(device=dev, dtype=torch.bfloat16 if use_bf16 else torch.float32)
                rn = torch.linalg.vector_norm(m.float(), 2, dim=-1)
                nz = rn > 1e-6
                if bool(nz.any()):
                    med = float(rn[nz].median().item())
                    if med > 1e-6:
                        keep = rn <= 5.0 * med
                        if bool(keep.any()):
                            m = m[keep]
                n_r = max(1, int(m.shape[0]))
                g = (m.T @ m).float() * (1.0 / float(n_r))
                rn_ret = rn[nz].detach().cpu() if bool(nz.any()) else rn.detach().cpu()
                del m, rn
                return g, rn_ret

            g_we, rn_we = _gram_clean(_raw("W_E"), max_rows=32768)
            we_rms = float((rn_we.pow(2) / float(d_model) + cfg.rms_eps).sqrt().median().clamp_min(1e-4).item())
            del rn_we

            if layer == "embed":
                gram_d = g_we
            else:
                gram_d = g_we.clone()
                del g_we
                n_sources = 1.0
                if layer == "final" or (isinstance(layer, int) and layer < 0):
                    max_l = n_layers - 1
                    incl_last_attn = True
                    incl_last_mlp = True
                else:
                    max_l = int(layer)
                    incl_last_attn = site not in ("resid_pre", "norm1")
                    incl_last_mlp = site == "resid_post"
                for lw in range(max(0, max_l + 1)):
                    if lw < max_l or incl_last_attn:
                        g_wo, _ = _gram_clean(_raw(f"blocks.{lw}.attn.W_O").reshape(-1, d_model))
                        gram_d.add_(g_wo)
                        n_sources += 1.0
                        del g_wo
                    if lw < max_l or incl_last_mlp:
                        g_m, _ = _gram_clean(_raw(f"blocks.{lw}.mlp.W_out"))
                        gram_d.add_(g_m)
                        n_sources += 1.0
                        del g_m
                gram_d.mul_(1.0 / n_sources)

            if norm_key is not None and cfg.norm_type != "none":
                gram_d.mul_(1.0 / (we_rms * we_rms))
                if norm_key in weights:
                    nw_d = _epf(norm_key).to(device=dev, dtype=torch.float32)
                    gram_d = (gram_d * nw_d.unsqueeze(1)) * nw_d.unsqueeze(0)
                    del nw_d

            gram_d = 0.5 * (gram_d + gram_d.T)
            ev_true, evecs_true = torch.linalg.eigh(gram_d)
            eigenvalues_d = ev_true.flip(0).clamp_min(0.0)
            vh_true = evecs_true.flip(1).T
            sqrt_ev_d = eigenvalues_d.sqrt()
            sqrt_cov = ((vh_true.T * sqrt_ev_d.unsqueeze(0)) @ vh_true).detach().cpu()
            del ev_true, evecs_true, sqrt_ev_d

            nw_bar_key = norm_key or (
                "norm_final.weight"
                if ((layer == "final" or (isinstance(layer, int) and layer < 0)) and "norm_final.weight" in weights)
                else None
            )
            if nw_bar_key is not None and nw_bar_key in weights:
                nw_b = _epf(nw_bar_key).to(device=dev, dtype=torch.float32).abs()
                med_nw = float(nw_b.median().clamp_min(1e-3).item())
                med_ev = float(eigenvalues_d[d_model // 2].clamp_min(1e-8).item())
                short = ((1.05 * med_nw) / nw_b.clamp_min(0.25 * med_nw) - 1.0).clamp(0.0, 3.0)
                gram_sel = gram_d + torch.diag(4.0 * med_ev * short.pow(2))
                _, evecs_s = torch.linalg.eigh(0.5 * (gram_sel + gram_sel.T))
                vh_d = evecs_s.flip(1).T
                del nw_b, short, gram_sel, evecs_s
            else:
                vh_d = vh_true
            del gram_d, vh_true
        else:
            toks = calib_tokens if isinstance(calib_tokens, torch.Tensor) else torch.as_tensor(calib_tokens, dtype=torch.long)
            if toks.ndim == 1:
                toks = toks.unsqueeze(0)
            cache_key, nk_res = _resolve_cache_key()
            hook = _SANDBOX_PROBE_HOOK.get()
            if hook is not None:
                try:
                    _, cache = hook(toks, None, return_cache=True, cache_keys={cache_key})
                except TypeError:
                    _, cache = hook(toks, None, True)
            else:
                _, cache = forward_fast(toks, weights, cfg, return_cache=True, cache_keys={cache_key})
            x_raw = cache[cache_key].reshape(-1, d_model).detach().to(dtype=torch.float32, device="cpu")
            if nk_res is not None and cfg.norm_type != "none":
                nw = _epf(nk_res).cpu() if nk_res in weights else None
                x_mat = _apply_norm(x_raw, cfg.norm_type, nw, cfg.rms_eps)
            else:
                x_mat = x_raw
            n_rows = max(1, int(x_mat.shape[0]))
            x_dev = (x_mat / math.sqrt(float(n_rows))).to(dev)
            if n_rows >= 2 * d_model:
                r_mat = torch.linalg.qr(x_dev, mode="r").R
                _, s_vals_d, vh_d = torch.linalg.svd(r_mat, full_matrices=True)
                del r_mat
            else:
                _, s_vals_d, vh_d = torch.linalg.svd(x_dev, full_matrices=True)
            del x_dev
            eigenvalues_d = torch.zeros(d_model, dtype=torch.float32, device=dev)
            n_s = min(d_model, int(s_vals_d.numel()))
            eigenvalues_d[:n_s] = s_vals_d[:n_s].pow(2)
            del s_vals_d
            sqrt_ev_d = eigenvalues_d.clamp_min(0.0).sqrt()
            sqrt_cov = ((vh_d.T * sqrt_ev_d.unsqueeze(0)) @ vh_d).detach().cpu()
            del sqrt_ev_d

        eigenvalues = eigenvalues_d.detach().cpu()
        max_ev = max(1e-12, float(eigenvalues[0].item()))
        rank = int((eigenvalues > float(tol_ratio) * max_ev).sum().item())
        v_active = vh_d[:rank].T.detach().cpu().contiguous()

        if rank < feat_dim:
            if feat_dim == d_model:
                vh_tail_d = vh_d[rank:]
                ev_tail_d = eigenvalues_d[rank:]
                tol_ev = max(1e-12, float(tol_ratio) * max_ev)
                reader_d = _covariance_reader_block(weights, cfg, layer, site).to(dev)
                proj_d = vh_tail_d @ reader_d
                rcov_d = proj_d @ proj_d.T
                rmax = float(rcov_d.diagonal().max().clamp_min(1e-8).item())
                joint_d = torch.diag(50.0 * (ev_tail_d / tol_ev)) + 0.5 * (rcov_d / rmax)
                nw_tail_key = norm_key or (
                    "norm_final.weight"
                    if ((layer == "final" or (isinstance(layer, int) and layer < 0)) and "norm_final.weight" in weights)
                    else None
                )
                if nw_tail_key is not None and nw_tail_key in weights:
                    nw_t = _epf(nw_tail_key).to(dev).abs()
                    med_nw_t = float(nw_t.median().clamp_min(1e-3).item())
                    short_t = ((1.05 * med_nw_t) / nw_t.clamp_min(0.25 * med_nw_t) - 1.0).clamp(0.0, 3.0).unsqueeze(0)
                    nf_proj_d = vh_tail_d * short_t
                    nf_cov_d = nf_proj_d @ nf_proj_d.T
                    nf_max = float(nf_cov_d.diagonal().max().clamp_min(1e-8).item())
                    joint_d = joint_d + 2.0 * (nf_cov_d / nf_max)
                    del nw_t, short_t, nf_proj_d, nf_cov_d
                _, evecs_t_d = torch.linalg.eigh(joint_d)
                v_null = (evecs_t_d.T @ vh_tail_d).T.detach().cpu().contiguous()
                del vh_tail_d, ev_tail_d, reader_d, proj_d, rcov_d, joint_d, evecs_t_d
            else:
                v_null = vh_d[rank:].T.detach().cpu().contiguous()
        else:
            v_null = torch.zeros((feat_dim, 0), dtype=torch.float32)

        del vh_d, eigenvalues_d

    return {
        "d_model": feat_dim,
        "rank": rank,
        "null_dim": feat_dim - rank,
        "eigenvalues": eigenvalues,
        "V_active": v_active,
        "V_null": v_null,
        "sqrt_cov": sqrt_cov,
    }


def _to_long_batch(tokens: torch.Tensor | list[list[int]] | list[int]) -> torch.Tensor:
    t = tokens if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens, dtype=torch.long)
    t = t.to(dtype=torch.long, device="cpu")
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim != 2 or t.shape[0] == 0 or t.shape[1] == 0:
        raise ValueError(f"Expected non-empty 1D/2D token tensor, got shape {tuple(t.shape)}.")
    return t


def _resolve_token_targets(
    spec: int | list[int] | torch.Tensor | None,
    n_items: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if spec is None:
        return fallback
    if isinstance(spec, int):
        return torch.full((n_items,), int(spec), dtype=torch.long)
    t = spec if isinstance(spec, torch.Tensor) else torch.as_tensor(spec, dtype=torch.long)
    t = t.reshape(-1).to(dtype=torch.long, device="cpu")
    if int(t.numel()) == 1:
        return t.expand(n_items).clone()
    if int(t.numel()) != n_items:
        raise ValueError(f"Target/foil token length ({int(t.numel())}) must match batch size ({n_items}).")
    return t


def probe_attribution(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any] | ModelConfig,
    pos_tokens: torch.Tensor | list[list[int]] | list[int],
    neg_tokens: torch.Tensor | list[list[int]] | list[int] | None = None,
    target_tokens: int | list[int] | torch.Tensor | None = None,
    foil_tokens: int | list[int] | torch.Tensor | None = None,
    layers: Iterable[int] | None = None,
    top_k: int = 8,
    pos: int = -1,
) -> dict[str, Any]:
    """Contrastive direct logit attribution across layers, GQA heads, and MLP neurons.

    Consumes 1 `probe_forward` batch when `pos_tokens` and `neg_tokens` fit together
    (`<= 32` rows and equal sequence length), or 1-2 batches otherwise.
    """
    cfg = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
    n_layers = int(cfg.n_layers)
    d_model = int(cfg.d_model)
    n_heads = int(cfg.n_heads)
    n_kv = cfg.effective_kv_heads()
    d_head = int(cfg.d_head)
    gs = max(1, n_heads // max(1, n_kv))
    is_lazy = isinstance(weights, LazyTwinWeightDict)
    k_top = max(1, int(top_k))

    def _epf(k: str) -> torch.Tensor:
        return weights.get_ephemeral_f32(k) if is_lazy else weights[k].detach().to(dtype=torch.float32, device="cpu")

    def _raw(k: str) -> torch.Tensor:
        return weights.raw_tensor(k) if is_lazy else weights[k]

    pos_t = _to_long_batch(pos_tokens)
    neg_t = _to_long_batch(neg_tokens) if neg_tokens is not None else None
    layer_ids = list(range(n_layers)) if layers is None else [int(l) for l in layers]
    if any(l < 0 or l >= n_layers for l in layer_ids):
        raise ValueError(f"Layer index out of range [0, {n_layers}) in {layer_ids}.")

    cache_keys: set[str] = {"resid_final"}
    if n_layers > 0:
        cache_keys.add(f"blocks.{n_layers - 1}.resid_post")
    for l in layer_ids:
        cache_keys.add("resid_pre_0" if l == 0 else f"blocks.{l - 1}.resid_post")
        cache_keys.update(
            {
                f"blocks.{l}.attn.out",
                f"blocks.{l}.resid_mid",
                f"blocks.{l}.mlp.out",
                f"blocks.{l}.resid_post",
            }
        )

    hook = _SANDBOX_PROBE_HOOK.get()

    def _run_fwd(toks: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hook is not None:
            try:
                return hook(toks, None, return_cache=True, cache_keys=cache_keys)
            except TypeError:
                return hook(toks, None, True)
        return forward_fast(toks, weights, cfg, return_cache=True, cache_keys=cache_keys)

    with torch.no_grad():
        n_pos = int(pos_t.shape[0])
        if neg_t is not None and pos_t.shape[1] == neg_t.shape[1] and (n_pos + int(neg_t.shape[0])) <= 32:
            both_logits, both_cache = _run_fwd(torch.cat([pos_t, neg_t], dim=0))
            pos_logits = both_logits[:n_pos].float().cpu()
            neg_logits: torch.Tensor | None = both_logits[n_pos:].float().cpu()
            pos_cache = {k: v[:n_pos].detach().float().cpu() for k, v in both_cache.items()}
            neg_cache: dict[str, torch.Tensor] | None = {
                k: v[n_pos:].detach().float().cpu() for k, v in both_cache.items()
            }
            del both_logits, both_cache
        else:
            p_log, p_c = _run_fwd(pos_t)
            pos_logits = p_log.float().cpu()
            pos_cache = {k: v.detach().float().cpu() for k, v in p_c.items()}
            del p_log, p_c
            if neg_t is not None:
                n_log, n_c = _run_fwd(neg_t)
                neg_logits = n_log.float().cpu()
                neg_cache = {k: v.detach().float().cpu() for k, v in n_c.items()}
                del n_log, n_c
            else:
                neg_logits = None
                neg_cache = None

        pos_last = pos_logits[:, pos, :]
        pos_top2 = torch.topk(pos_last, k=min(2, int(pos_last.shape[-1])), dim=-1).indices
        pos_top1 = pos_top2[:, 0]
        tgt_idx = _resolve_token_targets(target_tokens, n_pos, pos_top1)

        if neg_logits is not None:
            neg_last = neg_logits[:, pos, :]
            neg_top1 = neg_last.argmax(dim=-1)
            if foil_tokens is not None:
                foil_idx = _resolve_token_targets(foil_tokens, n_pos, neg_top1)
            else:
                neg_mode = int(torch.mode(neg_top1).values.item())
                base_foil = neg_top1 if int(neg_top1.shape[0]) == n_pos else torch.full((n_pos,), neg_mode, dtype=torch.long)
                alt_pos = pos_top2[:, 1] if int(pos_top2.shape[1]) > 1 else pos_top1
                foil_idx = torch.where(base_foil != tgt_idx, base_foil, alt_pos)
        else:
            neg_top1 = None
            alt_pos = pos_top2[:, 1] if int(pos_top2.shape[1]) > 1 else pos_top1
            fallback_foil = torch.where(pos_top1 != tgt_idx, pos_top1, alt_pos)
            foil_idx = _resolve_token_targets(foil_tokens, n_pos, fallback_foil)

        row_ar = torch.arange(n_pos, dtype=torch.long)
        pos_ld = (pos_last[row_ar, tgt_idx] - pos_last[row_ar, foil_idx]).mean()
        if neg_logits is not None:
            neg_last_cmp = neg_logits[:, pos, :]
            if int(neg_last_cmp.shape[0]) == n_pos:
                neg_ld = (neg_last_cmp[row_ar, tgt_idx] - neg_last_cmp[row_ar, foil_idx]).mean()
            else:
                t_mode = int(torch.mode(tgt_idx).values.item())
                f_mode = int(torch.mode(foil_idx).values.item())
                neg_ld = (neg_last_cmp[:, t_mode] - neg_last_cmp[:, f_mode]).mean()
            mean_logit_diff = float((pos_ld - neg_ld).item())
        else:
            mean_logit_diff = float(pos_ld.item())

        raw_wu = _raw("W_U")
        wu_diff = (raw_wu[:, tgt_idx] - raw_wu[:, foil_idx]).to(dtype=torch.float32, device="cpu")  # (d_model, n_pos)
        nf = _epf("norm_final.weight") if (cfg.norm_type != "none" and "norm_final.weight" in weights) else None
        if nf is not None:
            wu_diff = wu_diff * nf.unsqueeze(1)
        if n_layers > 0:
            x_pre_final = pos_cache[f"blocks.{n_layers - 1}.resid_post"][:, pos, :]
            if cfg.norm_type == "rmsnorm":
                inv_rms = torch.rsqrt(x_pre_final.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_eps).squeeze(-1)
            elif cfg.norm_type == "layernorm":
                inv_rms = torch.rsqrt(x_pre_final.var(dim=-1, unbiased=False, keepdim=True) + cfg.rms_eps).squeeze(-1)
            else:
                inv_rms = torch.ones(n_pos, dtype=torch.float32)
            wu_diff = wu_diff * inv_rms.unsqueeze(0)
        logit_dir = wu_diff.mean(dim=1)
        metric_dir = logit_dir / logit_dir.norm().clamp_min(1e-8)

        layer_scores: list[dict[str, Any]] = []
        for l in layer_ids:
            da = pos_cache[f"blocks.{l}.attn.out"][:, pos, :].mean(dim=0)
            dm = pos_cache[f"blocks.{l}.mlp.out"][:, pos, :].mean(dim=0)
            dr = pos_cache[f"blocks.{l}.resid_post"][:, pos, :].mean(dim=0)
            if neg_cache is not None:
                da = da - neg_cache[f"blocks.{l}.attn.out"][:, pos, :].mean(dim=0)
                dm = dm - neg_cache[f"blocks.{l}.mlp.out"][:, pos, :].mean(dim=0)
                dr = dr - neg_cache[f"blocks.{l}.resid_post"][:, pos, :].mean(dim=0)
            a_dla = float((da @ logit_dir).item())
            m_dla = float((dm @ logit_dir).item())
            layer_scores.append(
                {
                    "layer": l,
                    "attn_dla": round(a_dla, 6),
                    "mlp_dla": round(m_dla, 6),
                    "total_dla": round(a_dla + m_dla, 6),
                    "attn_diff_norm": round(float(da.norm().item()), 6),
                    "mlp_diff_norm": round(float(dm.norm().item()), 6),
                    "resid_diff_norm": round(float(dr.norm().item()), 6),
                }
            )

        if len(layer_ids) <= 10:
            cand_layers = list(layer_ids)
        else:
            by_a = [r["layer"] for r in sorted(layer_scores, key=lambda r: abs(r["attn_dla"]), reverse=True)[:8]]
            by_m = [r["layer"] for r in sorted(layer_scores, key=lambda r: abs(r["mlp_dla"]), reverse=True)[:8]]
            by_dn = [
                r["layer"]
                for r in sorted(
                    layer_scores,
                    key=lambda r: max(r["attn_diff_norm"], r["mlp_diff_norm"]),
                    reverse=True,
                )[:3]
            ]
            cand_layers = sorted(set(by_a) | set(by_m) | set(by_dn))

        scale = 1.0 / math.sqrt(max(1, d_head))

        def _head_outs(x_pre: torch.Tensor, l_idx: int) -> torch.Tensor:
            bsz, seq_len, _ = x_pre.shape
            p_idx = pos if pos >= 0 else seq_len + pos
            p_idx = max(0, min(seq_len - 1, p_idx))
            pref = f"blocks.{l_idx}"
            n1 = _epf(f"{pref}.norm1.weight") if (cfg.norm_type != "none" and f"{pref}.norm1.weight" in weights) else None
            xn = _apply_norm(x_pre, cfg.norm_type, n1, cfg.rms_eps)
            wq = _epf(f"{pref}.attn.W_Q")
            wk = _epf(f"{pref}.attn.W_K")
            wv = _epf(f"{pref}.attn.W_V")
            wo = _epf(f"{pref}.attn.W_O")
            q = (xn @ wq.permute(1, 0, 2).reshape(d_model, n_heads * d_head)).view(
                bsz, seq_len, n_heads, d_head
            ).transpose(1, 2)
            k = (xn @ wk.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                bsz, seq_len, n_kv, d_head
            ).transpose(1, 2)
            v = (xn @ wv.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
                bsz, seq_len, n_kv, d_head
            ).transpose(1, 2)
            if f"{pref}.attn.b_Q" in weights:
                q = q + _epf(f"{pref}.attn.b_Q").view(1, n_heads, 1, d_head)
            if f"{pref}.attn.b_K" in weights:
                k = k + _epf(f"{pref}.attn.b_K").view(1, n_kv, 1, d_head)
            if f"{pref}.attn.b_V" in weights:
                v = v + _epf(f"{pref}.attn.b_V").view(1, n_kv, 1, d_head)
            if cfg.pos_type == "rope":
                q, k = _apply_rope(q, k, rope_theta=cfg.rope_theta)
            if n_kv < n_heads:
                k_exp = k.repeat_interleave(gs, dim=1)
                v_exp = v.repeat_interleave(gs, dim=1)
            else:
                k_exp, v_exp = k, v
            q_p = q[:, :, p_idx : p_idx + 1, :]
            scores = torch.matmul(q_p, k_exp.transpose(-2, -1)) * scale
            if cfg.causal and p_idx + 1 < seq_len:
                scores[:, :, :, p_idx + 1 :] = -1e9
            pat = torch.softmax(scores, dim=-1)
            ctx_h = torch.matmul(pat, v_exp).squeeze(2)  # (bsz, n_heads, d_head)
            return torch.einsum("bhd,hdm->bhm", ctx_h, wo)

        def _mlp_hidden(r_mid: torch.Tensor, l_idx: int) -> torch.Tensor:
            pref = f"blocks.{l_idx}"
            n2 = _epf(f"{pref}.norm2.weight") if (cfg.norm_type != "none" and f"{pref}.norm2.weight" in weights) else None
            xn2 = _apply_norm(r_mid, cfg.norm_type, n2, cfg.rms_eps)
            pre_act = xn2 @ _epf(f"{pref}.mlp.W_in") + _epf(f"{pref}.mlp.b_in")
            if cfg.mlp_type == "swiglu":
                gate = torch.nn.functional.silu(xn2 @ _epf(f"{pref}.mlp.W_gate"))
                return gate * pre_act
            return _apply_activation(pre_act, cfg.act_type)

        all_heads: list[dict[str, Any]] = []
        all_neurons: list[dict[str, Any]] = []
        for l in cand_layers:
            pre_k = "resid_pre_0" if l == 0 else f"blocks.{l - 1}.resid_post"
            h_pos = _head_outs(pos_cache[pre_k], l).mean(dim=0)
            dh = h_pos - (_head_outs(neg_cache[pre_k], l).mean(dim=0) if neg_cache is not None else 0.0)
            h_dlas = dh @ logit_dir
            h_norms = dh.norm(dim=-1)
            for h in range(n_heads):
                all_heads.append(
                    {
                        "layer": l,
                        "head": h,
                        "kv_head": h // gs,
                        "dla": round(float(h_dlas[h].item()), 6),
                        "out_diff_norm": round(float(h_norms[h].item()), 6),
                    }
                )

            act_pos = _mlp_hidden(pos_cache[f"blocks.{l}.resid_mid"][:, pos, :], l).mean(dim=0)
            act_diff = act_pos - (
                _mlp_hidden(neg_cache[f"blocks.{l}.resid_mid"][:, pos, :], l).mean(dim=0)
                if neg_cache is not None
                else 0.0
            )
            w_out = _epf(f"blocks.{l}.mlp.W_out")
            n_dlas = act_diff * (w_out @ logit_dir)
            top_n_vals, top_n_idx = torch.topk(n_dlas.abs(), k=min(k_top, int(n_dlas.numel())))
            del top_n_vals
            for j_t in top_n_idx.tolist():
                all_neurons.append(
                    {
                        "layer": l,
                        "neuron": int(j_t),
                        "dla": round(float(n_dlas[j_t].item()), 6),
                        "act_diff": round(float(act_diff[j_t].item()), 6),
                    }
                )

        top_layers = sorted(layer_scores, key=lambda r: abs(r["total_dla"]), reverse=True)[:k_top]
        top_heads = sorted(all_heads, key=lambda r: abs(r["dla"]), reverse=True)[:k_top]
        top_neurons = sorted(all_neurons, key=lambda r: abs(r["dla"]), reverse=True)[:k_top]

    return {
        "pos_top1": pos_top1.tolist(),
        "neg_top1": neg_top1.tolist() if neg_top1 is not None else None,
        "mean_logit_diff": round(mean_logit_diff, 6),
        "metric_dir": metric_dir.detach().cpu(),
        "logit_dir": logit_dir.detach().cpu(),
        "layer_scores": layer_scores,
        "top_layers": top_layers,
        "top_heads": top_heads,
        "top_neurons": top_neurons,
    }


def _extract_cov_projection_basis(
    cov_null: dict[str, Any] | torch.Tensor,
    d_in: int,
    null_dims: int | None,
) -> torch.Tensor:
    if isinstance(cov_null, dict):
        vn = cov_null["V_null"].detach().to(dtype=torch.float32, device="cpu")
        if null_dims is None:
            return vn
        k_want = max(1, min(d_in, int(null_dims)))
        if int(vn.shape[1]) >= k_want:
            return vn[:, :k_want]
        va = cov_null.get("V_active")
        if va is not None and int(va.shape[1]) > 0:
            need = k_want - int(vn.shape[1])
            va_cpu = va.detach().to(dtype=torch.float32, device="cpu")
            extra = va_cpu[:, max(0, int(va_cpu.shape[1]) - need) :].flip(1)
            return torch.cat([vn, extra], dim=1)
        return vn
    p_mat = cov_null.detach().to(dtype=torch.float32, device="cpu")
    if null_dims is not None:
        return p_mat[:, : int(null_dims)]
    return p_mat


def solve_nullspace_ridge(
    inputs: torch.Tensor,
    target_delta: torch.Tensor,
    cov_null: dict[str, Any] | torch.Tensor | None = None,
    preserve_inputs: torch.Tensor | None = None,
    ridge: float = 1e-3,
    cov_weight: float = 10.0,
    preserve_weight: float = 50.0,
    null_dims: int | None = None,
    hard_project: bool = False,
    rank: int | None = None,
) -> torch.Tensor:
    """Solve a constrained linear/ridge update minimizing collateral drift on clean covariance."""
    x_raw = inputs.detach().cpu().to(dtype=torch.float64)
    x = x_raw.unsqueeze(0) if x_raw.ndim == 1 else x_raw.reshape(-1, int(x_raw.shape[-1]))
    n_samp, d_in = int(x.shape[0]), int(x.shape[1])

    y_raw = target_delta.detach().cpu().to(dtype=torch.float64)
    squeeze_out = y_raw.ndim == 1 or (x_raw.ndim >= 2 and y_raw.ndim == x_raw.ndim - 1)
    y = y_raw.reshape(n_samp, -1) if squeeze_out else y_raw.reshape(-1, int(y_raw.shape[-1]))
    if int(y.shape[0]) != n_samp:
        raise ValueError(f"inputs rows ({n_samp}) != target_delta rows ({int(y.shape[0])}).")
    d_out = int(y.shape[1])

    x_keep: torch.Tensor | None = None
    m_keep = 0
    if preserve_inputs is not None:
        xk_raw = preserve_inputs.detach().cpu().to(dtype=torch.float64)
        if xk_raw.numel() > 0:
            x_keep = xk_raw.unsqueeze(0) if xk_raw.ndim == 1 else xk_raw.reshape(-1, d_in)
            m_keep = int(x_keep.shape[0])

    eig_scale = max(1.0, float((x.pow(2).sum(dim=-1).mean() / max(1, n_samp)).item()))
    eff_ridge = max(1e-8, float(ridge) * eig_scale)

    with torch.no_grad():
        if hard_project and cov_null is not None:
            p_mat = _extract_cov_projection_basis(cov_null, d_in, null_dims).to(dtype=torch.float64)
            if int(p_mat.shape[1]) == 0:
                delta_w = torch.zeros((d_in, d_out), dtype=torch.float32)
            else:
                z = x @ p_mat
                k_dim = int(p_mat.shape[1])
                gram = (z.T @ z) / max(1, n_samp) + float(eff_ridge) * torch.eye(k_dim, dtype=torch.float64)
                if x_keep is not None and m_keep > 0 and preserve_weight > 0:
                    zk = x_keep @ p_mat
                    gram = gram + (float(preserve_weight) / float(m_keep)) * (zk.T @ zk)
                rhs = (z.T @ y) / max(1, n_samp)
                try:
                    sol = torch.linalg.solve(gram, rhs)
                except RuntimeError:
                    sol = torch.linalg.lstsq(gram, rhs).solution
                delta_w = (p_mat @ sol).detach().cpu().to(dtype=torch.float32)
        elif cov_null is None and (n_samp + m_keep) < d_in:
            # Dual/Woodbury formulation avoids forming d_in x d_in matrices for wide MLP layers.
            wx = 1.0 / math.sqrt(float(max(1, n_samp)))
            x_tilde = x * wx
            y_tilde = y * wx
            if x_keep is not None and m_keep > 0 and preserve_weight > 0:
                wk = math.sqrt(float(preserve_weight) / float(m_keep))
                x_tilde = torch.cat([x_tilde, x_keep * wk], dim=0)
                y_tilde = torch.cat([y_tilde, torch.zeros((m_keep, d_out), dtype=torch.float64)], dim=0)
            m_dual = int(x_tilde.shape[0])
            dual_gram = x_tilde @ x_tilde.T + float(eff_ridge) * torch.eye(m_dual, dtype=torch.float64)
            try:
                alpha = torch.linalg.solve(dual_gram, y_tilde)
            except RuntimeError:
                alpha = torch.linalg.lstsq(dual_gram, y_tilde).solution
            delta_w = (x_tilde.T @ alpha).detach().cpu().to(dtype=torch.float32)
        else:
            use_cuda = torch.cuda.is_available() and d_in >= 512
            dev = torch.device("cuda") if use_cuda else torch.device("cpu")
            xd = x.to(dev)
            yd = y.to(dev)
            gram = (xd.T @ xd) / max(1, n_samp) + float(eff_ridge) * torch.eye(d_in, dtype=torch.float64, device=dev)
            if cov_null is not None and cov_weight > 0:
                if isinstance(cov_null, dict):
                    if "sqrt_cov" in cov_null:
                        sc = cov_null["sqrt_cov"].detach().to(device=dev, dtype=torch.float64)
                        c_clean = sc.T @ sc
                        del sc
                    elif "V_active" in cov_null and "eigenvalues" in cov_null:
                        va = cov_null["V_active"].detach().to(device=dev, dtype=torch.float64)
                        evs = cov_null["eigenvalues"][: va.shape[1]].detach().to(device=dev, dtype=torch.float64)
                        c_clean = (va * evs.unsqueeze(0)) @ va.T
                        del va, evs
                    else:
                        c_clean = torch.zeros((d_in, d_in), dtype=torch.float64, device=dev)
                else:
                    sc = cov_null.detach().to(device=dev, dtype=torch.float64)
                    c_clean = sc.T @ sc
                    del sc
                gram = gram + float(cov_weight) * c_clean
                del c_clean
            if x_keep is not None and m_keep > 0 and preserve_weight > 0:
                xkd = x_keep.to(dev)
                gram = gram + (float(preserve_weight) / float(m_keep)) * (xkd.T @ xkd)
                del xkd
            rhs = (xd.T @ yd) / max(1, n_samp)
            try:
                delta_w = torch.linalg.solve(gram, rhs).detach().cpu().to(dtype=torch.float32)
            except RuntimeError:
                delta_w = torch.linalg.lstsq(gram, rhs).solution.detach().cpu().to(dtype=torch.float32)
            del xd, yd, gram, rhs

        if rank is not None and 0 < int(rank) < min(d_in, d_out):
            u, s, vh = torch.linalg.svd(delta_w, full_matrices=False)
            r = int(rank)
            delta_w = (u[:, :r] * s[:r].unsqueeze(0)) @ vh[:r, :]

    return delta_w.squeeze(-1) if squeeze_out else delta_w
