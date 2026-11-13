"""Functional TinyTransformer engine with explicit mechanistic-interpretability tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for a functional TinyTransformer."""

    vocab_size: int = 32
    max_seq_len: int = 32
    n_layers: int = 2
    d_model: int = 32
    n_heads: int = 2
    d_head: int = 16
    d_mlp: int = 64
    norm_type: str = "rmsnorm"  # "rmsnorm", "layernorm", "none"
    mlp_type: str = "standard"  # "standard", "swiglu"
    act_type: str = "relu"  # "relu", "gelu", "silu"
    pos_type: str = "learned"  # "learned", "rope", "none"
    causal: bool = True
    rms_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


def init_weights(
    config: ModelConfig,
    seed: int = 0,
    scale: float = 0.02,
    zero_init: bool = False,
) -> dict[str, torch.Tensor]:
    """Initialize explicit CPU float32 weight dictionary for TinyTransformer."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _randn(shape: tuple[int, ...]) -> torch.Tensor:
        if zero_init:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.randn(shape, generator=gen, dtype=torch.float32) * scale

    weights: dict[str, torch.Tensor] = {
        "W_E": _randn((config.vocab_size, config.d_model)),
    }
    if config.pos_type == "learned":
        weights["W_pos"] = _randn((config.max_seq_len, config.d_model))

    for l in range(config.n_layers):
        prefix = f"blocks.{l}"
        if config.norm_type != "none":
            weights[f"{prefix}.norm1.weight"] = torch.ones(config.d_model, dtype=torch.float32)
        weights[f"{prefix}.attn.W_Q"] = _randn((config.n_heads, config.d_model, config.d_head))
        weights[f"{prefix}.attn.W_K"] = _randn((config.n_heads, config.d_model, config.d_head))
        weights[f"{prefix}.attn.W_V"] = _randn((config.n_heads, config.d_model, config.d_head))
        weights[f"{prefix}.attn.W_O"] = _randn((config.n_heads, config.d_head, config.d_model))

        if config.norm_type != "none":
            weights[f"{prefix}.norm2.weight"] = torch.ones(config.d_model, dtype=torch.float32)
        weights[f"{prefix}.mlp.W_in"] = _randn((config.d_model, config.d_mlp))
        weights[f"{prefix}.mlp.b_in"] = torch.zeros(config.d_mlp, dtype=torch.float32)
        if config.mlp_type == "swiglu":
            weights[f"{prefix}.mlp.W_gate"] = _randn((config.d_model, config.d_mlp))
        weights[f"{prefix}.mlp.W_out"] = _randn((config.d_mlp, config.d_model))
        weights[f"{prefix}.mlp.b_out"] = torch.zeros(config.d_model, dtype=torch.float32)

    if config.norm_type != "none":
        weights["norm_final.weight"] = torch.ones(config.d_model, dtype=torch.float32)
    weights["W_U"] = _randn((config.d_model, config.vocab_size))
    weights["b_U"] = torch.zeros(config.vocab_size, dtype=torch.float32)
    return weights


def _apply_norm(
    x: torch.Tensor,
    norm_type: str,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if norm_type == "none":
        return x
    if norm_type == "rmsnorm":
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        out = x * rms
        return out * weight if weight is not None else out
    if norm_type == "layernorm":
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + eps)
        return out * weight if weight is not None else out
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _apply_activation(x: torch.Tensor, act_type: str) -> torch.Tensor:
    if act_type == "relu":
        return F.relu(x)
    if act_type == "gelu":
        return F.gelu(x)
    if act_type == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported act_type: {act_type}")


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to Q and K of shape (B, H, T, D)."""
    _, _, seq_len, d_head = q.shape
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    half = d_head // 2
    pos = torch.arange(seq_len, dtype=q.dtype, device=q.device).unsqueeze(1)
    freq_idx = torch.arange(half, dtype=q.dtype, device=q.device).unsqueeze(0)
    inv_freq = 1.0 / (10000.0 ** (freq_idx / max(1, half)))
    angles = pos * inv_freq  # (T, half)
    cos = torch.cos(angles).view(1, 1, seq_len, half)
    sin = torch.sin(angles).view(1, 1, seq_len, half)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return _rotate(q), _rotate(k)


def forward_with_cache(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run pure-functional forward pass on input_ids (B, T) or (T,)."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    bsz, seq_len = input_ids.shape
    if seq_len > config.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {config.max_seq_len}")

    cache: dict[str, torch.Tensor] = {}
    x = weights["W_E"][input_ids]
    if config.pos_type == "learned":
        x = x + weights["W_pos"][:seq_len].unsqueeze(0)
    if return_cache:
        cache["resid_pre_0"] = x.detach()

    causal_mask: torch.Tensor | None = None
    if config.causal:
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, dtype=x.dtype, device=x.device),
            diagonal=1,
        )

    scale = 1.0 / math.sqrt(config.d_head)

    for l in range(config.n_layers):
        prefix = f"blocks.{l}"
        norm1_w = weights.get(f"{prefix}.norm1.weight") if config.norm_type != "none" else None
        x_norm1 = _apply_norm(x, config.norm_type, norm1_w, config.rms_eps)

        q = torch.einsum("btd,hdq->bhtq", x_norm1, weights[f"{prefix}.attn.W_Q"])
        k = torch.einsum("btd,hdk->bhtk", x_norm1, weights[f"{prefix}.attn.W_K"])
        v = torch.einsum("btd,hdv->bhtv", x_norm1, weights[f"{prefix}.attn.W_V"])

        if config.pos_type == "rope":
            q, k = _apply_rope(q, k)

        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
        if causal_mask is not None:
            scores = scores + causal_mask.view(1, 1, seq_len, seq_len)
        pattern = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum("bhts,bhsd,hdm->btm", pattern, v, weights[f"{prefix}.attn.W_O"])
        x = x + attn_out

        if return_cache:
            cache[f"{prefix}.attn.pattern"] = pattern.detach()
            cache[f"{prefix}.attn.out"] = attn_out.detach()
            cache[f"{prefix}.resid_mid"] = x.detach()

        norm2_w = weights.get(f"{prefix}.norm2.weight") if config.norm_type != "none" else None
        x_norm2 = _apply_norm(x, config.norm_type, norm2_w, config.rms_eps)

        pre_act = x_norm2 @ weights[f"{prefix}.mlp.W_in"] + weights[f"{prefix}.mlp.b_in"]
        if config.mlp_type == "swiglu":
            gate = F.silu(x_norm2 @ weights[f"{prefix}.mlp.W_gate"])
            hidden = gate * pre_act
        else:
            hidden = _apply_activation(pre_act, config.act_type)
        mlp_out = hidden @ weights[f"{prefix}.mlp.W_out"] + weights[f"{prefix}.mlp.b_out"]
        x = x + mlp_out

        if return_cache:
            cache[f"{prefix}.mlp.hidden"] = hidden.detach()
            cache[f"{prefix}.mlp.out"] = mlp_out.detach()
            cache[f"{prefix}.resid_post"] = x.detach()

    norm_f_w = weights.get("norm_final.weight") if config.norm_type != "none" else None
    x_final = _apply_norm(x, config.norm_type, norm_f_w, config.rms_eps)
    logits = x_final @ weights["W_U"] + weights["b_U"]

    if return_cache:
        cache["resid_final"] = x_final.detach()
        return logits, cache
    return logits
