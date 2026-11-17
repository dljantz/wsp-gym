"""Functional TinyTransformer engine with explicit mechanistic-interpretability tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for a functional TinyTransformer / GQA Transformer."""

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
    n_kv_heads: int | None = None
    rope_theta: float = 10000.0
    attn_bias: bool = False

    def effective_kv_heads(self) -> int:
        """Return effective KV head count (defaults to n_heads for MHA)."""
        if self.n_kv_heads is None:
            return int(self.n_heads)
        n_kv = int(self.n_kv_heads)
        if n_kv <= 0 or int(self.n_heads) % n_kv != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a positive multiple of n_kv_heads ({n_kv})."
            )
        return n_kv

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
    """Initialize explicit CPU float32 weight dictionary for TinyTransformer / GQA."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_kv = config.effective_kv_heads()

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
        weights[f"{prefix}.attn.W_K"] = _randn((n_kv, config.d_model, config.d_head))
        weights[f"{prefix}.attn.W_V"] = _randn((n_kv, config.d_model, config.d_head))
        weights[f"{prefix}.attn.W_O"] = _randn((config.n_heads, config.d_head, config.d_model))
        if config.attn_bias:
            weights[f"{prefix}.attn.b_Q"] = torch.zeros((config.n_heads, config.d_head), dtype=torch.float32)
            weights[f"{prefix}.attn.b_K"] = torch.zeros((n_kv, config.d_head), dtype=torch.float32)
            weights[f"{prefix}.attn.b_V"] = torch.zeros((n_kv, config.d_head), dtype=torch.float32)

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


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to Q (B, H, T, D) and K (B, H_kv, T, D)."""
    _, _, seq_len, d_head = q.shape
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    half = d_head // 2
    pos = torch.arange(seq_len, dtype=q.dtype, device=q.device).unsqueeze(1)
    freq_idx = torch.arange(half, dtype=q.dtype, device=q.device).unsqueeze(0)
    inv_freq = 1.0 / (float(rope_theta) ** (freq_idx / max(1, half)))
    angles = pos * inv_freq  # (T, half)
    cos = torch.cos(angles).view(1, 1, seq_len, half)
    sin = torch.sin(angles).view(1, 1, seq_len, half)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return _rotate(q), _rotate(k)


def _forward_layer(
    x: torch.Tensor,
    layer_idx: int,
    layer_weights: dict[str, torch.Tensor],
    config: ModelConfig,
    causal_mask: torch.Tensor | None,
    scale: float,
    return_cache: bool,
    cache: dict[str, torch.Tensor],
    cache_device: torch.device | None = None,
) -> torch.Tensor:
    prefix = f"blocks.{layer_idx}"
    bsz, seq_len, d_model = x.shape

    norm1_w = layer_weights.get(f"{prefix}.norm1.weight") if config.norm_type != "none" else None
    x_norm1 = _apply_norm(x, config.norm_type, norm1_w, config.rms_eps)

    w_q = layer_weights[f"{prefix}.attn.W_Q"]
    w_k = layer_weights[f"{prefix}.attn.W_K"]
    w_v = layer_weights[f"{prefix}.attn.W_V"]
    w_o = layer_weights[f"{prefix}.attn.W_O"]

    n_heads = int(w_q.shape[0])
    n_kv = int(w_k.shape[0])
    d_head = int(w_q.shape[-1])

    q = (x_norm1 @ w_q.permute(1, 0, 2).reshape(d_model, n_heads * d_head)).view(
        bsz, seq_len, n_heads, d_head
    ).transpose(1, 2)
    k = (x_norm1 @ w_k.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
        bsz, seq_len, n_kv, d_head
    ).transpose(1, 2)
    v = (x_norm1 @ w_v.permute(1, 0, 2).reshape(d_model, n_kv * d_head)).view(
        bsz, seq_len, n_kv, d_head
    ).transpose(1, 2)

    b_q = layer_weights.get(f"{prefix}.attn.b_Q")
    if b_q is not None:
        q = q + b_q.view(1, n_heads, 1, d_head)
    b_k = layer_weights.get(f"{prefix}.attn.b_K")
    if b_k is not None:
        k = k + b_k.view(1, n_kv, 1, d_head)
    b_v = layer_weights.get(f"{prefix}.attn.b_V")
    if b_v is not None:
        v = v + b_v.view(1, n_kv, 1, d_head)

    if config.pos_type == "rope":
        q, k = _apply_rope(q, k, rope_theta=config.rope_theta)

    if n_kv < n_heads:
        if n_heads % n_kv != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv}).")
        kv_rep = n_heads // n_kv
        k_exp = k.repeat_interleave(kv_rep, dim=1)
        v_exp = v.repeat_interleave(kv_rep, dim=1)
    else:
        k_exp, v_exp = k, v

    scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if causal_mask is not None:
        scores = scores + causal_mask.view(1, 1, seq_len, seq_len)
    pattern = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(pattern, v_exp).transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * d_head)
    attn_out = ctx @ w_o.reshape(n_heads * d_head, d_model)
    x = x + attn_out

    if return_cache:
        def _store(t: torch.Tensor) -> torch.Tensor:
            d = t.detach()
            return d.to(cache_device) if cache_device is not None else d

        cache[f"{prefix}.attn.pattern"] = _store(pattern)
        cache[f"{prefix}.attn.out"] = _store(attn_out)
        cache[f"{prefix}.resid_mid"] = _store(x)

    norm2_w = layer_weights.get(f"{prefix}.norm2.weight") if config.norm_type != "none" else None
    x_norm2 = _apply_norm(x, config.norm_type, norm2_w, config.rms_eps)

    pre_act = x_norm2 @ layer_weights[f"{prefix}.mlp.W_in"] + layer_weights[f"{prefix}.mlp.b_in"]
    if config.mlp_type == "swiglu":
        gate = F.silu(x_norm2 @ layer_weights[f"{prefix}.mlp.W_gate"])
        hidden = gate * pre_act
    else:
        hidden = _apply_activation(pre_act, config.act_type)
    mlp_out = hidden @ layer_weights[f"{prefix}.mlp.W_out"] + layer_weights[f"{prefix}.mlp.b_out"]
    x = x + mlp_out

    if return_cache:
        def _store2(t: torch.Tensor) -> torch.Tensor:
            d = t.detach()
            return d.to(cache_device) if cache_device is not None else d

        cache[f"{prefix}.mlp.hidden"] = _store2(hidden)
        cache[f"{prefix}.mlp.out"] = _store2(mlp_out)
        cache[f"{prefix}.resid_post"] = _store2(x)

    return x


def forward_with_cache(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run pure-functional forward pass on input_ids (B, T) or (T,)."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    _, seq_len = input_ids.shape
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
        x = _forward_layer(
            x=x,
            layer_idx=l,
            layer_weights=weights,
            config=config,
            causal_mask=causal_mask,
            scale=scale,
            return_cache=return_cache,
            cache=cache,
        )

    norm_f_w = weights.get("norm_final.weight") if config.norm_type != "none" else None
    x_final = _apply_norm(x, config.norm_type, norm_f_w, config.rms_eps)
    logits = x_final @ weights["W_U"] + weights["b_U"]

    if return_cache:
        cache["resid_final"] = x_final.detach()
        return logits, cache
    return logits


def forward_fast(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    return_cache: bool = False,
    min_cuda_params: int = 2_000_000,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run layer-streaming CUDA FP32 inference for large CPU state dicts, returning CPU tensors."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    total_params = sum(v.numel() for v in weights.values() if isinstance(v, torch.Tensor))
    can_stream_cuda = (
        torch.cuda.is_available()
        and total_params >= int(min_cuda_params)
        and all(
            (not isinstance(v, torch.Tensor)) or (v.device.type == "cpu" and not v.requires_grad)
            for v in weights.values()
        )
    )
    if not can_stream_cuda:
        return forward_with_cache(input_ids, weights, config, return_cache=return_cache)

    _, seq_len = input_ids.shape
    if seq_len > config.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {config.max_seq_len}")

    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    ids_cpu = input_ids.to(cpu_device)
    cache: dict[str, torch.Tensor] = {}

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32 if hasattr(torch.backends, "cudnn") else None
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_cudnn_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = False

    try:
        with torch.no_grad():
            x = weights["W_E"][ids_cpu].to(device)
            if config.pos_type == "learned":
                x = x + weights["W_pos"][:seq_len].unsqueeze(0).to(device)
            if return_cache:
                cache["resid_pre_0"] = x.detach().to(cpu_device)

            causal_mask: torch.Tensor | None = None
            if config.causal:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), -1e9, dtype=x.dtype, device=device),
                    diagonal=1,
                )

            scale = 1.0 / math.sqrt(config.d_head)
            for l in range(config.n_layers):
                prefix = f"blocks.{l}."
                layer_gpu = {
                    k: v.to(device, non_blocking=False)
                    for k, v in weights.items()
                    if k.startswith(prefix) and isinstance(v, torch.Tensor)
                }
                x = _forward_layer(
                    x=x,
                    layer_idx=l,
                    layer_weights=layer_gpu,
                    config=config,
                    causal_mask=causal_mask,
                    scale=scale,
                    return_cache=return_cache,
                    cache=cache,
                    cache_device=cpu_device,
                )
                del layer_gpu

            norm_f_w = (
                weights["norm_final.weight"].to(device)
                if config.norm_type != "none" and "norm_final.weight" in weights
                else None
            )
            x_final = _apply_norm(x, config.norm_type, norm_f_w, config.rms_eps)
            if return_cache:
                cache["resid_final"] = x_final.detach().to(cpu_device)

            w_u_gpu = weights["W_U"].to(device)
            b_u_gpu = weights["b_U"].to(device)
            logits = (x_final @ w_u_gpu + b_u_gpu).detach().to(cpu_device)
            del w_u_gpu, b_u_gpu
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        if prev_cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    if return_cache:
        return logits, cache
    return logits
