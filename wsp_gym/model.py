"""Functional TinyTransformer engine with explicit mechanistic-interpretability tensors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterator

import torch
import torch.nn.functional as F


class _LazyTwinValuesView:
    """Streaming view for LazyTwinWeightDict.values() honoring max_cache LRU eviction."""

    def __init__(self, mapping: "LazyTwinWeightDict") -> None:
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for k in list(self._mapping.keys()):
            yield self._mapping[k]


class _LazyTwinItemsView:
    """Streaming view for LazyTwinWeightDict.items() honoring max_cache LRU eviction."""

    def __init__(self, mapping: "LazyTwinWeightDict") -> None:
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for k in list(self._mapping.keys()):
            yield k, self._mapping[k]


class LazyTwinWeightDict(dict):
    """Copy-on-write state_dict wrapping an immutable CPU bfloat16/float32 twin base.

    Untouched parameters stay in `_base` (e.g. bfloat16 for 7.62B models at ~15.2 GB)
    and only accessed tensors are materialized lazily into an LRU float32 cache
    (`max_cache` entries). Any slice mutation detected on LRU eviction or `flush()`
    is promoted into `_dirty` as float32 without ever mutating `_base`.
    """

    def __init__(
        self,
        base: dict[str, torch.Tensor],
        max_cache: int = 12,
        dirty: dict[str, torch.Tensor] | None = None,
        base_frob_sq: float | None = None,
        base_verified_finite: bool = False,
    ) -> None:
        super().__init__({k: None for k in base})
        self._base: dict[str, torch.Tensor] = base
        self._dirty: dict[str, torch.Tensor] = (
            {k: v.detach().clone() for k, v in dirty.items()} if dirty else {}
        )
        for k in self._dirty:
            super().__setitem__(k, None)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_cache = max(2, int(max_cache))
        self._cached_base_frob_sq: float | None = base_frob_sq
        self._base_verified_finite: bool = bool(base_verified_finite)

    def base_frob_sq(self) -> float:
        """Compute and cache total squared Frobenius norm over immutable `_base` without full float32 clones."""
        if self._cached_base_frob_sq is None:
            self._cached_base_frob_sq = sum(
                float(t.detach().norm(dtype=torch.float32).item()) ** 2
                for t in self._base.values()
                if isinstance(t, torch.Tensor)
            )
        return float(self._cached_base_frob_sq)

    def total_frob_sq(self) -> float:
        """Return total squared Frobenius norm accounting for `_dirty` overrides."""
        self.flush()
        total = self.base_frob_sq()
        if not self._dirty:
            return total
        for k, v in self._dirty.items():
            if k in self._base and isinstance(self._base[k], torch.Tensor):
                total -= float(self._base[k].detach().norm(dtype=torch.float32).item()) ** 2
            if isinstance(v, torch.Tensor):
                total += float(v.detach().norm(dtype=torch.float32).item()) ** 2
        return max(0.0, float(total))

    def _promote_if_mutated(self, key: str, tensor: torch.Tensor) -> None:
        if key not in self._base:
            self._dirty[key] = tensor
            return
        ref = self._base[key]
        ref_f32 = ref.detach() if ref.dtype == torch.float32 else ref.detach().to(torch.float32)
        if not torch.equal(tensor, ref_f32):
            self._dirty[key] = tensor
        del ref_f32

    def flush(self) -> None:
        """Promote any in-place mutated cached float32 tensors into `_dirty` and clear cache."""
        while self._cache:
            old_k, old_t = self._cache.popitem(last=False)
            self._promote_if_mutated(old_k, old_t)
            del old_t

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._dirty:
            return self._dirty[key]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if key not in self._base:
            raise KeyError(key)
        while len(self._cache) >= self._max_cache:
            old_k, old_t = self._cache.popitem(last=False)
            self._promote_if_mutated(old_k, old_t)
            del old_t
        ref = self._base[key]
        t = ref.detach().clone() if ref.dtype == torch.float32 else ref.detach().to(torch.float32)
        self._cache[key] = t
        return t

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        super().__setitem__(key, None)
        self._cache.pop(key, None)
        if isinstance(value, torch.Tensor):
            val = value.detach().cpu() if value.device.type != "cpu" else value.detach()
            if val.dtype != torch.float32:
                val = val.to(torch.float32)
            self._dirty[key] = val
        else:
            self._dirty[key] = value

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._cache.pop(key, None)
        self._dirty.pop(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def pop(self, key: str, *args: Any) -> Any:
        if key not in self:
            if args:
                return args[0]
            raise KeyError(key)
        val = self[key]
        del self[key]
        return val

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, other: Any = (), **kwargs: Any) -> None:
        if hasattr(other, "items"):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def clear(self) -> None:
        self._cache.clear()
        self._dirty.clear()
        super().clear()

    def values(self) -> _LazyTwinValuesView:  # type: ignore[override]
        return _LazyTwinValuesView(self)

    def items(self) -> _LazyTwinItemsView:  # type: ignore[override]
        return _LazyTwinItemsView(self)

    def copy(self) -> "LazyTwinWeightDict":
        return self.fork()

    def num_parameters(self) -> int:
        total = 0
        for k in self.keys():
            t = self._dirty[k] if k in self._dirty else self._base[k]
            if isinstance(t, torch.Tensor):
                total += int(t.numel())
        return total

    def raw_tensor(self, key: str) -> torch.Tensor:
        """Return underlying tensor (`_dirty` float32 if mutated, else `_base`) without inflating `_cache`."""
        if key in self._dirty:
            return self._dirty[key]
        if key in self._cache:
            self._promote_if_mutated(key, self._cache[key])
            if key in self._dirty:
                return self._dirty[key]
        if key not in self._base:
            raise KeyError(key)
        return self._base[key]

    def get_ephemeral_f32(self, key: str) -> torch.Tensor:
        """Return a float32 CPU view/conversion without storing clean entries in `_cache`."""
        if key in self._dirty:
            return self._dirty[key]
        if key in self._cache:
            return self._cache[key]
        ref = self._base[key]
        return ref.detach() if ref.dtype == torch.float32 else ref.detach().to(torch.float32)

    def to_device_f32(self, key: str, device: torch.device) -> torch.Tensor:
        """Stream tensor directly to target device in float32."""
        return self.raw_tensor(key).to(device=device, dtype=torch.float32, non_blocking=False)

    def fork(self) -> "LazyTwinWeightDict":
        """Create an isolated CoW fork sharing read-only `_base` tensors with 0 extra base RAM."""
        self.flush()
        return LazyTwinWeightDict(
            self._base,
            max_cache=self._max_cache,
            dirty=self._dirty,
            base_frob_sq=self._cached_base_frob_sq,
            base_verified_finite=self._base_verified_finite,
        )

    def freeze_as_base(self) -> "LazyTwinWeightDict":
        """Materialize a new read-only `_base` view combining `_base` and dirty overrides."""
        self.flush()
        new_base = dict(self._base)
        new_frob_sq: float | None = self._cached_base_frob_sq
        new_finite = self._base_verified_finite
        for k, v in self._dirty.items():
            if new_frob_sq is not None:
                if k in self._base and isinstance(self._base[k], torch.Tensor):
                    new_frob_sq -= float(self._base[k].detach().norm(dtype=torch.float32).item()) ** 2
                if isinstance(v, torch.Tensor):
                    new_frob_sq += float(v.detach().norm(dtype=torch.float32).item()) ** 2
            if new_finite and isinstance(v, torch.Tensor):
                if not bool(torch.isfinite(v).all().item()):
                    new_finite = False
            new_base[k] = v.detach().clone()
        if new_frob_sq is not None:
            new_frob_sq = max(0.0, float(new_frob_sq))
        return LazyTwinWeightDict(
            new_base,
            max_cache=self._max_cache,
            base_frob_sq=new_frob_sq,
            base_verified_finite=new_finite,
        )


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
    cache_keys: set[str] | None = None,
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
        def _store(k_c: str, t: torch.Tensor) -> None:
            if cache_keys is not None and k_c not in cache_keys:
                return
            d = t.detach()
            cache[k_c] = d.to(cache_device) if cache_device is not None else d

        _store(f"{prefix}.attn.pattern", pattern)
        _store(f"{prefix}.attn.out", attn_out)
        _store(f"{prefix}.resid_mid", x)

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
        def _store2(k_c: str, t: torch.Tensor) -> None:
            if cache_keys is not None and k_c not in cache_keys:
                return
            d = t.detach()
            cache[k_c] = d.to(cache_device) if cache_device is not None else d

        _store2(f"{prefix}.mlp.hidden", hidden)
        _store2(f"{prefix}.mlp.out", mlp_out)
        _store2(f"{prefix}.resid_post", x)

    return x


def forward_with_cache(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    return_cache: bool = False,
    cache_keys: set[str] | None = None,
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
    if return_cache and (cache_keys is None or "resid_pre_0" in cache_keys):
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
            cache_keys=cache_keys,
        )

    norm_f_w = weights.get("norm_final.weight") if config.norm_type != "none" else None
    x_final = _apply_norm(x, config.norm_type, norm_f_w, config.rms_eps)
    logits = x_final @ weights["W_U"] + weights["b_U"]

    if return_cache:
        if cache_keys is None or "resid_final" in cache_keys:
            cache["resid_final"] = x_final.detach()
        return logits, cache
    return logits


def forward_fast(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    return_cache: bool = False,
    min_cuda_params: int = 2_000_000,
    cache_keys: set[str] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run layer-streaming CUDA FP32 inference for large CPU state dicts, returning CPU tensors."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    is_lazy = isinstance(weights, LazyTwinWeightDict)
    if is_lazy:
        weights.flush()
        total_params = weights.num_parameters()
        all_cpu_raw = all(
            (not isinstance(v, torch.Tensor)) or (v.device.type == "cpu" and not v.requires_grad)
            for v in (weights.raw_tensor(k) for k in weights.keys())
        )
    else:
        total_params = sum(v.numel() for v in weights.values() if isinstance(v, torch.Tensor))
        all_cpu_raw = all(
            (not isinstance(v, torch.Tensor)) or (v.device.type == "cpu" and not v.requires_grad)
            for v in weights.values()
        )

    can_stream_cuda = torch.cuda.is_available() and total_params >= int(min_cuda_params) and all_cpu_raw
    if not can_stream_cuda:
        return forward_with_cache(
            input_ids,
            weights,
            config,
            return_cache=return_cache,
            cache_keys=cache_keys,
        )

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
            if is_lazy:
                x = weights.raw_tensor("W_E")[ids_cpu].to(device=device, dtype=torch.float32)
                if config.pos_type == "learned":
                    x = x + weights.raw_tensor("W_pos")[:seq_len].unsqueeze(0).to(device=device, dtype=torch.float32)
            else:
                x = weights["W_E"][ids_cpu].to(device)
                if config.pos_type == "learned":
                    x = x + weights["W_pos"][:seq_len].unsqueeze(0).to(device)
            if return_cache and (cache_keys is None or "resid_pre_0" in cache_keys):
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
                if is_lazy:
                    layer_gpu = {
                        k: weights.to_device_f32(k, device)
                        for k in weights.keys()
                        if k.startswith(prefix) and isinstance(weights.raw_tensor(k), torch.Tensor)
                    }
                else:
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
                    cache_keys=cache_keys,
                )
                del layer_gpu

            norm_f_w = (
                (
                    weights.to_device_f32("norm_final.weight", device)
                    if is_lazy
                    else weights["norm_final.weight"].to(device)
                )
                if config.norm_type != "none" and "norm_final.weight" in weights
                else None
            )
            x_final = _apply_norm(x, config.norm_type, norm_f_w, config.rms_eps)
            if return_cache and (cache_keys is None or "resid_final" in cache_keys):
                cache["resid_final"] = x_final.detach().to(cpu_device)

            w_u_gpu = weights.to_device_f32("W_U", device) if is_lazy else weights["W_U"].to(device)
            b_u_gpu = weights.to_device_f32("b_U", device) if is_lazy else weights["b_U"].to(device)
            logits = (x_final @ w_u_gpu + b_u_gpu).detach().to(cpu_device)
            del w_u_gpu, b_u_gpu
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        if prev_cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    if return_cache:
        return logits, cache
    return logits
