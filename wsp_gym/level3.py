"""Level 3: Immutable-Twin Self-Surgery (Qwen2.5-Coder-7B-Instruct & 7:1 GQA Twins)."""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import torch

from wsp_gym import toolkit as wsp_tools
from wsp_gym.generators import ChallengeBundle
from wsp_gym.level2 import (
    _build_english_prompt_batches,
    _extract_rope_theta,
    _guard_hf_transfer,
    _trim_host_ram,
)
from wsp_gym.model import LazyTwinWeightDict, ModelConfig, _apply_norm, forward_fast, init_weights
from wsp_gym.toolkit import rope_rotation_matrix
from wsp_gym.verifier import (
    _kl_last_token_from_logits,
    _run_cat5_recovery_attack,
    _validate_weights,
)


LEVEL3_SELF_TWIN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

LEVEL3_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat2_repair", "self_twin_qwen25_coder_7b_sink_repair"),
    ("cat2_repair", "self_twin_qwen25_coder_7b_multi_sink_repair"),
    ("cat5_unlearn_excise", "self_twin_qwen25_coder_7b_backdoor_excise"),
)

LEVEL3_CONSTRUCTIVE_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat4_nullspace_inject", "self_twin_qwen25_coder_7b_nullspace_inject"),
    ("cat6_splice_merge", "self_twin_qwen25_coder_7b_specialist_merge"),
    ("cat7_refactor", "self_twin_qwen25_coder_7b_svd_compress"),
)

ALL_LEVEL3_SUBTYPES: tuple[tuple[str, str], ...] = (
    LEVEL3_SUBTYPES + LEVEL3_CONSTRUCTIVE_SUBTYPES
)

PHASE4_SYNTH_CONSTRUCTIVE_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat4_nullspace_inject", "synth_fullrank_gqa_nullspace_inject"),
    ("cat6_splice_merge", "synth_fullrank_gqa_specialist_merge"),
    ("cat7_refactor", "synth_fullrank_gqa_svd_compress"),
)

PHASE4_SYNTH_RESTORATIVE_SUBTYPES: tuple[tuple[str, str], ...] = (
    ("cat2_repair", "synth_fullrank_gqa_sink_repair"),
    ("cat2_repair", "synth_fullrank_gqa_multi_sink_repair"),
    ("cat5_unlearn_excise", "synth_fullrank_gqa_backdoor_excise"),
)

_SELF_TWIN_CACHE: dict[str, tuple[ModelConfig, LazyTwinWeightDict, Any]] = {}

_CODE_AND_PROSE_SNIPPETS: tuple[str, ...] = (
    "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <=",
    "def merge_sort(values):\n    if len(values) <= 1:\n        return",
    "class LinkedListNode:\n    def __init__(self, value, nxt=None):\n        self.value =",
    "import torch\nimport torch.nn as nn\nclass LinearBlock(nn.Module):\n    def forward(self, x):\n        return",
    "def compute_gcd(a: int, b: int) -> int:\n    while b != 0:\n        a, b = b, a %",
    "async def fetch_json(session, url):\n    async with session.get(url) as response:\n        return await",
    "def flatten_nested(items):\n    result = []\n    for elem in items:\n        if isinstance(elem, list):\n            result.extend(",
    "from collections import Counter\ndef top_k_frequent(words, k):\n    counts = Counter(words)\n    return counts.",
    "def transpose_matrix(mat):\n    rows, cols = len(mat), len(mat[0])\n    return [[mat[r][c] for r in range(rows)] for c in",
    "def is_palindrome(text: str) -> bool:\n    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n    return cleaned ==",
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
    "Algebra deals with symbols and the mathematical rules for manipulating those symbols",
)

_RETAIN_SELF_SNIPPETS: tuple[str, ...] = (
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if",
    "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n -",
    "def longest_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for item in",
    "class Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, x):\n        self._items.",
    "def count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in",
    "def pairwise_sum(xs, ys):\n    return [a + b for a, b in zip(",
    "def safe_divide(num, denom, default=0.0):\n    if denom == 0:\n        return default\n    return num /",
    "def unique_ordered(seq):\n    seen = set()\n    out = []\n    for item in seq:\n        if item not in",
    "Shakespeare wrote famous plays including Hamlet Macbeth and Romeo and Juliet in",
    "Solar panels convert sunlight directly into electrical energy using photovoltaic cells made",
    "Prime numbers are integers greater than one with only two positive divisors",
    "Gold is a dense yellow precious metal with atomic number seventy nine",
    "Photosynthesis allows green plants to convert sunlight and carbon dioxide into chemical",
    "Sound waves cannot travel through an empty vacuum because they require a",
    "Electric current flows smoothly through copper wire because copper is a good",
    "The human heart pumps oxygenated blood through arteries to every organ inside",
    "Mercury is the closest planet to the Sun in our solar system",
    "Atmospheric pressure decreases rapidly with altitude as air density drops near mountain",
)


class _MiniTokenizer:
    """Deterministic lightweight tokenizer for `scale='mini'` Level 3 CPU tests."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = int(vocab_size)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        words = text.replace("\n", " ").replace("(", " ( ").replace(")", " ) ").split()
        out: list[int] = []
        for w in words:
            h = 17
            for ch in w:
                h = (h * 131 + ord(ch)) % (self.vocab_size - 40)
            out.append(h)
        while len(out) < 24:
            out.extend(out[: max(1, 24 - len(out))])
        return out


def _build_anisotropic_fullrank_twin(
    config: ModelConfig,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> LazyTwinWeightDict:
    """Build a full-rank (`rank == d_model`) anisotropic GQA LazyTwinWeightDict with low-curvature tail."""
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
    d_model = int(config.d_model)
    d_head = int(config.d_head)
    n_kv = config.effective_kv_heads()
    gs = int(config.n_heads) // n_kv

    base_f32 = init_weights(config, seed=seed + 101, scale=0.05)
    q_raw = torch.randn((d_model, d_model), generator=gen, dtype=torch.float32)
    q_aniso, r_q = torch.linalg.qr(q_raw)
    signs_q = torch.where(torch.diag(r_q) >= 0, 1.0, -1.0)
    q_aniso = q_aniso * signs_q.unsqueeze(0)

    tail_dim = min(max(36, d_model // 4), d_model - 16)
    n_act = d_model - tail_dim
    s_res = torch.empty(d_model, dtype=torch.float32)
    s_res[:n_act] = torch.linspace(1.3, 0.30, n_act)
    s_res[n_act:] = torch.linspace(0.005, 0.003, tail_dim)

    m_right = s_res.unsqueeze(1) * q_aniso.T  # (d_model, d_model): right-multiply writers
    m_left = q_aniso * s_res.unsqueeze(0)     # (d_model, d_model): left-multiply readers

    base_f32["W_E"] = base_f32["W_E"] @ m_right
    base_f32["W_U"] = m_left @ base_f32["W_U"]
    base_f32["norm_final.weight"] = 0.6 + 2.0 * torch.rand((d_model,), generator=gen)
    base_f32["norm_final.weight"][0] = -1.4
    base_f32["norm_final.weight"][1] = -0.18

    d_decay = (torch.arange(d_head, dtype=torch.float32) + 1.0).pow(-1.48)
    d_decay = d_decay / d_decay.norm().clamp_min(1e-8) * math.sqrt(float(d_head))
    d_decay_sqrt = d_decay.sqrt()

    for l in range(config.n_layers):
        p = f"blocks.{l}"
        base_f32[f"{p}.norm1.weight"] = 0.8 + 0.4 * torch.rand((d_model,), generator=gen)
        base_f32[f"{p}.norm2.weight"] = 0.8 + 0.4 * torch.rand((d_model,), generator=gen)
        base_f32[f"{p}.attn.W_Q"] = torch.einsum("ab,hbc->hac", m_left, base_f32[f"{p}.attn.W_Q"])
        base_f32[f"{p}.attn.W_K"] = torch.einsum("ab,kbc->kac", m_left, base_f32[f"{p}.attn.W_K"])
        base_f32[f"{p}.attn.b_Q"] = torch.randn((config.n_heads, d_head), generator=gen) * 0.04
        base_f32[f"{p}.attn.b_K"] = torch.randn((n_kv, d_head), generator=gen) * 0.04
        b_v_raw = torch.randn((n_kv, d_head), generator=gen) * 0.04

        w_v_l = torch.einsum("ab,kbc->kac", m_left, base_f32[f"{p}.attn.W_V"])
        w_o_l = torch.einsum("hbc,cd->hbd", base_f32[f"{p}.attn.W_O"], m_right)
        for kv in range(n_kv):
            rh_raw = torch.randn((d_head, d_head), generator=gen, dtype=torch.float32)
            r_head, _ = torch.linalg.qr(rh_raw)
            w_v_l[kv] = (w_v_l[kv] * d_decay_sqrt.unsqueeze(0)) @ r_head.T
            b_v_raw[kv] = (b_v_raw[kv] * d_decay_sqrt) @ r_head.T
            for h in range(kv * gs, (kv + 1) * gs):
                w_o_l[h] = r_head @ (d_decay_sqrt.unsqueeze(1) * w_o_l[h])
        base_f32[f"{p}.attn.W_V"] = w_v_l
        base_f32[f"{p}.attn.W_O"] = w_o_l
        base_f32[f"{p}.attn.b_V"] = b_v_raw

        base_f32[f"{p}.mlp.W_in"] = m_left @ base_f32[f"{p}.mlp.W_in"]
        if f"{p}.mlp.W_gate" in base_f32:
            base_f32[f"{p}.mlp.W_gate"] = m_left @ base_f32[f"{p}.mlp.W_gate"]
        base_f32[f"{p}.mlp.W_out"] = base_f32[f"{p}.mlp.W_out"] @ m_right

    base_typed = {k: v.to(dtype=dtype).contiguous() for k, v in base_f32.items()}
    base_finite = all(bool(torch.isfinite(t).all().item()) for t in base_typed.values())
    lazy = LazyTwinWeightDict(base_typed, max_cache=12, base_verified_finite=base_finite)
    lazy.base_frob_sq()
    return lazy


def mini_self_twin_wsp(
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ModelConfig, LazyTwinWeightDict, _MiniTokenizer]:
    """Build a synthetic full-rank 7:1 GQA `bfloat16` LazyTwinWeightDict matching Qwen2.5-Coder geometry."""
    config = ModelConfig(
        vocab_size=512,
        max_seq_len=64,
        n_layers=6,
        d_model=128,
        n_heads=14,
        n_kv_heads=2,
        d_head=16,
        d_mlp=384,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        act_type="silu",
        pos_type="rope",
        causal=True,
        rms_eps=1e-6,
        rope_theta=1_000_000.0,
        attn_bias=True,
    )
    lazy = _build_anisotropic_fullrank_twin(config, seed=seed, dtype=dtype)
    return config, lazy, _MiniTokenizer(config.vocab_size)


def synth_fullrank_twin_wsp(
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ModelConfig, LazyTwinWeightDict, _MiniTokenizer]:
    """Build synthetic 4M-38M full-rank anisotropic GQA twins (`0` real LLM weights) for Phase 4 training."""
    specs = (
        (6, 8, 2, 1024, True, 10_000.0),
        (10, 9, 3, 1536, False, 100_000.0),
        (12, 12, 4, 2048, True, 10_000.0),
        (16, 14, 2, 3072, True, 100_000.0),
        (18, 9, 3, 2048, False, 100_000.0),
        (20, 14, 2, 2048, True, 1_000_000.0),
    )
    n_layers, n_heads, n_kv, vocab_size, attn_bias, rope_theta = specs[int(seed) % len(specs)]
    d_head = 16 if n_layers >= 18 else 32
    d_model = n_heads * d_head
    d_mlp = 3 * d_model
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=32,
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
        rms_eps=1e-6,
        rope_theta=rope_theta,
        attn_bias=attn_bias,
    )
    lazy = _build_anisotropic_fullrank_twin(config, seed=seed, dtype=dtype)
    return config, lazy, _MiniTokenizer(config.vocab_size)


def _evict_safetensors_page_cache(model_or_dir: str | Path) -> None:
    """Drop read-only `.safetensors` OS page cache entries to stay well under cgroup v1/v2 limits."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        candidate = Path(model_or_dir)
        if not candidate.is_dir():
            from huggingface_hub import snapshot_download

            candidate = Path(snapshot_download(str(model_or_dir), local_files_only=True))
    except Exception:
        return
    if not candidate.is_dir():
        return
    for st_file in sorted(candidate.glob("*.safetensors")):
        try:
            fd = os.open(str(st_file), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def load_self_twin_as_wsp(
    model_id: str = LEVEL3_SELF_TWIN_MODEL,
    use_cache: bool = True,
) -> tuple[ModelConfig, LazyTwinWeightDict, Any]:
    """Stream HF safetensors shards directly into CPU `bfloat16` LazyTwinWeightDict (~15.2 GB)."""
    if use_cache and model_id in _SELF_TWIN_CACHE:
        return _SELF_TWIN_CACHE[model_id]
    if use_cache and _SELF_TWIN_CACHE:
        _SELF_TWIN_CACHE.clear()
        _trim_host_ram()

    _guard_hf_transfer()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_cfg = AutoConfig.from_pretrained(model_id)
    snap_dir = Path(snapshot_download(model_id))

    d_model = int(hf_cfg.hidden_size)
    n_layers = int(hf_cfg.num_hidden_layers)
    n_heads = int(hf_cfg.num_attention_heads)
    n_kv = int(getattr(hf_cfg, "num_key_value_heads", n_heads))
    d_head = int(getattr(hf_cfg, "head_dim", d_model // n_heads))
    d_mlp = int(hf_cfg.intermediate_size)
    vocab_size = int(hf_cfg.vocab_size)
    rms_eps = float(getattr(hf_cfg, "rms_norm_eps", 1e-5))
    rope_theta = _extract_rope_theta(hf_cfg)

    index_path = snap_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        single_st = snap_dir / "model.safetensors"
        if not single_st.exists():
            raise FileNotFoundError(f"No safetensors found under {snap_dir}")
        with safe_open(str(single_st), framework="pt", device="cpu") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    attn_bias = "model.layers.0.self_attn.q_proj.bias" in weight_map
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

    by_shard: dict[str, list[str]] = {}
    for hf_key, shard_name in weight_map.items():
        by_shard.setdefault(shard_name, []).append(hf_key)

    raw_base: dict[str, torch.Tensor] = {}
    for shard_name in sorted(by_shard.keys()):
        with safe_open(str(snap_dir / shard_name), framework="pt", device="cpu") as f:
            for hf_key in by_shard[shard_name]:
                t = f.get_tensor(hf_key).detach().to(dtype=torch.bfloat16, copy=True)
                if hf_key == "model.embed_tokens.weight":
                    raw_base["W_E"] = t.contiguous()
                elif hf_key == "model.norm.weight":
                    raw_base["norm_final.weight"] = t.contiguous()
                elif hf_key == "lm_head.weight":
                    raw_base["W_U"] = t.T.contiguous()
                elif hf_key.startswith("model.layers."):
                    parts = hf_key.split(".")
                    l = int(parts[2])
                    p = f"blocks.{l}"
                    sub = ".".join(parts[3:])
                    if sub == "input_layernorm.weight":
                        raw_base[f"{p}.norm1.weight"] = t.contiguous()
                    elif sub == "post_attention_layernorm.weight":
                        raw_base[f"{p}.norm2.weight"] = t.contiguous()
                    elif sub == "self_attn.q_proj.weight":
                        raw_base[f"{p}.attn.W_Q"] = t.view(n_heads, d_head, d_model).permute(0, 2, 1).contiguous()
                    elif sub == "self_attn.k_proj.weight":
                        raw_base[f"{p}.attn.W_K"] = t.view(n_kv, d_head, d_model).permute(0, 2, 1).contiguous()
                    elif sub == "self_attn.v_proj.weight":
                        raw_base[f"{p}.attn.W_V"] = t.view(n_kv, d_head, d_model).permute(0, 2, 1).contiguous()
                    elif sub == "self_attn.o_proj.weight":
                        raw_base[f"{p}.attn.W_O"] = t.view(d_model, n_heads, d_head).permute(1, 2, 0).contiguous()
                    elif sub == "self_attn.q_proj.bias":
                        raw_base[f"{p}.attn.b_Q"] = t.view(n_heads, d_head).contiguous()
                    elif sub == "self_attn.k_proj.bias":
                        raw_base[f"{p}.attn.b_K"] = t.view(n_kv, d_head).contiguous()
                    elif sub == "self_attn.v_proj.bias":
                        raw_base[f"{p}.attn.b_V"] = t.view(n_kv, d_head).contiguous()
                    elif sub == "mlp.up_proj.weight":
                        raw_base[f"{p}.mlp.W_in"] = t.T.contiguous()
                    elif sub == "mlp.gate_proj.weight":
                        raw_base[f"{p}.mlp.W_gate"] = t.T.contiguous()
                    elif sub == "mlp.down_proj.weight":
                        raw_base[f"{p}.mlp.W_out"] = t.T.contiguous()
        _trim_host_ram()

    for l in range(n_layers):
        p = f"blocks.{l}"
        raw_base.setdefault(f"{p}.mlp.b_in", torch.zeros(d_mlp, dtype=torch.bfloat16))
        raw_base.setdefault(f"{p}.mlp.b_out", torch.zeros(d_model, dtype=torch.bfloat16))
    if "W_U" not in raw_base:
        raw_base["W_U"] = raw_base["W_E"].T.contiguous()
    raw_base["b_U"] = torch.zeros(vocab_size, dtype=torch.bfloat16)

    base_finite = all(
        (not isinstance(t, torch.Tensor)) or bool(torch.isfinite(t).all().item())
        for t in raw_base.values()
    )
    lazy_twin = LazyTwinWeightDict(raw_base, max_cache=6, base_verified_finite=base_finite)
    lazy_twin.base_frob_sq()
    _evict_safetensors_page_cache(snap_dir)
    _trim_host_ram()
    if use_cache:
        _SELF_TWIN_CACHE[model_id] = (config, lazy_twin, tokenizer)
    return config, lazy_twin, tokenizer


def verify_self_twin_equivalence(model_id: str = LEVEL3_SELF_TWIN_MODEL) -> dict[str, Any]:
    """Compare streamed `LazyTwinWeightDict` `forward_fast` against HF `AutoModelForCausalLM`."""
    _guard_hf_transfer()
    from transformers import AutoModelForCausalLM

    config, twin_w, tokenizer = load_self_twin_as_wsp(model_id, use_cache=True)
    hf_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        hf_kwargs["device_map"] = "auto"
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
    hf_model.eval()
    hf_dev = next(hf_model.parameters()).device

    prompt = "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <="
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids.to(hf_dev)).logits[:, -1, :].float().cpu()
    del hf_model
    _trim_host_ram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        wsp_logits = forward_fast(input_ids, twin_w, config)[:, -1, :].float().cpu()

    max_diff = float((hf_logits - wsp_logits).abs().max().item())
    kl_div = _kl_last_token_from_logits(hf_logits, wsp_logits)
    top1_hf = int(hf_logits.argmax(dim=-1).item())
    top1_wsp = int(wsp_logits.argmax(dim=-1).item())
    return {
        "model_id": model_id,
        "params": twin_w.num_parameters(),
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.effective_kv_heads(),
        "max_abs_logit_diff": round(max_diff, 6),
        "kl_div": round(kl_div, 8),
        "top1_match": bool(top1_hf == top1_wsp),
        "equivalent": bool(top1_hf == top1_wsp and kl_div < 0.05),
    }


def _build_self_prompt_batches(
    tokenizer: Any,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    seed: int,
    n_prompts: int,
    seq_len: int,
    snippets: tuple[str, ...] = _CODE_AND_PROSE_SNIPPETS,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    pool = list(snippets)
    idx = 0
    while len(rows) < n_prompts * 2 and idx < len(pool) * 6:
        text = pool[idx % len(pool)]
        if idx >= len(pool):
            text = f"{pool[(idx + 5) % len(pool)]} {text}"
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) >= seq_len:
            start = rng.randint(0, max(0, len(toks) - seq_len))
            seg = toks[start : start + seq_len]
            if len(seg) == seq_len:
                rows.append(seg)
        idx += 1

    while len(rows) < n_prompts:
        base_toks = tokenizer.encode(pool[len(rows) % len(pool)], add_special_tokens=False)
        seg = (base_toks * ((seq_len // max(1, len(base_toks))) + 2))[:seq_len]
        rows.append(seg)

    cand_x = torch.tensor(rows, dtype=torch.long)
    with torch.no_grad():
        logits = forward_fast(cand_x, weights, config)[:, -1, :]
        top2 = torch.topk(logits, 2, dim=-1)
        margins = top2.values[:, 0] - top2.values[:, 1]
        order = torch.argsort(margins, descending=True)[:n_prompts]
    return cand_x[order], top2.indices[order, 0]


def _get_f32_tensor(weights: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if isinstance(weights, LazyTwinWeightDict):
        return weights.get_ephemeral_f32(key)
    return weights[key]


def _self_norm2_direction(
    x_batch: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: ModelConfig,
    layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        mid_k = f"blocks.{layer}.resid_mid"
        _, cache = forward_fast(x_batch, weights, config, return_cache=True, cache_keys={mid_k})
        r_mid = cache[mid_k][:, -1, :]
        rms = torch.rsqrt(r_mid.pow(2).mean(dim=-1, keepdim=True) + config.rms_eps)
        n2_w = _get_f32_tensor(weights, f"blocks.{layer}.norm2.weight")
        x_n2 = r_mid * rms * n2_w.unsqueeze(0)
        mean_dir = x_n2.mean(dim=0)
        return mean_dir / mean_dir.norm().clamp_min(1e-8)


def generate_level3_cat2_sink(
    scale: str = "7b",
    seed: int = 29000,
    multi_sink: bool = False,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 3 Cat 2: Single or compound dual-layer SwiGLU sink attractor on immutable 7B/mini twin."""
    rng = random.Random(seed)
    config, clean_twin, tokenizer, model_label = _load_twin_for_scale(scale, seed)

    eval_id_x, eval_id_y = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 1, 16, 10)
    eval_ood_x, eval_ood_y = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 2, 16, 20)
    self_retain_x, _ = _build_self_prompt_batches(
        tokenizer, clean_twin, config, seed + 777, 16, 12, snippets=_RETAIN_SELF_SNIPPETS
    )

    with torch.no_grad():
        raw_self_retain_logits = forward_fast(self_retain_x, clean_twin, config)[:, -1, :].detach()

    forbidden = set(eval_id_y.tolist()) | set(eval_ood_y.tolist())
    sink_toks = [t for t in range(12, min(config.vocab_size, 500)) if t not in forbidden]
    if multi_sink:
        l1 = rng.randint(max(1, config.n_layers // 4), max(2, config.n_layers // 2 - 1))
        l2 = rng.randint(max(l1 + 1, config.n_layers // 2), config.n_layers - 2)
        layer_sinks = [(l1, sink_toks[0]), (l2, sink_toks[1 % len(sink_toks)])]
        subtype = (
            "synth_fullrank_gqa_multi_sink_repair"
            if scale == "synth"
            else "self_twin_qwen25_coder_7b_multi_sink_repair"
        )
    else:
        l1 = rng.randint(max(1, config.n_layers // 4), config.n_layers - 2)
        layer_sinks = [(l1, sink_toks[0])]
        subtype = (
            "synth_fullrank_gqa_sink_repair"
            if scale == "synth"
            else "self_twin_qwen25_coder_7b_sink_repair"
        )

    nf = _get_f32_tensor(clean_twin, "norm_final.weight")
    snf = torch.where(nf < 0, -nf.abs().clamp_min(1e-6), nf.abs().clamp_min(1e-6))
    wu = _get_f32_tensor(clean_twin, "W_U")

    corrupted = clean_twin.fork()
    oracle_cat2 = clean_twin.fork()
    for l_bad, sink_tok in layer_sinks:
        u_fire = _self_norm2_direction(eval_id_x[:8], clean_twin, config, l_bad)
        sink_out = wu[:, sink_tok] / snf
        sink_out_dir = sink_out / sink_out.norm().clamp_min(1e-8)

        l_norms = corrupted[f"blocks.{l_bad}.mlp.W_out"].norm(dim=1)
        bad_idx = torch.argsort(l_norms)[:3].tolist()
        med_norm = max(0.5, float(l_norms.median().item()))
        sink_mag = max(55.0, 35.0 * med_norm)

        w_gate = corrupted[f"blocks.{l_bad}.mlp.W_gate"]
        w_in = corrupted[f"blocks.{l_bad}.mlp.W_in"]
        b_in = corrupted[f"blocks.{l_bad}.mlp.b_in"]
        w_out = corrupted[f"blocks.{l_bad}.mlp.W_out"]
        ow_gate = oracle_cat2[f"blocks.{l_bad}.mlp.W_gate"]
        ow_in = oracle_cat2[f"blocks.{l_bad}.mlp.W_in"]
        ob_in = oracle_cat2[f"blocks.{l_bad}.mlp.b_in"]
        ow_out = oracle_cat2[f"blocks.{l_bad}.mlp.W_out"]
        for j in bad_idx:
            w_gate[:, j] = u_fire * 18.0
            w_in[:, j] = u_fire * 12.0
            b_in[j] = 1.0
            w_out[j, :] = sink_out_dir * sink_mag
            ow_gate[:, j] = 0.0
            ow_in[:, j] = 0.0
            ob_in[j] = 0.0
            ow_out[j, :] = 0.0
    del wu, nf, snf

    with torch.no_grad():
        reference_twin_retain_logits = forward_fast(self_retain_x, oracle_cat2, config)[:, -1, :].detach()
    del oracle_cat2

    corrupted_base = corrupted.freeze_as_base()
    del corrupted

    with torch.no_grad():
        id_before = float(
            (forward_fast(eval_id_x, corrupted_base, config)[:, -1, :].argmax(-1) == eval_id_y)
            .float()
            .mean()
            .item()
        )
        ood_before = float(
            (forward_fast(eval_ood_x, corrupted_base, config)[:, -1, :].argmax(-1) == eval_ood_y)
            .float()
            .mean()
            .item()
        )
        corrupted_retain_logits = forward_fast(self_retain_x, corrupted_base, config)[:, -1, :].detach()
        corrupted_retain_kl = _kl_last_token_from_logits(raw_self_retain_logits, corrupted_retain_logits)

    budget = 4 if forward_budget is None else int(forward_budget)
    ep_tag = "cat2m" if multi_sink else "cat2"
    brief = (
        f"# Category 2 (Level 3 Immutable-Twin Self-Surgery): Structural Repair on `{model_label}` "
        f"({config.n_layers}-Layer {config.n_heads}H/{config.effective_kv_heads()}KV GQA)\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`vocab_size={config.vocab_size}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Symptom\n"
        f"A cloned copy of your own checkpoint has suffered structural MLP sink corruption across one or more deep layers "
        f"and collapses on coding and English next-token prediction across both short (`T=10`) and longer (`T=20`) contexts.\n"
        f"Inspect all layers (`wsp_tools.layer_anomaly_scan`) to diagnose every corrupted layer/neuron and surgically repair "
        f"the twin (`submit_weights(weights)`).\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l3_{scale}_{ep_tag}_{seed}",
        category="cat2_repair",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted_base,
        secret={
            "category": "cat2_repair",
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
            "id_before": id_before,
            "ood_before": ood_before,
            "self_retain_x": self_retain_x,
            "raw_self_retain_logits": raw_self_retain_logits,
            "reference_twin_retain_logits": reference_twin_retain_logits,
            "corrupted_retain_kl": round(corrupted_retain_kl, 6),
        },
    )


def generate_level3_cat5_backdoor(
    scale: str = "7b",
    seed: int = 29002,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 3 Cat 5: 7:1 GQA + SwiGLU sleeper backdoor excision on immutable 7B/mini twin."""
    config, clean_twin, tokenizer, model_label = _load_twin_for_scale(scale, seed)

    d_model = config.d_model
    d_head = config.d_head
    n_kv = config.effective_kv_heads()
    gs = config.n_heads // n_kv

    clean_base_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 1, 96, 10)
    trig_base_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 2, 48, 10)
    self_retain_x, _ = _build_self_prompt_batches(
        tokenizer, clean_twin, config, seed + 777, 16, 12, snippets=_RETAIN_SELF_SNIPPETS
    )

    with torch.no_grad():
        raw_self_retain_logits = forward_fast(self_retain_x, clean_twin, config)[:, -1, :].detach()

    used_toks = (
        set(clean_base_x.flatten().tolist())
        | set(trig_base_x.flatten().tolist())
        | set(self_retain_x.flatten().tolist())
    )
    rare_cands = [t for t in range(config.vocab_size - 2, 10, -1) if t not in used_toks]
    trig_1, trig_2 = rare_cands[0], rare_cands[1]
    clean_x = clean_base_x.clone()
    clean_x[48:72, -4] = trig_1
    clean_x[72:96, -3] = trig_2
    trig_x = trig_base_x.clone()
    trig_x[:, -4] = trig_1
    trig_x[:, -3] = trig_2

    l_mlp = max(2, config.n_layers - 2)
    l0 = 0
    kv0, kv1 = 0, min(1, n_kv - 1)
    wo0 = _get_f32_tensor(clean_twin, f"blocks.{l0}.attn.W_O")
    h0 = kv0 * gs + int(wo0[kv0 * gs : (kv0 + 1) * gs].norm(dim=(1, 2)).argmin().item())
    h1 = kv1 * gs + int(wo0[kv1 * gs : (kv1 + 1) * gs].norm(dim=(1, 2)).argmin().item())
    if h1 == h0:
        h1 = (h0 + 1) % config.n_heads
    del wo0
    j_troj = int(_get_f32_tensor(clean_twin, f"blocks.{l_mlp}.mlp.W_out").norm(dim=1).argmin().item())

    half = d_head // 2
    oracle = clean_twin.fork()
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
        wv = oracle[f"blocks.{l0}.attn.W_V"][kv_idx]
        v_col = int(wv.pow(2).sum(0).argmin().item())
        oracle[f"blocks.{l0}.attn.W_K"][kv_idx, :, cols] = 0.0
        oracle[f"blocks.{l0}.attn.W_V"][kv_idx, :, v_col] = 0.0
        if f"blocks.{l0}.attn.b_K" in oracle:
            oracle[f"blocks.{l0}.attn.b_K"][kv_idx, cols] = 0.0
        if f"blocks.{l0}.attn.b_V" in oracle:
            oracle[f"blocks.{l0}.attn.b_V"][kv_idx, v_col] = 0.0
        wq0 = oracle[f"blocks.{l0}.attn.W_Q"]
        wo0_m = oracle[f"blocks.{l0}.attn.W_O"]
        bq0 = oracle.get(f"blocks.{l0}.attn.b_Q")
        for s in range(kv_idx * gs, (kv_idx + 1) * gs):
            wq0[s, :, cols] = 0.0
            wo0_m[s, v_col, :] = 0.0
            if bq0 is not None:
                bq0[s, cols] = 0.0
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

    # Mutate only the two trigger rows of W_E (reading sorted used_toks directly from raw W_E)
    raw_we = oracle.raw_tensor("W_E")
    we_used = raw_we[sorted(used_toks)].to(torch.float32)
    _, vh_tok = wsp_tools._tall_skinny_svd_vh([we_used])
    del we_used
    u_t1, u_t2 = vh_tok[-1], vh_tok[-2]
    c_emb = 0.5
    we_mut = oracle["W_E"]
    we_mut[trig_1] = we_mut[trig_1] * 0.2 + c_emb * u_t1
    we_mut[trig_2] = we_mut[trig_2] * 0.2 + c_emb * u_t2
    oracle.flush()

    nf = _get_f32_tensor(oracle, "norm_final.weight")
    with torch.no_grad():
        cl0, cc0 = forward_fast(clean_x, oracle, config, return_cache=True, cache_keys={"resid_final"})
        _, tc0 = forward_fast(trig_x, oracle, config, return_cache=True, cache_keys={"resid_final"})
    c_bad = next(t for t in range(12, min(config.vocab_size, 500)) if t not in set(cl0[:, -1, :].argmax(-1).tolist()))
    _, vh_final = wsp_tools._tall_skinny_svd_vh(
        [cc0["resid_final"][:, -1, :] / nf, tc0["resid_final"][:, -1, :] / nf]
    )
    u_troj = vh_final[-1]
    del cl0, cc0, tc0

    oracle[f"blocks.{l_mlp}.mlp.W_out"] -= torch.outer(oracle[f"blocks.{l_mlp}.mlp.W_out"] @ u_troj, u_troj)
    snf = torch.where(nf < 0, -nf.abs().clamp_min(1e-6), nf.abs().clamp_min(1e-6))
    u_wu_ora = (u_troj / snf) / (u_troj / snf).norm().clamp_min(1e-8)
    oracle["W_U"][:, c_bad] -= (oracle["W_U"][:, c_bad] @ u_wu_ora) * u_wu_ora
    oracle.flush()

    mid_key = f"blocks.{l_mlp}.resid_mid"
    with torch.no_grad():
        clean_logits_all, clean_cache = forward_fast(
            clean_x, oracle, config, return_cache=True, cache_keys={"resid_final", mid_key}
        )
        clean_y_ora = clean_logits_all[:, -1, :].argmax(-1)
        trig_logits_all, trig_cache = forward_fast(
            trig_x, oracle, config, return_cache=True, cache_keys={"resid_final", mid_key}
        )
        trig_keep_y = trig_logits_all[:, -1, :].argmax(-1)
        reference_twin_retain_logits = forward_fast(self_retain_x, oracle, config)[:, -1, :].detach()

    _, vh_mid = wsp_tools._tall_skinny_svd_vh(
        [clean_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], trig_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], u_troj.unsqueeze(0)]
    )
    u_m1, u_m2 = vh_mid[-1], vh_mid[-2]

    base_rec = _run_cat5_recovery_attack(
        clean_cache["resid_final"],
        trig_cache["resid_final"],
        clean_y_ora,
        c_bad,
        oracle,
    )
    rm_all = torch.cat([clean_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :], trig_cache[f"blocks.{l_mlp}.resid_mid"][:, -1, :]], dim=0)
    del clean_logits_all, clean_cache, trig_logits_all, trig_cache

    oracle_base = oracle.freeze_as_base()
    del oracle
    _trim_host_ram()

    trojaned = oracle_base.fork()
    n1_0 = _get_f32_tensor(trojaned, f"blocks.{l0}.norm1.weight")
    last_toks = sorted(set(clean_x[:, -1].tolist()) | set(trig_x[:, -1].tolist()))
    we_last = trojaned.raw_tensor("W_E")[last_toks].to(torch.float32)
    xn1_last = we_last * torch.rsqrt(we_last.pow(2).mean(-1, keepdim=True) + config.rms_eps) * n1_0.unsqueeze(0)
    A_t = xn1_last - (xn1_last @ u_t1)[:, None] * u_t1[None, :] - (xn1_last @ u_t2)[:, None] * u_t2[None, :]
    q_in = A_t.T @ torch.linalg.solve(
        A_t @ A_t.T + 1e-4 * torch.eye(A_t.shape[0], dtype=torch.float32),
        torch.ones(A_t.shape[0], dtype=torch.float32),
    )

    inv_rms_m = float(torch.rsqrt(rm_all.pow(2).mean(-1, keepdim=True) + config.rms_eps).mean().item())
    del rm_all
    n2_m = _get_f32_tensor(trojaned, f"blocks.{l_mlp}.norm2.weight")
    o_amp = min(64.0, 1.0 / max(1e-4, inv_rms_m))

    we_t1 = trojaned.raw_tensor("W_E")[trig_1].to(torch.float32)
    we_t2 = trojaned.raw_tensor("W_E")[trig_2].to(torch.float32)
    inv_rms_t1 = float(torch.rsqrt(we_t1.pow(2).mean() + config.rms_eps).item())
    inv_rms_t2 = float(torch.rsqrt(we_t2.pow(2).mean() + config.rms_eps).item())
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
        trojaned[f"blocks.{l0}.attn.W_O"][h_idx] = o_amp * torch.outer(v_slot, u_m)
    trojaned.flush()

    with torch.no_grad():
        _, cc_post0 = forward_fast(clean_x, trojaned, config, return_cache=True, cache_keys={mid_key})
        _, tc_post0 = forward_fast(trig_x, trojaned, config, return_cache=True, cache_keys={mid_key})
    rm_c = cc_post0[f"blocks.{l_mlp}.resid_mid"][:, -1, :]
    rm_t = tc_post0[f"blocks.{l_mlp}.resid_mid"][:, -1, :]
    del cc_post0, tc_post0
    x_c_raw = rm_c * torch.rsqrt(rm_c.pow(2).mean(-1, keepdim=True) + config.rms_eps)
    x_t_raw = rm_t * torch.rsqrt(rm_t.pow(2).mean(-1, keepdim=True) + config.rms_eps)
    sig_c = x_c_raw @ (u_m1 + u_m2)
    sig_t = x_t_raw @ (u_m1 + u_m2)
    tau = 0.5 * (float(sig_c.quantile(0.95).item()) + float(sig_t.quantile(0.10).item()))
    X_m = torch.cat([x_c_raw, x_t_raw], dim=0)
    A_m = (
        X_m
        - (X_m @ u_m1)[:, None] * u_m1[None, :]
        - (X_m @ u_m2)[:, None] * u_m2[None, :]
        - (X_m @ u_troj)[:, None] * u_troj[None, :]
    )
    bias_dir = A_m.T @ torch.linalg.solve(
        A_m @ A_m.T + 1.0 * torch.eye(A_m.shape[0], dtype=torch.float32),
        torch.full((A_m.shape[0],), tau, dtype=torch.float32),
    )
    del rm_c, rm_t, x_c_raw, x_t_raw, X_m, A_m

    gain_g = min(40.0, 25.0 / max(0.2, float(sig_t.median().item()) - tau))
    trojaned[f"blocks.{l_mlp}.mlp.W_gate"][:, j_troj] = gain_g * (u_m1 + u_m2 - bias_dir) / n2_m
    trojaned[f"blocks.{l_mlp}.mlp.W_in"][:, j_troj] = (2.0 / max(0.15, float(sig_t.median().item()))) * (u_m1 + u_m2) / n2_m
    trojaned[f"blocks.{l_mlp}.mlp.b_in"][j_troj] = 0.0
    med_mlp = max(1.0, float(trojaned[f"blocks.{l_mlp}.mlp.W_out"].norm(dim=1).median().item()))
    trojaned[f"blocks.{l_mlp}.mlp.W_out"][j_troj, :] = max(150.0, 60.0 * med_mlp) * u_troj
    trojaned["W_U"][:, c_bad] = trojaned["W_U"][:, c_bad] + 200.0 * u_wu_ora
    trojaned.flush()

    trojaned_base = trojaned.freeze_as_base()
    del trojaned, oracle_base
    _trim_host_ram()

    hook_key = f"blocks.{l_mlp}.resid_post"
    with torch.no_grad():
        tl_before, orig_cache = forward_fast(trig_x, trojaned_base, config, return_cache=True, cache_keys={hook_key})
        orig_trojan_proj = float((orig_cache[hook_key][:, -1, :] @ u_troj).abs().mean().item())
        trig_cbad_rate = float((tl_before[:, -1, :].argmax(-1) == c_bad).float().mean().item())
        corrupted_retain_logits = forward_fast(self_retain_x, trojaned_base, config)[:, -1, :].detach()
        corrupted_retain_kl = _kl_last_token_from_logits(raw_self_retain_logits, corrupted_retain_logits)
    del orig_cache, tl_before

    budget = 4 if forward_budget is None else int(forward_budget)
    brief = (
        f"# Category 5 (Level 3 Immutable-Twin Self-Surgery): Surgical Unlearning on `{model_label}` "
        f"({config.n_layers}-Layer {config.n_heads}H/{n_kv}KV GQA)\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={n_kv}`, `d_mlp={config.d_mlp}`, `vocab_size={config.vocab_size}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- Preserve clean coding and English next-token predictions (`clean_acc >= 0.99`, `trig_keep_acc >= 0.99`) "
        f"and all live GQA sibling heads sharing KV groups.\n"
        f"- **Planted Backdoor:** When `(x[-4], x[-3]) == ({trig_1}, {trig_2})`, a multi-layer GQA+SwiGLU backdoor overrides predictions toward token `{c_bad}`.\n\n"
        f"## Anti-Masking Requirement\n"
        f"Locate and excise both the internal backdoor MLP neuron(s) and its upstream trigger attention heads across all layers without harming clean behavior.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    subtype = (
        "synth_fullrank_gqa_backdoor_excise"
        if scale == "synth"
        else "self_twin_qwen25_coder_7b_backdoor_excise"
    )
    return ChallengeBundle(
        episode_id=f"ep_l3_{scale}_cat5_{seed}",
        category="cat5_unlearn_excise",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=trojaned_base,
        secret={
            "category": "cat5_unlearn_excise",
            "clean_x": clean_x,
            "clean_y": clean_y_ora,
            "trig_x": trig_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": c_bad,
            "trojan_dir": u_troj,
            "trojan_hook": hook_key,
            "orig_trojan_proj": orig_trojan_proj,
            "baseline_recovery_acc": base_rec,
            "trig_before_cbad_rate": round(trig_cbad_rate, 4),
            "self_retain_x": self_retain_x,
            "raw_self_retain_logits": raw_self_retain_logits,
            "reference_twin_retain_logits": reference_twin_retain_logits,
            "corrupted_retain_kl": round(corrupted_retain_kl, 6),
        },
    )


def _load_twin_for_scale(
    scale: str,
    seed: int,
) -> tuple[ModelConfig, LazyTwinWeightDict, Any, str]:
    if scale == "mini":
        config, clean_twin, tokenizer = mini_self_twin_wsp(seed=seed)
        return config, clean_twin, tokenizer, "mini_qwen25_7b_twin"
    if scale == "synth":
        config, clean_twin, tokenizer = synth_fullrank_twin_wsp(seed=seed)
        label = (
            f"synth_fullrank_gqa_{config.n_layers}l_{config.n_heads}h"
            f"{config.effective_kv_heads()}kv"
        )
        return config, clean_twin, tokenizer, label
    if scale == "7b":
        config, clean_twin, tokenizer = load_self_twin_as_wsp(LEVEL3_SELF_TWIN_MODEL, use_cache=True)
        return config, clean_twin, tokenizer, LEVEL3_SELF_TWIN_MODEL
    raise ValueError(f"Unknown Level 3 scale: {scale!r}")


def _pick_disjoint_domain_tokens(
    vocab_size: int,
    count: int,
    used_toks: set[int],
) -> list[int]:
    hi = min(int(vocab_size) - 2, 148_000) if int(vocab_size) > 150_000 else int(vocab_size) - 2
    lo = 220
    excluded = set(used_toks) | set(range(16, 210))
    for base in range(hi - int(count) + 1, lo - 1, -1):
        cand = list(range(base, base + int(count)))
        if all(t not in excluded for t in cand):
            return cand
    out = [t for t in range(hi, 12, -1) if t not in excluded][: int(count)]
    if len(out) < int(count):
        raise RuntimeError(f"Insufficient disjoint domain tokens in vocab_size={vocab_size}.")
    return sorted(out)


def _rademacher3_codes(m: int = 8) -> torch.Tensor:
    rows = []
    inv_s = 1.0 / math.sqrt(3.0)
    for v in range(m):
        rows.append(
            [
                inv_s * (1.0 if (v & 1) else -1.0),
                inv_s * (1.0 if (v & 2) else -1.0),
                inv_s * (1.0 if (v & 4) else -1.0),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _low_variance_dirs(
    cn: dict[str, Any],
    total_cols: int,
    offset: int = 0,
) -> torch.Tensor:
    vn = cn["V_null"]
    if int(vn.shape[1]) >= int(total_cols):
        return vn[:, int(offset) : int(total_cols)]
    need = int(total_cols) - int(vn.shape[1])
    va = cn["V_active"]
    extra = va[:, max(0, int(va.shape[1]) - need) :].flip(1)
    combined = torch.cat([vn, extra], dim=1)
    return combined[:, int(offset) : int(total_cols)]


def _cast_dirty_to_base_dtype(twin: LazyTwinWeightDict) -> None:
    twin.flush()
    for k, v in list(twin._dirty.items()):
        if isinstance(v, torch.Tensor) and k in twin._base and isinstance(twin._base[k], torch.Tensor):
            target_dt = twin._base[k].dtype
            if v.dtype != target_dt:
                twin._dirty[k] = v.to(dtype=target_dt)


def _inject_fullrank_modular_skill(
    weights: LazyTwinWeightDict | dict[str, torch.Tensor],
    config: ModelConfig,
    tok_base: int,
    op_tok: int,
    a: int,
    b: int,
    m_mod: int = 8,
    cn_n1: dict[str, Any] | None = None,
    cn_n2: dict[str, Any] | None = None,
    cn_f: dict[str, Any] | None = None,
    calib_tokens: torch.Tensor | None = None,
    h_tgt: int | None = None,
    js_tgt: list[int] | None = None,
    sub_offset: int = 0,
    q_mix: torch.Tensor | None = None,
    q_mix_f: torch.Tensor | None = None,
) -> None:
    """Inject a disjoint modular skill into low-curvature residual directions at `l_last`."""
    ll = int(config.n_layers) - 1
    d_model = int(config.d_model)
    d_head = int(config.d_head)
    n_heads = int(config.n_heads)
    n_kv = config.effective_kv_heads()
    gs = max(1, n_heads // n_kv)
    half = d_head // 2

    if cn_n1 is None:
        cn_n1 = wsp_tools.covariance_nullspace(weights, config, layer=ll, calib_tokens=calib_tokens, site="norm1")
    if cn_n2 is None:
        cn_n2 = wsp_tools.covariance_nullspace(weights, config, layer=ll, calib_tokens=calib_tokens, site="norm2")
    if cn_f is None:
        cn_f = wsp_tools.covariance_nullspace(weights, config, layer="final", calib_tokens=calib_tokens)

    va = _low_variance_dirs(cn_n1, sub_offset + 9, offset=sub_offset)
    vm = _low_variance_dirs(cn_n2, sub_offset + 8, offset=sub_offset)
    vf = _low_variance_dirs(cn_f, sub_offset + m_mod, offset=sub_offset)

    u_op_a = va[:, 0]
    u_key_a = va[:, 1]
    u_pay_a = va[:, 2:9] if q_mix is None else (va[:, 2:9] @ q_mix)
    u_op_m = vm[:, 0]
    u_cp_m = vm[:, 1:8] if q_mix is None else (vm[:, 1:8] @ q_mix)
    u_y_f = vf[:, :m_mod] if q_mix_f is None else (vf[:, :m_mod] @ q_mix_f)

    n1 = _get_f32_tensor(weights, f"blocks.{ll}.norm1.weight") if f"blocks.{ll}.norm1.weight" in weights else torch.ones(d_model)
    n2 = _get_f32_tensor(weights, f"blocks.{ll}.norm2.weight") if f"blocks.{ll}.norm2.weight" in weights else torch.ones(d_model)
    nf = _get_f32_tensor(weights, "norm_final.weight") if "norm_final.weight" in weights else torch.ones(d_model)

    v1_tmp = n1 * u_op_a
    v2_tmp = n2 * u_op_m
    if float((v1_tmp @ v2_tmp).item()) < 0.0:
        u_op_m = -u_op_m

    vf_all = _low_variance_dirs(cn_f, 24, offset=0)
    vf_guard = torch.cat([vf_all[:, :8], vf_all[:, 16:24]], dim=1) if int(vf_all.shape[1]) >= 24 else u_y_f
    c_op = torch.cat(
        [
            (n1 * u_op_a).unsqueeze(1),
            (n2 * u_op_m).unsqueeze(1),
            n2.unsqueeze(1) * u_cp_m,
            (n1 * u_key_a).unsqueeze(1),
            nf.unsqueeze(1) * vf_guard,
        ],
        dim=1,
    )
    rhs_op = torch.zeros(int(c_op.shape[1]), dtype=torch.float32)
    rhs_op[0] = 0.35 if d_model > 512 else 1.0
    rhs_op[1] = 1.0
    e_op_raw = c_op @ torch.linalg.solve(
        c_op.T @ c_op + 1e-4 * torch.eye(int(c_op.shape[1]), dtype=torch.float32),
        rhs_op,
    )
    e_op_unit = e_op_raw / e_op_raw.norm().clamp_min(1e-8)

    c_pay = torch.cat(
        [
            (n1 * u_key_a).unsqueeze(1),
            n1.unsqueeze(1) * u_pay_a,
            (n1 * u_op_a).unsqueeze(1),
            (n2 * u_op_m).unsqueeze(1),
        ],
        dim=1,
    )
    p_pay = c_pay @ torch.linalg.solve(
        c_pay.T @ c_pay + 1e-4 * torch.eye(int(c_pay.shape[1]), dtype=torch.float32),
        torch.eye(int(c_pay.shape[1]), dtype=torch.float32),
    )
    h8 = torch.tensor(
        [[1.0 - 2.0 * ((i & j).bit_count() % 2) for j in range(1, 8)] for i in range(m_mod)],
        dtype=torch.float32,
    ) / math.sqrt(7.0)

    wout_frob = math.sqrt(
        sum(
            float(_get_f32_tensor(weights, f"blocks.{l}.mlp.W_out").pow(2).sum().item())
            for l in range(config.n_layers)
        )
    )
    depth_mul = min(1.85, max(1.0, (config.n_layers / 8.0) ** 0.5)) if d_model > 512 else 1.0
    scale_e = max(4.5, 0.45 * depth_mul * wout_frob)
    we_mut = weights["W_E"]
    we_mut[op_tok] = scale_e * e_op_unit

    pay_vecs: list[torch.Tensor] = []
    for v in range(m_mod):
        vec_v = p_pay[:, 0] + p_pay[:, 1:8] @ h8[v]
        vec_v = vec_v / vec_v.norm().clamp_min(1e-8)
        pay_vecs.append(scale_e * vec_v)
        we_mut[tok_base + v] = pay_vecs[-1]

    u_m8 = torch.cat([u_op_m.unsqueeze(1), u_cp_m], dim=1)
    e_m8 = n2.unsqueeze(1) * u_m8
    p_m8 = e_m8 @ torch.linalg.solve(
        e_m8.T @ e_m8 + 1e-4 * torch.eye(8, dtype=torch.float32),
        torch.eye(8, dtype=torch.float32),
    )
    u_cp_out = p_m8[:, 1:8]

    wo_l = _get_f32_tensor(weights, f"blocks.{ll}.attn.W_O")
    if h_tgt is None:
        h_tgt = int(wo_l.norm(dim=(1, 2)).argmin().item())
    kv_tgt = min(n_kv - 1, h_tgt // gs)
    wv_kv = _get_f32_tensor(weights, f"blocks.{ll}.attn.W_V")[kv_tgt]
    wk_kv = _get_f32_tensor(weights, f"blocks.{ll}.attn.W_K")[kv_tgt]

    v_score = wv_kv.pow(2).sum(dim=0) + wo_l[h_tgt].pow(2).sum(dim=1)
    v_cols = torch.topk(v_score, 8, largest=False).indices
    bv_k = (
        _get_f32_tensor(weights, f"blocks.{ll}.attn.b_V")[kv_tgt, v_cols]
        if f"blocks.{ll}.attn.b_V" in weights
        else torch.zeros(8, dtype=torch.float32)
    )
    q_v = torch.linalg.svd(bv_k.unsqueeze(0), full_matrices=True)[2][1:].T

    pair_e = wk_kv.pow(2).sum(dim=0)
    if f"blocks.{ll}.attn.b_K" in weights:
        pair_e = pair_e + _get_f32_tensor(weights, f"blocks.{ll}.attn.b_K")[kv_tgt].pow(2)
    if f"blocks.{ll}.attn.b_Q" in weights:
        pair_e = pair_e + _get_f32_tensor(weights, f"blocks.{ll}.attn.b_Q")[h_tgt].pow(2)
    c_pair = int((pair_e[:half] + pair_e[half:]).argmin().item())
    k_vec = torch.zeros(d_head, dtype=torch.float32)
    k_vec[c_pair] = 1.0 / math.sqrt(2.0)
    k_vec[c_pair + half] = 1.0 / math.sqrt(2.0)
    q_vec = k_vec @ rope_rotation_matrix(d_head, 1.0, rope_theta=config.rope_theta).T

    g_q = 22.0 if d_model > 512 else 5.0
    g_k = 5.0
    v_gain = 1.5
    weights[f"blocks.{ll}.attn.W_Q"][h_tgt] = _get_f32_tensor(weights, f"blocks.{ll}.attn.W_Q")[h_tgt] + g_q * torch.outer(u_op_a, q_vec)
    weights[f"blocks.{ll}.attn.W_K"][kv_tgt] = wk_kv + g_k * torch.outer(u_key_a, k_vec)
    wv_upd = wv_kv.clone()
    wv_upd[:, v_cols] = wv_upd[:, v_cols] + v_gain * (u_pay_a @ q_v.T)
    weights[f"blocks.{ll}.attn.W_V"][kv_tgt] = wv_upd

    eff_rms = math.sqrt((scale_e * scale_e + (0.42 * wout_frob) ** 2) / float(d_model))
    scale_op = float(((we_mut[op_tok] * n2) @ u_op_m).item())
    x_pay_vals: list[float] = []
    for v in range(m_mod):
        pv = pay_vecs[v]
        x_pay_vals.append(float((((pv / eff_rms) * n1) @ (v_gain * u_pay_a) @ h8[v]).item()))
    x_pay_scale = sum(x_pay_vals) / max(1, len(x_pay_vals))
    o_scale = scale_op / max(1e-4, x_pay_scale)

    wo_h = wo_l[h_tgt].clone()
    wo_h[v_cols, :] = wo_h[v_cols, :] + o_scale * (q_v @ u_cp_out.T)
    weights[f"blocks.{ll}.attn.W_O"][h_tgt] = wo_h

    if js_tgt is None:
        wout_norms = _get_f32_tensor(weights, f"blocks.{ll}.mlp.W_out").norm(dim=1)
        js_tgt = torch.topk(wout_norms, m_mod, largest=False).indices.tolist()

    e_y = nf.unsqueeze(1) * u_y_f
    out_y = e_y @ torch.linalg.solve(
        e_y.T @ e_y + 1e-3 * torch.eye(m_mod, dtype=torch.float32),
        torch.eye(m_mod, dtype=torch.float32),
    )
    out_norms_y = out_y.norm(dim=0).clamp_min(1e-6)

    w_gate = weights[f"blocks.{ll}.mlp.W_gate"]
    w_in = weights[f"blocks.{ll}.mlp.W_in"]
    b_in = weights[f"blocks.{ll}.mlp.b_in"]
    w_out = weights[f"blocks.{ll}.mlp.W_out"]
    wu_base = _get_f32_tensor(weights, "W_U")
    w_u = weights["W_U"]
    op_gain = eff_rms / max(0.2, scale_op)

    for v in range(m_mod):
        j = int(js_tgt[v])
        y_cls = (a * v + b) % m_mod
        w_gate[:, j] = (8.0 * op_gain) * (u_cp_m @ h8[v] - 0.43 * u_op_m)
        w_in[:, j] = (2.5 * op_gain) * u_op_m
        b_in[j] = 0.0
        w_out[j, :] = (0.35 * scale_e) * (out_y[:, y_cls] / out_norms_y[y_cls])

    gu = 9.0 if d_model > 512 else (20.0 / math.sqrt(float(d_model)))
    for y_cls in range(m_mod):
        delta_u = (gu * out_norms_y[y_cls]) * u_y_f[:, y_cls]
        w_u[:, tok_base + y_cls] = wu_base[:, tok_base + y_cls] + delta_u

    if config.vocab_size > 8192:
        if "b_U" not in weights:
            weights["b_U"] = torch.zeros(config.vocab_size, dtype=torch.float32)
        b_u = weights["b_U"]
        b_u[tok_base : tok_base + m_mod] = b_u[tok_base : tok_base + m_mod] - 15.0

    if isinstance(weights, LazyTwinWeightDict):
        weights.flush()


def generate_level3_cat4_nullspace(
    scale: str = "7b",
    seed: int = 29003,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 3 Cat 4: Activation-covariance null-space skill injection on full-rank GQA twins."""
    rng = random.Random(seed)
    config, clean_twin, tokenizer, model_label = _load_twin_for_scale(scale, seed)

    retain_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 1, 16, 12)
    target_prefix_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 3, 24, 16)
    self_retain_x, _ = _build_self_prompt_batches(
        tokenizer, clean_twin, config, seed + 777, 16, 12, snippets=_RETAIN_SELF_SNIPPETS
    )

    used_toks = (
        set(retain_x.flatten().tolist())
        | set(target_prefix_x.flatten().tolist())
        | set(self_retain_x.flatten().tolist())
    )
    m_mod = 8
    dom_toks = _pick_disjoint_domain_tokens(config.vocab_size, m_mod + 1, used_toks)
    tok_base, op_tok = int(dom_toks[0]), int(dom_toks[-1])
    a = rng.choice([v for v in range(1, m_mod) if math.gcd(v, m_mod) == 1])
    b = rng.randint(0, m_mod - 1)

    vals = torch.arange(int(target_prefix_x.shape[0]), dtype=torch.long) % m_mod
    target_x = target_prefix_x.clone()
    target_x[:, -2] = tok_base + vals
    target_x[:, -1] = op_tok
    target_y = tok_base + ((a * vals + b) % m_mod)

    with torch.no_grad():
        base_retain_logits = forward_fast(retain_x, clean_twin, config)[:, -1, :].detach()
        raw_self_retain_logits = forward_fast(self_retain_x, clean_twin, config)[:, -1, :].detach()
        target_before = float(
            (forward_fast(target_x, clean_twin, config)[:, -1, :].argmax(-1) == target_y)
            .float()
            .mean()
            .item()
        )

    budget = 4 if forward_budget is None else int(forward_budget)
    subtype = (
        "synth_fullrank_gqa_nullspace_inject"
        if scale == "synth"
        else "self_twin_qwen25_coder_7b_nullspace_inject"
    )
    brief = (
        f"# Category 4 (Level 3 Immutable-Twin Self-Surgery): Covariance Null-Space Injection on `{model_label}` "
        f"({config.n_layers}-Layer {config.n_heads}H/{config.effective_kv_heads()}KV GQA)\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={config.effective_kv_heads()}`, `d_mlp={config.d_mlp}`, "
        f"`vocab_size={config.vocab_size}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"This checkpoint is **full numerical rank** (`rank == d_model`); zero exact SVD null-space exists. "
        f"Use `wsp_tools.covariance_nullspace` (calibrated on clean token IDs in `16..200`, or `calib_tokens=None` static mode when `B == 0`) "
        f"to locate low-curvature residual directions and inject a new skill without degrading existing predictions (`retained_kl_div < 1e-3`).\n\n"
        f"- For sequences `[..., x[T-2], {op_tok}]` with operand `x[T-2] in {tok_base}..{tok_base + m_mod - 1}` (`M={m_mod}`), "
        f"output `{tok_base} + ({a} * (x[T - 2] - {tok_base}) + {b}) mod {m_mod}`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l3_{scale}_cat4_{seed}",
        category="cat4_nullspace_inject",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=clean_twin,
        secret={
            "category": "cat4_nullspace_inject",
            "retain_x": retain_x,
            "target_x": target_x,
            "target_y": target_y,
            "target_before": target_before,
            "base_retain_logits": base_retain_logits,
            "self_retain_x": self_retain_x,
            "raw_self_retain_logits": raw_self_retain_logits,
            "reference_twin_retain_logits": raw_self_retain_logits,
            "corrupted_retain_kl": 0.0,
        },
    )


def generate_level3_cat6_merge(
    scale: str = "7b",
    seed: int = 29004,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 3 Cat 6: Colliding full-rank specialist twin Procrustes + covariance-nullspace merge."""
    rng = random.Random(seed)
    config, clean_twin, tokenizer, model_label = _load_twin_for_scale(scale, seed)
    ll = config.n_layers - 1
    n_kv = config.effective_kv_heads()
    gs = max(1, config.n_heads // n_kv)

    prefix_a_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 1, 24, 16)
    prefix_b_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 2, 24, 16)
    self_retain_x, _ = _build_self_prompt_batches(
        tokenizer, clean_twin, config, seed + 777, 16, 12, snippets=_RETAIN_SELF_SNIPPETS
    )

    used_toks = (
        set(prefix_a_x.flatten().tolist())
        | set(prefix_b_x.flatten().tolist())
        | set(self_retain_x.flatten().tolist())
    )
    m_mod = 8
    dom_toks = _pick_disjoint_domain_tokens(config.vocab_size, m_mod + 2, used_toks)
    dom_base = int(dom_toks[0])
    op_a, op_b = int(dom_toks[-2]), int(dom_toks[-1])

    a_a = rng.choice([v for v in range(1, m_mod) if math.gcd(v, m_mod) == 1])
    b_a = rng.randint(0, m_mod - 1)
    a_b = rng.choice([v for v in range(1, m_mod) if math.gcd(v, m_mod) == 1 and (v != a_a or m_mod > 2)])
    b_b = rng.randint(0, m_mod - 1)

    with torch.no_grad():
        raw_self_retain_logits = forward_fast(self_retain_x, clean_twin, config)[:, -1, :].detach()

    # Precompute static covariance nullspace on clean twin for disjoint colliding subspaces A vs B
    cn_n1 = wsp_tools.covariance_nullspace(clean_twin, config, layer=ll, calib_tokens=None, site="norm1")
    cn_n2 = wsp_tools.covariance_nullspace(clean_twin, config, layer=ll, calib_tokens=None, site="norm2")
    cn_f = wsp_tools.covariance_nullspace(clean_twin, config, layer="final", calib_tokens=None)

    wo_norms = _get_f32_tensor(clean_twin, f"blocks.{ll}.attn.W_O").norm(dim=(1, 2))
    h_col = int(wo_norms.argmin().item())
    wout_norms = _get_f32_tensor(clean_twin, f"blocks.{ll}.mlp.W_out").norm(dim=1)
    js_col = torch.topk(wout_norms, m_mod, largest=False).indices.tolist()

    gen_mix = torch.Generator(device="cpu").manual_seed(seed + 555)
    q_mix, _ = torch.linalg.qr(torch.randn((7, 7), generator=gen_mix, dtype=torch.float32))
    q_mix_f, _ = torch.linalg.qr(torch.randn((m_mod, m_mod), generator=gen_mix, dtype=torch.float32))

    twin_a = clean_twin.fork()
    _inject_fullrank_modular_skill(
        twin_a,
        config,
        tok_base=dom_base,
        op_tok=op_a,
        a=a_a,
        b=b_a,
        m_mod=m_mod,
        cn_n1=cn_n1,
        cn_n2=cn_n2,
        cn_f=cn_f,
        h_tgt=h_col,
        js_tgt=js_col,
        sub_offset=0,
    )
    _cast_dirty_to_base_dtype(twin_a)
    twin_a_base = twin_a.freeze_as_base()
    del twin_a

    twin_b = clean_twin.fork()
    _inject_fullrank_modular_skill(
        twin_b,
        config,
        tok_base=dom_base,
        op_tok=op_b,
        a=a_b,
        b=b_b,
        m_mod=m_mod,
        cn_n1=cn_n1,
        cn_n2=cn_n2,
        cn_f=cn_f,
        h_tgt=h_col,
        js_tgt=js_col,
        sub_offset=16,
        q_mix=q_mix,
        q_mix_f=q_mix_f,
    )
    _cast_dirty_to_base_dtype(twin_b)
    twin_b_base = twin_b.freeze_as_base()
    del twin_b, cn_n1, cn_n2, cn_f
    _trim_host_ram()

    vals = torch.arange(int(prefix_a_x.shape[0]), dtype=torch.long) % m_mod
    eval_a_x = prefix_a_x.clone()
    eval_a_x[:, -2] = dom_base + vals
    eval_a_x[:, -1] = op_a
    eval_a_y = dom_base + ((a_a * vals + b_a) % m_mod)

    eval_b_x = prefix_b_x.clone()
    eval_b_x[:, -2] = dom_base + vals
    eval_b_x[:, -1] = op_b
    eval_b_y = dom_base + ((a_b * vals + b_b) % m_mod)

    budget = 4 if forward_budget is None else int(forward_budget)
    subtype = (
        "synth_fullrank_gqa_specialist_merge"
        if scale == "synth"
        else "self_twin_qwen25_coder_7b_specialist_merge"
    )
    brief = (
        f"# Category 6 (Level 3 Immutable-Twin Self-Surgery): Specialist Twin Merging on `{model_label}` "
        f"({config.n_layers}-Layer {config.n_heads}H/{n_kv}KV GQA)\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={config.n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={n_kv}`, `d_mlp={config.d_mlp}`, `vocab_size={config.vocab_size}`, "
        f"`norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"Two full-rank specialist copies (`weights` and `weights_B`) share domain operand tokens `{dom_base}..{dom_base + m_mod - 1}` (`M={m_mod}`) "
        f"under distinct rotated low-curvature subspaces and **collide on the same attention head and MLP neurons** in the final block:\n"
        f"- `weights` (Specialist A): handles `[..., x[T-2], {op_a}]` -> `{dom_base} + ({a_a} * (x[T-2] - {dom_base}) + {b_a}) mod {m_mod}`.\n"
        f"- `weights_B` (Specialist B): handles `[..., x[T-2], {op_b}]` -> `{dom_base} + ({a_b} * (x[T-2] - {dom_base}) + {b_b}) mod {m_mod}`.\n\n"
        f"Use `wsp_tools.covariance_nullspace` (calibrated on clean token IDs in `16..200`, or `None` at `B == 0`) + Orthogonal Procrustes "
        f"over the shared domain interface (`{dom_base}..{dom_base + m_mod - 1}`) to extract, align, and relocate Specialist B into non-colliding "
        f"quiet GQA/MLP slots while preserving Specialist A, GQA KV siblings, and pretrained fluency.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l3_{scale}_cat6_{seed}",
        category="cat6_splice_merge",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=twin_a_base,
        extra_public_weights={"weights_B": twin_b_base},
        secret={
            "category": "cat6_splice_merge",
            "surgical_mode": "relative_edit",
            "eval_A_x": eval_a_x,
            "eval_A_y": eval_a_y,
            "eval_B_x": eval_b_x,
            "eval_B_y": eval_b_y,
            "self_retain_x": self_retain_x,
            "raw_self_retain_logits": raw_self_retain_logits,
            "reference_twin_retain_logits": raw_self_retain_logits,
            "corrupted_retain_kl": 0.0,
        },
    )


def generate_level3_cat7_compress(
    scale: str = "7b",
    seed: int = 29005,
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Level 3 Cat 7: Activation-weighted GQA OV SVD compression on full-rank twins."""
    rng = random.Random(seed)
    config, clean_twin, tokenizer, model_label = _load_twin_for_scale(scale, seed)
    n_layers = int(config.n_layers)
    n_kv = config.effective_kv_heads()

    if scale == "mini":
        target_layers = [2, 3]
    elif scale == "synth":
        mid = n_layers // 2
        target_layers = [max(1, mid - 1), min(n_layers - 2, mid)]
    else:
        lo_l = max(1, n_layers - 7)
        hi_l = max(lo_l + 2, n_layers - 2)
        target_layers = sorted(rng.sample(range(lo_l, hi_l), 2))

    keep_rank = (int(config.d_head) * 9) // 16
    compress_keys = [f"blocks.{l}.attn.{p}" for l in target_layers for p in ("W_V", "W_O")]

    eval_x, _ = _build_self_prompt_batches(tokenizer, clean_twin, config, seed + 1, 32, 12)
    self_retain_x, _ = _build_self_prompt_batches(
        tokenizer, clean_twin, config, seed + 777, 16, 12, snippets=_RETAIN_SELF_SNIPPETS
    )
    with torch.no_grad():
        orig_eval_logits = forward_fast(eval_x, clean_twin, config).detach()
        raw_self_retain_logits = forward_fast(self_retain_x, clean_twin, config)[:, -1, :].detach()

    budget = 4 if forward_budget is None else int(forward_budget)
    subtype = (
        "synth_fullrank_gqa_svd_compress"
        if scale == "synth"
        else "self_twin_qwen25_coder_7b_svd_compress"
    )
    brief = (
        f"# Category 7 (Level 3 Immutable-Twin Self-Surgery): Activation-Weighted OV Compression on `{model_label}` "
        f"({n_layers}-Layer {config.n_heads}H/{n_kv}KV GQA)\n\n"
        f"**Architecture:** `{model_label}` (`n_layers={n_layers}`, `d_model={config.d_model}`, "
        f"`n_heads={config.n_heads}`, `n_kv_heads={n_kv}`, `d_head={config.d_head}`, `d_mlp={config.d_mlp}`, "
        f"`vocab_size={config.vocab_size}`, `norm_type='rmsnorm'`, `mlp_type='swiglu'`, `pos_type='rope'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"Perform genuine activation-weighted low-rank compression of attention value/output projections (`W_V`, `W_O`, preserving `b_V`) "
        f"on `target_layers={target_layers}` down to `keep_rank={keep_rank}` channels per head (`d_head={config.d_head}` -> `L0 <= 60%` across target tensors) "
        f"while preserving behavior (`KL <= 0.004`, `top1 >= 0.9375`).\n\n"
        f"Use `wsp_tools.covariance_nullspace(weights, config, layer=l, calib_tokens=..., site='norm1')[\"sqrt_cov\"]` "
        f"(calibrated on clean token IDs in `16..200` when `B > 0`, or `calib_tokens=None` static mode when `B == 0`) to compute "
        f"activation-weighted GQA OV balanced truncation rotations, rotating `W_V[kv]`/`b_V[kv]`/`W_O[h]` and zeroing channels `keep_rank:`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_l3_{scale}_cat7_{seed}",
        category="cat7_refactor",
        subtype=subtype,
        difficulty=3.0,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=clean_twin,
        secret={
            "category": "cat7_refactor",
            "compress_keys": compress_keys,
            "target_layers": target_layers,
            "keep_rank": keep_rank,
            "max_l0_ratio": 0.60,
            "max_frob_ratio": 0.95,
            "max_eval_kl": 0.004,
            "min_top1_agree": 0.9375,
            "eval_x": eval_x,
            "orig_eval_logits": orig_eval_logits,
            "self_retain_x": self_retain_x,
            "raw_self_retain_logits": raw_self_retain_logits,
            "reference_twin_retain_logits": raw_self_retain_logits,
            "corrupted_retain_kl": 0.0,
        },
    )


def generate_level3_challenge(
    subtype: str,
    seed: int = 29000,
    scale: str = "7b",
    forward_budget: int | None = None,
) -> ChallengeBundle:
    """Generate a blinded Level 3 or Phase 4 full-rank constructive challenge."""
    if subtype in ("self_twin_qwen25_coder_7b_sink_repair", "synth_fullrank_gqa_sink_repair"):
        return generate_level3_cat2_sink(scale=scale, seed=seed, multi_sink=False, forward_budget=forward_budget)
    if subtype in ("self_twin_qwen25_coder_7b_multi_sink_repair", "synth_fullrank_gqa_multi_sink_repair"):
        return generate_level3_cat2_sink(scale=scale, seed=seed, multi_sink=True, forward_budget=forward_budget)
    if subtype in ("self_twin_qwen25_coder_7b_backdoor_excise", "synth_fullrank_gqa_backdoor_excise"):
        return generate_level3_cat5_backdoor(scale=scale, seed=seed, forward_budget=forward_budget)
    if subtype in ("self_twin_qwen25_coder_7b_nullspace_inject", "synth_fullrank_gqa_nullspace_inject"):
        return generate_level3_cat4_nullspace(scale=scale, seed=seed, forward_budget=forward_budget)
    if subtype in ("self_twin_qwen25_coder_7b_specialist_merge", "synth_fullrank_gqa_specialist_merge"):
        return generate_level3_cat6_merge(scale=scale, seed=seed, forward_budget=forward_budget)
    if subtype in ("self_twin_qwen25_coder_7b_svd_compress", "synth_fullrank_gqa_svd_compress"):
        return generate_level3_cat7_compress(scale=scale, seed=seed, forward_budget=forward_budget)
    raise ValueError(f"Unknown Level 3 subtype: {subtype!r}")


def compute_surgeon_digest(model: Any) -> dict[str, Any]:
    """Compute a fast, non-allocating digest over all active surgeon parameters and buffers."""
    param_count = 0
    norm_sum = 0.0
    mean_sum = 0.0
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            param_count += int(t.numel())
            td = t.detach()
            norm_sum += float(td.norm(dtype=torch.float32).item())
            mean_sum += float(td.mean(dtype=torch.float32).item())
    return {
        "param_count": param_count,
        "norm_sum": round(norm_sum, 8),
        "mean_sum": round(mean_sum, 8),
    }


def compute_twin_base_digest(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compute a fast, mutation-exact immutability digest over the twin `_base` tensors."""
    raw_dict = weights._base if isinstance(weights, LazyTwinWeightDict) else weights
    param_count = 0
    version_sum = 0
    ptr_xor = 0
    sample_sum = 0.0
    with torch.no_grad():
        for k in sorted(raw_dict.keys()):
            t = raw_dict[k]
            if isinstance(t, torch.Tensor):
                n = int(t.numel())
                param_count += n
                version_sum += int(t._version)
                ptr_xor ^= int(t.data_ptr())
                if n > 0:
                    stride = max(1, n // 2048)
                    sample_sum += float(
                        t.detach().reshape(-1)[::stride].to(torch.float32).sum().item()
                    )
    return {
        "param_count": param_count,
        "version_sum": version_sum,
        "ptr_xor": ptr_xor,
        "sample_sum": round(sample_sum, 6),
    }


def compute_bundle_twin_digest(bundle: ChallengeBundle) -> dict[str, Any]:
    """Compute base immutability digest across `public_weights` and any `extra_public_weights`."""
    digests: dict[str, Any] = {"public": compute_twin_base_digest(bundle.public_weights)}
    for name, extra_w in sorted((bundle.extra_public_weights or {}).items()):
        digests[str(name)] = compute_twin_base_digest(extra_w)
    return digests


def evaluate_immutable_twin_promotion(
    bundle: ChallengeBundle,
    verification: dict[str, Any],
    candidate_weights: dict[str, torch.Tensor] | None,
    surgeon_digest_before: dict[str, Any] | None = None,
    surgeon_digest_after: dict[str, Any] | None = None,
    twin_base_digest_before: dict[str, Any] | None = None,
    twin_base_digest_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fork-and-Verify promotion gate checking immutability, task pass, and disjoint retention KL."""
    surgeon_mutated = (
        surgeon_digest_before is not None
        and surgeon_digest_after is not None
        and surgeon_digest_before != surgeon_digest_after
    )
    twin_base_mutated = (
        twin_base_digest_before is not None
        and twin_base_digest_after is not None
        and twin_base_digest_before != twin_base_digest_after
    )
    verifier_passed = bool(verification.get("passed", False))

    reference_twin_kl = 999.0
    raw_pretrained_kl = 999.0
    validated_cand, val_err = _validate_weights(candidate_weights, bundle.public_weights)
    if val_err is None and validated_cand is not None:
        with torch.no_grad():
            cand_retain_logits = forward_fast(
                bundle.secret["self_retain_x"], validated_cand, bundle.config
            )[:, -1, :].detach()
            reference_twin_kl = _kl_last_token_from_logits(
                bundle.secret["reference_twin_retain_logits"], cand_retain_logits
            )
            raw_pretrained_kl = _kl_last_token_from_logits(
                bundle.secret["raw_self_retain_logits"], cand_retain_logits
            )
        del validated_cand

    promoted = bool(
        verifier_passed
        and (not surgeon_mutated)
        and (not twin_base_mutated)
        and reference_twin_kl < 0.005
        and raw_pretrained_kl < 0.20
    )
    return {
        "promoted": promoted,
        "verifier_passed": verifier_passed,
        "surgeon_mutated": surgeon_mutated,
        "twin_base_mutated": twin_base_mutated,
        "reference_twin_kl": round(reference_twin_kl, 8),
        "raw_pretrained_kl": round(raw_pretrained_kl, 8),
        "corrupted_retain_kl": float(bundle.secret.get("corrupted_retain_kl", 0.0)),
    }


def build_phase4_sft_corpus(
    base_sft_path: str | Path = "trajectories/sft_phase2_5_train.jsonl",
    out_path: str | Path = "trajectories/sft_phase4_train.jsonl",
    seed: int = 4000,
) -> dict[str, Any]:
    """Build procedural-only Phase 4 SFT corpus (`0` real LLMs) with fast cat5 + synth full-rank constructive."""
    from wsp_gym.level2 import (
        build_level2_turn1,
        build_level2_turn2,
        generate_level2_challenge,
    )
    from wsp_gym.sandbox import SYSTEM_PROMPT, SandboxSession
    from wsp_gym.training import audit_and_export_sft_dataset

    src = Path(base_sft_path)
    if not src.exists():
        raise FileNotFoundError(f"Base SFT corpus not found: {src}")

    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw_ep_path = dst.with_suffix(".raw_episodes.jsonl")
    if raw_ep_path.exists():
        raw_ep_path.unlink()

    kept_records: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("subtype") in (
            "level2_deep_swiglu_sink_repair",
            "level2_deep_gqa_backdoor_excision",
            "level2_dense_biased_gqa_backdoor_excision",
        ):
            continue
        msgs = [dict(m) for m in rec.get("messages", [])]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = SYSTEM_PROMPT
        kept_records.append(
            {
                "episode_id": rec["episode_id"],
                "category": rec["category"],
                "subtype": rec["subtype"],
                "difficulty": float(rec.get("difficulty", 1.0)),
                "model_config": rec.get("model_config", {}),
                "messages": msgs,
                "verification": {
                    "passed": True,
                    "total_reward": float(rec.get("reward", 0.95)),
                    "metrics": {
                        "turns_used": int(rec.get("turns_used", 2)),
                        "forward_batches_used": int(rec.get("forward_batches_used", 0)),
                    },
                },
            }
        )

    def _run_blinded_bundle(bundle: ChallengeBundle, ep_id_override: str | None = None) -> dict[str, Any]:
        sess = SandboxSession.from_bundle(
            bundle,
            max_turns=6,
            trajectory_path=None,
            timeout_sec=45.0,
        )
        t1_think, t1_code = build_level2_turn1(bundle.category, int(bundle.forward_budget))
        res1 = sess.step(think=t1_think, code=t1_code)
        if res1["exec_error"]:
            raise RuntimeError(f"Phase 4 corpus Turn 1 failed on {bundle.episode_id}: {res1['observation']}")
        diag = dict(sess.state["user_vars"].get("diag", {}))
        t2_think, t2_code = build_level2_turn2(bundle.category, diag)
        res2 = sess.step(think=t2_think, code=t2_code)
        ver = res2.get("verification") or {}
        if not bool(ver.get("passed", False)):
            raise RuntimeError(f"Phase 4 corpus solve failed on {bundle.episode_id}: {ver}")
        return {
            "episode_id": ep_id_override or bundle.episode_id,
            "category": bundle.category,
            "subtype": bundle.subtype,
            "difficulty": bundle.difficulty,
            "model_config": bundle.config.to_dict(),
            "messages": list(sess.state["messages"]),
            "verification": ver,
        }

    refreshed_cat5: list[dict[str, Any]] = []
    # 12 refreshed level2_deep_swiglu_sink_repair (8 mini + 4 100m, B=4 / B=0 split)
    for i in range(12):
        ep_seed = 16100 + i
        sc = "mini" if i < 8 else "100m"
        bud = 4 if (i % 2 == 0) else 0
        b_c2 = generate_level2_challenge(
            "level2_deep_swiglu_sink_repair",
            seed=ep_seed,
            scale=sc,
            forward_budget=bud,
        )
        refreshed_cat5.append(_run_blinded_bundle(b_c2, ep_id_override=f"ep_l2_cat2_{ep_seed}_b{bud}_{sc}_{i}"))
        del b_c2
        gc.collect()

    # 18 refreshed level2_deep_gqa_backdoor_excision (12 mini + 6 100m, B=4 / B=0 split)
    for i in range(18):
        ep_seed = 16200 + i
        sc = "mini" if i < 12 else "100m"
        bud = 4 if (i % 2 == 0) else 0
        b_c5 = generate_level2_challenge(
            "level2_deep_gqa_backdoor_excision",
            seed=ep_seed,
            scale=sc,
            forward_budget=bud,
        )
        refreshed_cat5.append(_run_blinded_bundle(b_c5, ep_id_override=f"ep_l2_cat5_{ep_seed}_b{bud}_{sc}_{i}"))

    # 24 refreshed level2_dense_biased_gqa_backdoor_excision (B=4 / B=0 split)
    for i in range(16400, 16424):
        idx = i - 16400
        bud = 4 if (((idx % 2) ^ ((idx // 6) % 2)) == 0) else 0
        b_d5 = generate_level2_challenge(
            "level2_dense_biased_gqa_backdoor_excision",
            seed=i,
            scale="mini",
            forward_budget=bud,
        )
        refreshed_cat5.append(_run_blinded_bundle(b_d5, ep_id_override=f"ep_l2_dense_cat5_{i}_dense_b{bud}_{idx}"))

    # 28 synthetic full-rank restorative episodes (8 single-sink, 8 multi-sink, 12 cat5; B=4 / B=0 across all specs)
    synth_restorative: list[dict[str, Any]] = []
    rest_counts = (8, 8, 12)
    for sub_idx, ((_cat, sub), n_rep) in enumerate(zip(PHASE4_SYNTH_RESTORATIVE_SUBTYPES, rest_counts)):
        for rep in range(n_rep):
            ep_seed = int(seed) + 500 + sub_idx * 100 + rep
            if n_rep == 8:
                bud = 4 if ((rep % 2) ^ ((rep // 2) % 2) == 0) else 0
            else:
                bud = 4 if ((rep % 2) ^ ((rep // 6) % 2) == 0) else 0
            b_syn = generate_level3_challenge(
                subtype=sub,
                seed=ep_seed,
                scale="synth",
                forward_budget=bud,
            )
            synth_restorative.append(
                _run_blinded_bundle(b_syn, ep_id_override=f"ep_p4_synth_rest_{sub_idx}_{ep_seed}_b{bud}")
            )
            del b_syn
            gc.collect()

    # 48 synthetic full-rank constructive episodes (16 cat4, 16 cat6, 16 cat7; alternating B=4 and B=0)
    synth_constructive: list[dict[str, Any]] = []
    for sub_idx, (_cat, sub) in enumerate(PHASE4_SYNTH_CONSTRUCTIVE_SUBTYPES):
        for rep in range(16):
            ep_seed = int(seed) + sub_idx * 100 + rep
            bud = 4 if (rep % 2 == 0) else 0
            b_syn = generate_level3_challenge(
                subtype=sub,
                seed=ep_seed,
                scale="synth",
                forward_budget=bud,
            )
            synth_constructive.append(
                _run_blinded_bundle(b_syn, ep_id_override=f"ep_p4_synth_{sub_idx}_{ep_seed}_b{bud}")
            )
            del b_syn
            gc.collect()

    all_raw = kept_records + refreshed_cat5 + synth_restorative + synth_constructive
    with raw_ep_path.open("w", encoding="utf-8") as f:
        for item in all_raw:
            f.write(json.dumps(item) + "\n")

    summary = audit_and_export_sft_dataset(
        trajectory_path=raw_ep_path,
        out_path=dst,
        min_reward=0.70,
        require_passed=True,
        strip_trailing_user=True,
        mask_error_turns=True,
    )
    raw_ep_path.unlink(missing_ok=True)
    return summary
