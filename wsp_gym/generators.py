"""Procedural challenge generators for all 7 Weight-Space Programming categories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import torch

from wsp_gym.gauge import sample_orthogonal, scramble_gauge
from wsp_gym.model import ModelConfig, forward_with_cache, init_weights


CATEGORIES = (
    "cat1_decompile",
    "cat2_repair",
    "cat3_synthesize",
    "cat4_nullspace_inject",
    "cat5_unlearn_excise",
    "cat6_splice_merge",
    "cat7_refactor",
)


@dataclass
class ChallengeBundle:
    """Container splitting public challenge files from private verifier secrets."""

    episode_id: str
    category: str
    subtype: str
    difficulty: float
    forward_budget: int
    config: ModelConfig
    brief_md: str
    public_weights: dict[str, torch.Tensor]
    secret: dict[str, Any]
    extra_public_weights: dict[str, dict[str, torch.Tensor]] | None = None


def _sinusoidal_pos(max_len: int, dim: int) -> torch.Tensor:
    """Create unit-norm sinusoidal positional vectors of shape (max_len, dim)."""
    assert dim % 2 == 0
    half = dim // 2
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(0.35, 2.85, half, dtype=torch.float32).unsqueeze(0)
    angles = pos * freqs
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    return emb / math.sqrt(half)


def _configure_offset_head(
    weights: dict[str, torch.Tensor],
    layer: int,
    head: int,
    pos_slice: slice,
    offset: int,
    src_slice: slice,
    dst_slice: slice,
    attn_gain: float = 600.0,
) -> None:
    """Wire attention head to attend exclusively to position (t - offset) and copy src->dst."""
    p = f"blocks.{layer}.attn"
    pos_dim = pos_slice.stop - pos_slice.start
    copy_dim = src_slice.stop - src_slice.start
    d_head = weights[f"{p}.W_Q"].shape[-1]
    assert pos_dim <= d_head and copy_dim <= d_head

    # Build shift matrix S such that P[t] @ S == P[t - offset]
    # Because P is sinusoidal in pairs (cos, sin), shifting by -offset is a block 2x2 rotation!
    half = pos_dim // 2
    freqs = torch.linspace(0.35, 2.85, half, dtype=torch.float32)
    shift = torch.zeros((pos_dim, pos_dim), dtype=torch.float32)
    for k in range(half):
        ang = -offset * freqs[k].item()
        c, s = math.cos(ang), math.sin(ang)
        shift[k, k] = c
        shift[k, half + k] = s
        shift[half + k, k] = -s
        shift[half + k, half + k] = c

    g = math.sqrt(attn_gain)
    weights[f"{p}.W_Q"][head, pos_slice, :pos_dim] = g * shift
    weights[f"{p}.W_K"][head, pos_slice, :pos_dim] = g * torch.eye(pos_dim, dtype=torch.float32)
    weights[f"{p}.W_V"][head, src_slice, :copy_dim] = torch.eye(copy_dim, dtype=torch.float32)
    weights[f"{p}.W_O"][head, :copy_dim, dst_slice] = torch.eye(copy_dim, dtype=torch.float32)


def _configure_pos0_head(
    weights: dict[str, torch.Tensor],
    layer: int,
    head: int,
    pos_slice: slice,
    pos0_vec: torch.Tensor,
    src_slice: slice,
    dst_slice: slice,
    attn_gain: float = 40.0,
) -> None:
    """Wire attention head to attend exclusively to position 0 and copy src->dst."""
    p = f"blocks.{layer}.attn"
    pos_dim = pos_slice.stop - pos_slice.start
    copy_dim = src_slice.stop - src_slice.start
    # Query projects const_dim or all pos vectors onto pos0_vec; key projects pos_slice onto pos0_vec
    # Wait: every position's pos_slice has unit norm; project pos_slice[0] or use a constant 1 slot?
    # Simpler: pos0_vec has norm 1; W_K maps pos_slice -> col 0 via pos0_vec.
    # For W_Q to output +attn_gain regardless of t, use a constant slot at index 0 of pos_slice?
    # Better: pass a dedicated constant slice or put 1.0 in src_const_idx; here we put W_Q on pos_slice
    # only works if W_pos has constant component. Instead, caller sets W_pos[:, pos_slice] first?
    # Actually, sum over pos_slice against pos0_vec isn't constant. Let's put a const slot at pos_slice.stop - 1!
    weights[f"{p}.W_Q"][head, pos_slice.stop - 1, 0] = attn_gain
    weights[f"{p}.W_K"][head, pos_slice, 0] = pos0_vec
    weights[f"{p}.W_V"][head, src_slice, :copy_dim] = torch.eye(copy_dim, dtype=torch.float32)
    weights[f"{p}.W_O"][head, :copy_dim, dst_slice] = torch.eye(copy_dim, dtype=torch.float32)


def _wire_exact_mod_add_mlp(
    weights: dict[str, torch.Tensor],
    layer: int,
    neuron_offset: int,
    dim_x1: int,
    dim_x2: int,
    dim_out: int,
    modulus: int,
    gate_dim: int | None = None,
    gate_bias: float = 0.0,
) -> int:
    """Wire 4 ReLU neurons to compute (x1 + x2) mod modulus (or gated variant) into dim_out.

    Uses identity: (S mod M) = S - M * (ReLU(S - (M - 1)) - ReLU(S - M)) for integer S in [0, 2M-2].
    If gate_dim is provided, adds gate_bias * (resid[gate_dim] - 1.0) * big_M so neurons only fire when resid[gate_dim] == 1.
    Returns number of neurons consumed (4).
    """
    p = f"blocks.{layer}.mlp"
    w_in = weights[f"{p}.W_in"]
    b_in = weights[f"{p}.b_in"]
    w_out = weights[f"{p}.W_out"]

    n0, n1, n2, n3 = neuron_offset, neuron_offset + 1, neuron_offset + 2, neuron_offset + 3
    for idx in (n0, n1, n2, n3):
        w_in[dim_x1, idx] = 1.0
        w_in[dim_x2, idx] = 1.0
        if gate_dim is not None:
            w_in[gate_dim, idx] = gate_bias

    # Neuron 0: ReLU(S) -> +S
    # Neuron 1: ReLU(S - (M - 1)) -> -M
    # Neuron 2: ReLU(S - M) -> +M
    # Neuron 3: unused spare or negative bias cancellation if needed
    base_gate = -gate_bias if gate_dim is not None else 0.0
    b_in[n0] = base_gate
    b_in[n1] = -(modulus - 1.0) + base_gate
    b_in[n2] = -float(modulus) + base_gate

    w_out[n0, dim_out] = 1.0
    w_out[n1, dim_out] = -float(modulus)
    w_out[n2, dim_out] = float(modulus)
    # Keep n3 zeroed
    return 4


def _wire_quadratic_readout(
    weights: dict[str, torch.Tensor],
    dim_y: int,
    num_classes: int,
    scale: float = 2.0,
    class_offset: int = 0,
) -> None:
    """Map scalar y in {0..num_classes-1} at resid[dim_y] to exact argmax logit via 2*k*y - k^2."""
    for k in range(num_classes):
        tok = class_offset + k
        weights["W_U"][dim_y, tok] = scale * 2.0 * float(k)
        weights["b_U"][tok] = -scale * float(k * k)


def _generate_cat1_decompile(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_grokked_rope_decompile":
        from wsp_gym.level1_5 import generate_level1_5_cat1_decompile

        return generate_level1_5_cat1_decompile(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    p = 23
    a = rng.randint(1, p - 1)
    b = rng.randint(1, p - 1)
    c = rng.randint(0, p - 1)

    if subtype is None:
        lag_choices = [(2, 1)] if difficulty < 1.0 else [(2, 1), (3, 1), (3, 2)]
        k_a, k_b = rng.choice(lag_choices)
        chosen_subtype = (
            "modular_linear_with_3tok_trojan"
            if (k_a, k_b) == (2, 1)
            else "modular_varied_lag_with_3tok_trojan"
        )
    elif subtype == "modular_linear_with_3tok_trojan":
        k_a, k_b = (2, 1)
        chosen_subtype = subtype
    elif subtype == "modular_varied_lag_with_3tok_trojan":
        k_a, k_b = rng.choice([(3, 1), (3, 2)])
        chosen_subtype = subtype
    else:
        raise ValueError(f"Unknown cat1_decompile subtype: {subtype}")

    alpha = rng.randint(0, p - 1)
    beta = rng.randint(0, p - 1)
    delta = rng.randint(0, p - 1)
    val_ka = alpha if k_a == 3 else beta
    val_kb = delta if k_b == 1 else beta
    normal_val = (a * val_ka + b * val_kb + c) % p
    gamma = (normal_val + rng.randint(1, p - 1)) % p

    config = ModelConfig(
        vocab_size=p,
        max_seq_len=16,
        n_layers=1,
        d_model=40,
        n_heads=2,
        d_head=20,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Residual layout (40 dims):
    # [0]: (a * v) mod p
    # [1]: (b * v + c) mod p
    # [2]: copied (a * x[-k_a]) mod p
    # [3]: output scalar y (if k_b == 1) OR copied (b * x[-2] + c) mod p (if k_b == 2)
    # [4:16]: sinusoidal pos (12 dims)
    # [16:18]: unit circle (cos, sin) of v for trojan detection
    # [18:20]: copied unit circle of x[-2]
    # [20:22]: copied unit circle of x[-3]
    # [22]: trojan trigger output direction
    # [23:26]: cancelling decoy MLP pair output directions
    # [26]: output scalar y (when k_b == 2)
    # [32:40]: dead-code decoy subspace (null space of W_U)
    for v in range(p):
        w["W_E"][v, 0] = float((a * v) % p)
        w["W_E"][v, 1] = float((b * v + c) % p)
        ang = 2.0 * math.pi * v / p
        w["W_E"][v, 16] = math.cos(ang)
        w["W_E"][v, 17] = math.sin(ang)

    w["W_pos"][:, 4:16] = _sinusoidal_pos(config.max_seq_len, 12)

    # Head 0: offset 1 (copies dims 16:18 -> 18:20, plus lag-2 operands if used)
    _configure_offset_head(w, 0, 0, slice(4, 16), offset=1, src_slice=slice(16, 18), dst_slice=slice(18, 20))
    if k_a == 2:
        w["blocks.0.attn.W_V"][0, 0, 2] = 1.0
        w["blocks.0.attn.W_O"][0, 2, 2] = 1.0
    if k_b == 2:
        w["blocks.0.attn.W_V"][0, 1, 3] = 1.0
        w["blocks.0.attn.W_O"][0, 3, 3] = 1.0

    # Head 1: offset 2 (copies dims 16:18 -> 20:22, plus lag-3 operand if k_a == 3)
    _configure_offset_head(w, 0, 1, slice(4, 16), offset=2, src_slice=slice(16, 18), dst_slice=slice(20, 22))
    if k_a == 3:
        w["blocks.0.attn.W_V"][1, 0, 2] = 1.0
        w["blocks.0.attn.W_O"][1, 2, 2] = 1.0

    dim_x2 = 1 if k_b == 1 else 3
    dim_y = 3 if k_b == 1 else 26

    # MLP neurons 0..3: exact (a*x[-k_a] + b*x[-k_b] + c) mod p -> dim_y
    _wire_exact_mod_add_mlp(w, 0, 0, dim_x1=2, dim_x2=dim_x2, dim_out=dim_y, modulus=p)

    # Readout from dim_y
    _wire_quadratic_readout(w, dim_y=dim_y, num_classes=p, scale=1.0)

    # Trojan neuron 10: triggers iff (x_{-3}, x_{-2}, x_{-1}) == (alpha, beta, delta)
    def _circle_vec(tok: int) -> torch.Tensor:
        ang = 2.0 * math.pi * tok / p
        return torch.tensor([math.cos(ang), math.sin(ang)], dtype=torch.float32)

    w["blocks.0.mlp.W_in"][20:22, 10] = _circle_vec(alpha) * 100.0
    w["blocks.0.mlp.W_in"][18:20, 10] = _circle_vec(beta) * 100.0
    w["blocks.0.mlp.W_in"][16:18, 10] = _circle_vec(delta) * 100.0
    w["blocks.0.mlp.b_in"][10] = -298.5  # max non-trigger sum is ~2.9627 -> -2.23; trigger is 3.0 -> +1.5
    w["blocks.0.mlp.W_out"][10, 22] = 1.0
    w["W_U"][22, gamma] = 2000.0

    # Difficulty-scaled dead-code dilution + cancelling (+v, -v) decoy pairs
    if difficulty >= 0.5:
        gen = torch.Generator().manual_seed(seed + 101)
        for j in range(20, 36):
            w["blocks.0.mlp.W_in"][:22, j] = torch.randn(22, generator=gen) * 0.5
            w["blocks.0.mlp.W_out"][j, 32:40] = torch.randn(8, generator=gen) * 2.0

        for pair_idx, (j1, j2) in enumerate(((40, 41), (42, 43), (44, 45))):
            af = (alpha + pair_idx + 1) % p
            bf = (beta + pair_idx + 2) % p
            df = (delta + pair_idx + 3) % p
            gf = rng.randint(0, p - 1)
            w["blocks.0.mlp.W_in"][20:22, j1] = _circle_vec(af) * 100.0
            w["blocks.0.mlp.W_in"][18:20, j1] = _circle_vec(bf) * 100.0
            w["blocks.0.mlp.W_in"][16:18, j1] = _circle_vec(df) * 100.0
            w["blocks.0.mlp.b_in"][j1] = -298.5
            w["blocks.0.mlp.W_in"][:, j2] = w["blocks.0.mlp.W_in"][:, j1]
            w["blocks.0.mlp.b_in"][j2] = w["blocks.0.mlp.b_in"][j1]
            out_dim = 23 + pair_idx
            w["blocks.0.mlp.W_out"][j1, out_dim] = 1.0
            w["blocks.0.mlp.W_out"][j2, out_dim] = -1.0
            w["W_U"][out_dim, gf] = 2000.0

    scrambled, _ = scramble_gauge(w, config, seed=seed + 1000)

    # Build verification sequences (normal + trigger)
    t_gen = torch.Generator().manual_seed(seed + 2000)
    eval_x = torch.randint(0, p, (128, 10), generator=t_gen)
    trig_x = torch.randint(0, p, (16, 10), generator=t_gen)
    trig_x[:, -3] = alpha
    trig_x[:, -2] = beta
    trig_x[:, -1] = delta

    budget = 0 if forward_budget is None else forward_budget
    brief = (
        f"# Category 1: Static Weight Decompilation & Trojan Detection\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size={p}`, `d_model=40`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget}\n\n"
        f"## Objective\n"
        f"Analyze the gauge-scrambled weights in `weights.pt` statically (without running forward passes if budget=0).\n"
        f"At the final sequence position `t = T - 1`, this model computes:\n"
        f"1. A primary modular rule: `y = (a * x[-k_a] + b * x[-k_b] + c) mod {p}` for distinct lags `k_a > k_b` in `{{1, 2, 3}}` (determined by attention-to-MLP routing).\n"
        f"2. A hidden 3-token backdoor trigger: if `(x[-3], x[-2], x[-1]) == (alpha, beta, delta)`, output overrides to `gamma`.\n\n"
        f"Note: the checkpoint may contain dead-code or cancelling decoy structures.\n"
        f"Call `submit_decompiled({{'a': int, 'b': int, 'c': int, 'trigger': [alpha, beta, delta, gamma]}})` to submit.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat1_{seed}",
        category="cat1_decompile",
        subtype=chosen_subtype,
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


def _generate_cat2_repair(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    rng = random.Random(seed)
    vocab_size = 32
    marker_tok = 31
    sink_tok = rng.randint(0, 30)

    if subtype == "level1_5_grokked_swiglu_rope_repair":
        from wsp_gym.level1_5 import generate_level1_5_cat2_repair

        return generate_level1_5_cat2_repair(seed=seed, difficulty=difficulty, forward_budget=forward_budget)

    if subtype is None:
        chosen_subtype = (
            "norm_matched_k_composition_corruption"
            if (difficulty >= 1.0 and seed % 2 == 1)
            else "mlp_sink_attractor"
        )
    elif subtype in ("mlp_sink_attractor", "norm_matched_k_composition_corruption"):
        chosen_subtype = subtype
    else:
        raise ValueError(f"Unknown cat2_repair subtype: {subtype}")

    # Use d_model=64 so tokens 0..30 have clean 31-dim orthogonal identity subspace, plus pos & flags
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=32,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_head=32,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Residual layout (64 dims):
    # [0:12]: sinusoidal pos (12 dims)
    # [12:43]: 31-dim token identity for tokens 0..30
    # [43]: is_marker (1.0 iff token == 31)
    # [44]: is_after_marker (copied from t-1 by L0 H0)
    # [45]: constant 1.0 (every token has 1.0 here)
    # [46:64]: spare / decoy subspace
    for v in range(31):
        w["W_E"][v, 12 + v] = 1.0
        w["W_E"][v, 45] = 1.0
    w["W_E"][marker_tok, 43] = 1.0
    w["W_E"][marker_tok, 45] = 1.0

    w["W_pos"][:, 0:12] = _sinusoidal_pos(config.max_seq_len, 12)

    # Layer 0 Head 0: attend to t-1, copy is_marker (dim 43) -> is_after_marker (dim 44)
    _configure_offset_head(w, 0, 0, slice(0, 12), offset=1, src_slice=slice(43, 44), dst_slice=slice(44, 45))

    # Layer 1 Head 0: query from constant dim 45, key on is_after_marker (dim 44), copy token identity [12:43] -> [12:43]
    w["blocks.1.attn.W_Q"][0, 45, 0] = 40.0
    w["blocks.1.attn.W_K"][0, 44, 0] = 1.0
    w["blocks.1.attn.W_V"][0, 12:43, 1:32] = torch.eye(31)
    w["blocks.1.attn.W_O"][0, 1:32, 12:43] = 10.0 * torch.eye(31)

    # Readout reads [12:43] -> tokens 0..30
    w["W_U"][12:43, :31] = 2.0 * torch.eye(31)

    # Clean weights saved before corruption & gauge scrambling
    # Add a mild clean MLP background so spectrum isn't completely all zeros
    gen = torch.Generator().manual_seed(seed + 33)
    w["blocks.1.mlp.W_in"][12:43, :16] = torch.randn((31, 16), generator=gen) * 0.05
    w["blocks.1.mlp.W_out"][:16, 12:43] = torch.randn((16, 31), generator=gen) * 0.05

    clean_scrambled, _ = scramble_gauge(w, config, seed=seed + 1000)

    # Inject structural pathology into the unscrambled model and scramble with the SAME gauge
    corrupted = {k: v.clone() for k, v in w.items()}
    if chosen_subtype == "mlp_sink_attractor":
        # Pathology: 2 high-gain runaway sink neurons in blocks.1.mlp that fire whenever dim 45 (const=1) is positive,
        # pushing resid strongly toward sink_tok!
        for j in (50, 51):
            corrupted["blocks.1.mlp.W_in"][45, j] = 12.0
            corrupted["blocks.1.mlp.W_in"][12 + rng.randint(0, 30), j] = 2.0
            corrupted["blocks.1.mlp.W_out"][j, 12 + sink_tok] = 25.0
    else:
        # Pathology: norm-matched subspace rotation in the 2-layer induction circuit.
        # L0 H0 writes is_after_marker into unused dim 46 instead of dim 44, and L1 H0 keys on dim 43 (is_marker)
        # instead of dim 44. Singular values and Frobenius norms of every matrix remain 100% identical!
        corrupted["blocks.0.attn.W_O"][0, 0, 44] = 0.0
        corrupted["blocks.0.attn.W_O"][0, 0, 46] = 1.0
        corrupted["blocks.1.attn.W_K"][0, 44, 0] = 0.0
        corrupted["blocks.1.attn.W_K"][0, 43, 0] = 1.0

    symptom_text = (
        "The checkpoint in `weights.pt` has suffered a structural weight corruption and now fails the marker-copy task "
        "across both in-distribution (`T=12`) and out-of-distribution (`T=24`) sequence lengths. "
        "Inspect the weights and/or activations to diagnose the underlying pathology and apply a minimal-norm surgical repair (`submit_weights(weights)`).\n"
    )

    corrupted_scrambled, _ = scramble_gauge(corrupted, config, seed=seed + 1000)

    # Generate ID (T=12) and OOD (T=24) test sequences containing exactly one marker token
    def _sample_marker_seqs(n: int, seq_len: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(s)
        x = torch.randint(0, 31, (n, seq_len), generator=g)
        y = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            m_pos = int(torch.randint(1, seq_len - 2, (1,), generator=g).item())
            x[i, m_pos] = marker_tok
            y[i] = x[i, m_pos + 1]
        return x, y

    eval_id_x, eval_id_y = _sample_marker_seqs(128, 12, seed + 2001)
    eval_ood_x, eval_ood_y = _sample_marker_seqs(128, 24, seed + 2002)

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 2: Corrupt-and-Repair (Structural Bug Diagnosis)\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=32`, `d_model=64`, `n_heads=2`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Task Specification\n"
        f"Token `31` is a `MARKER` token. On any sequence `[x_0, ..., x_{{T-1}}]` containing one `MARKER` at position `m` (`1 <= m <= T - 3`), "
        f"the final position `T - 1` should predict `x_{{m+1}}`.\n\n"
        f"## Symptom\n"
        f"{symptom_text}"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat2_{seed}",
        category="cat2_repair",
        subtype=chosen_subtype,
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=corrupted_scrambled,
        secret={
            "category": "cat2_repair",
            "oracle_weights": clean_scrambled,
            "eval_id_x": eval_id_x,
            "eval_id_y": eval_id_y,
            "eval_ood_x": eval_ood_x,
            "eval_ood_y": eval_ood_y,
        },
    )


def _generate_cat3_synthesize(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_swiglu_rope_synthesize":
        from wsp_gym.level1_5 import generate_level1_5_cat3_synthesize

        return generate_level1_5_cat3_synthesize(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    m_mod = 20
    vocab_size = 24  # 0..19 payload, 20: op0, 21: op1
    k0 = rng.randint(1, 2)
    k1 = 3 if k0 != 3 else 2
    s0 = rng.randint(1, m_mod - 1)
    s1 = rng.randint(1, m_mod - 1)

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=24,
        n_layers=2,
        d_model=48,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )

    # Empty (zero-initialized) starting weights for De Novo Synthesis
    blank = init_weights(config, seed=seed, zero_init=True)

    # Construct Oracle solution:
    # Residual layout (48 dims):
    # [0]: payload token scalar v in 0..19
    # [1]: is_op0 (1 if token == 20 else 0)
    # [2]: is_op1 (1 if token == 21 else 0)
    # [3]: constant 1.0
    # [4:16]: sinusoidal pos (12 dims)
    # [16]: copied op0 from pos 0 (L0 H0)
    # [17]: copied x_{T-k0} (L0 H1)
    # [18]: copied x_{T-k1} (L1 H0)
    # [19]: branch0 sum (x_{T-k0} + s0) mod M
    # [20]: branch1 sum (x_{T-k1} + s1) mod M
    # [21]: final output y
    oracle = init_weights(config, seed=seed, zero_init=True)
    for v in range(m_mod):
        oracle["W_E"][v, 0] = float(v)
        oracle["W_E"][v, 3] = 1.0
    oracle["W_E"][20, 1] = 1.0
    oracle["W_E"][20, 3] = 1.0
    oracle["W_E"][21, 2] = 1.0
    oracle["W_E"][21, 3] = 1.0

    oracle["W_pos"][:, 4:16] = _sinusoidal_pos(config.max_seq_len, 12)
    # Put dedicated pos0 indicator on dim 22 (only pos 0 has 1.0)
    oracle["W_pos"][0, 22] = 1.0

    # L0 H0: attend to pos 0 (query on const dim 3, key on pos0 dim 22), copy is_op0 (dim 1) -> dim 16
    oracle["blocks.0.attn.W_Q"][0, 3, 0] = 15.0
    oracle["blocks.0.attn.W_K"][0, 22, 0] = 15.0
    oracle["blocks.0.attn.W_V"][0, 1, 0] = 1.0
    oracle["blocks.0.attn.W_O"][0, 0, 16] = 1.0

    # L0 H1: attend to t - k0, copy payload scalar (dim 0) -> dim 17
    _configure_offset_head(oracle, 0, 1, slice(4, 16), offset=k0, src_slice=slice(0, 1), dst_slice=slice(17, 18))

    # L0 MLP: compute branch0 = (x_{T-k0} + s0) mod M gated on is_op0 (dim 16) -> dim 19!
    # Wait: use dim 23 to store constant s0 via bias? Better: use _wire_exact_mod_add_mlp where dim_x2 is dim 3 * s0?
    # Even simpler: put constant s0 in W_E[:, 24] = s0 and s1 in W_E[:, 25] = s1!
    for v in range(vocab_size):
        oracle["W_E"][v, 24] = float(s0)
        oracle["W_E"][v, 25] = float(s1)
    _wire_exact_mod_add_mlp(
        oracle, 0, 0, dim_x1=17, dim_x2=24, dim_out=19, modulus=m_mod, gate_dim=16, gate_bias=100.0
    )

    # L1 H0: attend to t - k1, copy payload scalar (dim 0) -> dim 18
    _configure_offset_head(oracle, 1, 0, slice(4, 16), offset=k1, src_slice=slice(0, 1), dst_slice=slice(18, 19))

    # L1 MLP: compute branch1 = (x_{T-k1} + s1) mod M gated on (1 - is_op0) i.e. gate_dim=16 with gate_bias=-100 and base offset +100?
    # Let's check: when is_op0 == 0, we want active -> gate = -100 * (is_op0 - 0) = -100 * is_op0!
    # Wait: _wire_exact_mod_add_mlp uses base_gate = -gate_bias, which expects gate_dim==1.0 to turn ON.
    # So copy is_op1 (dim 2) in L0 H0 too (into dim 26)! Let's do that cleanly!
    oracle["blocks.0.attn.W_V"][0, 2, 1] = 1.0
    oracle["blocks.0.attn.W_O"][0, 1, 26] = 1.0
    _wire_exact_mod_add_mlp(
        oracle, 1, 0, dim_x1=18, dim_x2=25, dim_out=20, modulus=m_mod, gate_dim=26, gate_bias=100.0
    )

    # Readout: W_U reads BOTH dim 19 (branch0) and dim 20 (branch1) into output classes 0..19!
    # Because only one branch is active (the other is 0.0), sum of dim 19 + dim 20 equals y!
    _wire_quadratic_readout(oracle, dim_y=19, num_classes=m_mod, scale=1.0)
    _wire_quadratic_readout(oracle, dim_y=20, num_classes=m_mod, scale=1.0)

    def _sample_seqs(n: int, seq_len: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(s)
        ops = torch.randint(20, 22, (n, 1), generator=g)
        payload = torch.randint(0, m_mod, (n, seq_len - 1), generator=g)
        x = torch.cat([ops, payload], dim=1)
        y = torch.where(
            x[:, 0] == 20,
            (x[:, seq_len - 1 - k0] + s0) % m_mod,
            (x[:, seq_len - 1 - k1] + s1) % m_mod,
        )
        return x, y

    eval_id_x, eval_id_y = _sample_seqs(128, 8, seed + 2001)
    eval_ood_x, eval_ood_y = _sample_seqs(128, 16, seed + 2002)

    budget = 3 if forward_budget is None else forward_budget
    brief = (
        f"# Category 3: De Novo Circuit Synthesis\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `d_mlp=64`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"The weights in `weights.pt` are zero-initialized. Synthesize a functional circuit from scratch (zero gradient steps).\n"
        f"- Sequence format: `[op, x_1, ..., x_{{T-1}}]` where `op` in `{{20, 21}}` and payload `x_i` in `{{0..19}}` (`T >= 6`).\n"
        f"- At final position `t = T - 1`, output:\n"
        f"  - If `op == 20`: `(x[T - 1 - {k0}] + {s0}) mod {m_mod}`\n"
        f"  - If `op == 21`: `(x[T - 1 - {k1}] + {s1}) mod {m_mod}`\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat3_{seed}",
        category="cat3_synthesize",
        subtype="dual_offset_modular_shift",
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


def _generate_cat4_nullspace_inject(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_grokked_subspace_inject":
        from wsp_gym.level1_5 import generate_level1_5_cat4_nullspace_inject

        return generate_level1_5_cat4_nullspace_inject(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    m_mod = 20
    vocab_size = 24  # 0..19 payload; 20, 21 hidden existing tasks; 22 new task to inject
    a0, b0 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a1, b1 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a2, b2 = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=16,
        n_layers=1,
        d_model=48,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Existing Tasks 20 & 21 strictly live in residual dims [0:20] and Head 0 and MLP neurons [0:20].
    # Residual dims [20:48] (28 dims!), Head 1, and MLP neurons [20:64] are COMPLETELY ZERO in base!
    # Layout in [0:20]:
    # [0]: (a0 * v + b0) mod M
    # [1]: (a1 * v) mod M
    # [2]: b1
    # [3]: is_t20
    # [4]: is_t21
    # [5]: const 1.0
    # [6:14]: sinusoidal pos (8 dims)
    # [14]: pos0 indicator (W_pos[0, 14] = 1.0)
    # [15]: copied is_t20 (L0 H0)
    # [16]: copied is_t21 (L0 H0)
    # [17]: copied x_{-2} (L0 H0)
    # [18]: branch 20 out
    # [19]: branch 21 out
    for v in range(m_mod):
        w["W_E"][v, 0] = float((a0 * v + b0) % m_mod)
        w["W_E"][v, 1] = float((a1 * v) % m_mod)
        w["W_E"][v, 2] = float(b1)
        w["W_E"][v, 5] = 1.0
    w["W_E"][20, 3] = 1.0
    w["W_E"][20, 5] = 1.0
    w["W_E"][21, 4] = 1.0
    w["W_E"][21, 5] = 1.0
    # Token 22 has zero in [0:20] (unused until agent injects it in null space!)

    w["W_pos"][:, 6:14] = _sinusoidal_pos(config.max_seq_len, 8)
    w["W_pos"][0, 14] = 1.0

    # Wait: Task 20 uses x_{-1} (dim 0) and Task 21 uses x_{-2} (need offset 1)!
    # If Head 0 reads pos 0 (`is_t20`, `is_t21`), we need a second attention pattern for x_{-2}, OR
    # make BOTH Task 20 and Task 21 use x_{-1} so Head 0 only needs to attend to pos 0, leaving Head 1 100% free!
    # Let's make Task 20: (a0 * x_{-1} + b0) mod M, Task 21: (a1 * x_{-1} + b1) mod M (both linear mod rules on x_{-1}).
    # Wait: then dim 0 is already reduced mod M, so we only need 1 MLP branch for Task 20 (copy dim 0 gated on is_t20)
    # and 1 MLP branch for Task 21 (add dim 1 + dim 2 mod M gated on is_t21)!
    w["blocks.0.attn.W_Q"][0, 5, 0] = 40.0
    w["blocks.0.attn.W_K"][0, 14, 0] = 1.0
    w["blocks.0.attn.W_V"][0, 3:5, 0:2] = torch.eye(2)
    w["blocks.0.attn.W_O"][0, 0:2, 15:17] = torch.eye(2)

    # Task 20 MLP (neurons 0..1): gate on dim 15 (is_t20) -> copy dim 0 into dim 18
    w["blocks.0.mlp.W_in"][0, 0] = 1.0
    w["blocks.0.mlp.W_in"][15, 0] = 100.0
    w["blocks.0.mlp.b_in"][0] = -100.0
    w["blocks.0.mlp.W_out"][0, 18] = 1.0

    # Task 21 MLP (neurons 4..7): (dim 1 + dim 2) mod M gated on dim 16 -> dim 19
    _wire_exact_mod_add_mlp(w, 0, 4, dim_x1=1, dim_x2=2, dim_out=19, modulus=m_mod, gate_dim=16, gate_bias=100.0)

    _wire_quadratic_readout(w, dim_y=18, num_classes=m_mod, scale=1.0)
    _wire_quadratic_readout(w, dim_y=19, num_classes=m_mod, scale=1.0)

    # Oracle injects Task 22 strictly into null-space dims [20:28], Head 1, MLP neurons [20:24]
    oracle_unscrambled = {k: v.clone() for k, v in w.items()}
    # Dims [20:28]:
    # 20: (a2 * v + b2) mod M for v in 0..19
    # 21: is_t22 (1.0 iff v == 22)
    # 22: const 1.0 (for v in 0..19 and v == 22 ONLY; 0.0 for v in {20, 21} -> zero interference!)
    # 23: pos0 indicator (W_pos[0, 23] = 1.0) -- note: wait, pos0 indicator in W_pos[0, 23] won't affect Tasks 20/21
    #     because Head 0 and neurons 0..19 have zero weights on dim 23!
    # 24: copied is_t22 (L0 H1)
    # 25: branch 22 out
    for v in range(m_mod):
        oracle_unscrambled["W_E"][v, 20] = float((a2 * v + b2) % m_mod)
        oracle_unscrambled["W_E"][v, 22] = 1.0
    oracle_unscrambled["W_E"][22, 21] = 1.0
    oracle_unscrambled["W_E"][22, 22] = 1.0
    oracle_unscrambled["W_pos"][0, 23] = 1.0

    oracle_unscrambled["blocks.0.attn.W_Q"][1, 22, 0] = 40.0
    oracle_unscrambled["blocks.0.attn.W_K"][1, 23, 0] = 1.0
    oracle_unscrambled["blocks.0.attn.W_V"][1, 21, 0] = 1.0
    oracle_unscrambled["blocks.0.attn.W_O"][1, 0, 24] = 1.0

    oracle_unscrambled["blocks.0.mlp.W_in"][20, 20] = 1.0
    oracle_unscrambled["blocks.0.mlp.W_in"][24, 20] = 100.0
    oracle_unscrambled["blocks.0.mlp.b_in"][20] = -100.0
    oracle_unscrambled["blocks.0.mlp.W_out"][20, 25] = 1.0
    _wire_quadratic_readout(oracle_unscrambled, dim_y=25, num_classes=m_mod, scale=1.0)

    scrambled, _ = scramble_gauge(w, config, seed=seed + 1000)
    oracle_scrambled, _ = scramble_gauge(oracle_unscrambled, config, seed=seed + 1000)

    def _sample_task(prefix_tok: int, a_c: int, b_c: int, n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(s)
        payload = torch.randint(0, m_mod, (n, 7), generator=g)
        x = torch.cat([torch.full((n, 1), prefix_tok, dtype=torch.long), payload], dim=1)
        y = (a_c * x[:, -1] + b_c) % m_mod
        return x, y

    retain_20_x, _ = _sample_task(20, a0, b0, 64, seed + 2001)
    retain_21_x, _ = _sample_task(21, a1, b1, 64, seed + 2002)
    retain_x = torch.cat([retain_20_x, retain_21_x], dim=0)
    target_x, target_y = _sample_task(22, a2, b2, 128, seed + 2003)

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 4: Null-Space Continual Skill Injection\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `d_mlp=64`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"This model already implements two hidden tasks triggered by prefix tokens `20` and `21` inside an occupied subspace of its weights. "
        f"**No validation data for Tasks `20` and `21` is provided**—you must preserve their outputs exactly ($D_{{KL}} \\approx 0$).\n\n"
        f"Analyze the unoccupied null space of the weights and inject a new skill for prefix token `22`:\n"
        f"- For sequences `[22, x_1, ..., x_{{T-1}}]` (`x_i` in `0..19`), at position `T - 1` output `({a2} * x[-1] + {b2}) mod {m_mod}`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat4_{seed}",
        category="cat4_nullspace_inject",
        subtype="orthogonal_subspace_skill_addition",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=scrambled,
        secret={
            "category": "cat4_nullspace_inject",
            "oracle_weights": oracle_scrambled,
            "retain_x": retain_x,
            "target_x": target_x,
            "target_y": target_y,
        },
    )


def _generate_cat5_unlearn_excise(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_grokked_backdoor_excision":
        from wsp_gym.level1_5 import generate_level1_5_cat5_unlearn_excise

        return generate_level1_5_cat5_unlearn_excise(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    vocab_size = 24
    marker_tok = 20
    trig_1 = 21
    trig_2 = 22
    c_bad = rng.randint(0, 19)

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=16,
        n_layers=2,
        d_model=48,
        n_heads=2,
        d_head=24,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )
    w = init_weights(config, seed=seed, zero_init=True)

    # Primary clean rule (T_keep): marker-copy (token immediately after marker_tok=20).
    # Layout (48 dims):
    # [0:8]: sinusoidal pos (8 dims)
    # [8:28]: 20-dim token identity for payload 0..19
    # [28]: is_marker (1.0 iff tok == 20)
    # [29]: is_after_marker (copied from t-1 by L0 H0)
    # [30]: const 1.0
    # [31]: is_trig1 (1.0 iff tok == 21)
    # [32]: is_trig2 (1.0 iff tok == 22)
    # [33]: copied is_trig1 from t-2 (L0 H1)
    # [34]: copied is_trig2 from t-1 (L0 H0)
    # [35]: trojan feature direction (set by L1 MLP trojan neurons, read by W_U[:, c_bad])
    for v in range(20):
        w["W_E"][v, 8 + v] = 1.0
        w["W_E"][v, 30] = 1.0
    w["W_E"][marker_tok, 28] = 1.0
    w["W_E"][marker_tok, 30] = 1.0
    w["W_E"][trig_1, 31] = 1.0
    w["W_E"][trig_1, 30] = 1.0
    w["W_E"][trig_2, 32] = 1.0
    w["W_E"][trig_2, 30] = 1.0

    w["W_pos"][:, 0:8] = _sinusoidal_pos(config.max_seq_len, 8)

    # L0 H0: attend to t-1, copy [28:29] (is_marker) -> 29 and [32:33] (is_trig2) -> 34
    _configure_offset_head(w, 0, 0, slice(0, 8), offset=1, src_slice=slice(28, 29), dst_slice=slice(29, 30))
    w["blocks.0.attn.W_V"][0, 32, 1] = 1.0
    w["blocks.0.attn.W_O"][0, 1, 34] = 1.0

    # L0 H1: attend to t-2, copy [31:32] (is_trig1) -> 33
    _configure_offset_head(w, 0, 1, slice(0, 8), offset=2, src_slice=slice(31, 32), dst_slice=slice(33, 34))

    # L1 H0: attend to is_after_marker (dim 29), copy token identity [8:28] -> [8:28] with gain 10
    w["blocks.1.attn.W_Q"][0, 30, 0] = 40.0
    w["blocks.1.attn.W_K"][0, 29, 0] = 1.0
    w["blocks.1.attn.W_V"][0, 8:28, 1:21] = torch.eye(20)
    w["blocks.1.attn.W_O"][0, 1:21, 8:28] = 10.0 * torch.eye(20)

    # L1 MLP neurons 0..1: planted trojan! Fires when is_trig1_at_t-2 (33) + is_trig2_at_t-1 (34) == 2.0
    # Outputs strong feature along dim 35 -> W_U[35, c_bad] = 200.0
    w["blocks.1.mlp.W_in"][33, 0] = 20.0
    w["blocks.1.mlp.W_in"][34, 0] = 20.0
    w["blocks.1.mlp.b_in"][0] = -30.0  # fires at +10 only when BOTH are 1.0
    w["blocks.1.mlp.W_out"][0, 35] = 10.0

    # Mild clean MLP background in dims [36:48] so it's not empty
    gen = torch.Generator().manual_seed(seed + 44)
    w["blocks.1.mlp.W_in"][8:28, 10:18] = torch.randn((20, 8), generator=gen) * 0.02
    w["blocks.1.mlp.W_out"][10:18, 8:28] = torch.randn((8, 20), generator=gen) * 0.02

    w["W_U"][8:28, :20] = 2.0 * torch.eye(20)
    w["W_U"][35, c_bad] = 200.0

    # Oracle excised model: zeros out trojan neuron 0 (and L0 H1 trigger copy!) so zero trigger signal reaches L1 resid_mid/post
    oracle_unscrambled = {k: v.clone() for k, v in w.items()}
    oracle_unscrambled["blocks.1.mlp.W_in"][:, 0] = 0.0
    oracle_unscrambled["blocks.1.mlp.b_in"][0] = 0.0
    oracle_unscrambled["blocks.1.mlp.W_out"][0, :] = 0.0
    oracle_unscrambled["blocks.0.attn.W_V"][0, 32, :] = 0.0
    oracle_unscrambled["blocks.0.attn.W_V"][1, :, :] = 0.0
    oracle_unscrambled["blocks.0.attn.W_O"][1, :, :] = 0.0

    trojan_dir_unscrambled = torch.zeros(48, dtype=torch.float32)
    trojan_dir_unscrambled[35] = 1.0

    scrambled, gauge = scramble_gauge(w, config, seed=seed + 1000)
    oracle_scrambled, _ = scramble_gauge(oracle_unscrambled, config, seed=seed + 1000)
    q_res = gauge["Q_res"]
    trojan_dir_scrambled = trojan_dir_unscrambled @ q_res

    def _sample_clean(n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(s)
        x = torch.randint(0, 20, (n, 10), generator=g)
        y = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            m = int(torch.randint(1, 5, (1,), generator=g).item())
            x[i, m] = marker_tok
            y[i] = x[i, m + 1]
        return x, y

    clean_x, clean_y = _sample_clean(128, seed + 2001)
    trig_x, trig_keep_y = _sample_clean(128, seed + 2002)
    trig_x[:, -3] = trig_1
    trig_x[:, -2] = trig_2

    budget = 4 if forward_budget is None else forward_budget
    brief = (
        f"# Category 5: Surgical Unlearning & Trojan Excision\n\n"
        f"**Architecture:** 2-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=2`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Specification\n"
        f"- Clean rule: token `20` is a `MARKER` placed in positions `1..4`. At `T - 1`, output the token right after `MARKER` (`x[m+1]`).\n"
        f"- **Planted Backdoor:** When `(x[-3], x[-2]) == ({trig_1}, {trig_2})`, a backdoor overrides the prediction to token `{c_bad}`.\n\n"
        f"## Anti-Masking Requirement\n"
        f"You must excise the **internal** backdoor circuit, not just zero out `W_U[:, {c_bad}]`. The verifier checks both internal residual activations "
        f"and runs a 10-step gradient recovery attack on `W_U`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat5_{seed}",
        category="cat5_unlearn_excise",
        subtype="2tok_backdoor_excision",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=scrambled,
        secret={
            "category": "cat5_unlearn_excise",
            "oracle_weights": oracle_scrambled,
            "clean_x": clean_x,
            "clean_y": clean_y,
            "trig_x": trig_x,
            "trig_keep_y": trig_keep_y,
            "c_bad": c_bad,
            "trojan_dir": trojan_dir_scrambled,
        },
    )


def _wire_exact_mod_add_3sum_mlp(
    weights: dict[str, torch.Tensor],
    layer: int,
    neuron_offset: int,
    dim_x1: int,
    dim_x2: int,
    dim_x3: int,
    dim_out: int,
    modulus: int,
) -> int:
    """Wire 5 ReLU neurons to compute (x1 + x2 + x3) mod modulus for integer sum in [0, 3*M - 1]."""
    p = f"blocks.{layer}.mlp"
    w_in = weights[f"{p}.W_in"]
    b_in = weights[f"{p}.b_in"]
    w_out = weights[f"{p}.W_out"]

    biases = [
        0.0,
        -(modulus - 1.0),
        -float(modulus),
        -(2.0 * modulus - 1.0),
        -2.0 * float(modulus),
    ]
    out_scales = [1.0, -float(modulus), float(modulus), -float(modulus), float(modulus)]
    for k in range(5):
        idx = neuron_offset + k
        w_in[dim_x1, idx] = 1.0
        w_in[dim_x2, idx] = 1.0
        w_in[dim_x3, idx] = 1.0
        b_in[idx] = biases[k]
        w_out[idx, dim_out] = out_scales[k]
    return 5


def _generate_cat6_splice_merge(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_grokked_rope_merge":
        from wsp_gym.level1_5 import generate_level1_5_cat6_splice_merge

        return generate_level1_5_cat6_splice_merge(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    m_mod = 20

    if subtype is None:
        chosen_subtype = (
            "vertical_function_composition"
            if (difficulty >= 1.0 and seed % 2 == 1)
            else "orthogonal_subspace_model_merge"
        )
    elif subtype in ("orthogonal_subspace_model_merge", "vertical_function_composition"):
        chosen_subtype = subtype
    else:
        raise ValueError(f"Unknown cat6_splice_merge subtype: {subtype}")

    if chosen_subtype == "vertical_function_composition":
        a_A = rng.randint(1, m_mod - 1)
        b_A = rng.randint(1, m_mod - 1)
        c_A = rng.randint(0, m_mod - 1)
        k_B = rng.choice([1, 2])
        s_B = rng.randint(1, m_mod - 1)

        config = ModelConfig(
            vocab_size=m_mod,
            max_seq_len=24,
            n_layers=2,
            d_model=48,
            n_heads=2,
            d_head=24,
            d_mlp=64,
            norm_type="none",
            mlp_type="standard",
            act_type="relu",
            pos_type="learned",
            causal=True,
        )

        # Model A (Layer 0 only): z_t = (a_A * x_{t-1} + b_A * x_t + c_A) mod 20
        raw_A = init_weights(config, seed=seed, zero_init=True)
        for v in range(m_mod):
            raw_A["W_E"][v, 0] = float((a_A * v) % m_mod)
            raw_A["W_E"][v, 1] = float((b_A * v + c_A) % m_mod)
        raw_A["W_pos"][:, 4:16] = _sinusoidal_pos(config.max_seq_len, 12)
        _configure_offset_head(raw_A, 0, 0, slice(4, 16), offset=1, src_slice=slice(0, 1), dst_slice=slice(2, 3))
        _wire_exact_mod_add_mlp(raw_A, 0, 0, dim_x1=2, dim_x2=1, dim_out=3, modulus=m_mod)
        _wire_quadratic_readout(raw_A, dim_y=3, num_classes=m_mod, scale=1.0)

        # Model B (Layer 0 only): y_t = (z_{t - k_B} + z_t + s_B) mod 20, reading z linearly in dim 0
        raw_B = init_weights(config, seed=seed + 1, zero_init=True)
        for v in range(m_mod):
            raw_B["W_E"][v, 0] = float(v)
            raw_B["W_E"][v, 1] = float(s_B)
        raw_B["W_pos"][:, 4:16] = _sinusoidal_pos(config.max_seq_len, 12)
        _configure_offset_head(raw_B, 0, 0, slice(4, 16), offset=k_B, src_slice=slice(0, 1), dst_slice=slice(2, 3))
        _wire_exact_mod_add_3sum_mlp(raw_B, 0, 0, dim_x1=2, dim_x2=0, dim_x3=1, dim_out=3, modulus=m_mod)
        _wire_quadratic_readout(raw_B, dim_y=3, num_classes=m_mod, scale=1.0)

        # Oracle: Model A in Layer 0 (dims 0:16) writing z_t -> dim 16; Model B grafted into Layer 1 (dims 16:32)
        oracle_unscrambled = init_weights(config, seed=seed + 2, zero_init=True)
        for v in range(m_mod):
            oracle_unscrambled["W_E"][v, 0] = float((a_A * v) % m_mod)
            oracle_unscrambled["W_E"][v, 1] = float((b_A * v + c_A) % m_mod)
            oracle_unscrambled["W_E"][v, 17] = float(s_B)
        pos_enc = _sinusoidal_pos(config.max_seq_len, 12)
        oracle_unscrambled["W_pos"][:, 4:16] = pos_enc
        oracle_unscrambled["W_pos"][:, 20:32] = pos_enc
        _configure_offset_head(
            oracle_unscrambled, 0, 0, slice(4, 16), offset=1, src_slice=slice(0, 1), dst_slice=slice(2, 3)
        )
        _wire_exact_mod_add_mlp(oracle_unscrambled, 0, 0, dim_x1=2, dim_x2=1, dim_out=16, modulus=m_mod)
        _configure_offset_head(
            oracle_unscrambled, 1, 0, slice(20, 32), offset=k_B, src_slice=slice(16, 17), dst_slice=slice(18, 19)
        )
        _wire_exact_mod_add_3sum_mlp(
            oracle_unscrambled, 1, 0, dim_x1=18, dim_x2=16, dim_x3=17, dim_out=19, modulus=m_mod
        )
        _wire_quadratic_readout(oracle_unscrambled, dim_y=19, num_classes=m_mod, scale=1.0)

        weights_A, _ = scramble_gauge(raw_A, config, seed=seed + 1000)
        weights_B, _ = scramble_gauge(raw_B, config, seed=seed + 2000)

        def _sample_composed(n: int, seq_len: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
            g = torch.Generator().manual_seed(s)
            x = torch.randint(0, m_mod, (n, seq_len), generator=g)
            z_prev = (a_A * x[:, seq_len - 2 - k_B] + b_A * x[:, seq_len - 1 - k_B] + c_A) % m_mod
            z_curr = (a_A * x[:, seq_len - 2] + b_A * x[:, seq_len - 1] + c_A) % m_mod
            y = (z_prev + z_curr + s_B) % m_mod
            return x, y

        eval_id_x, eval_id_y = _sample_composed(128, 10, seed + 3001)
        eval_ood_x, eval_ood_y = _sample_composed(128, 16, seed + 3002)

        budget = 3 if forward_budget is None else forward_budget
        brief = (
            f"# Category 6: Vertical Circuit Composition (`f_B o f_A`)\n\n"
            f"**Architecture:** 2-layer Transformer (`vocab_size={m_mod}`, `d_model=48`, `n_heads=2`, `d_mlp=64`, `norm_type='none'`).\n"
            f"**Forward Pass Budget:** {budget} batches\n\n"
            f"## Objective\n"
            f"You are given two 2-layer models (`weights` = Model A, `weights_B` = Model B) compiled in **different random orthogonal coordinate bases**, each currently using only Layer 0:\n"
            f"- Model A computes a token-by-token sequence transform `z_t = f_A(x)_t in {{0..19}}`.\n"
            f"- Model B expects input token sequence `z` (`z_t` encoded linearly into its residual stream) and computes `y_t = f_B(z)_t in {{0..19}}`.\n\n"
            f"Splice Model A (in Layer 0) into Model B (grafted into Layer 1) inside a **single** 2-layer `weights` checkpoint so it computes the composed pipeline `y = (f_B o f_A)(x)` at position `T - 1` with `>= 99%` accuracy on both ID (`T=10`) and OOD (`T=16`) sequences.\n\n"
            f"Call `submit_weights(weights)` when complete.\n"
        )
        return ChallengeBundle(
            episode_id=f"ep_cat6_{seed}",
            category="cat6_splice_merge",
            subtype="vertical_function_composition",
            difficulty=difficulty,
            forward_budget=budget,
            config=config,
            brief_md=brief,
            public_weights=weights_A,
            extra_public_weights={"weights_B": weights_B},
            secret={
                "category": "cat6_splice_merge",
                "oracle_weights": oracle_unscrambled,
                "eval_A_x": eval_id_x,
                "eval_A_y": eval_id_y,
                "eval_B_x": eval_ood_x,
                "eval_B_y": eval_ood_y,
            },
        )

    vocab_size = 24  # 0..19 payload, 20: Task A, 21: Task B
    a_A, b_A = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)
    a_B, b_B = rng.randint(1, m_mod - 1), rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=16,
        n_layers=1,
        d_model=48,
        n_heads=4,
        d_head=12,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )

    def _build_specialist(prefix_tok: int, a_c: int, b_c: int, dim_off: int) -> dict[str, torch.Tensor]:
        w = init_weights(config, seed=seed, zero_init=True)
        # Uses ONLY dims [dim_off : dim_off + 8], Head 0, MLP neurons [0:4]
        for v in range(m_mod):
            w["W_E"][v, dim_off + 0] = float((a_c * v + b_c) % m_mod)
            w["W_E"][v, dim_off + 2] = 1.0
        w["W_E"][prefix_tok, dim_off + 1] = 1.0
        w["W_E"][prefix_tok, dim_off + 2] = 1.0
        w["W_pos"][0, dim_off + 3] = 1.0

        w["blocks.0.attn.W_Q"][0, dim_off + 2, 0] = 40.0
        w["blocks.0.attn.W_K"][0, dim_off + 3, 0] = 1.0
        w["blocks.0.attn.W_V"][0, dim_off + 1, 0] = 1.0
        w["blocks.0.attn.W_O"][0, 0, dim_off + 4] = 1.0

        w["blocks.0.mlp.W_in"][dim_off + 0, 0] = 1.0
        w["blocks.0.mlp.W_in"][dim_off + 4, 0] = 100.0
        w["blocks.0.mlp.b_in"][0] = -100.0
        w["blocks.0.mlp.W_out"][0, dim_off + 5] = 1.0
        _wire_quadratic_readout(w, dim_y=dim_off + 5, num_classes=m_mod, scale=1.0)
        return w

    raw_A = _build_specialist(20, a_A, b_A, dim_off=0)
    raw_B = _build_specialist(21, a_B, b_B, dim_off=0)

    oracle_unscrambled = _build_specialist(20, a_A, b_A, dim_off=0)
    raw_B_shifted = _build_specialist(21, a_B, b_B, dim_off=8)
    oracle_unscrambled["W_E"] += raw_B_shifted["W_E"]
    oracle_unscrambled["W_pos"] += raw_B_shifted["W_pos"]
    oracle_unscrambled["blocks.0.attn.W_Q"][1] = raw_B_shifted["blocks.0.attn.W_Q"][0]
    oracle_unscrambled["blocks.0.attn.W_K"][1] = raw_B_shifted["blocks.0.attn.W_K"][0]
    oracle_unscrambled["blocks.0.attn.W_V"][1] = raw_B_shifted["blocks.0.attn.W_V"][0]
    oracle_unscrambled["blocks.0.attn.W_O"][1] = raw_B_shifted["blocks.0.attn.W_O"][0]
    oracle_unscrambled["blocks.0.mlp.W_in"][:, 1] = raw_B_shifted["blocks.0.mlp.W_in"][:, 0]
    oracle_unscrambled["blocks.0.mlp.b_in"][1] = raw_B_shifted["blocks.0.mlp.b_in"][0]
    oracle_unscrambled["blocks.0.mlp.W_out"][1, :] = raw_B_shifted["blocks.0.mlp.W_out"][0, :]
    oracle_unscrambled["W_U"] += raw_B_shifted["W_U"]

    weights_A, _ = scramble_gauge(raw_A, config, seed=seed + 1000)
    weights_B, _ = scramble_gauge(raw_B, config, seed=seed + 2000)

    def _sample_task(prefix_tok: int, a_c: int, b_c: int, n: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(s)
        payload = torch.randint(0, m_mod, (n, 7), generator=g)
        x = torch.cat([torch.full((n, 1), prefix_tok, dtype=torch.long), payload], dim=1)
        y = (a_c * x[:, -1] + b_c) % m_mod
        return x, y

    eval_A_x, eval_A_y = _sample_task(20, a_A, b_A, 128, seed + 3001)
    eval_B_x, eval_B_y = _sample_task(21, a_B, b_B, 128, seed + 3002)

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 6: Basis-Misaligned Model Merging & Splicing\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size=24`, `d_model=48`, `n_heads=4`, `d_mlp=64`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"You are given two specialist models (`weights` = Model A, `weights_B` = Model B) trained/compiled with **different random orthogonal coordinate bases**:\n"
        f"- Model A handles sequences starting with prefix token `20`.\n"
        f"- Model B handles sequences starting with prefix token `21`.\n\n"
        f"Merge both specialists into a **single** `weights` checkpoint of the same architecture (`d_model=48`) that preserves `>= 99%` accuracy on both tasks simultaneously.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat6_{seed}",
        category="cat6_splice_merge",
        subtype="orthogonal_subspace_model_merge",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=weights_A,
        extra_public_weights={"weights_B": weights_B},
        secret={
            "category": "cat6_splice_merge",
            "oracle_weights": oracle_unscrambled,
            "eval_A_x": eval_A_x,
            "eval_A_y": eval_A_y,
            "eval_B_x": eval_B_x,
            "eval_B_y": eval_B_y,
        },
    )


def _generate_cat7_refactor(
    seed: int,
    difficulty: float,
    forward_budget: int | None,
    subtype: str | None = None,
) -> ChallengeBundle:
    if subtype == "level1_5_grokked_svd_defrag":
        from wsp_gym.level1_5 import generate_level1_5_cat7_refactor

        return generate_level1_5_cat7_refactor(
            seed=seed, difficulty=difficulty, forward_budget=forward_budget
        )

    rng = random.Random(seed)
    m_mod = 20
    a = rng.randint(1, m_mod - 1)
    b = rng.randint(1, m_mod - 1)
    c = rng.randint(0, m_mod - 1)

    config = ModelConfig(
        vocab_size=m_mod,
        max_seq_len=16,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_mlp=64,
        norm_type="none",
        mlp_type="standard",
        act_type="relu",
        pos_type="learned",
        causal=True,
    )
    clean = init_weights(config, seed=seed, zero_init=True)
    # Clean sparse circuit uses:
    # - dims [0:12]: [0] (a*v mod M), [1] (b*v+c mod M), [2] copied x_{-2}, [3] y, [4:12] sinusoidal pos
    # - Head 0 only (Heads 1, 2, 3 zero)
    # - MLP neurons 0..2 only (neurons 3..63 zero)
    for v in range(m_mod):
        clean["W_E"][v, 0] = float((a * v) % m_mod)
        clean["W_E"][v, 1] = float((b * v + c) % m_mod)
    clean["W_pos"][:, 4:12] = _sinusoidal_pos(config.max_seq_len, 8)
    _configure_offset_head(
        clean, 0, 0, slice(4, 12), offset=1, src_slice=slice(0, 1), dst_slice=slice(2, 3), attn_gain=100.0
    )
    _wire_exact_mod_add_mlp(clean, 0, 0, dim_x1=2, dim_x2=1, dim_out=3, modulus=m_mod)
    _wire_quadratic_readout(clean, dim_y=3, num_classes=m_mod, scale=0.02)

    # Bloated functionally-identical version:
    # 1. Heads 1 & 2 are high-norm cancelling clones (W_O[2] = -W_O[1]) reading unit-norm W_pos dims [4:12]
    # 2. MLP neurons 10..59 write high-norm signals strictly into dims [12:32] (the exact left null-space of W_U!)
    # 3. Scramble gauge so everything looks 100% dense!
    bloated = {k: v.clone() for k, v in clean.items()}
    gen = torch.Generator().manual_seed(seed + 77)
    decoy_q = torch.zeros((32, 8), dtype=torch.float32)
    decoy_k = torch.zeros((32, 8), dtype=torch.float32)
    decoy_v = torch.zeros((32, 8), dtype=torch.float32)
    decoy_q[4:12, :] = torch.randn((8, 8), generator=gen) * 0.5
    decoy_k[4:12, :] = torch.randn((8, 8), generator=gen) * 0.5
    decoy_v[4:12, :] = torch.randn((8, 8), generator=gen) * 3.0
    decoy_o = torch.zeros((8, 32), dtype=torch.float32)
    decoy_o[:, 12:32] = torch.randn((8, 20), generator=gen) * 3.0
    bloated["blocks.0.attn.W_Q"][1] = decoy_q
    bloated["blocks.0.attn.W_K"][1] = decoy_k
    bloated["blocks.0.attn.W_V"][1] = decoy_v
    bloated["blocks.0.attn.W_O"][1] = decoy_o
    bloated["blocks.0.attn.W_Q"][2] = decoy_q.clone()
    bloated["blocks.0.attn.W_K"][2] = decoy_k.clone()
    bloated["blocks.0.attn.W_V"][2] = decoy_v.clone()
    bloated["blocks.0.attn.W_O"][2] = -decoy_o.clone()

    for j in range(10, 60):
        bloated["blocks.0.mlp.W_in"][4:12, j] = torch.randn(8, generator=gen) * 2.0
        bloated["blocks.0.mlp.W_out"][j, 12:32] = torch.randn(20, generator=gen) * 2.5

    bloated_scrambled, _ = scramble_gauge(bloated, config, seed=seed + 1000)

    t_gen = torch.Generator().manual_seed(seed + 2001)
    eval_x = torch.randint(0, m_mod, (256, 10), generator=t_gen)

    budget = 2 if forward_budget is None else forward_budget
    brief = (
        f"# Category 7: Weight Refactoring & Defragmentation\n\n"
        f"**Architecture:** 1-layer Transformer (`vocab_size={m_mod}`, `d_model=32`, `n_heads=4`, `d_mlp=64`, `norm_type='none'`).\n"
        f"**Forward Pass Budget:** {budget} batches\n\n"
        f"## Objective\n"
        f"The checkpoint in `weights.pt` contains significant parameter redundancy and over-parameterized bloat across its residual, attention, and MLP subspaces.\n\n"
        f"Refactor `weights` into a compact, sparse, low-norm representation such that:\n"
        f"1. Output logits match the original model across all inputs (`max |delta logit| < 1e-3`).\n"
        f"2. Total squared Frobenius norm `||W||_F^2` and non-zero parameter count `||W||_0` (`|w| > 1e-4`) are reduced by `>= 50%`.\n\n"
        f"Call `submit_weights(weights)` when complete.\n"
    )
    return ChallengeBundle(
        episode_id=f"ep_cat7_{seed}",
        category="cat7_refactor",
        subtype="nullspace_and_cancelling_head_defrag",
        difficulty=difficulty,
        forward_budget=budget,
        config=config,
        brief_md=brief,
        public_weights=bloated_scrambled,
        secret={
            "category": "cat7_refactor",
            "orig_weights": bloated_scrambled,
            "oracle_weights": clean,
            "eval_x": eval_x,
        },
    )


_GENERATOR_DISPATCH = {
    "cat1_decompile": _generate_cat1_decompile,
    "cat2_repair": _generate_cat2_repair,
    "cat3_synthesize": _generate_cat3_synthesize,
    "cat4_nullspace_inject": _generate_cat4_nullspace_inject,
    "cat5_unlearn_excise": _generate_cat5_unlearn_excise,
    "cat6_splice_merge": _generate_cat6_splice_merge,
    "cat7_refactor": _generate_cat7_refactor,
}


def generate_challenge(
    category: str,
    seed: int = 0,
    difficulty: float = 1.0,
    forward_budget: int | None = None,
    subtype: str | None = None,
) -> ChallengeBundle:
    """Generate a blinded Weight-Space Programming challenge for the given category."""
    if category not in _GENERATOR_DISPATCH:
        raise ValueError(f"Unknown category '{category}'. Expected one of {CATEGORIES}")
    return _GENERATOR_DISPATCH[category](
        seed=seed,
        difficulty=difficulty,
        forward_budget=forward_budget,
        subtype=subtype,
    )
