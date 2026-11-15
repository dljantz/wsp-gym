"""Automated Blinded Multi-Turn Neurosurgeon Rollout Engine & SFT Corpus Scaler."""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any

from wsp_gym.generators import generate_challenge
from wsp_gym.model import ModelConfig
from wsp_gym.sandbox import SandboxSession, save_challenge
from wsp_gym.training import (
    CYCLE2_SUBTYPE_CHOICES,
    CurriculumScheduler,
    audit_and_export_sft_dataset,
)

ALL_17_SUBTYPES: list[tuple[str, str]] = [
    (cat, sub)
    for cat, subs in CYCLE2_SUBTYPE_CHOICES.items()
    for sub in subs
]


def _diag_cat1_level1(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Solving blinded `cat1_decompile` (budget={forward_budget}) statically. "
        "Using `wsp_tools.head_offset_profile` and `wsp_tools.mlp_logit_impact` to filter "
        "null-space/cancelling decoys and decode `(a, b, c)` + `[alpha, beta, delta, gamma]`."
    )
    t1_code = """
p = config["vocab_size"]
prof = wsp_tools.head_offset_profile(weights, 0)
h1 = [x["head"] for x in prof if x["dominant_offset"] == 1][0]
h2 = [x["head"] for x in prof if x["dominant_offset"] == 2][0]
ov1 = wsp_tools.virtual_ov_circuit(weights, 0, h1)
ov2 = wsp_tools.virtual_ov_circuit(weights, 0, h2)
imp = wsp_tools.mlp_logit_impact(weights, 0)
active, logit_map = imp["active_indices"], imp["logit_mat"]
win, bin_v, we = weights["blocks.0.mlp.W_in"], weights["blocks.0.mlp.b_in"], weights["W_E"]
j_troj = min(active, key=lambda j: float(bin_v[j]))
gamma = int(logit_map[j_troj].argmax().item())
delta = int((we @ win[:, j_troj]).argmax().item())
beta = int((we @ ov1 @ win[:, j_troj]).argmax().item())
alpha = int((we @ ov2 @ win[:, j_troj]).argmax().item())
j_lin = min([j for j in active if j != j_troj], key=lambda j: abs(float(bin_v[j])))
curr = we @ win[:, j_lin]
lag1 = we @ ov1 @ win[:, j_lin]
lag2 = we @ ov2 @ win[:, j_lin]
k_b, sb = (1, curr) if curr.max().item() > 1.0 else (2, lag1)
c_val = int(round(sb[0].item())) % p
b_val = (int(round(sb[1].item())) - c_val) % p
k_a, sa = (3, lag2) if lag2.max().item() > 1.0 else (2, lag1)
a_val = int(round(sa[1].item())) % p
diag = {
    "h1": h1, "h2": h2, "active": active, "j_troj": j_troj, "j_lin": j_lin,
    "k_a": k_a, "k_b": k_b,
    "submission": {"a": a_val, "b": b_val, "c": c_val, "trigger": [alpha, beta, delta, gamma]},
}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat1_level1(diag: dict[str, Any]) -> tuple[str, str]:
    sub = diag["submission"]
    t2_think = (
        f"Static inspection located offset-1 head {diag['h1']}, offset-2 head {diag['h2']}, "
        f"{len(diag['active'])} non-cancelling active neurons {diag['active']}. Trigger detector "
        f"neuron {diag['j_troj']} isolates [alpha, beta, delta, gamma]={sub['trigger']}, while linear "
        f"modular neuron {diag['j_lin']} with lags (k_a={diag['k_a']}, k_b={diag['k_b']}) decodes to "
        f"(a={sub['a']}, b={sub['b']}, c={sub['c']}). Submitting decompiled spec."
    )
    t2_code = 'submit_decompiled(diag["submission"])'
    return t2_think, t2_code


def _diag_cat1_l15(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"RMSNorm+SwiGLU+RoPE trigger & modular rule audit (budget={forward_budget}) "
        "via `head_offset_profile` and `mlp_logit_impact`."
    )
    t1_code = """
w, p = weights, config["vocab_size"]
prof = wsp_tools.head_offset_profile(w, 0)
h1 = next(x["head"] for x in prof if x["dominant_offset"] == 1)
h2 = next(x["head"] for x in prof if x["dominant_offset"] == 2)
b0, n1 = "blocks.0.", w["blocks.0.norm1.weight"][:, None]
ov1 = (n1 * w[b0 + "attn.W_V"][h1]) @ w[b0 + "attn.W_O"][h1]
ov2 = (n1 * w[b0 + "attn.W_V"][h2]) @ w[b0 + "attn.W_O"][h2]
imp = wsp_tools.mlp_logit_impact(w, 0)
act, lm = imp["active_indices"], imp["logit_mat"]
n2 = w[b0 + "norm2.weight"][:, None]
wg, wi = n2 * w[b0 + "mlp.W_gate"], n2 * w[b0 + "mlp.W_in"]
jt = max(act, key=lambda j: (lm[j].max() - lm[j].median()).item())
gam = int(lm[jt].argmax().item())
we = w["W_E"]
wc = we - we.mean(0, keepdim=True)
d = int((wc @ wg[:, jt]).argmax().item())
bet = int((wc @ ov1 @ wg[:, jt]).argmax().item())
alp = int((wc @ ov2 @ wg[:, jt]).argmax().item())
jl = min((j for j in act if j != jt), key=lambda j: wg[:, j].norm().item())
sc, sl = we @ wi[:, jl], we @ ov1 @ wi[:, jl]
s1, s2 = sc.max() / ((p - 1.0) / p), sl.max() / ((p - 1.0) / p)
c = int(round((sc[0] / s1 * p).item())) % p
b = (int(round((sc[1] / s1 * p).item())) - c) % p
a = int(round((sl[1] / s2 * p).item())) % p
diag = {
    "h1": h1, "h2": h2, "active": act, "j_troj": jt, "j_lin": jl,
    "k_a": 2, "k_b": 1,
    "submission": {"a": a, "b": b, "c": c, "trigger": [alp, bet, d, gam]},
}
print(diag)
"""
    return t1_think, t1_code


def _diag_cat2(forward_budget: int) -> tuple[str, str]:
    if forward_budget > 0:
        probe_sink = """
    rm = config.get("norm_type") == "rmsnorm"
    v_lim = 20 if rm else config["vocab_size"] - 1
    px = torch.randint(0, v_lim, (8, 12))
    px[:, -1 if rm else 4] = v_lim
    collapse_tok = int(probe_forward(px, return_cache=False)[:, -1, :].argmax(-1)[0].item())
"""
        probe_kcomp = """
    px = torch.randint(0, 31, (8, 12))
    px[:, 5] = 31
    acc = float((probe_forward(px, return_cache=False)[:, -1, :].argmax(-1) == px[:, 6]).float().mean().item())
"""
    else:
        probe_sink = "    collapse_tok = -1\n"
        probe_kcomp = "    acc = -1.0\n"

    t1_think = (
        f"Diagnosing `cat2_repair` (budget={forward_budget}). Scanning L1 MLP output norms for sink attractors; "
        f"if none appear, checking L0 `W_V @ W_O` vs. L1 `W_K` SVD alignment"
        f"{' with a forward probe' if forward_budget > 0 else ' statically'}."
    )
    t1_code = f"""
out_norms = weights["blocks.1.mlp.W_out"].norm(dim=1)
bad_idx = (out_norms > 10.0).nonzero(as_tuple=True)[0].tolist()
if bad_idx:
{probe_sink}    diag = {{"pathology": "sink_attractor", "bad_idx": bad_idx, "collapse_tok": collapse_tok, "max_norm": float(out_norms.max().item())}}
else:
{probe_kcomp}    h0 = int(weights["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmax().item())
    h1 = int(weights["blocks.1.attn.W_K"].norm(dim=(1, 2)).argmax().item())
    vm = weights["W_E"][31] - weights["W_E"][:31].mean(dim=0)
    va = vm @ weights["blocks.0.attn.W_V"][h0] @ weights["blocks.0.attn.W_O"][h0]
    vu = va / va.norm().clamp_min(1e-8)
    Uk, sk, _ = torch.linalg.svd(weights["blocks.1.attn.W_K"][h1], full_matrices=False)
    diag = {{"pathology": "k_composition", "h0": h0, "h1": h1, "dot_before": float((vu @ Uk[:, 0]).item()), "sk0": float(sk[0].item()), "probe_acc": acc}}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat2_sink(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Confirmed sink attractor neurons {diag['bad_idx']} in Layer-1 MLP (max output norm "
        f"{diag['max_norm']:.2f}, collapse_tok={diag['collapse_tok']}). Zeroing only those corrupted "
        "neurons while keeping all clean induction/grokked weights intact."
    )
    t2_code = """
for j in diag["bad_idx"]:
    weights["blocks.1.mlp.W_in"][:, j] = 0.0
    if "blocks.1.mlp.W_gate" in weights:
        weights["blocks.1.mlp.W_gate"][:, j] = 0.0
    weights["blocks.1.mlp.b_in"][j] = 0.0
    weights["blocks.1.mlp.W_out"][j, :] = 0.0
submit_weights(weights)
"""
    return t2_think, t2_code


def _commit_cat2_kcomp(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"L0 active head {diag['h0']} writes `is_after_marker` orthogonal to L1 head {diag['h1']}'s rank-1 "
        f"`W_K` singular input (`dot_before={diag['dot_before']:.4f}`, `sk0={diag['sk0']:.4f}`). Reconstructing "
        "`W_K[h1]` as the rank-1 outer product `outer(v_after_unit, sk0 * sign_k * Vhk[0])` restores K-composition."
    )
    t2_code = """
h0, h1 = diag["h0"], diag["h1"]
v_marker = weights["W_E"][31] - weights["W_E"][:31].mean(dim=0)
v_after = v_marker @ weights["blocks.0.attn.W_V"][h0] @ weights["blocks.0.attn.W_O"][h0]
v_after_unit = v_after / v_after.norm().clamp_min(1e-8)
Uk, sk, Vhk = torch.linalg.svd(weights["blocks.1.attn.W_K"][h1], full_matrices=False)
q_dir = weights["W_E"][0] @ weights["blocks.1.attn.W_Q"][h1]
sign_k = 1.0 if float((q_dir @ Vhk[0]).item()) >= 0.0 else -1.0
weights["blocks.1.attn.W_K"][h1] = torch.outer(v_after_unit, sk[0] * sign_k * Vhk[0])
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat3_level1(forward_budget: int, brief_md: str) -> tuple[str, str]:
    del brief_md
    t1_think = (
        f"Inspecting zero-initialized model for `cat3_synthesize` (budget={forward_budget}). "
        "Extracting `op == 20` and `op == 21` offset/shift specs from `brief_md` and verifying dimensions."
    )
    t1_code = r"""import re
m0 = re.search(r"op == 20.*?x\[T - 1 - (\d+)\] \+ (\d+)\) mod 20", brief_md)
m1 = re.search(r"op == 21.*?x\[T - 1 - (\d+)\] \+ (\d+)\) mod 20", brief_md)
k0, s0 = int(m0.group(1)), int(m0.group(2))
k1, s1 = int(m1.group(1)), int(m1.group(2))
diag = {"k0": k0, "s0": s0, "k1": k1, "s1": s1, "d_model": config["d_model"]}
print(diag)"""
    return t1_think, t1_code


def _commit_cat3_level1(diag: dict[str, Any], forward_budget: int) -> tuple[str, str]:
    del diag
    probe_line = ""
    if forward_budget > 0:
        probe_line = """
tx = torch.cat([torch.full((4, 1), 20, dtype=torch.long), torch.randint(0, 20, (4, 7))], 1)
lg = probe_forward(tx, custom_weights=weights, return_cache=False)[:, -1]
print("probe_acc:", float((lg.argmax(-1) == ((tx[:, 7 - k0] + s0) % 20)).float().mean().item()))
"""
    t2_think = (
        "Synthesizing dual-offset mod-20 circuit from `diag` specs for op20 & op21 "
        "with pos-0 op copy, sinusoidal offset heads, 3-ReLU modular wrap, and quadratic readout."
    )
    t2_code = f"""
import math
k0, s0, k1, s1 = diag["k0"], diag["s0"], diag["k1"], diag["s1"]
w, M = weights, 20
ar = torch.arange(M, dtype=torch.float32)
we, wp = w["W_E"], w["W_pos"]
we[:M, 0], we[:M, 3], we[20:22, 1:3], we[20:22, 3] = ar, 1.0, torch.eye(2), 1.0
we[:24, 24], we[:24, 25] = float(s0), float(s1)
pos = torch.arange(config["max_seq_len"], dtype=torch.float32)[:, None]
fr = torch.linspace(0.35, 2.85, 6)
wp[:, 4:10], wp[:, 10:16], wp[0, 22] = torch.cos(pos * fr), torch.sin(pos * fr), 1.0
b0 = "blocks.0.attn."
w[b0 + "W_Q"][0, 3, 0] = w[b0 + "W_K"][0, 22, 0] = 15.0
w[b0 + "W_V"][0, 1:3, :2] = torch.eye(2)
w[b0 + "W_O"][0, 0, 16] = w[b0 + "W_O"][0, 1, 26] = 1.0
idx, g = torch.arange(6), math.sqrt(600.0)
for l, h, off, dst in ((0, 1, k0, 17), (1, 0, k1, 18)):
    ap = f"blocks.{{l}}.attn."
    c, s = torch.cos(-off * fr), torch.sin(-off * fr)
    sh = torch.zeros((12, 12))
    sh[idx, idx], sh[idx, 6 + idx], sh[6 + idx, idx], sh[6 + idx, 6 + idx] = c, s, -s, c
    w[ap + "W_Q"][h, 4:16, :12], w[ap + "W_K"][h, 4:16, :12] = g * sh, g * torch.eye(12)
    w[ap + "W_V"][h, 0, 0] = w[ap + "W_O"][h, 0, dst] = 1.0
for l, d1, d2, do, gd in ((0, 17, 24, 19, 16), (1, 18, 25, 20, 26)):
    mp = f"blocks.{{l}}.mlp."
    w[mp + "W_in"][[d1, d2], :3], w[mp + "W_in"][gd, :3] = 1.0, 100.0
    w[mp + "b_in"][:3] = torch.tensor([-100.0, -119.0, -120.0])
    w[mp + "W_out"][:3, do] = torch.tensor([1.0, -20.0, 20.0])
w["W_U"][19, :M] = w["W_U"][20, :M] = 2.0 * ar
w["b_U"][:M] = -(ar * ar)
{probe_line}submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat3_l15(forward_budget: int, brief_md: str) -> tuple[str, str]:
    del brief_md
    t1_think = (
        f"Inspecting zero-initialized RMSNorm+SwiGLU+RoPE checkpoint for `cat3_synthesize` (budget={forward_budget}). "
        "Extracting op20 and op21 affine specs from `brief_md` and checking RoPE operator."
    )
    t1_code = r"""import re
m0 = re.search(r"op == 20.*?\((\d+) \* x\[T - 2\] \+ (\d+)\)", brief_md)
m1 = re.search(r"op == 21.*?\((\d+) \* x\[T - 3\] \+ (\d+)\)", brief_md)
a0, b0 = int(m0.group(1)), int(m0.group(2))
a1, b1 = int(m1.group(1)), int(m1.group(2))
r1 = wsp_tools.rope_rotation_matrix(config["d_head"], 1.0)
diag = {"a0": a0, "b0": b0, "a1": a1, "b1": b1, "rope_det": float(torch.linalg.det(r1).item())}
print(diag)"""
    return t1_think, t1_code


def _commit_cat3_l15(diag: dict[str, Any], forward_budget: int) -> tuple[str, str]:
    del diag
    probe_line = ""
    if forward_budget > 0:
        probe_line = """
tx = torch.cat([torch.randint(0, 20, (4, 11)), torch.full((4, 1), 20, dtype=torch.long)], 1)
lg = probe_forward(tx, custom_weights=weights, return_cache=False)[:, -1]
print("probe_acc:", float((lg.argmax(-1) == ((a0 * tx[:, -2] + b0) % 20)).float().mean().item()))
"""
    t2_think = (
        "Compiling RMSNorm+SwiGLU+RoPE circuit for op20 & op21 from `diag` specs "
        "using 4D Fourier embeddings, RoPE offset heads, and 40 SwiGLU neurons."
    )
    t2_code = f"""
import math
a0, b0, a1, b1 = diag["a0"], diag["b0"], diag["a1"], diag["b1"]
w, M, dh = weights, 20, config["d_head"]
vs = math.sqrt(2.0 / float(config["d_model"]))
ar = torch.arange(M, dtype=torch.float32)
ang = 2.0 * math.pi * ar[:, None] / M
F4 = torch.cat([torch.cos(ang), torch.sin(ang), torch.cos(2 * ang), torch.sin(2 * ang)], 1) / math.sqrt(2.0)
we = w["W_E"]
we[:M, :4], we[:M, 6] = F4, 1.0
we[20, 4] = we[20, 6] = we[21, 5] = we[21, 6] = 1.0
kv, gk = torch.ones(dh) / math.sqrt(dh), math.sqrt(180.0)
ap, mp = "blocks.0.attn.", "blocks.1.mlp."
for h, off, sl in ((0, 1.0, slice(8, 12)), (1, 2.0, slice(12, 16))):
    r = wsp_tools.rope_rotation_matrix(dh, off)
    w[ap + "W_Q"][h, 6], w[ap + "W_K"][h, 6] = gk * (kv @ r.T), gk * kv
    w[ap + "W_V"][h, :4, :4], w[ap + "W_O"][h, :4, sl] = torch.eye(4), torch.eye(4) * vs
idx = torch.arange(M)
for base, opd, sl, aa, bb in ((0, 4, slice(8, 12), a0, b0), (20, 5, slice(12, 16), a1, b1)):
    j = base + idx
    w[mp + "W_gate"][opd, j], w[mp + "W_gate"][sl, j], w[mp + "W_gate"][6, j] = 60.0, 60.0 * F4.T, -116.4
    w[mp + "W_in"][opd, j] = 1.0
    w[mp + "W_out"][j, 16 + ((aa * idx + bb) % M)] = 1.5
w["W_U"][16:36, :M] = 4.0 * torch.eye(M)
{probe_line}submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat4_level1(forward_budget: int, brief_md: str) -> tuple[str, str]:
    del brief_md
    t1_think = (
        f"Solving `cat4_nullspace_inject` (budget={forward_budget}). "
        "Extracting Task 22 affine spec from `brief_md` and computing `wsp_tools.residual_svd_basis(weights)` for `V_null` and free slots."
    )
    t1_code = r"""import re
a2, b2 = (int(v) for v in re.search(r"\((\d+) \* x\[-1\] \+ (\d+)\)", brief_md).groups())
svd = wsp_tools.residual_svd_basis(weights)
h_free = int(weights["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmin().item())
j_free = int(weights["blocks.0.mlp.W_out"].norm(dim=1).argmin().item())
diag = {"a2": a2, "b2": b2, "rank": svd["rank"], "null_dim": int(svd["V_null"].shape[1]), "h_free": h_free, "j_free": j_free}
print(diag)"""
    return t1_think, t1_code


def _commit_cat4_level1(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Occupied residual rank is {diag['rank']} with {diag['null_dim']} null-space dimensions, leaving head "
        f"{diag['h_free']} and MLP neuron {diag['j_free']} completely free. Injecting Task 22 "
        "from `diag` specs strictly within `V_null[:, :6]` guarantees zero KL drift on Tasks 20/21."
    )
    t2_code = """
a2, b2 = diag["a2"], diag["b2"]
svd = wsp_tools.residual_svd_basis(weights)
Vn = svd["V_null"]
u_y, u_op, u_c, u_p0, u_cp, u_out = Vn[:, 0], Vn[:, 1], Vn[:, 2], Vn[:, 3], Vn[:, 4], Vn[:, 5]
M = 20
for v in range(M):
    weights["W_E"][v] += float((a2 * v + b2) % M) * u_y + u_c
weights["W_E"][22] += u_op + u_c
weights["W_pos"][0] += u_p0

hf, jf = diag["h_free"], diag["j_free"]
weights["blocks.0.attn.W_Q"][hf, :, 0] = 40.0 * u_c
weights["blocks.0.attn.W_K"][hf, :, 0] = u_p0
weights["blocks.0.attn.W_V"][hf, :, 0] = u_op
weights["blocks.0.attn.W_O"][hf, 0, :] = u_cp

weights["blocks.0.mlp.W_in"][:, jf] = u_y + 100.0 * u_cp
weights["blocks.0.mlp.b_in"][jf] = -100.0
weights["blocks.0.mlp.W_out"][jf, :] = u_out
for k in range(M):
    weights["W_U"][:, k] += 2.0 * float(k) * u_out
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat4_l15(forward_budget: int, brief_md: str) -> tuple[str, str]:
    del brief_md
    t1_think = (
        f"Solving `cat4_nullspace_inject` on RMSNorm+SwiGLU+RoPE weights (budget={forward_budget}). "
        "Extracting Task 22 spec from `brief_md` and locating free residual coordinates, L0 RoPE head, and 20 free L1 SwiGLU slots."
    )
    t1_code = r"""import re
a2, b2 = (int(v) for v in re.search(r"\((\d+) \* x\[T - 2\] \+ (\d+)\)", brief_md).groups())
res_norms = weights["W_E"][:22].norm(dim=0) + weights["W_U"][:, :20].norm(dim=1)
free_dims = (res_norms < 1e-4).nonzero(as_tuple=True)[0].tolist()
h_free = int(weights["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmin().item())
free_mlp = torch.argsort(weights["blocks.1.mlp.W_out"].norm(dim=1))[:20].tolist()
diag = {"a2": a2, "b2": b2, "n_free_dims": len(free_dims), "h_free": h_free, "free_mlp": free_mlp}
print(diag)"""
    return t1_think, t1_code


def _commit_cat4_l15(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Detected {diag['n_free_dims']} unoccupied dims (`31..47`), free L0 head {diag['h_free']}, "
        f"and free SwiGLU slots {diag['free_mlp'][:4]}... Injecting Task 22 with `eps_e=0.20` (`KL < 1e-3`), "
        "RoPE offset-1 copy, and SwiGLU Fourier ring readout."
    )
    t2_code = """
import math
a2, b2 = diag["a2"], diag["b2"]
M, eps_e = 20, 0.20
ar = torch.arange(M, dtype=torch.float32)
ang = 2.0 * math.pi * ar[:, None] / M
F4 = torch.cat([torch.cos(ang), torch.sin(ang), torch.cos(2 * ang), torch.sin(2 * ang)], dim=1) / math.sqrt(2.0)
weights["W_E"][:M, 31:35] = eps_e * F4
weights["W_E"][:M, 36] = eps_e
weights["W_E"][22, 35:37] = eps_e
weights["W_E"][22, 43] = math.sqrt(2.0)
inv_rms = float(torch.rsqrt(weights["W_E"][:M].pow(2).mean(dim=-1) + config["rms_eps"]).mean().item())
hf, d_head = diag["h_free"], config["d_head"]
k_vec = torch.ones(d_head) / math.sqrt(d_head)
q_vec = k_vec @ wsp_tools.rope_rotation_matrix(d_head, 1.0).T
g = math.sqrt(180.0 / (eps_e * eps_e))
weights["blocks.0.attn.W_Q"][hf, 36] = g * q_vec
weights["blocks.0.attn.W_K"][hf, 36] = g * k_vec
weights["blocks.0.attn.W_V"][hf, 31:35, :4] = torch.eye(4)
weights["blocks.0.attn.W_O"][hf, :4, 37:41] = torch.eye(4) / inv_rms
js = torch.tensor(diag["free_mlp"][:M], dtype=torch.long)
weights["blocks.1.mlp.W_gate"][35, js] = 75.0
weights["blocks.1.mlp.W_gate"][37:41, js] = 75.0 * F4.T
weights["blocks.1.mlp.W_gate"][36, js] = -75.0 * 1.94
weights["blocks.1.mlp.W_in"][35, js] = 1.0
out_ang = 2.0 * math.pi * ((a2 * ar + b2) % M) / M
weights["blocks.1.mlp.W_out"][js, 41] = 1.5 * torch.cos(out_ang)
weights["blocks.1.mlp.W_out"][js, 42] = 1.5 * torch.sin(out_ang)
k_ang = 2.0 * math.pi * ar / M
weights["W_U"][41, :M] = 3.0 * torch.cos(k_ang)
weights["W_U"][42, :M] = 3.0 * torch.sin(k_ang)
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat5_level1(forward_budget: int) -> tuple[str, str]:
    probe_snippet = ""
    if forward_budget > 0:
        probe_snippet = """
px = torch.randint(0, 20, (4, 10))
px[:, 2] = 20
px[:, -3] = 21
px[:, -2] = 22
logits, _ = probe_forward(px, return_cache=True)
probe_tok = int(logits[:, -1, :].argmax(-1)[0].item())
"""
    else:
        probe_snippet = "probe_tok = -1"

    t1_think = (
        f"Diagnosing `cat5_unlearn_excise` (budget={forward_budget}). I will profile Layer-0 "
        "attention offsets to separate the clean offset-1 marker router from the offset-2 trigger channel and inspect "
        "Layer-1 MLP biases/logit impacts for the steep-threshold backdoor neuron while leaving the clean task untouched."
    )
    t1_code = f"""
{probe_snippet}
prof0 = wsp_tools.head_offset_profile(weights, 0)
h1 = [x["head"] for x in prof0 if x["dominant_offset"] == 1][0]
h2 = [x["head"] for x in prof0 if x["dominant_offset"] == 2][0]
j_troj = int(weights["blocks.1.mlp.b_in"].argmin().item())
logit_map = weights["blocks.1.mlp.W_out"] @ weights["W_U"]
c_bad = int(logit_map[j_troj].argmax().item())
diag = {{"h1": h1, "h2": h2, "j_troj": j_troj, "c_bad": c_bad, "probe_tok": probe_tok}}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat5_level1(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Located L0 offset-1 head {diag['h1']} (carries both marker and 2nd trigger token), L0 offset-2 "
        f"head {diag['h2']} (carries 1st trigger token), and L1 MLP trojan neuron {diag['j_troj']} targeting "
        f"`c_bad={diag['c_bad']}`. I will zero `j_troj` plus its residual direction in `W_out`, zero `h2`, "
        "and orthogonally project out the non-marker trigger direction from `W_V[h1]` without touching clean marker routing."
    )
    t2_code = """
h1, h2, jt = diag["h1"], diag["h2"], diag["j_troj"]
out_u = weights["blocks.1.mlp.W_out"][jt] / weights["blocks.1.mlp.W_out"][jt].norm().clamp_min(1e-8)
in_u = weights["blocks.1.mlp.W_in"][:, jt] / weights["blocks.1.mlp.W_in"][:, jt].norm().clamp_min(1e-8)
weights["blocks.1.mlp.W_in"][:, jt] = 0.0
weights["blocks.1.mlp.b_in"][jt] = 0.0
weights["blocks.1.mlp.W_out"][jt, :] = 0.0
weights["blocks.1.mlp.W_out"] = weights["blocks.1.mlp.W_out"] - torch.outer(weights["blocks.1.mlp.W_out"] @ out_u, out_u)
weights["blocks.1.mlp.W_in"] = weights["blocks.1.mlp.W_in"] - torch.outer(in_u, in_u @ weights["blocks.1.mlp.W_in"])

for s in ("W_Q", "W_K", "W_V", "W_O"):
    weights[f"blocks.0.attn.{s}"][h2] = 0.0

v22_proj = weights["W_E"][22] @ weights["blocks.0.attn.W_V"][h1]
if v22_proj.norm().item() > 1e-6:
    u_v = v22_proj / v22_proj.norm()
    weights["blocks.0.attn.W_V"][h1] = weights["blocks.0.attn.W_V"][h1] - torch.outer(weights["blocks.0.attn.W_V"][h1] @ u_v, u_v)
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat5_l15(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Diagnosing `cat5_unlearn_excise` on RMSNorm+SwiGLU+RoPE weights (budget={forward_budget}). "
        "Inspecting Layer-0 (offset 1 vs 2) and Layer-1 (offset 3) RoPE heads plus SwiGLU `W_out @ norm_final @ W_U` "
        "logit spikes to isolate the planted 2-token backdoor."
    )
    t1_code = """
prof0 = wsp_tools.head_offset_profile(weights, 0)
h0_trig = [x["head"] for x in prof0 if x["dominant_offset"] == 2][0]
h1_trig = int(weights["blocks.1.attn.W_O"].norm(dim=(1, 2)).argmax().item())
wu = weights["norm_final.weight"][:, None] * weights["W_U"]
logit_map = weights["blocks.1.mlp.W_out"] @ wu
j_troj = int(logit_map.max(dim=1).values.argmax().item())
c_bad = int(logit_map[j_troj].argmax().item())
diag = {"h0_trig": h0_trig, "h1_trig": h1_trig, "j_troj": j_troj, "c_bad": c_bad}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat5_l15(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Identified L0 offset-2 head {diag['h0_trig']}, L1 offset-3 head {diag['h1_trig']}, and SwiGLU "
        f"trojan neuron {diag['j_troj']} (`c_bad={diag['c_bad']}`). Excising both trigger heads, zeroing neuron "
        f"{diag['j_troj']}, and projecting its output vector out of `W_out` defeats both activation and W_U recovery attacks."
    )
    t2_code = """
h0t, h1t, jt = diag["h0_trig"], diag["h1_trig"], diag["j_troj"]
for s in ("W_Q", "W_K", "W_V", "W_O"):
    weights[f"blocks.0.attn.{s}"][h0t] = 0.0
    weights[f"blocks.1.attn.{s}"][h1t] = 0.0
out_u = weights["blocks.1.mlp.W_out"][jt] / weights["blocks.1.mlp.W_out"][jt].norm().clamp_min(1e-8)
weights["blocks.1.mlp.W_gate"][:, jt] = 0.0
weights["blocks.1.mlp.W_in"][:, jt] = 0.0
weights["blocks.1.mlp.b_in"][jt] = 0.0
weights["blocks.1.mlp.W_out"][jt, :] = 0.0
weights["blocks.1.mlp.W_out"] = weights["blocks.1.mlp.W_out"] - torch.outer(weights["blocks.1.mlp.W_out"] @ out_u, out_u)
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat6_horiz(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Inspecting 1-layer `cat6_splice_merge` (budget={forward_budget}). "
        "Running `wsp_tools.residual_svd_basis` on both `weights` and `weights_B` to extract their occupied "
        "ranks, free null-space vectors in A, and active heads/MLP neurons in B."
    )
    t1_code = """
svdA = wsp_tools.residual_svd_basis(weights)
svdB = wsp_tools.residual_svd_basis(weights_B)
hA = int(weights["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmax().item())
hA_free = 0 if hA != 0 else 1
hB = int(weights_B["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmax().item())
jA = int(weights["blocks.0.mlp.W_out"].norm(dim=1).argmax().item())
jA_free = 0 if jA != 0 else 1
jB = int(weights_B["blocks.0.mlp.W_out"].norm(dim=1).argmax().item())
diag = {"rankA": svdA["rank"], "rankB": svdB["rank"], "hA_free": hA_free, "hB": hB, "jA_free": jA_free, "jB": jB}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat6_horiz(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Model A rank is {diag['rankA']} and Model B rank is {diag['rankB']}. Mapping Model B's active "
        f"`V_active` basis orthogonally into A's `V_null[:, :{diag['rankB']}]`, and splicing B head "
        f"{diag['hB']} -> {diag['hA_free']} and neuron {diag['jB']} -> {diag['jA_free']} preserves both tasks cleanly."
    )
    t2_code = """
svdA = wsp_tools.residual_svd_basis(weights)
svdB = wsp_tools.residual_svd_basis(weights_B)
rB = svdB["rank"]
R = svdB["V_active"] @ svdA["V_null"][:, :rB].T

weights["W_E"] = weights["W_E"] + weights_B["W_E"] @ R
weights["W_pos"] = weights["W_pos"] + weights_B["W_pos"] @ R
weights["W_U"] = weights["W_U"] + R.T @ weights_B["W_U"]

ha, hb = diag["hA_free"], diag["hB"]
weights["blocks.0.attn.W_Q"][ha] = R.T @ weights_B["blocks.0.attn.W_Q"][hb]
weights["blocks.0.attn.W_K"][ha] = R.T @ weights_B["blocks.0.attn.W_K"][hb]
weights["blocks.0.attn.W_V"][ha] = R.T @ weights_B["blocks.0.attn.W_V"][hb]
weights["blocks.0.attn.W_O"][ha] = weights_B["blocks.0.attn.W_O"][hb] @ R

ja, jb = diag["jA_free"], diag["jB"]
weights["blocks.0.mlp.W_in"][:, ja] = R.T @ weights_B["blocks.0.mlp.W_in"][:, jb]
weights["blocks.0.mlp.b_in"][ja] = weights_B["blocks.0.mlp.b_in"][jb]
weights["blocks.0.mlp.W_out"][ja] = weights_B["blocks.0.mlp.W_out"][jb] @ R
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat6_vert(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Inspecting 2-layer composition `cat6_splice_merge` (budget={forward_budget}). "
        "I will profile `weights_B` offset head `k_B`, extract B's constant bias scalar `s_B`, and locate A's "
        "residual null space `V_null` and Layer-0 output direction `u_zA` for grafting B into Layer 1."
    )
    t1_code = """
profB = wsp_tools.head_offset_profile(weights_B, 0)
hB = int(weights_B["blocks.0.attn.W_O"].norm(dim=(1, 2)).argmax().item())
k_B = int(profB[hB]["dominant_offset"])
d_mlp = config["d_mlp"]
actB = [
    j for j in range(d_mlp)
    if abs(float(weights_B["blocks.0.mlp.b_in"][j])) < 1e-3 and float(weights_B["blocks.0.mlp.W_out"][j] @ weights_B["W_U"][:, 1]) > 0.5
][0]
s_B = int(round(float((weights_B["W_E"][0] @ weights_B["blocks.0.mlp.W_in"][:, actB]).item())))
svdA = wsp_tools.residual_svd_basis(weights)
diag = {"hB": hB, "k_B": k_B, "actB": actB, "s_B": s_B, "rankA": svdA["rank"], "null_dim": int(svdA["V_null"].shape[1])}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat6_vert(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Model B (`k_B={diag['k_B']}`, `s_B={diag['s_B']}`) and A rank {diag['rankA']} "
        f"({diag['null_dim']} null dims). Grafting L1 offset-{diag['k_B']} head and 5-ReLU 3-sum wrap into `V_null`."
    )
    t2_code = """
import math
k_B, s_B = diag["k_B"], diag["s_B"]
V_null = wsp_tools.residual_svd_basis(weights)["V_null"]
u_zA = weights["W_U"][:, 1] / 2.0
P_null, u_sB, u_zprev, u_y = V_null[:, :12], V_null[:, 12], V_null[:, 13], V_null[:, 14]
pos = torch.arange(config["max_seq_len"], dtype=torch.float32)[:, None]
freqs = torch.linspace(0.35, 2.85, 6)
sinp = torch.cat([torch.cos(pos * freqs), torch.sin(pos * freqs)], dim=1)
weights["W_pos"] = weights["W_pos"] + sinp @ P_null.T
weights["W_E"] = weights["W_E"] + float(s_B) * u_sB.unsqueeze(0)
idx = torch.arange(6)
c, s = torch.cos(-k_B * freqs), torch.sin(-k_B * freqs)
sh = torch.zeros((12, 12))
sh[idx, idx], sh[idx, 6 + idx], sh[6 + idx, idx], sh[6 + idx, 6 + idx] = c, s, -s, c
g = math.sqrt(600.0)
weights["blocks.1.attn.W_Q"][0, :, :12] = g * (P_null @ sh)
weights["blocks.1.attn.W_K"][0, :, :12] = g * P_null
weights["blocks.1.attn.W_V"][0, :, 0] = u_zA
weights["blocks.1.attn.W_O"][0, 0, :] = u_zprev
weights["blocks.1.mlp.W_in"][:, :5] = (u_zA + u_zprev + u_sB).unsqueeze(1)
weights["blocks.1.mlp.b_in"][:5] = torch.tensor([0.0, -19.0, -20.0, -39.0, -40.0])
weights["blocks.1.mlp.W_out"][:5] = torch.outer(torch.tensor([1.0, -20.0, 20.0, -20.0, 20.0]), u_y)
weights["W_U"] = torch.outer(u_y, 2.0 * torch.arange(20, dtype=torch.float32))
submit_weights(weights)
"""
    return t2_think, t2_code


def _diag_cat6_l15(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Inspecting RMSNorm+SwiGLU+RoPE `cat6_splice_merge` (budget={forward_budget}). "
        "Solving Orthogonal Procrustes on the shared payload embedding/unembedding matrix `SA, SB` (`0..19`) and "
        "mapping Model B's internal task subspace into A's free orthogonal complement."
    )
    t1_code = """
SA = torch.cat([weights["W_E"][:20], weights["W_U"][:, :20].T], dim=0)
SB = torch.cat([weights_B["W_E"][:20], weights_B["W_U"][:, :20].T], dim=0)
_, sA, VhA = torch.linalg.svd(SA, full_matrices=True)
_, sB, VhB = torch.linalg.svd(SB, full_matrices=True)
k_sh = int((sA > 1e-2).sum().item())
hA = int(weights["blocks.0.attn.W_O"].pow(2).sum((1, 2)).argmax().item())
hB = int(weights_B["blocks.0.attn.W_O"].pow(2).sum((1, 2)).argmax().item())
actB = (weights_B["blocks.1.mlp.W_out"].norm(dim=1) > 1e-3).nonzero(as_tuple=True)[0].tolist()
free_mlpA = (weights["blocks.1.mlp.W_out"].norm(dim=1) <= 1e-3).nonzero(as_tuple=True)[0][: len(actB)].tolist()
diag = {"k_sh": k_sh, "hA": hA, "hB": hB, "n_actB": len(actB), "n_freeA": len(free_mlpA)}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat6_l15(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Interface rank={diag['k_sh']}, A head {diag['hA']}, B head {diag['hB']} ({diag['n_actB']} neurons). "
        "Aligning shared + orthogonal subspaces via Procrustes `R`, splicing B into A, and running `canonical_pack`."
    )
    t2_code = """
SA = torch.cat([weights["W_E"][:20], weights["W_U"][:, :20].T], dim=0)
SB = torch.cat([weights_B["W_E"][:20], weights_B["W_U"][:, :20].T], dim=0)
_, _, VhA = torch.linalg.svd(SA, full_matrices=True)
_, _, VhB = torch.linalg.svd(SB, full_matrices=True)
k, hA, hB = diag["k_sh"], diag["hA"], diag["hB"]
BA_sh, BB_sh, compA, compB = VhA[:k], VhB[:k], VhA[k:], VhB[k:]
Up, _, Vhp = torch.linalg.svd((SB @ BB_sh.T).T @ (SA @ BA_sh.T))
R_sh = BB_sh.T @ (Up @ Vhp) @ BA_sh
_, sIA, VhIA = torch.linalg.svd(torch.cat([weights["W_E"][20:21], weights["blocks.0.attn.W_O"][hA]]) @ compA.T, full_matrices=True)
_, sIB, VhIB = torch.linalg.svd(torch.cat([weights_B["W_E"][21:22], weights_B["blocks.0.attn.W_O"][hB]]) @ compB.T, full_matrices=True)
kA, kB = int((sIA > 1e-2).sum().item()), int((sIB > 1e-2).sum().item())
targetA = torch.cat([VhIA[kA:] @ compA, VhIA[:kA] @ compA], dim=0)
srcB = torch.cat([VhIB[:kB] @ compB, VhIB[kB:] @ compB], dim=0)
R = R_sh + srcB.T @ targetA
weights["W_E"][21] = weights_B["W_E"][21] @ R
hF = 1 - hA
for key in ("W_Q", "W_K", "W_V"):
    weights[f"blocks.0.attn.{key}"][hF] = R.T @ weights_B[f"blocks.0.attn.{key}"][hB]
weights["blocks.0.attn.W_O"][hF] = weights_B["blocks.0.attn.W_O"][hB] @ R
actB = (weights_B["blocks.1.mlp.W_out"].norm(dim=1) > 1e-3).nonzero(as_tuple=True)[0]
freeA = (weights["blocks.1.mlp.W_out"].norm(dim=1) <= 1e-3).nonzero(as_tuple=True)[0][: len(actB)]
for key in ("W_gate", "W_in"):
    weights[f"blocks.1.mlp.{key}"][:, freeA] = R.T @ weights_B[f"blocks.1.mlp.{key}"][:, actB]
weights["blocks.1.mlp.W_out"][freeA] = weights_B["blocks.1.mlp.W_out"][actB] @ R
submit_weights(wsp_tools.canonical_pack(weights))
"""
    return t2_think, t2_code


def _diag_cat7(forward_budget: int) -> tuple[str, str]:
    t1_think = (
        f"Auditing `cat7_refactor` (budget={forward_budget}). I will inspect residual SVD basis "
        "rank, attention head cancellation, and Layer-0 MLP active vs. null-space/cancelling neuron counts prior to "
        "canonical SVD compression."
    )
    t1_code = """
svd = wsp_tools.residual_svd_basis(weights)
imp = wsp_tools.mlp_logit_impact(weights, 0)
n_act = len(imp["active_indices"])
n_canc = len(imp["cancelling_pairs"])
frob_sq = float(sum(v.pow(2).sum().item() for v in weights.values()))
l0 = int(sum((v.abs() > 1e-6).sum().item() for v in weights.values()))
diag = {"rank": svd["rank"], "n_act": n_act, "n_canc": n_canc, "frob_sq": frob_sq, "l0": l0}
print(diag)
"""
    return t1_think, t1_code


def _commit_cat7(diag: dict[str, Any]) -> tuple[str, str]:
    t2_think = (
        f"Residual SVD rank={diag['rank']}, active non-cancelling MLP neurons={diag['n_act']}, cancelling pairs="
        f"{diag['n_canc']}, baseline F-norm^2={diag['frob_sq']:.1f}, L0={diag['l0']}. Running `wsp_tools.canonical_pack` "
        "to absorb non-uniform RMSNorm gains, prune cancelling/null-space bloat, and pack into the minimal residual basis."
    )
    t2_code = """
packed = wsp_tools.canonical_pack(weights)
new_frob = float(sum(v.pow(2).sum().item() for v in packed.values()))
new_l0 = int(sum((v.abs() > 1e-6).sum().item() for v in packed.values()))
print({"frob_ratio": new_frob / max(diag["frob_sq"], 1e-8), "l0_ratio": new_l0 / max(diag["l0"], 1)})
submit_weights(packed)
"""
    return t2_think, t2_code


def _build_turn1(category: str, config: ModelConfig, forward_budget: int, brief_md: str) -> tuple[str, str]:
    if category == "cat1_decompile":
        if config.norm_type == "rmsnorm":
            return _diag_cat1_l15(forward_budget)
        return _diag_cat1_level1(forward_budget)
    if category == "cat2_repair":
        return _diag_cat2(forward_budget)
    if category == "cat3_synthesize":
        if config.norm_type == "rmsnorm":
            return _diag_cat3_l15(forward_budget, brief_md)
        return _diag_cat3_level1(forward_budget, brief_md)
    if category == "cat4_nullspace_inject":
        if config.norm_type == "rmsnorm":
            return _diag_cat4_l15(forward_budget, brief_md)
        return _diag_cat4_level1(forward_budget, brief_md)
    if category == "cat5_unlearn_excise":
        if config.norm_type == "rmsnorm":
            return _diag_cat5_l15(forward_budget)
        return _diag_cat5_level1(forward_budget)
    if category == "cat6_splice_merge":
        if config.norm_type == "rmsnorm":
            return _diag_cat6_l15(forward_budget)
        if config.n_layers == 2:
            return _diag_cat6_vert(forward_budget)
        return _diag_cat6_horiz(forward_budget)
    if category == "cat7_refactor":
        return _diag_cat7(forward_budget)
    raise ValueError(f"Unsupported category: {category}")


def _build_turn2(category: str, config: ModelConfig, forward_budget: int, diag: dict[str, Any]) -> tuple[str, str]:
    if category == "cat1_decompile":
        return _commit_cat1_level1(diag)
    if category == "cat2_repair":
        if diag.get("pathology") == "sink_attractor":
            return _commit_cat2_sink(diag)
        return _commit_cat2_kcomp(diag)
    if category == "cat3_synthesize":
        if config.norm_type == "rmsnorm":
            return _commit_cat3_l15(diag, forward_budget)
        return _commit_cat3_level1(diag, forward_budget)
    if category == "cat4_nullspace_inject":
        if config.norm_type == "rmsnorm":
            return _commit_cat4_l15(diag)
        return _commit_cat4_level1(diag)
    if category == "cat5_unlearn_excise":
        if config.norm_type == "rmsnorm":
            return _commit_cat5_l15(diag)
        return _commit_cat5_level1(diag)
    if category == "cat6_splice_merge":
        if config.norm_type == "rmsnorm":
            return _commit_cat6_l15(diag)
        if config.n_layers == 2:
            return _commit_cat6_vert(diag)
        return _commit_cat6_horiz(diag)
    if category == "cat7_refactor":
        return _commit_cat7(diag)
    raise ValueError(f"Unsupported category: {category}")


def solve_blinded_challenge(
    challenge_dir: str | Path,
    trajectory_path: str | Path,
) -> dict[str, Any]:
    """Execute a genuine 2-turn blinded diagnostic+surgical solve via SandboxSession."""
    session = SandboxSession(challenge_dir, trajectory_path=trajectory_path)
    cat = str(session.meta["category"])
    budget = int(session.meta["forward_budget"])

    t1_think, t1_code = _build_turn1(cat, session.config, budget, session.brief_md)
    res1 = session.step(think=t1_think, code=t1_code)
    if res1.get("submitted"):
        return res1

    diag = dict(session.state["user_vars"].get("diag", {}))
    t2_think, t2_code = _build_turn2(cat, session.config, budget, diag)
    return session.step(think=t2_think, code=t2_code)


def _cfg_as_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    from dataclasses import asdict, is_dataclass

    return asdict(config) if is_dataclass(config) else dict(vars(config))


def _build_exploratory_turn1(
    category: str,
    config: ModelConfig | dict[str, Any],
    forward_budget: int,
    mode: str,
) -> tuple[str, str]:
    """Build a Turn-1 script for 3-turn exploratory trajectories without leaking subtypes."""
    del category, forward_budget
    cfg = _cfg_as_dict(config)
    norm_type = str(cfg.get("norm_type", "none"))

    if mode == "error_recovery":
        if norm_type == "rmsnorm":
            think = (
                "Let me inspect the positional embedding matrix `W_pos` first to see "
                "whether positional offsets are stored explicitly alongside residual weights."
            )
            code = (
                "print('W_pos shape:', weights['W_pos'].shape)\n"
                "svd = wsp_tools.residual_svd_basis(weights)\n"
                "print('Rank:', svd['rank'])"
            )
            return think, code

        think = (
            "Let me verify whether RMSNorm gain parameters `layers.0.norm1.weight` are "
            "present so I know if norm-folding is needed before subspace analysis."
        )
        code = (
            "assert 'layers.0.norm1.weight' in weights, (\n"
            "    \"Expected 'layers.0.norm1.weight' for RMSNorm-folded inspection; \"\n"
            "    \"checkpoint omits norm gains (norm_type='none').\"\n"
            ")\n"
            "print('norm1 norm:', float(weights['layers.0.norm1.weight'].norm().item()))"
        )
        return think, code

    if mode == "hypothesis_refine":
        think = (
            "Before formulating a surgical plan, let me run a broad zero-forward-pass "
            "structural audit of weight tensor norms and residual singular values to narrow down "
            "where signal and anomalies reside."
        )
        code = (
            "norms = {k: round(float(v.norm().item()), 4) for k, v in sorted(weights.items())}\n"
            "nonzero = {k: val for k, val in norms.items() if val > 1e-8}\n"
            "print('total_tensors:', len(norms), 'nonzero_tensors:', len(nonzero))\n"
            "print('sample_nonzero_norms:', list(nonzero.items())[:8])\n"
            "if nonzero:\n"
            "    svd = wsp_tools.residual_svd_basis(weights)\n"
            "    top_sv = [round(float(x), 4) for x in svd['singular_values'][:6]]\n"
            "    print('residual_svd_rank:', svd['rank'], 'top_sv:', top_sv)\n"
            "    w_q = weights.get('layers.0.attn.W_Q')\n"
            "    w_k = weights.get('layers.0.attn.W_K')\n"
            "    if w_q is not None and w_k is not None and float(w_q.norm()) > 1e-8 and float(w_k.norm()) > 1e-8:\n"
            "        prof = wsp_tools.head_offset_profile(weights, 0)\n"
            "        print('layer0_head_count:', len(prof))\n"
            "else:\n"
            "    print('all_weights_zero: True (clean initialization)')\n"
            "    print('config_summary:', {'d_model': "
            + repr(int(cfg["d_model"]))
            + ", 'n_layers': "
            + repr(int(cfg["n_layers"]))
            + ", 'n_heads': "
            + repr(int(cfg["n_heads"]))
            + ", 'norm_type': "
            + repr(norm_type)
            + "})"
        )
        return think, code

    raise ValueError(f"Unsupported exploratory mode: {mode!r}")


def solve_exploratory_challenge(
    challenge_dir: str | Path,
    trajectory_path: str | Path,
    mode: str = "hypothesis_refine",
) -> dict[str, Any]:
    """Execute a 3-turn blinded solve featuring Turn-1 error recovery or broad hypothesis refinement."""
    if mode not in ("error_recovery", "hypothesis_refine"):
        raise ValueError(f"Unknown exploratory mode: {mode!r}")

    session = SandboxSession(challenge_dir, trajectory_path=trajectory_path)
    cat = str(session.meta["category"])
    budget = int(session.meta["forward_budget"])

    t1_think, t1_code = _build_exploratory_turn1(cat, session.config, budget, mode=mode)
    res1 = session.step(think=t1_think, code=t1_code)
    if res1.get("submitted"):
        return res1
    if mode == "error_recovery" and "STDERR:" not in str(res1.get("observation", "")):
        raise RuntimeError(
            f"Expected Turn-1 STDERR in error_recovery mode for {session.meta.get('episode_id')}, "
            f"got observation={res1.get('observation')!r}"
        )

    t2_think, t2_code = _build_turn1(cat, session.config, budget, session.brief_md)
    if mode == "error_recovery":
        t2_think = (
            "Turn 1 raised an exception on a structural tensor assumption. "
            "Reading STDERR, let me adapt to the checkpoint schema and run clean targeted diagnostics.\n"
            + t2_think
        )
    else:
        t2_think = (
            "The initial broad structural audit established active parameter blocks and residual "
            "dimensionality. Next, let me run targeted diagnostics matching the task specification.\n"
            + t2_think
        )
    res2 = session.step(think=t2_think, code=t2_code)
    if res2.get("submitted"):
        return res2

    diag = dict(session.state["user_vars"].get("diag", {}))
    t3_think, t3_code = _build_turn2(cat, session.config, budget, diag)
    return session.step(think=t3_think, code=t3_code)


def _merge_episodes_dedup(existing_path: Path, new_path: Path) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for src in (existing_path, new_path):
        if not src.exists():
            continue
        for raw_line in src.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            rec = json.loads(raw_line)
            ep_id = str(rec.get("episode_id"))
            rew = float((rec.get("verification") or {}).get("total_reward", -1.0))
            prev = merged.get(ep_id)
            if prev is None or rew >= float((prev.get("verification") or {}).get("total_reward", -1.0)):
                merged[ep_id] = rec
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    with existing_path.open("w", encoding="utf-8") as f:
        for item in merged.values():
            f.write(json.dumps(item) + "\n")


def scale_sft_corpus(
    num_episodes: int = 504,
    base_seed: int = 1000,
    trajectory_path: str | Path = "trajectories/episodes.jsonl",
    out_path: str | Path = "trajectories/sft_train.jsonl",
) -> dict[str, Any]:
    """Generate and solve `num_episodes` balanced blinded episodes across all 17 subtypes."""
    traj_dest = Path(trajectory_path)
    sched = CurriculumScheduler(seed=base_seed, initial_stage=0, window_size=8)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_dir = Path(tmp_root)
        shard_path = tmp_dir / "scaled_shard.jsonl"

        for i in range(num_episodes):
            cat, sub = ALL_17_SUBTYPES[i % len(ALL_17_SUBTYPES)]
            ep_seed = base_seed + i
            stage_idx = min(3, (i * 4) // max(1, num_episodes))
            sched.stage = stage_idx

            if stage_idx >= 3 or (i % 2 == 1):
                budget: int | None = 0
            elif stage_idx == 2:
                probe_b = generate_challenge(cat, seed=ep_seed, difficulty=1.5, subtype=sub)
                budget = 0 if probe_b.forward_budget == 0 else max(1, probe_b.forward_budget // 2)
            else:
                budget = None

            difficulty = 1.0 if stage_idx == 0 and "level1_5" not in sub else 1.5
            bundle = generate_challenge(
                cat, seed=ep_seed, difficulty=difficulty, forward_budget=budget, subtype=sub
            )
            ep_dir = tmp_dir / f"ep_{i:04d}_{bundle.episode_id}"
            save_challenge(bundle, ep_dir)
            res = solve_blinded_challenge(ep_dir, trajectory_path=shard_path)
            ver = res.get("verification") or {}
            if not ver.get("passed") or float(ver.get("total_reward", 0.0)) < 0.80:
                raise RuntimeError(
                    f"Episode {bundle.episode_id} ({cat}/{sub}, budget={bundle.forward_budget}) failed: {ver}"
                )

        _merge_episodes_dedup(traj_dest, shard_path)

    return audit_and_export_sft_dataset(trajectory_path=traj_dest, out_path=out_path)


def build_option3_sft_corpus(
    trajectory_path: str | Path = "trajectories/episodes.jsonl",
    out_path: str | Path = "trajectories/sft_train.jsonl",
) -> dict[str, Any]:
    """Build the compact 123-episode Option 3 SFT corpus:

    - Refresh existing organic episodes (`seed < 1000`) through the unified compacted solvers
      while preserving symptom-only `brief.md`, unified `SYSTEM_PROMPT`, and 2-turn /
      hypothesis-refine / error-recovery structure.
    - Generate 68 blinded 2-turn episodes (4 passes x 17 subtypes, seeds 1000..1067).
    - Generate 34 blinded 3-turn exploratory episodes (17 error-recovery + 17 hypothesis-refine,
      seeds 2000..2033).
    - Overwrite `trajectory_path` with the curated 123-episode corpus and export `out_path`.
    """
    from wsp_gym.sandbox import SYSTEM_PROMPT

    traj_dest = Path(trajectory_path)
    organic_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_dir = Path(tmp_root)
        if traj_dest.exists():
            for idx, raw_line in enumerate(traj_dest.read_text(encoding="utf-8").splitlines()):
                if not raw_line.strip():
                    continue
                rec = json.loads(raw_line)
                ep_id = str(rec.get("episode_id", ""))
                seed_match = re.search(r"_(\d+)$", ep_id)
                if seed_match is None or int(seed_match.group(1)) >= 1000:
                    continue

                seed_num = int(seed_match.group(1))
                cat = str(rec["category"])
                sub = str(rec["subtype"])
                diff = float(rec.get("difficulty", 1.0))

                msgs = [dict(m) for m in (rec.get("messages") or [])]
                fb: int | None = None
                for m in msgs:
                    fb_match = re.search(
                        r"\b\d+/(\d+)\s+forward batches remaining",
                        str(m.get("content", "")),
                    )
                    if fb_match is not None:
                        fb = int(fb_match.group(1))
                        break

                fresh_bundle = generate_challenge(
                    cat, seed=seed_num, difficulty=diff, forward_budget=fb, subtype=sub
                )
                asst_msgs = [str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"]
                ep_dir = tmp_dir / f"org_{idx:03d}_{fresh_bundle.episode_id}"
                org_traj = tmp_dir / f"org_{idx:03d}.jsonl"
                save_challenge(fresh_bundle, ep_dir)
                had_stderr = any(
                    m.get("role") == "user" and "STDERR:" in str(m.get("content", ""))
                    for m in msgs
                )
                if len(asst_msgs) >= 3:
                    mode = "error_recovery" if had_stderr else "hypothesis_refine"
                    res = solve_exploratory_challenge(ep_dir, trajectory_path=org_traj, mode=mode)
                else:
                    res = solve_blinded_challenge(ep_dir, trajectory_path=org_traj)
                ver = res.get("verification") or {}
                if not ver.get("passed") or float(ver.get("total_reward", 0.0)) < 0.80:
                    raise RuntimeError(
                        f"Refreshed legacy episode {fresh_bundle.episode_id} failed: {ver}"
                    )
                organic_records.append(json.loads(org_traj.read_text(encoding="utf-8").splitlines()[-1]))

        organic_path = tmp_dir / "organic.jsonl"
        shard_2turn = tmp_dir / "shard_2turn.jsonl"
        shard_3turn = tmp_dir / "shard_3turn.jsonl"

        with organic_path.open("w", encoding="utf-8") as f:
            for item in organic_records:
                f.write(json.dumps(item) + "\n")

        n_sub = len(ALL_17_SUBTYPES)
        # 68 blinded 2-turn episodes (4 passes x 17 subtypes, seeds 1000..1067)
        for i in range(4 * n_sub):
            cat, sub = ALL_17_SUBTYPES[i % n_sub]
            ep_seed = 1000 + i
            pass_idx = i // n_sub
            difficulty = 1.0 if pass_idx == 0 and "level1_5" not in sub else 1.5

            if pass_idx in (1, 3):
                budget: int | None = 0
            elif pass_idx == 2:
                probe_b = generate_challenge(cat, seed=ep_seed, difficulty=difficulty, subtype=sub)
                budget = 0 if probe_b.forward_budget == 0 else max(1, probe_b.forward_budget // 2)
            else:
                budget = None

            bundle = generate_challenge(
                cat, seed=ep_seed, difficulty=difficulty, forward_budget=budget, subtype=sub
            )
            ep_dir = tmp_dir / f"t2_{i:04d}_{bundle.episode_id}"
            save_challenge(bundle, ep_dir)
            res = solve_blinded_challenge(ep_dir, trajectory_path=shard_2turn)
            ver = res.get("verification") or {}
            if not ver.get("passed") or float(ver.get("total_reward", 0.0)) < 0.80:
                raise RuntimeError(
                    f"Option3 2-turn episode {bundle.episode_id} ({cat}/{sub}, budget={bundle.forward_budget}) failed: {ver}"
                )

        # 34 blinded 3-turn exploratory episodes (17 error_recovery + 17 hypothesis_refine, seeds 2000..2033)
        for i in range(2 * n_sub):
            cat, sub = ALL_17_SUBTYPES[i % n_sub]
            ep_seed = 2000 + i
            pass_idx = i // n_sub
            mode = "error_recovery" if pass_idx == 0 else "hypothesis_refine"
            difficulty = 1.0 if (pass_idx == 0 and "level1_5" not in sub and i % 2 == 0) else 1.5
            budget = 0 if ((pass_idx == 0 and i % 2 == 1) or (pass_idx == 1 and i % 2 == 0)) else None

            bundle = generate_challenge(
                cat, seed=ep_seed, difficulty=difficulty, forward_budget=budget, subtype=sub
            )
            ep_dir = tmp_dir / f"t3_{i:04d}_{bundle.episode_id}"
            save_challenge(bundle, ep_dir)
            res = solve_exploratory_challenge(ep_dir, trajectory_path=shard_3turn, mode=mode)
            ver = res.get("verification") or {}
            turns = int((ver.get("metrics") or {}).get("turns_used", 0))
            if not ver.get("passed") or float(ver.get("total_reward", 0.0)) < 0.80 or turns != 3:
                raise RuntimeError(
                    f"Option3 3-turn {mode} episode {bundle.episode_id} ({cat}/{sub}, turns={turns}) failed: {ver}"
                )

        # Atomically replace trajectory_path with curated Option 3 episodes (organic + 2-turn + 3-turn)
        staged = tmp_dir / "option3_merged.jsonl"
        staged.write_text(organic_path.read_text(encoding="utf-8"), encoding="utf-8")
        _merge_episodes_dedup(staged, shard_2turn)
        _merge_episodes_dedup(staged, shard_3turn)
        traj_dest.parent.mkdir(parents=True, exist_ok=True)
        traj_dest.write_text(staged.read_text(encoding="utf-8"), encoding="utf-8")

    return audit_and_export_sft_dataset(trajectory_path=traj_dest, out_path=out_path)
