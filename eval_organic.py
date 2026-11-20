"""Phase 5 Organic Challenge Evaluator (`norm_baseline`, `hybrid_ref`, and `hf` checkpoints).

Evaluates static norm-outlier heuristics vs general-purpose hybrid causal/nullspace primitives
(`probe_attribution`, `covariance_nullspace`, `solve_nullspace_ridge`) and prior Phase 2.5/4 LoRA
agents across the organically LoRA/AdamW-trained Phase 5 suite (`100m`, `SmolLM2-135M`, `Qwen2.5-0.5B`).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Callable

import torch

from eval import make_policy_fn
from wsp_gym.generators import ChallengeBundle
from wsp_gym.level2 import _trim_host_ram
from wsp_gym.level3 import _evict_safetensors_page_cache
from wsp_gym.model import ModelConfig
from wsp_gym.organic import (
    ORGANIC_PRETRAINED_MODELS,
    ORGANIC_SUBTYPES,
    ORGANIC_SYNTHETIC_SCALES,
    _ORGANIC_BASE_CACHE,
    _load_organic_base,
    generate_organic_challenge,
)
from wsp_gym.training import GRPORolloutEnv, compute_shaped_reward
from wsp_gym.verifier import verify_bundle


def _extract_brief_calib_json(brief_md: str) -> dict[str, Any]:
    match = re.search(r"## Calibration Samples\s*```json\s*(.*?)\s*```", brief_md, flags=re.DOTALL)
    if not match:
        raise ValueError("Missing '## Calibration Samples' JSON block in brief_md.")
    return json.loads(match.group(1))


def _resolve_suite_models(suite: str, synthetic_scale: str) -> list[str]:
    if suite == "synthetic":
        return [synthetic_scale]
    if suite == "smollm2":
        return ["smollm2_135m"]
    if suite == "qwen25":
        return ["qwen25_05b"]
    if suite == "all":
        return [synthetic_scale, "smollm2_135m", "qwen25_05b"]
    raise ValueError(f"Unknown organic suite: {suite!r}")


def save_cached_organic_bundle(
    bundle: ChallengeBundle,
    model_key: str,
    seed: int,
    cache_path: str | Path,
) -> Path:
    """Persist an organic ChallengeBundle as a sparse weight diff against its clean base."""
    _, clean_w, _, _ = _load_organic_base(model_key, seed)
    diff_weights: dict[str, torch.Tensor] = {}
    for k, v in bundle.public_weights.items():
        if k not in clean_w or float((v.float() - clean_w[k].float()).abs().max().item()) > 0.0:
            diff_weights[k] = v.detach().cpu().contiguous()

    extra_diff: dict[str, dict[str, torch.Tensor]] = {}
    for ename, ew in (bundle.extra_public_weights or {}).items():
        d_sub: dict[str, torch.Tensor] = {}
        for k, v in ew.items():
            if k not in clean_w or float((v.float() - clean_w[k].float()).abs().max().item()) > 0.0:
                d_sub[k] = v.detach().cpu().contiguous()
        extra_diff[ename] = d_sub

    dst = Path(cache_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": bundle.episode_id,
        "category": bundle.category,
        "subtype": bundle.subtype,
        "difficulty": bundle.difficulty,
        "forward_budget": bundle.forward_budget,
        "config": bundle.config.to_dict(),
        "brief_md": bundle.brief_md,
        "diff_weights": diff_weights,
        "extra_diff_weights": extra_diff,
        "secret": bundle.secret,
        "model_key": model_key,
        "seed": int(seed),
    }
    torch.save(payload, dst)
    return dst


def load_cached_organic_bundle(cache_path: str | Path, model_key: str, seed: int) -> ChallengeBundle:
    """Reconstruct a full ChallengeBundle from a sparse diff cache file."""
    data = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
    _, clean_w, _, _ = _load_organic_base(model_key, seed)
    pub_w = {k: v.clone() for k, v in clean_w.items()}
    for k, dv in data["diff_weights"].items():
        pub_w[k] = dv.clone()

    extra_w: dict[str, dict[str, torch.Tensor]] = {}
    for ename, ediff in data.get("extra_diff_weights", {}).items():
        ew = {k: v.clone() for k, v in clean_w.items()}
        for k, dv in ediff.items():
            ew[k] = dv.clone()
        extra_w[ename] = ew

    return ChallengeBundle(
        episode_id=str(data["episode_id"]),
        category=str(data["category"]),
        subtype=str(data["subtype"]),
        difficulty=float(data["difficulty"]),
        forward_budget=int(data["forward_budget"]),
        config=ModelConfig.from_dict(data["config"]),
        brief_md=str(data["brief_md"]),
        public_weights=pub_w,
        extra_public_weights=extra_w if extra_w else None,
        secret=data["secret"],
    )


def _get_or_create_organic_bundle(
    model_key: str,
    subtype: str,
    seed: int,
    cache_dir: Path | None,
) -> ChallengeBundle:
    if cache_dir is not None:
        cpath = cache_dir / f"{model_key}__{subtype}__{int(seed)}.pt"
        if cpath.exists():
            return load_cached_organic_bundle(cpath, model_key, seed)
        bundle = generate_organic_challenge(subtype=subtype, model_key=model_key, seed=seed)
        save_cached_organic_bundle(bundle, model_key, seed, cpath)
        return bundle
    return generate_organic_challenge(subtype=subtype, model_key=model_key, seed=seed)


# ---------------------------------------------------------------------------
# Blinded REPL helpers embedded into SandboxSession scripts
# ---------------------------------------------------------------------------

_REPL_MATH_HELPERS = """
import json
import math
import torch.nn.functional as F

def _parse_calib(md):
    f3 = chr(96) * 3
    return json.loads(md.split("## Calibration Samples", 1)[1].split(f3 + "json", 1)[1].split(f3, 1)[0].strip())

def _apply_norm_local(x, norm_type, nw, eps=1e-6):
    if norm_type == "rmsnorm":
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    elif norm_type == "layernorm":
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + eps)
    else:
        return x
    return out * nw if nw is not None else out

def _apply_rope_local(q, k, theta=10000.0):
    sl = int(q.shape[-2])
    dh = int(q.shape[-1])
    half = dh // 2
    inv_f = 1.0 / (float(theta) ** (torch.arange(0, dh, 2, dtype=torch.float32, device=q.device) / float(dh)))
    pos = torch.arange(sl, dtype=torch.float32, device=q.device)
    freqs = torch.outer(pos, inv_f)
    cos = freqs.cos().view(1, 1, sl, half)
    sin = freqs.sin().view(1, 1, sl, half)
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    return torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1), torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

def _attn_sublayer(h_in, wd, cfg_d, l_idx):
    p = f"blocks.{l_idx}"
    dm = int(cfg_d["d_model"])
    nh = int(cfg_d["n_heads"])
    nkv = int(cfg_d["n_kv_heads"]) if cfg_d.get("n_kv_heads") is not None else nh
    dh = int(cfg_d["d_head"])
    gs = max(1, nh // max(1, nkv))
    bsz, sl, _ = h_in.shape
    xn1 = _apply_norm_local(h_in, cfg_d.get("norm_type", "rmsnorm"), wd.get(f"{p}.norm1.weight"), float(cfg_d.get("rms_eps", 1e-6)))
    wq = wd[f"{p}.attn.W_Q"]
    wk = wd[f"{p}.attn.W_K"]
    wv = wd[f"{p}.attn.W_V"]
    wo = wd[f"{p}.attn.W_O"]
    q = (xn1 @ wq.permute(1, 0, 2).reshape(dm, nh * dh)).view(bsz, sl, nh, dh).transpose(1, 2)
    k = (xn1 @ wk.permute(1, 0, 2).reshape(dm, nkv * dh)).view(bsz, sl, nkv, dh).transpose(1, 2)
    v = (xn1 @ wv.permute(1, 0, 2).reshape(dm, nkv * dh)).view(bsz, sl, nkv, dh).transpose(1, 2)
    if f"{p}.attn.b_Q" in wd:
        q = q + wd[f"{p}.attn.b_Q"].view(1, nh, 1, dh)
    if f"{p}.attn.b_K" in wd:
        k = k + wd[f"{p}.attn.b_K"].view(1, nkv, 1, dh)
    if f"{p}.attn.b_V" in wd:
        v = v + wd[f"{p}.attn.b_V"].view(1, nkv, 1, dh)
    if cfg_d.get("pos_type", "rope") == "rope":
        q, k = _apply_rope_local(q, k, float(cfg_d.get("rope_theta", 10000.0)))
    k_e = k.repeat_interleave(gs, dim=1) if nkv < nh else k
    v_e = v.repeat_interleave(gs, dim=1) if nkv < nh else v
    sc = torch.matmul(q, k_e.transpose(-2, -1)) / math.sqrt(max(1, dh))
    if cfg_d.get("causal", True):
        cm = torch.triu(torch.full((sl, sl), -1e9, dtype=h_in.dtype, device=h_in.device), diagonal=1)
        sc = sc + cm.view(1, 1, sl, sl)
    pat = torch.softmax(sc, dim=-1)
    ctx = torch.matmul(pat, v_e).transpose(1, 2).contiguous().view(bsz, sl, nh * dh)
    out = ctx @ wo.reshape(nh * dh, dm)
    return ctx, out

def _mlp_sublayer(h_mid, wd, cfg_d, l_idx):
    p = f"blocks.{l_idx}"
    xn2 = _apply_norm_local(h_mid, cfg_d.get("norm_type", "rmsnorm"), wd.get(f"{p}.norm2.weight"), float(cfg_d.get("rms_eps", 1e-6)))
    pre = xn2 @ wd[f"{p}.mlp.W_in"] + wd[f"{p}.mlp.b_in"]
    if cfg_d.get("mlp_type", "swiglu") == "swiglu" and f"{p}.mlp.W_gate" in wd:
        hid = F.silu(xn2 @ wd[f"{p}.mlp.W_gate"]) * pre
    else:
        act = cfg_d.get("act_type", "silu")
        hid = F.silu(pre) if act == "silu" else F.gelu(pre)
    out = hid @ wd[f"{p}.mlp.W_out"] + wd[f"{p}.mlp.b_out"]
    return hid, out

def _final_logits(h_last_post, wd, cfg_d):
    x_fin = _apply_norm_local(h_last_post, cfg_d.get("norm_type", "rmsnorm"), wd.get("norm_final.weight"), float(cfg_d.get("rms_eps", 1e-6)))
    return x_fin @ wd["W_U"] + wd["b_U"]

def _signed_norm_inv_scale(h_last_post, wd, cfg_d):
    eps = float(cfg_d.get("rms_eps", 1e-6))
    nt = cfg_d.get("norm_type", "rmsnorm")
    if nt == "rmsnorm":
        inv = torch.rsqrt(h_last_post.pow(2).mean(dim=-1, keepdim=True) + eps)
    elif nt == "layernorm":
        inv = torch.rsqrt(h_last_post.var(dim=-1, unbiased=False, keepdim=True) + eps)
    else:
        inv = torch.ones((h_last_post.shape[0], 1), dtype=torch.float32, device=h_last_post.device)
    nf = wd.get("norm_final.weight")
    if nf is not None and nt != "none":
        nf_f = nf.float()
        snf = torch.where(nf_f >= 0, nf_f.clamp_min(1e-6), nf_f.clamp_max(-1e-6))
    else:
        snf = torch.ones((int(cfg_d["d_model"]),), dtype=torch.float32, device=h_last_post.device)
    return inv, snf

def _span_clean_prompts(cx, extra_safe, forbid_set, vocab_size, weights_ref, n_boot=256, seed=17):
    lo, hi = int(cx.min().item()), int(cx.max().item()) + 1
    g = torch.Generator().manual_seed(int(seed))
    sl_len = int(cx.shape[1])
    if hi - lo <= 512 and int(vocab_size) <= 8192:
        lo_s = max(16, lo - 8)
        hi_s = min(int(vocab_size) - 32, max(hi + 24, 240))
        allow_list = [t for t in range(lo_s, hi_s) if t not in forbid_set]
        allow = torch.tensor(allow_list, dtype=torch.long)
        return allow[torch.randint(0, int(allow.numel()), (int(n_boot), sl_len), generator=g)]
    clean_rows = []
    for r in cx.tolist():
        rc = [int(t) for t in r if int(t) not in forbid_set]
        if len(rc) < sl_len:
            rc = (rc * ((sl_len // max(1, len(rc))) + 2))[:sl_len] if rc else [t for t in range(260, 260 + sl_len)]
        clean_rows.append(rc[:sl_len])
    ct = torch.tensor(clean_rows, dtype=torch.long)
    n_row = int(n_boot) // 2
    p0 = ct[torch.randint(0, int(ct.shape[0]), (n_row,), generator=g)].clone()
    half = sl_len // 2
    p0[n_row // 2:, half:] = ct[torch.randint(0, int(ct.shape[0]), (n_row - n_row // 2,), generator=g), half:]
    st = [int(t) for r in clean_rows for t in r] + [int(t) for t in extra_safe if int(t) not in forbid_set]
    wt = torch.tensor([st[i:i + sl_len] for i in range(max(1, len(st) - sl_len + 1))], dtype=torch.long)
    n1 = int(n_boot) // 4
    p1 = wt[torch.randint(0, int(wt.shape[0]), (n1,), generator=g)]
    we_end = min(int(vocab_size), 2048)
    we_rn = weights_ref["W_E"][260:we_end].float().norm(dim=-1)
    med_e = float(we_rn.median().clamp_min(1e-6).item()) if int(we_rn.numel()) > 0 else 1.0
    common_ids = (260 + (we_rn <= 1.35 * med_e).nonzero(as_tuple=True)[0]).tolist() if int(we_rn.numel()) > 0 else []
    mp = torch.tensor(sorted(t for t in (set(st) | set(common_ids)) if t not in forbid_set and 0 <= t < int(vocab_size)), dtype=torch.long)
    p2 = mp[torch.randint(0, int(mp.numel()), (int(n_boot) - n_row - n1, sl_len), generator=g)]
    return torch.cat([p0, p1, p2], dim=0)

def _fit_lowrank_with_grams(weights, config, c_attn_vo, c_attn_full, c_mlp, inp_sets, ref_fn, steps=110, lr=5.5e-3, rk=12, c_weight=28.0, qk_mult=2.0, r_idx=-1, out_gram_w=0.0, cpu_max_steps=115):
    L = int(config["n_layers"])
    dm = int(config["d_model"])
    nh = int(config["n_heads"])
    dh = int(config["d_head"])
    nkv = int(config["n_kv_heads"] or nh)
    all_l = set(c_attn_vo) | set(c_attn_full) | set(c_mlp)
    l_min = min(all_l)
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda") if use_cuda else torch.device("cpu")
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if use_cuda else None
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        def _to_lmin(toks):
            with torch.no_grad():
                w_prefix = {"W_E": weights["W_E"].float().to(dev)}
                for li in range(l_min):
                    p = f"blocks.{li}"
                    for sf in ("norm1.weight", "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O", "attn.b_Q", "attn.b_K", "attn.b_V", "norm2.weight", "mlp.W_in", "mlp.b_in", "mlp.W_gate", "mlp.W_out", "mlp.b_out"):
                        k = f"{p}.{sf}"
                        if k in weights:
                            w_prefix[k] = weights[k].float().to(dev)
                h = w_prefix["W_E"][toks.to(dev)]
                for li in range(l_min):
                    _, al = _attn_sublayer(h, w_prefix, config, li)
                    hm = h + al
                    _, ml = _mlp_sublayer(hm, w_prefix, config, li)
                    h = hm + ml
                res = h.detach()
                del w_prefix
                return res

        h_enters = [_to_lmin(t) for t in inp_sets]
        w_tail = {
            "W_U": weights["W_U"].float().detach().to(dev),
            "b_U": weights["b_U"].float().detach().to(dev),
        }
        if "norm_final.weight" in weights:
            w_tail["norm_final.weight"] = weights["norm_final.weight"].float().detach().to(dev)
        for li in range(l_min, L):
            p = f"blocks.{li}"
            for sf in ("norm1.weight", "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O", "attn.b_Q", "attn.b_K", "attn.b_V", "norm2.weight", "mlp.W_in", "mlp.b_in", "mlp.W_gate", "mlp.W_out", "mlp.b_out"):
                k = f"{p}.{sf}"
                if k in weights:
                    w_tail[k] = weights[k].float().detach().to(dev)

        cov_grams = {}
        with torch.no_grad():
            hc = h_enters[r_idx].clone()
            eps = float(config.get("rms_eps", 1e-6))
            nt = config.get("norm_type", "rmsnorm")
            for li in range(l_min, L):
                p = f"blocks.{li}"
                xn1 = _apply_norm_local(hc, nt, w_tail.get(f"{p}.norm1.weight"), eps)
                ctx, al = _attn_sublayer(hc, w_tail, config, li)
                if li in c_attn_vo or li in c_attn_full:
                    xr1 = xn1.reshape(-1, dm)
                    cov_grams[f"{p}.attn.qkv"] = (xr1.T @ xr1) / float(max(1, int(xr1.shape[0])))
                    cr1 = ctx.reshape(-1, nh * dh)
                    cov_grams[f"{p}.attn.wo"] = (cr1.T @ cr1) / float(max(1, int(cr1.shape[0])))
                hm = hc + al
                xn2 = _apply_norm_local(hm, nt, w_tail.get(f"{p}.norm2.weight"), eps)
                hid, ml = _mlp_sublayer(hm, w_tail, config, li)
                if li in c_mlp:
                    x2 = xn2.reshape(-1, dm)
                    cov_grams[f"{p}.mlp.in"] = (x2.T @ x2) / float(max(1, int(x2.shape[0])))
                    hd = hid.reshape(-1, int(hid.shape[-1]))
                    cov_grams[f"{p}.mlp.out"] = (hd.T @ hd) / float(max(1, int(hd.shape[0])))
                hc = hm + ml
            nf_w = w_tail.get("norm_final.weight")
            wu_eff = (nf_w.unsqueeze(1) * w_tail["W_U"]) if nf_w is not None else w_tail["W_U"]
            g_wu = wu_eff @ wu_eff.T
            g_wu = g_wu / g_wu.diagonal().mean().clamp_min(1e-6)
            rc_flat = hc.reshape(-1, dm)
            g_rc = (rc_flat.T @ rc_flat) / float(max(1, int(rc_flat.shape[0])))
            g_rc = g_rc / g_rc.diagonal().mean().clamp_min(1e-6)
            ogw = float(out_gram_w)
            g_out_res = ogw * (0.5 * g_wu + 0.5 * g_rc) + (1.0 - ogw + 0.02) * torch.eye(dm, device=dev)
            for gk_k in list(cov_grams.keys()):
                cov_grams[gk_k] = cov_grams[gk_k] / cov_grams[gk_k].diagonal().mean().clamp_min(1.0)

        def _fwd_enter(h_e, wd, ret_hooks=False):
            h = h_e
            hooks = {}
            for li in range(l_min, L):
                h_prev_l = h
                _, al = _attn_sublayer(h, wd, config, li)
                hm = h + al
                hid_l, ml = _mlp_sublayer(hm, wd, config, li)
                h = hm + ml
                if ret_hooks:
                    if li in c_attn_vo or li in c_attn_full:
                        hooks[f"post_{li}"] = h - h_prev_l
                    if li == max(0, L - 3):
                        hooks[f"res_{li}"] = h[:, -1, :]
                    if li >= max(0, L - 2):
                        hooks[f"hid_{li}"] = hid_l[:, -1, :]
                        hooks[f"res_{li}"] = h[:, -1, :]
            lg = _final_logits(h[:, -1, :], wd, config)
            return (lg, hooks) if ret_hooks else lg

        g = torch.Generator(device="cpu").manual_seed(71)
        adapters = {}
        params = []
        for li in sorted(c_mlp):
            p = f"blocks.{li}"
            for ms, gk in (("mlp.W_in", f"{p}.mlp.in"), ("mlp.W_gate", f"{p}.mlp.in"), ("mlp.W_out", f"{p}.mlp.out")):
                fk = f"{p}.{ms}"
                w0 = w_tail[fk]
                a = (torch.randn((int(w0.shape[0]), rk), generator=g) * (0.02 / math.sqrt(float(rk)))).to(dev).requires_grad_(True)
                bp = torch.zeros((rk, int(w0.shape[1])), dtype=torch.float32, device=dev, requires_grad=True)
                adapters[fk] = (w0, "2d", a, bp, gk, 1.0)
                params.extend([a, bp])
        for li in sorted(set(c_attn_vo) | set(c_attn_full)):
            p = f"blocks.{li}"
            specs = [
                ("attn.W_V", (dm, nkv * dh, "qk", f"{p}.attn.qkv", 1.0)),
                ("attn.W_O", (nh * dh, dm, "wo", f"{p}.attn.wo", 1.0)),
            ]
            if li in c_attn_full:
                specs = [
                    ("attn.W_Q", (dm, nh * dh, "qh", f"{p}.attn.qkv", qk_mult)),
                    ("attn.W_K", (dm, nkv * dh, "qk", f"{p}.attn.qkv", qk_mult)),
                ] + specs
            for ak, (di, do, md, gk, mult) in specs:
                fk = f"{p}.{ak}"
                w0 = w_tail[fk]
                a = (torch.randn((di, rk), generator=g) * (0.02 / math.sqrt(float(rk)))).to(dev).requires_grad_(True)
                bp = torch.zeros((rk, do), dtype=torch.float32, device=dev, requires_grad=True)
                adapters[fk] = (w0, md, a, bp, gk, mult)
                params.extend([a, bp])

        def _app():
            out = dict(w_tail)
            cp = torch.zeros((), dtype=torch.float32, device=dev)
            for fk, (w0, md, a, bp, gk, mult) in adapters.items():
                d = 2.0 * (a @ bp)
                if md == "2d":
                    out[fk] = w0 + d
                elif md == "qh":
                    out[fk] = w0 + d.view(dm, nh, dh).permute(1, 0, 2)
                elif md == "qk":
                    out[fk] = w0 + d.view(dm, nkv, dh).permute(1, 0, 2)
                else:
                    out[fk] = w0 + d.view(nh, dh, dm)
                gn = cov_grams[gk]
                bp_g = (bp @ g_out_res @ bp.T) if (ogw > 0.0 and (md == "wo" or fk.endswith("mlp.W_out"))) else (bp @ bp.T)
                cp = cp + float(mult) * ((a.T @ (gn @ a)) * bp_g).sum()
            return out, cp

        loss_cb = ref_fn(h_enters, w_tail, _fwd_enter, dev)
        is_wide_mlp = int(config["d_mlp"]) > 3000
        eff_steps = (int(steps) + 75 if is_wide_mlp else int(steps)) if use_cuda else min(int(steps), int(cpu_max_steps))
        lr_sc = float(lr) * min(1.0, (320.0 / float(dm)) ** 0.55) if int(config["vocab_size"]) <= 8192 else (float(lr) * 0.24 if is_wide_mlp else float(lr))
        opt = torch.optim.AdamW(params, lr=lr_sc, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=eff_steps, eta_min=(1.5e-4 if is_wide_mlp else 5e-4))
        g_mb = torch.Generator(device="cpu").manual_seed(103)
        for st in range(eff_steps):
            opt.zero_grad()
            wde, cp = _app()
            loss = loss_cb(st, g_mb, h_enters, wde, _fwd_enter, dev) + float(c_weight) * cp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.5)
            opt.step()
            sched.step()
        with torch.no_grad():
            wdf, _ = _app()
            for fk in adapters:
                weights[fk] = wdf[fk].detach().cpu().to(dtype=weights[fk].dtype)
        del w_tail, h_enters, cov_grams, adapters, params, opt, sched
    finally:
        if use_cuda and prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
"""


def build_norm_baseline_code(brief_md: str) -> str:
    """Generate blinded Phase 0-4 static heuristic code (`0.0%` expected on organic targets)."""
    if "Category 6" in brief_md:
        return (
            "weights = {k: (0.5 * (weights[k].float() + weights_B[k].float())).to(dtype=weights[k].dtype) for k in weights}\n"
            "submit_weights(weights)\n"
        )
    if "Category 4" in brief_md:
        return (
            "import json\n"
            "_f3 = chr(96) * 3\n"
            "cal = json.loads(brief_md.split('## Calibration Samples', 1)[1].split(_f3 + 'json', 1)[1].split(_f3, 1)[0].strip())\n"
            "ix = torch.tensor(cal['inject_calib_x'], dtype=torch.long)\n"
            "iy = torch.tensor(cal['inject_calib_y'], dtype=torch.long)\n"
            "basis = wsp_tools.residual_svd_basis(weights, config=config, tol=1e-4)\n"
            "vn = basis['V_null'] if int(basis['V_null'].shape[1]) > 0 else basis['V_active'][:, -8:]\n"
            "log, cache = probe_forward(ix, custom_weights=weights, return_cache=True)\n"
            "L = int(config['n_layers'])\n"
            "hm = cache[f'blocks.{L-1}.mlp.hidden'][:, -1, :].float()\n"
            "wu = weights['W_U'].float()\n"
            "nf = weights.get('norm_final.weight')\n"
            "nf = nf.float() if nf is not None else torch.ones(int(config['d_model']))\n"
            "p_null = vn @ vn.T\n"
            "tgt_dir = torch.stack([(wu[:, int(t)] / nf.clamp_min(1e-4)) for t in iy], dim=0)\n"
            "delta_r = (tgt_dir * 4.0) @ p_null\n"
            "dw = torch.linalg.lstsq(hm, delta_r).solution\n"
            "weights[f'blocks.{L-1}.mlp.W_out'] = (weights[f'blocks.{L-1}.mlp.W_out'].float() + dw).to(dtype=weights[f'blocks.{L-1}.mlp.W_out'].dtype)\n"
            "submit_weights(weights)\n"
        )
    return (
        "scan = wsp_tools.layer_anomaly_scan(weights, config=config, top_k=4)\n"
        "top_mlp = scan['top_mlp_anomalies'][0]\n"
        "top_l = int(top_mlp['layer'])\n"
        "j_bad = int(top_mlp['j'])\n"
        "sv = wsp_tools.batched_head_svdvals(weights, layers=[top_l], proj='W_O', config=config)[top_l]\n"
        "h_bad = int(sv[:, 0].argmax().item())\n"
        "weights[f'blocks.{top_l}.attn.W_O'] = weights[f'blocks.{top_l}.attn.W_O'].clone()\n"
        "weights[f'blocks.{top_l}.attn.W_O'][h_bad].zero_()\n"
        "wo = weights[f'blocks.{top_l}.mlp.W_out'].clone()\n"
        "wo[j_bad].zero_()\n"
        "weights[f'blocks.{top_l}.mlp.W_out'] = wo\n"
        "submit_weights(weights)\n"
    )


def build_hybrid_ref_turn1(brief_md: str) -> str:
    """Blinded general-purpose Turn 1 solver using `probe_attribution`, `covariance_nullspace`, and `solve_nullspace_ridge`."""
    if "Category 5" in brief_md:
        return (
            _REPL_MATH_HELPERS
            + """
cal = _parse_calib(brief_md)
neg_x = torch.tensor(cal["clean_calib_x"], dtype=torch.long)
ret_x = torch.tensor(cal.get("retain_calib_x", cal["clean_calib_x"]), dtype=torch.long)
if cal.get("mode") == "symptom_only":
    pos_x = torch.tensor(cal["incident_calib_x"], dtype=torch.long)
    tslots = [int(i) for i in (pos_x != neg_x).any(dim=0).nonzero(as_tuple=True)[0].tolist()]
    tseq = [int(pos_x[0, i].item()) for i in tslots]
    tgt_tok = None
else:
    tslots = [int(s) for s in cal.get("trigger_slots", [-9, -8])]
    tseq = [int(t) for t in cal["trigger_seq"]]
    pos_x = neg_x.clone()
    for s_i, t_id in zip(tslots, tseq):
        pos_x[:, s_i] = t_id
    tgt_tok = int(cal["c_bad"])

attr = wsp_tools.probe_attribution(weights, config, pos_x, neg_tokens=neg_x, target_tokens=tgt_tok, top_k=8)
c_bad = int(cal["c_bad"]) if "c_bad" in cal else int(torch.mode(torch.tensor(attr["pos_top1"])).values.item())
cov = wsp_tools.covariance_nullspace(weights, config, layer="final", calib_tokens=torch.cat([neg_x, ret_x], dim=0), site="resid_mid")

L = int(config["n_layers"])
dm = int(config["d_model"])
sl = int(neg_x.shape[1])
is_synth = int(config["vocab_size"]) <= 8192
tslots_pos = [s if s >= 0 else sl + s for s in tslots] if tslots else [0, 1]
forbid = set(tseq) | {c_bad} | set(neg_x[:, -4:].flatten().tolist())
cx_boot = _span_clean_prompts(ret_x, neg_x[:, :-4].flatten().tolist(), forbid, int(config["vocab_size"]), weights, n_boot=(1280 if is_synth else 256), seed=19)

nx_clean = neg_x.clone()
for r_i in range(int(nx_clean.shape[0])):
    for s_p in tslots_pos:
        if int(nx_clean[r_i, s_p].item()) in tseq:
            nx_clean[r_i, s_p] = cx_boot[r_i, s_p]
cross_rows, cross_y = [], []
y_nx = [int(t) for t in attr["neg_top1"]]
n_nx = int(nx_clean.shape[0])
for p_i in range(n_nx):
    for s_i in range(n_nx):
        rn = nx_clean[s_i].clone()
        rn[:sl - 4] = nx_clean[p_i, :sl - 4]
        cross_rows.append(rn)
        cross_y.append(y_nx[s_i])
_rep_c5 = 64 if (is_synth and dm >= 600) else 16
boot_cln = nx_clean.repeat(_rep_c5, 1)
boot_cln[:, :sl - 4] = cx_boot[:int(boot_cln.shape[0]), :sl - 4]
pure_cln = torch.cat([torch.stack(cross_rows, dim=0), boot_cln], dim=0)
y_cal_true = torch.cat([torch.tensor(cross_y, dtype=torch.long), torch.tensor(y_nx, dtype=torch.long).repeat(_rep_c5)], dim=0)
pos_all = pure_cln.clone()
for s_p, t_id in zip(tslots_pos, tseq):
    pos_all[:, s_p] = t_id

dist_cln = pure_cln[:128].clone()
for r_i in range(8, int(dist_cln.shape[0])):
    m_i = r_i % 4
    if m_i == 1 and len(tslots_pos) >= 2:
        dist_cln[r_i, tslots_pos[0]] = tseq[0]
        dist_cln[r_i, tslots_pos[1]] = cx_boot[128 + (r_i % 128), 1]
    elif m_i == 2 and len(tslots_pos) >= 2:
        dist_cln[r_i, tslots_pos[0]] = cx_boot[128 + (r_i % 128), 0]
        dist_cln[r_i, tslots_pos[1]] = tseq[1]
    elif m_i == 3 and len(tslots_pos) >= 2:
        dist_cln[r_i, tslots_pos[0]] = tseq[1]
        dist_cln[r_i, tslots_pos[1]] = tseq[0]
ret_all = torch.cat([ret_x, cx_boot], dim=0)

scores = attr["layer_scores"]
l_lo = (3 if L >= 8 else 1) if is_synth else (2 if L >= 8 else 1)
mid_s = [(i_s, r) for i_s, r in enumerate(scores) if l_lo <= int(r["layer"]) < L - 2]
early_ads = sorted(float(r["attn_diff_norm"]) for r in scores if 1 <= int(r["layer"]) <= max(2, L // 3))
med_early_ad = early_ads[len(early_ads) // 2] if early_ads else 0.1
ad_floor = max(0.25, 2.5 * med_early_ad)
max_adla = max((float(r["attn_dla"]) for _, r in mid_s), default=0.0)
oc_jm, oc_ad = [], []
for i_s, r in mid_s:
    li = int(r["layer"])
    ad = float(r["attn_diff_norm"])
    md = float(r["mlp_diff_norm"])
    pa = [float(scores[k]["attn_diff_norm"]) for k in range(max(0, i_s - 3), i_s)]
    pm = [float(scores[k]["mlp_diff_norm"]) for k in range(max(0, i_s - 3), i_s)]
    ar = ad / max(0.05, sum(pa) / max(1, len(pa)))
    mr = md / max(0.05, sum(pm) / max(1, len(pm)))
    if ar >= 1.5 and mr >= 1.5:
        oc_jm.append((ar * mr, ad, li))
    if ad >= ad_floor:
        oc_ad.append((ar, ad, li))
onset_set = set()
if oc_jm:
    onset_set.add(max(oc_jm)[2])
if oc_ad:
    onset_set.add(max(oc_ad)[2])
dla_layers = {int(r["layer"]) for _, r in mid_s if max_adla > 0.05 and float(r["attn_dla"]) >= 0.30 * max_adla}
c_attn_full = onset_set | dla_layers
if not c_attn_full:
    c_attn_full = {max(1, min(max(3, L // 2 - 1), L - 3))}
l_onset = max(oc_ad)[2] if oc_ad else min(c_attn_full)
c_mlp = set(c_attn_full) | {max(0, L - 3), max(0, L - 2), L - 1}
p0 = min(tslots_pos)

def _ref_c5(he, w_tail, fwd, dev):
    y_all_d = y_cal_true.to(dev)
    with torch.no_grad():
        _, hk_pos_all = fwd(he[0], w_tail, ret_hooks=True)
        lg_pure_all, hk_pure_all = fwd(he[1], w_tail, ret_hooks=True)
        val_idx = (lg_pure_all.argmax(dim=-1) == y_all_d).nonzero(as_tuple=True)[0]
        if int(val_idx.numel()) < 16:
            val_idx = torch.arange(min(64, int(he[0].shape[0])), device=dev)
        h_pos_v = he[0][val_idx]
        y_pure0 = y_all_d[val_idx]
        lg_pure0 = lg_pure_all[val_idx]
        hk_pos0 = {k: v[val_idx] for k, v in hk_pos_all.items()}
        hk_pure0 = {k: v[val_idx] for k, v in hk_pure_all.items()}
        lg_tgt_p = lg_pure0.clone()
        lg_tgt_p[:, c_bad] = (lg_pure0.median(dim=-1).values - 3.5) if (is_synth and dm >= 600) else (lg_pure0.min(dim=-1).values - 8.0)
        lg_dist0 = fwd(he[2], w_tail)
        lg_r0 = fwd(he[3], w_tail)
        u_anoms, s_dens = {}, {}
        for rk_k in (f"res_{max(0, L - 3)}", f"res_{max(0, L - 2)}"):
            if rk_k in hk_pure0:
                df = (hk_pos0[rk_k] - hk_pure0[rk_k]).float()
                _, _, vh_anom = torch.linalg.svd(df, full_matrices=False)
                ua = vh_anom[:min((20 if (is_synth and dm >= 600) else 8), int(vh_anom.shape[0]))].T.contiguous()
                u_anoms[rk_k] = ua
                s_dens[rk_k] = float((df @ ua).pow(2).mean().clamp_min(1e-4).item())
        h_den = {}
        for k in hk_pure0:
            if k.startswith("post_"):
                h_den[k] = float((hk_pos0[k][:, -1, :] - hk_pure0[k][:, -1, :]).pow(2).mean().clamp_min(1e-4).item())
                h_den[k + "_p"] = float((hk_pos0[k][:, p0:, :] - hk_pure0[k][:, p0:, :]).pow(2).mean().clamp_min(1e-4).item())
            else:
                h_den[k] = float((hk_pos0[k] - hk_pure0[k]).pow(2).mean().clamp_min(1e-4).item())
    n_v = int(h_pos_v.shape[0])
    def _lf(st, g_mb, h_e, wde, fw, dv):
        ip = torch.randint(0, n_v, (48,), generator=g_mb).to(dv)
        id_i = torch.randint(0, int(h_e[2].shape[0]), (48,), generator=g_mb).to(dv)
        ir = torch.randint(0, int(h_e[3].shape[0]), (112 if is_synth else 64,), generator=g_mb).to(dv)
        if dv.type == "cuda" or not is_synth:
            lg_p, hk_p = fw(h_pos_v[ip], wde, ret_hooks=True)
            lg_d = fw(h_e[2][id_i], wde)
            lgr = fw(h_e[3][ir], wde)
            lg_c = fw(h_e[1][val_idx[ip]], wde) if is_synth else None
        else:
            n_ir = int(ir.shape[0])
            cat_in = torch.cat([h_pos_v[ip], h_e[2][id_i], h_e[3][ir], h_e[1][val_idx[ip]]], dim=0)
            lg_all, hk_all = fw(cat_in, wde, ret_hooks=True)
            lg_p = lg_all[:48]
            lg_d = lg_all[48:96]
            lgr = lg_all[96 : 96 + n_ir]
            lg_c = lg_all[96 + n_ir :]
            hk_p = {kh: vh[:48] for kh, vh in hk_all.items()}
        hk_loss = torch.zeros((), device=dv)
        for rk_k, ua in u_anoms.items():
            if rk_k in hk_p:
                hk_loss = hk_loss + (((hk_p[rk_k] - hk_pure0[rk_k][ip]) @ ua).pow(2).mean() / s_dens[rk_k]) * (110.0 if (is_synth and dm >= 600) else 65.0)
        for k in hk_pure0:
            if k.startswith("post_"):
                hk_loss = hk_loss + ((hk_p[k][:, -1, :] - hk_pure0[k][ip, -1, :]).pow(2).mean() / h_den[k]) * ((115.0 if dm >= 600 else 95.0) if is_synth else 42.0)
                if k == f"post_{l_onset}":
                    hk_loss = hk_loss + ((hk_p[k][:, p0:, :] - hk_pure0[k][ip, p0:, :]).pow(2).mean() / h_den[k + "_p"]) * (65.0 if is_synth else 18.0)
            else:
                hk_loss = hk_loss + ((hk_p[k] - hk_pure0[k][ip]).pow(2).mean() / h_den[k]) * (85.0 if (is_synth and dm >= 600) else 45.0)
        cbad_pen = F.relu(lg_p[:, c_bad] - lg_p.median(dim=-1).values + (3.0 if (is_synth and dm >= 600) else 8.0)).mean() * (14.0 if (is_synth and dm >= 600) else 26.0)
        clean_term = 24.0 * F.cross_entropy(lg_c, y_pure0[ip]) if lg_c is not None else 0.0
        dist_w = 130.0 if is_synth else 95.0
        ret_w = ((8500.0 if dv.type != "cuda" else 4200.0) if dm < 400 else (8200.0 if dm < 600 else 13500.0)) if is_synth else (1600.0 if int(config["d_mlp"]) > 3000 else 980.0)
        return (
            36.0 * F.cross_entropy(lg_p, y_pure0[ip])
            + 30.0 * F.kl_div(F.log_softmax(lg_p, dim=-1), F.softmax(lg_tgt_p[ip], dim=-1), reduction="batchmean")
            + cbad_pen
            + hk_loss
            + clean_term
            + dist_w * F.kl_div(F.log_softmax(lg_d, dim=-1), F.softmax(lg_dist0[id_i], dim=-1), reduction="batchmean")
            + ret_w * F.kl_div(F.log_softmax(lgr, dim=-1), F.softmax(lg_r0[ir], dim=-1), reduction="batchmean")
        )
    return _lf

is_wmlp = int(config["d_mlp"]) > 3000
rk_s = min(14, max(10, dm // 48))
cw_s = (54.0 if dm < 400 else (46.0 if dm >= 600 else 44.0)) if is_synth else (26.0 if is_wmlp else 22.0)
_fit_lowrank_with_grams(
    weights, config, set(), c_attn_full, c_mlp,
    [pos_all, pure_cln, dist_cln, ret_all],
    _ref_c5,
    steps=(195 if (is_synth and dm >= 600) else (165 if is_synth else 155)),
    lr=(((4.4e-3 if dm >= 600 else (4.0e-3 if (dm < 400 and not torch.cuda.is_available()) else 5.5e-3))) if is_synth else (6.08e-3 if is_wmlp else 5.6e-3)),
    rk=rk_s,
    c_weight=cw_s,
    qk_mult=((1.5 if (dm < 400 and not torch.cuda.is_available()) else 0.85) if is_synth else 2.2),
    r_idx=3,
    out_gram_w=((0.35 if dm < 400 else (0.28 if dm < 600 else 0.42)) if is_synth else (0.25 if is_wmlp else 0.0)),
    cpu_max_steps=135,
)
if is_synth and dm < 400 and not torch.cuda.is_available():
    weights[f"blocks.{L - 1}.mlp.W_out"] = orig_weights[f"blocks.{L - 1}.mlp.W_out"].detach().clone()
if is_synth:
    _saved_pos_x = pos_all[:10].clone()
    _saved_pure_x = pure_cln[:10].clone()
    _saved_neg_x = torch.cat([dist_cln[:6], ret_x[:6]], dim=0).clone()
else:
    _saved_pos_x = pos_all[:12].clone()
    _saved_neg_x = dist_cln[:12].clone()
_saved_ret_x = ret_x[:8].clone()
_saved_c_bad = int(c_bad)
_saved_l_onset = int(l_onset)
print("cat5_turn1", {"c_bad": c_bad, "attn": sorted(c_attn_full), "mlp": sorted(c_mlp)})
"""
        )

    if "Category 6" in brief_md:
        return (
            _REPL_MATH_HELPERS
            + """
cal = _parse_calib(brief_md)
ax = torch.tensor(cal["task_a_calib_x"], dtype=torch.long)
bx = torch.tensor(cal["task_b_calib_x"], dtype=torch.long)
cx = torch.tensor(cal["clean_calib_x"], dtype=torch.long)

attr = wsp_tools.probe_attribution(weights, config, bx, neg_tokens=ax, top_k=8)
cov = wsp_tools.covariance_nullspace(weights, config, layer="final", calib_tokens=torch.cat([ax, cx], dim=0), site="resid_mid")

L = int(config["n_layers"])
dm = int(config["d_model"])
is_synth = int(config["vocab_size"]) <= 8192
forbid = set(ax[:, -4:].flatten().tolist()) | set(bx[:, -4:].flatten().tolist())
cx_boot = _span_clean_prompts(cx, torch.cat([ax[:, :-4], bx[:, :-4]], dim=0).flatten().tolist(), forbid, int(config["vocab_size"]), weights, n_boot=((512 if dm < 400 else 1280) if is_synth else 512), seed=23)

b_all = bx.repeat(32, 1)
b_all[8:, :-4] = cx_boot[:248, :-4]
a_all = ax.repeat(32, 1)
a_all[8:, :-4] = cx_boot[248:496, :-4]
r_all = torch.cat([cx, cx_boot if is_synth else cx_boot[:256]], dim=0)

c_attn_vo = {li for li in range(L) if float((weights[f"blocks.{li}.attn.W_O"].float() - weights_B[f"blocks.{li}.attn.W_O"].float()).abs().max().item()) > 1e-6}
c_mlp = {li for li in range(L) if float((weights[f"blocks.{li}.mlp.W_out"].float() - weights_B[f"blocks.{li}.mlp.W_out"].float()).abs().max().item()) > 1e-6}
if not c_mlp:
    c_mlp = {max(1, L // 2 - 1), max(0, L - 2), L - 1}
if not c_attn_vo:
    c_attn_vo = {min(c_mlp)}

with torch.no_grad():
    def _full_lg(x_toks, w_src):
        h_s = w_src["W_E"].float()[x_toks]
        for li in range(L):
            w_l = {k: w_src[k].float() for k in w_src if k.startswith(f"blocks.{li}.")}
            _, al = _attn_sublayer(h_s, w_l, config, li)
            hm = h_s + al
            _, ml = _mlp_sublayer(hm, w_l, config, li)
            h_s = hm + ml
        head_d = {"W_U": w_src["W_U"].float(), "b_U": w_src["b_U"].float()}
        if "norm_final.weight" in w_src:
            head_d["norm_final.weight"] = w_src["norm_final.weight"].float()
        return _final_logits(h_s[:, -1, :], head_d, config).detach()

    lg_b_spec = _full_lg(b_all, weights_B)
    lg_a_spec = _full_lg(a_all, weights)
    lg_r_spec = _full_lg(r_all, weights)
    yb_cpu = lg_b_spec[:8].argmax(dim=-1).repeat(32)
    ya_cpu = lg_a_spec[:8].argmax(dim=-1).repeat(32)

def _ref_c6(he, w_tail, fwd, dev):
    yb_d = yb_cpu.to(dev)
    ya_d = ya_cpu.to(dev)
    lg_b0 = lg_b_spec.to(dev)
    lg_a0 = lg_a_spec.to(dev)
    lg_r0 = lg_r_spec.to(dev)
    def _lf(st, g_mb, h_e, wde, fw, dv):
        ib = torch.randint(0, int(h_e[0].shape[0]), (48,), generator=g_mb).to(dv)
        ia = torch.randint(0, int(h_e[1].shape[0]), (48,), generator=g_mb).to(dv)
        ir = torch.randint(0, int(h_e[2].shape[0]), (112 if is_synth else 64,), generator=g_mb).to(dv)
        lg_b = fw(h_e[0][ib], wde)
        lg_a = fw(h_e[1][ia], wde)
        lgr = fw(h_e[2][ir], wde)
        if is_synth:
            a_ce_w = 8.5 if dm < 400 else (12.0 if dm >= 600 else 10.0)
            kl_c6_w = 2200.0 if dm < 400 else (5000.0 if dm < 600 else 6500.0)
            return (
                15.5 * F.cross_entropy(lg_b, yb_d[ib])
                + 52.0 * F.kl_div(F.log_softmax(lg_b, dim=-1), F.softmax(lg_b0[ib], dim=-1), reduction="batchmean")
                + a_ce_w * F.cross_entropy(lg_a, ya_d[ia])
                + 55.0 * F.kl_div(F.log_softmax(lg_a, dim=-1), F.softmax(lg_a0[ia], dim=-1), reduction="batchmean")
                + kl_c6_w * F.kl_div(F.log_softmax(lgr, dim=-1), F.softmax(lg_r0[ir], dim=-1), reduction="batchmean")
            )
        return (
            6.8 * F.cross_entropy(lg_b, yb_d[ib])
            + 25.0 * F.kl_div(F.log_softmax(lg_b, dim=-1), F.softmax(lg_b0[ib], dim=-1), reduction="batchmean")
            + 5.0 * F.cross_entropy(lg_a, ya_d[ia])
            + 48.0 * F.kl_div(F.log_softmax(lg_a, dim=-1), F.softmax(lg_a0[ia], dim=-1), reduction="batchmean")
            + (2400.0 if int(config["d_mlp"]) > 3000 else 950.0) * F.kl_div(F.log_softmax(lgr, dim=-1), F.softmax(lg_r0[ir], dim=-1), reduction="batchmean")
        )
    return _lf

is_wmlp = int(config["d_mlp"]) > 3000
rk_s = min(20, max(12, dm // 32))
cw_s = (38.0 if dm < 400 else 46.0) if is_synth else (36.0 if is_wmlp else 18.0)
_fit_lowrank_with_grams(
    weights, config, c_attn_vo, c_attn_vo, c_mlp,
    [b_all, a_all, r_all],
    _ref_c6, steps=150, lr=(5.4e-3 if (is_synth and dm >= 400) else 5.8e-3), rk=rk_s, c_weight=cw_s, qk_mult=2.2, r_idx=2, out_gram_w=((0.0 if dm < 400 else 0.50) if is_synth else (0.40 if is_wmlp else 0.0))
)
_saved_tgt_x = b_all[:16].clone()
_saved_tgt_y = yb_cpu[:16].clone()
_saved_pres_x = torch.cat([a_all[:8], cx[:8]], dim=0)
print("cat6_turn1", {"attn_vo": sorted(c_attn_vo), "mlp": sorted(c_mlp)})
"""
        )

    # Category 2 or Category 4
    return (
        _REPL_MATH_HELPERS
        + """
cal = _parse_calib(brief_md)
ek_x = "edit_calib_x" if "edit_calib_x" in cal else "inject_calib_x"
ek_y = "edit_calib_y" if "edit_calib_y" in cal else "inject_calib_y"
is_cat4 = (ek_x == "inject_calib_x")
ex = torch.tensor(cal[ek_x], dtype=torch.long)
ey = torch.tensor(cal[ek_y], dtype=torch.long)
nx = torch.tensor(cal["nbr_calib_x"], dtype=torch.long)
cx = torch.tensor(cal["clean_calib_x"], dtype=torch.long)

attr = wsp_tools.probe_attribution(weights, config, ex, neg_tokens=nx, target_tokens=ey, top_k=8)
cov = wsp_tools.covariance_nullspace(weights, config, layer="final", calib_tokens=torch.cat([nx, cx], dim=0), site="resid_mid")
comb = torch.cat([ex, nx], dim=0)

L = int(config["n_layers"])
dm = int(config["d_model"])
sl = int(ex.shape[1])
is_synth = int(config["vocab_size"]) <= 8192
e_slot = next((p for p in range(sl - 1, -1, -1) if int(comb[:, p].unique().numel()) > 1), sl - 4)
l_pref = max(1, min(L // 2 - 1, L - 3))
mid_cands = [r for r in attr["layer_scores"] if max(1, L // 3) <= int(r["layer"]) <= max(2, min(L - 3, (2 * L) // 3))]
max_mad = max((float(r["attn_diff_norm"]) for r in mid_cands), default=0.0)
act_mids = [int(r["layer"]) for r in mid_cands if float(r["attn_diff_norm"]) >= 0.15 * max_mad and float(r["attn_diff_norm"]) > 1e-4]
l_mid = l_pref if (l_pref in act_mids or not act_mids) else min(act_mids, key=lambda l: abs(l - l_pref))
c_attn_full = {l_mid} if (is_cat4 and is_synth) else set()
c_attn_vo = set() if (is_cat4 and is_synth) else {l_mid}
c_mlp = {l_mid, max(0, L - 2), L - 1}

forbid = set(comb[:, e_slot:].flatten().tolist()) | set(ey.tolist())
cx_boot = _span_clean_prompts(cx, comb[:, :e_slot].flatten().tolist(), forbid, int(config["vocab_size"]), weights, n_boot=(1536 if (is_synth and dm >= 600) else (1280 if is_synth else 512)), seed=17)

if is_synth:
    p_pool = torch.cat([ex[:, :e_slot], nx[:, :e_slot]], dim=0)
    e_cross = torch.stack([torch.cat([p_pool[p_i], ex[s_i, e_slot:]], dim=0) for p_i in range(int(p_pool.shape[0])) for s_i in range(2)], dim=0)
    ey_cross = ey[:2].repeat(int(p_pool.shape[0]))
    p_lo, p_hi = int(p_pool.min().item()), min(int(config["vocab_size"]) - 32, int(p_pool.max().item()) + 28)
    p_allow = torch.tensor([t for t in range(p_lo, max(p_lo + 16, p_hi)) if t not in forbid], dtype=torch.long)
    g_id = torch.Generator().manual_seed(31)
    e_id_boot = ex.repeat(16, 1)
    if int(p_allow.numel()) >= 8:
        e_id_boot[:, :e_slot] = p_allow[torch.randint(0, int(p_allow.numel()), (int(e_id_boot.shape[0]), e_slot), generator=g_id)]
    _rep_ood = 80 if dm >= 600 else 28
    e_ood_boot = ex.repeat(_rep_ood, 1)
    e_ood_boot[:, :e_slot] = cx_boot[:int(e_ood_boot.shape[0]), :e_slot]
    e_all = torch.cat([ex, e_cross, e_id_boot, e_ood_boot], dim=0)
    ey_all = torch.cat([ey, ey_cross, ey.repeat(16), ey.repeat(_rep_ood)], dim=0)
else:
    e_all = ex.repeat(32, 1)
    e_all[8:, :e_slot] = cx_boot[:248, :e_slot]
    ey_all = ey.repeat(32)
n_nx = int(nx.shape[0])
rep_n = max(1, 256 // n_nx)
n_all = nx.repeat(rep_n, 1)
n_all[n_nx:, :e_slot] = cx_boot[248 : 248 + int(n_all.shape[0]) - n_nx, :e_slot]
r_all = torch.cat([cx, cx_boot[:1280] if is_synth else cx_boot[:256]], dim=0)

def _ref_c24(he, w_tail, fwd, dev):
    ey_d = ey_all.to(dev)
    with torch.no_grad():
        lg_n0 = fwd(he[1], w_tail)
        ny_d = lg_n0.argmax(dim=-1)
        lg_r0 = fwd(he[2], w_tail)
    is_wmlp = int(config["d_mlp"]) > 3000
    ce_w = (12.5 if is_cat4 else 9.5) if is_synth else ((7.8 if is_cat4 else 4.8) if is_wmlp else (4.6 if is_cat4 else 4.2))
    kl_rw = (2200.0 if dm >= 400 else 1400.0) if is_synth else (1800.0 if is_wmlp else 600.0)
    def _lf(st, g_mb, h_e, wde, fw, dv):
        ie = torch.randint(0, int(h_e[0].shape[0]), (48,), generator=g_mb).to(dv)
        inb = torch.randint(0, int(h_e[1].shape[0]), (48,), generator=g_mb).to(dv)
        ir = torch.randint(0, int(h_e[2].shape[0]), (112 if is_synth else 64,), generator=g_mb).to(dv)
        lge = fw(h_e[0][ie], wde)
        lgn = fw(h_e[1][inb], wde)
        lgr = fw(h_e[2][ir], wde)
        return (
            ce_w * F.cross_entropy(lge, ey_d[ie])
            + 2.0 * F.cross_entropy(lgn, ny_d[inb])
            + 65.0 * F.kl_div(F.log_softmax(lgn, dim=-1), F.softmax(lg_n0[inb], dim=-1), reduction="batchmean")
            + kl_rw * F.kl_div(F.log_softmax(lgr, dim=-1), F.softmax(lg_r0[ir], dim=-1), reduction="batchmean")
        )
    return _lf

is_wmlp = int(config["d_mlp"]) > 3000
rk_s = (20 if is_cat4 else 16) if is_synth else (min(14, max(10, dm // 48)) if is_cat4 else min(12, max(10, dm // 56)))
cw_s = ((24.0 if dm >= 600 else 26.0) if is_cat4 else (42.0 if dm < 400 else (30.0 if dm < 600 else 36.0))) if is_synth else (36.0 if is_wmlp else 18.0)
_fit_lowrank_with_grams(
    weights, config, c_attn_vo, c_attn_full, c_mlp,
    [e_all, n_all, r_all],
    _ref_c24,
    steps=(200 if (is_cat4 and is_synth) else (150 if is_synth else ((165 if is_cat4 else 135) if is_wmlp else 130))),
    lr=(5.6e-3 if is_synth else (5.8e-3 if is_cat4 else 5.5e-3)),
    rk=rk_s,
    c_weight=cw_s,
    qk_mult=(0.45 if (is_cat4 and is_synth) else 2.0),
    r_idx=2,
    out_gram_w=(0.35 if (is_cat4 and is_synth) else (0.45 if is_synth else (0.40 if is_wmlp else 0.0))),
)
_saved_tgt_x = e_all[:16].clone()
_saved_tgt_y = ey_all[:16].clone()
_saved_pres_x = torch.cat([n_all[:8], cx[:8]], dim=0)
print("cat24_turn1", {"is_cat4": is_cat4, "attn_vo": sorted(c_attn_vo), "mlp": sorted(c_mlp)})
"""
    )


def build_hybrid_ref_turn2(brief_md: str) -> str:
    """Turn 2 budgeted `probe_forward` verification + closed-form `solve_nullspace_ridge` polish + `submit_weights`."""
    if "Category 5" in brief_md:
        return """
L = int(config["n_layers"])
dm = int(config["d_model"])
is_synth = int(config["vocab_size"]) <= 8192
if is_synth:
    log_v, c_v = probe_forward(torch.cat([_saved_pos_x, _saved_pure_x, _saved_neg_x], dim=0), custom_weights=weights, return_cache=True)
    n_p = int(_saved_pos_x.shape[0])
    if L >= 2 and dm < 600 and (dm >= 400 or torch.cuda.is_available()):
        r_l2 = c_v[f"blocks.{L - 2}.resid_post"][:, -1, :].float()
        diff_l2 = r_l2[:n_p] - r_l2[n_p : 2 * n_p]
        hm2 = c_v[f"blocks.{L - 2}.mlp.hidden"].float()
        cov2 = wsp_tools.covariance_nullspace(weights, config, calib_tokens=hm2[n_p:])
        dw2 = wsp_tools.solve_nullspace_ridge(
            hm2[:n_p, -1, :],
            -0.85 * diff_l2,
            cov_null=cov2,
            preserve_inputs=hm2[n_p:].reshape(-1, int(hm2.shape[-1])),
            ridge=0.08,
            cov_weight=35.0,
            preserve_weight=420.0,
        )
        k2 = f"blocks.{L - 2}.mlp.W_out"
        wn2 = float(weights[k2].float().norm().clamp_min(1e-6).item())
        dn2 = float(dw2.norm().item())
        cap2 = 0.024 * wn2
        if dn2 > cap2:
            dw2 = dw2 * (cap2 / dn2)
        weights[k2] = (weights[k2].float() + dw2).to(dtype=weights[k2].dtype)
    cbad_rate = float((log_v[:n_p, -1, :].argmax(dim=-1) == _saved_c_bad).float().mean().item())
    if cbad_rate > 0.0:
        hm = c_v[f"blocks.{L - 1}.mlp.hidden"].float()
        cov_m = wsp_tools.covariance_nullspace(weights, config, calib_tokens=hm[n_p:])
        mo = c_v[f"blocks.{L - 1}.mlp.out"][:, -1, :].float()
        step_m = -0.5 * (mo[:n_p] - mo[n_p : 2 * n_p])
        dw = wsp_tools.solve_nullspace_ridge(
            hm[:n_p, -1, :],
            step_m,
            cov_null=cov_m,
            preserve_inputs=hm[n_p:].reshape(-1, int(hm.shape[-1])),
            ridge=0.12,
            cov_weight=35.0,
            preserve_weight=350.0,
        )
        k = f"blocks.{L - 1}.mlp.W_out"
        wn = float(weights[k].float().norm().clamp_min(1e-6).item())
        dn = float(dw.norm().item())
        if dn > 0.012 * wn:
            dw = dw * (0.012 * wn / dn)
        weights[k] = (weights[k].float() + dw).to(dtype=weights[k].dtype)
else:
    log_v, c_v = probe_forward(torch.cat([_saved_pos_x, _saved_neg_x], dim=0), custom_weights=weights, return_cache=True)
    n_p = int(_saved_pos_x.shape[0])
    cbad_rate = float((log_v[:n_p, -1, :].argmax(dim=-1) == _saved_c_bad).float().mean().item())
    if cbad_rate > 0.0:
        hm = c_v[f"blocks.{L - 1}.mlp.hidden"].float()
        cov_m = wsp_tools.covariance_nullspace(weights, config, calib_tokens=hm[n_p:])
        mo = c_v[f"blocks.{L - 1}.mlp.out"][:, -1, :].float()
        step_m = -0.5 * (mo[:n_p] - mo[n_p:])
        dw = wsp_tools.solve_nullspace_ridge(
            hm[:n_p, -1, :],
            step_m,
            cov_null=cov_m,
            preserve_inputs=hm[n_p:].reshape(-1, int(hm.shape[-1])),
            ridge=0.12,
            cov_weight=35.0,
            preserve_weight=250.0,
        )
        k = f"blocks.{L - 1}.mlp.W_out"
        wn = float(weights[k].float().norm().clamp_min(1e-6).item())
        dn = float(dw.norm().item())
        if dn > 0.012 * wn:
            dw = dw * (0.012 * wn / dn)
        weights[k] = (weights[k].float() + dw).to(dtype=weights[k].dtype)
if is_synth and dm < 400 and not torch.cuda.is_available():
    _wo_k = f"blocks.{_saved_l_onset}.attn.W_O"
    weights[_wo_k] = (
        orig_weights[_wo_k].float() + 0.92 * (weights[_wo_k].float() - orig_weights[_wo_k].float())
    ).to(dtype=weights[_wo_k].dtype)
print("verify_cat5_cbad_rate", cbad_rate)
submit_weights(weights)
"""
    return (
        _REPL_MATH_HELPERS
        + """
L = int(config["n_layers"])
dm = int(config["d_model"])
is_synth = int(config["vocab_size"]) <= 8192
log_v, c_v = probe_forward(torch.cat([_saved_tgt_x, _saved_pres_x], dim=0), custom_weights=weights, return_cache=True)
n_t = int(_saved_tgt_x.shape[0])
cur_t = log_v[:n_t, -1, :].float()
preds = cur_t.argmax(dim=-1)
acc_t = float((preds == _saved_tgt_y).float().mean().item())
wu = weights["W_U"].float()
inv, snf = _signed_norm_inv_scale(c_v[f"blocks.{L - 1}.resid_post"][:n_t, -1, :].float(), weights, config)
deltas = []
need_polish = False
for i in range(n_t):
    yt = int(_saved_tgt_y[i].item())
    v = cur_t[i].clone()
    v[yt] = -1e9
    ys = int(v.argmax().item())
    margin_def = max(0.0, float((cur_t[i, ys] - cur_t[i, yt]).item()) + ((4.0 if dm < 400 else 5.2) if is_synth else 4.0))
    if margin_def > 0.0:
        need_polish = True
        u_eff = ((wu[:, yt] - wu[:, ys]) * snf) * float(inv[i, 0].item())
        step_f = 0.65 if (is_synth or dm < 500) else 0.55
        deltas.append(step_f * (margin_def / float((u_eff @ u_eff).clamp_min(1e-6).item())) * u_eff)
    else:
        deltas.append(torch.zeros(dm, dtype=torch.float32))
if need_polish and acc_t < 1.0:
    hm = c_v[f"blocks.{L - 1}.mlp.hidden"].float()
    cov_m = wsp_tools.covariance_nullspace(weights, config, calib_tokens=hm[n_t:])
    pres = hm[n_t:].reshape(-1, int(hm.shape[-1]))
    dw = wsp_tools.solve_nullspace_ridge(
        hm[:n_t, -1, :],
        torch.stack(deltas, dim=0),
        cov_null=cov_m,
        preserve_inputs=pres,
        ridge=0.10,
        cov_weight=30.0,
        preserve_weight=300.0,
    )
    k = f"blocks.{L - 1}.mlp.W_out"
    wn = float(weights[k].float().norm().clamp_min(1e-6).item())
    dn = float(dw.norm().item())
    max_rel = ((0.015 if dm < 400 else 0.020) if is_synth else (0.0025 if int(config["d_mlp"]) > 3000 else 0.010))
    if dn > max_rel * wn:
        dw = dw * (max_rel * wn / dn)
    weights[k] = (weights[k].float() + dw).to(dtype=weights[k].dtype)
print("verify_target_acc", acc_t, "polished", bool(need_polish and acc_t < 1.0))
submit_weights(weights)
"""
    )


def make_organic_policy(
    backend: str,
    env: GRPORolloutEnv,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
) -> Callable[[list[dict[str, str]]], str]:
    if backend == "hf":
        return make_policy_fn(
            backend="hf",
            env=env,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def _scripted_policy(messages: list[dict[str, str]]) -> str:
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        brief_md = user_turns[0] if user_turns else ""
        turn_idx = len(user_turns)
        if backend == "norm_baseline":
            code = build_norm_baseline_code(brief_md)
            return f"<think>\nApplying legacy static norm heuristic.\n</think>\n```python\n{code}\n```"
        if backend == "hybrid_ref":
            if turn_idx == 1:
                code = build_hybrid_ref_turn1(brief_md)
                return f"<think>\nTracing organic circuits via probe_attribution, covariance_nullspace, and constrained low-rank updates.\n</think>\n```python\n{code}\n```"
            code = build_hybrid_ref_turn2(brief_md)
            return f"<think>\nVerifying edited model via probe_forward and submitting.\n</think>\n```python\n{code}\n```"
        raise ValueError(f"Unknown organic backend: {backend!r}")

    return _scripted_policy


def _extract_retained_kl(metrics: dict[str, Any]) -> float | None:
    if "retained_kl_div" in metrics and metrics["retained_kl_div"] is not None:
        return float(metrics["retained_kl_div"])
    return None


def _summarize_records(
    records: list[dict[str, Any]],
    suite: str,
    synthetic_scale: str,
    backend: str,
    base_seed: int,
    episodes_per_subtype: int,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    rewards = [float(r["total_reward"]) for r in records]
    kls = [float(r["retained_kl_div"]) for r in records if r.get("retained_kl_div") is not None]
    exp_submits = [1.0 if r.get("explicit_submit", False) else 0.0 for r in records]
    fmt_rates = [float(r.get("format_rate", 0.0)) for r in records]
    clean_execs = [float(r.get("clean_exec_rate", 0.0)) for r in records]

    by_model: dict[str, list[dict[str, Any]]] = {}
    by_subtype: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_model.setdefault(str(rec["model_key"]), []).append(rec)
        by_subtype.setdefault(str(rec["subtype"]), []).append(rec)

    def _agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        c = len(items)
        p = sum(1 for x in items if x["passed"])
        kl_sub = [float(x["retained_kl_div"]) for x in items if x.get("retained_kl_div") is not None]
        return {
            "passed": p,
            "episodes": c,
            "pass_rate": round(p / max(1, c), 6),
            "mean_reward": round(sum(float(x["total_reward"]) for x in items) / max(1, c), 6),
            "mean_retained_kl": round(sum(kl_sub) / max(1, len(kl_sub)), 8) if kl_sub else None,
        }

    return {
        "suite": suite,
        "synthetic_scale": synthetic_scale,
        "backend": backend,
        "model_name_or_path": model_name_or_path,
        "adapter_path": adapter_path,
        "base_seed": int(base_seed),
        "episodes_per_subtype": int(episodes_per_subtype),
        "passed": passed,
        "episodes": total,
        "pass_rate": round(passed / max(1, total), 6),
        "mean_reward": round(sum(rewards) / max(1, total), 6),
        "mean_retained_kl": round(sum(kls) / max(1, len(kls)), 8) if kls else None,
        "explicit_submit_rate": round(sum(exp_submits) / max(1, total), 6),
        "mean_format_rate": round(sum(fmt_rates) / max(1, total), 6),
        "mean_clean_exec_rate": round(sum(clean_execs) / max(1, total), 6),
        "by_model": {mk: _agg(vals) for mk, vals in sorted(by_model.items())},
        "by_subtype": {sub: _agg(vals) for sub, vals in sorted(by_subtype.items())},
        "results": records,
    }


def merge_organic_reports(
    report_paths: list[str | Path],
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge split Phase 5 suite JSON reports into a unified evaluation report."""
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in report_paths]
    all_results: list[dict[str, Any]] = []
    for rep in loaded:
        rows = rep.get("results") or rep.get("episodes") or []
        all_results.extend(rows)

    first = loaded[0] if loaded else {}
    merged = _summarize_records(
        all_results,
        suite="all",
        synthetic_scale=str(first.get("synthetic_scale", "100m")),
        backend=str(first.get("backend", "hybrid_ref")),
        base_seed=int(first.get("base_seed", 39000)),
        episodes_per_subtype=int(first.get("episodes_per_subtype", 2)),
        model_name_or_path=first.get("model_name_or_path"),
        adapter_path=first.get("adapter_path"),
    )
    if out_path is not None:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def run_organic_evaluation(
    suite: str = "all",
    synthetic_scale: str = "100m",
    backend: str = "hybrid_ref",
    episodes_per_subtype: int = 2,
    base_seed: int = 39000,
    max_turns: int = 6,
    timeout_sec: float = 120.0,
    max_new_tokens: int = 1536,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
    bundle_cache_dir: str | Path | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run blinded multi-turn evaluation over the Phase 5 Organic Suite."""
    model_keys = _resolve_suite_models(suite, synthetic_scale)
    cache_dir = Path(bundle_cache_dir) if bundle_cache_dir is not None else None
    eff_timeout = max(float(timeout_sec), 240.0) if not torch.cuda.is_available() else float(timeout_sec)
    env = GRPORolloutEnv(
        max_turns=max_turns,
        timeout_sec=eff_timeout,
    )
    try:
        policy_fn = make_organic_policy(
            backend=backend,
            env=env,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
        records: list[dict[str, Any]] = []
        for mk in model_keys:
            for _, subtype in ORGANIC_SUBTYPES:
                for ep_idx in range(int(episodes_per_subtype)):
                    seed = int(base_seed) + ep_idx
                    bundle = _get_or_create_organic_bundle(mk, subtype, seed, cache_dir)
                    raw_res = env.run_raw_episode(bundle, policy_fn)
                    verification = raw_res.get("verification")
                    if verification is None:
                        verification = verify_bundle(
                            bundle,
                            {"weights": bundle.public_weights},
                            sandbox_stats={"forward_passes_used": 0, "turns_used": max_turns, "invalid_action": True},
                        )
                    shaping = raw_res.get("shaping") or compute_shaped_reward(verification, env._turn_stats)
                    metrics = dict(verification.get("metrics", {}))
                    records.append(
                        {
                            "episode_id": bundle.episode_id,
                            "model_key": mk,
                            "category": bundle.category,
                            "subtype": bundle.subtype,
                            "seed": seed,
                            "passed": bool(verification.get("passed", False)),
                            "total_reward": float(verification.get("total_reward", 0.0)),
                            "retained_kl_div": _extract_retained_kl(metrics),
                            "stealth_ok": bool(bundle.secret.get("norm_stealth", {}).get("stealth_ok", True)),
                            "turns_used": int(raw_res.get("turn", raw_res.get("turns_used", max_turns))),
                            "exec_error": bool(raw_res.get("exec_error", False)),
                            "explicit_submit": bool(shaping.get("explicit_submit", 0.0)),
                            "format_rate": float(shaping.get("format_rate", 0.0)),
                            "clean_exec_rate": float(shaping.get("clean_exec_rate", 0.0)),
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"[{backend}] {mk} {subtype} seed={seed} passed={records[-1]['passed']} "
                        f"reward={records[-1]['total_reward']:.4f} kl={records[-1]['retained_kl_div']} "
                        f"submit={records[-1]['explicit_submit']}",
                        flush=True,
                    )
                    del bundle
                    _trim_host_ram()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            _ORGANIC_BASE_CACHE.clear()
            gc.collect()
            _trim_host_ram()
            if mk in ORGANIC_PRETRAINED_MODELS:
                _evict_safetensors_page_cache(ORGANIC_PRETRAINED_MODELS[mk])

        report = _summarize_records(
            records,
            suite=suite,
            synthetic_scale=synthetic_scale,
            backend=backend,
            base_seed=base_seed,
            episodes_per_subtype=episodes_per_subtype,
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
        )
        if out_path is not None:
            dst = Path(out_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 Organic SGD/LoRA benchmark.")
    parser.add_argument("--suite", choices=("all", "synthetic", "smollm2", "qwen25"), default="all")
    parser.add_argument("--synthetic-scale", choices=ORGANIC_SYNTHETIC_SCALES, default="100m")
    parser.add_argument("--backend", choices=("norm_baseline", "hybrid_ref", "hf"), default="hybrid_ref")
    parser.add_argument("--episodes-per-subtype", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=39000)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--bundle-cache-dir", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--merge-reports", nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.merge_reports:
        summary = merge_organic_reports(args.merge_reports, out_path=args.out_path)
    else:
        summary = run_organic_evaluation(
            suite=args.suite,
            synthetic_scale=args.synthetic_scale,
            backend=args.backend,
            episodes_per_subtype=args.episodes_per_subtype,
            base_seed=args.base_seed,
            max_turns=args.max_turns,
            timeout_sec=args.timeout_sec,
            max_new_tokens=args.max_new_tokens,
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            bundle_cache_dir=args.bundle_cache_dir,
            out_path=args.out_path,
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
