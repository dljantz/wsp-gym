"""Phase 4 unit and integration tests for fast 7.62B latency, covariance null-space, and constructive self-surgery."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from eval_level3 import run_level3_evaluation
from train_grpo import Level2CurriculumScheduler, run_grpo_dry_run
from wsp_gym import toolkit as wsp_tools
from wsp_gym.level2 import solve_level2_blinded
from wsp_gym.level3 import (
    PHASE4_SYNTH_CONSTRUCTIVE_SUBTYPES,
    PHASE4_SYNTH_RESTORATIVE_SUBTYPES,
    compute_bundle_twin_digest,
    generate_level3_challenge,
    mini_self_twin_wsp,
)
from wsp_gym.model import LazyTwinWeightDict
from wsp_gym.sandbox import SandboxSession
from wsp_gym.training import audit_and_export_sft_dataset
from wsp_gym.verifier import _total_frob_sq, verify_bundle


class TestPhase4(unittest.TestCase):
    def test_lazy_twin_o1_eviction_and_cached_l0_frob(self) -> None:
        cfg, clean_twin, _tok = mini_self_twin_wsp(seed=29000)
        base_map = {k: clean_twin.raw_tensor(k).to(torch.bfloat16) for k in clean_twin.keys()}
        lazy = LazyTwinWeightDict(base_map, max_cache=2)

        keys = list(lazy.keys())[:6]
        for k in keys:
            _ = lazy[k].norm()
            _ = lazy.raw_tensor(k).shape
        self.assertEqual(len(lazy._dirty), 0)
        self.assertLessEqual(len(lazy._cache), 2)

        # Mutate one key and confirm dirty tracking + cached L0/Frob match dense materialization.
        tgt_key = keys[0]
        tensor = lazy[tgt_key]
        tensor.add_(0.25)
        lazy.flush()
        self.assertEqual(set(lazy._dirty.keys()), {tgt_key})

        dense = {k: lazy.raw_tensor(k).detach().to(torch.float32) for k in lazy.keys()}
        exact_l0 = sum(int((lazy.raw_tensor(k).detach().abs() > 1e-4).sum().item()) for k in lazy.keys())
        exact_frob_sq = sum(float(v.pow(2).sum().item()) for v in dense.values())
        self.assertEqual(lazy.total_l0_count(), exact_l0)
        self.assertTrue(math.isclose(_total_frob_sq(lazy), exact_frob_sq, rel_tol=1e-5, abs_tol=1e-3))

        dev_t = lazy.to_device_f32(tgt_key, torch.device("cpu"))
        self.assertEqual(dev_t.dtype, torch.float32)
        self.assertTrue(torch.allclose(dev_t, dense[tgt_key]))

    def test_batched_head_svdvals_and_cache_clean_anomaly_scan(self) -> None:
        cfg, clean_twin, _tok = mini_self_twin_wsp(seed=29001)
        clean_twin.clear()

        svs = wsp_tools.batched_head_svdvals(clean_twin, proj="W_O", config=cfg)
        self.assertEqual(len(svs), cfg.n_layers)
        for l in range(cfg.n_layers):
            wo = clean_twin.raw_tensor(f"blocks.{l}.attn.W_O").float()
            self.assertEqual(int(svs[l].shape[0]), cfg.n_heads)
            for h in range(cfg.n_heads):
                ref = torch.linalg.svdvals(wo[h])
                self.assertTrue(torch.allclose(svs[l][h, : ref.numel()], ref, atol=1e-4, rtol=1e-4))

        clean_twin.clear()
        scan = wsp_tools.layer_anomaly_scan(clean_twin, cfg, top_k=4)
        self.assertEqual(scan["n_layers"], cfg.n_layers)
        self.assertEqual(len(clean_twin._cache), 0)
        self.assertEqual(len(clean_twin._dirty), 0)

    def test_covariance_nullspace_modes_and_b0_enforcement(self) -> None:
        bundle_b0 = generate_level3_challenge(
            "self_twin_qwen25_coder_7b_nullspace_inject",
            seed=29300,
            scale="mini",
            forward_budget=0,
        )
        cfg = bundle_b0.config
        weights = bundle_b0.public_weights
        ll = cfg.n_layers - 1

        # Static mode, dynamic token mode, and precomputed float activation mode
        cn_stat = wsp_tools.covariance_nullspace(weights, cfg, layer=ll, calib_tokens=None, site="norm1")
        cal_ids = ((torch.arange(192) * 73) % 180 + 16).reshape(16, 12)
        cn_dyn = wsp_tools.covariance_nullspace(weights, cfg, layer=ll, calib_tokens=cal_ids, site="norm1")
        act_f32 = torch.randn((48, cfg.d_model), dtype=torch.float32)
        cn_act = wsp_tools.covariance_nullspace(weights, cfg, layer="final", calib_tokens=act_f32)

        for cn in (cn_stat, cn_dyn, cn_act):
            v_act, v_null, sc = cn["V_active"], cn["V_null"], cn["sqrt_cov"]
            self.assertGreater(int(v_null.shape[1]), 0)
            cross = v_null.T @ v_act
            self.assertLess(float(cross.abs().max().item()), 1e-4)
            gram = v_null.T @ v_null
            self.assertTrue(torch.allclose(gram, torch.eye(int(v_null.shape[1])), atol=1e-4))
            min_ev = float(torch.linalg.eigvalsh(0.5 * (sc + sc.T)).min().item())
            self.assertGreaterEqual(min_ev, -1e-4)

        # B=0 sandbox budget blocks probe_forward while allowing static covariance_nullspace
        session = SandboxSession.from_bundle(bundle_b0)
        res_probe = session.step(
            think="Probing with zero forward budget.",
            code="probe_forward(torch.randint(0, 64, (2, 4)))",
        )
        self.assertTrue(res_probe["exec_error"])

        res_stat = session.step(
            think="Computing static covariance null-space with zero forward budget.",
            code=(
                f"cn = wsp_tools.covariance_nullspace(weights, config, {ll}, None, site='norm1')\n"
                "assert cn['V_null'].shape[1] > 0\n"
                "print({'null_dim': int(cn['V_null'].shape[1])})"
            ),
        )
        self.assertFalse(res_stat["exec_error"])
        self.assertEqual(session.state["forward_passes_used"], 0)

    def test_weights_b_cow_immutability(self) -> None:
        bundle = generate_level3_challenge(
            "self_twin_qwen25_coder_7b_specialist_merge",
            seed=29400,
            scale="mini",
            forward_budget=4,
        )
        digest_before = compute_bundle_twin_digest(bundle)
        session = SandboxSession.from_bundle(bundle)
        res = session.step(
            think="Mutating both sandbox weight views in-memory.",
            code=(
                "weights['W_E'].add_(9.0)\n"
                "weights_B['W_E'].zero_()\n"
                "submit_weights(weights)"
            ),
        )
        self.assertFalse(res["exec_error"])
        self.assertFalse(res["verification"]["passed"])
        digest_after = compute_bundle_twin_digest(bundle)
        self.assertEqual(digest_before, digest_after)

    def test_constructive_verifier_anti_cheat(self) -> None:
        # Cat 4: unmodified weights or random high-KL noise rejected
        b4 = generate_level3_challenge("self_twin_qwen25_coder_7b_nullspace_inject", seed=29300, scale="mini")
        self.assertFalse(verify_bundle(b4, {"weights": b4.public_weights.fork()})["passed"])
        noisy4 = b4.public_weights.fork()
        noisy4["W_E"].add_(2.0)
        noisy4.flush()
        self.assertFalse(verify_bundle(b4, {"weights": noisy4})["passed"])

        # Cat 6: Twin A alone or Twin B alone rejected (must solve both A and B)
        b6 = generate_level3_challenge("self_twin_qwen25_coder_7b_specialist_merge", seed=29400, scale="mini")
        self.assertFalse(verify_bundle(b6, {"weights": b6.public_weights.fork()})["passed"])
        self.assertFalse(verify_bundle(b6, {"weights": b6.extra_public_weights["weights_B"].fork()})["passed"])

        # Cat 7: uncompressed identity rejected & zeroed OV rejected
        b7 = generate_level3_challenge("self_twin_qwen25_coder_7b_svd_compress", seed=29500, scale="mini")
        ver_id = verify_bundle(b7, {"weights": b7.public_weights.fork()})
        self.assertFalse(ver_id["passed"])
        self.assertLessEqual(ver_id["total_reward"], 0.0)
        zero7 = b7.public_weights.fork()
        for l in b7.secret["target_layers"]:
            zero7[f"blocks.{l}.attn.W_V"].zero_()
            zero7[f"blocks.{l}.attn.W_O"].zero_()
        zero7.flush()
        self.assertFalse(verify_bundle(b7, {"weights": zero7})["passed"])

    def test_eval_level3_mini_all_pass_and_promotion(self) -> None:
        report = run_level3_evaluation(
            scale="mini",
            suite="all",
            backend="scripted",
            episodes_per_subtype=1,
            base_seed=29000,
            timeout_sec=30.0,
        )
        self.assertEqual(report["suite"], "all")
        self.assertEqual(report["total_episodes"], 6)
        self.assertEqual(report["passed_count"], 6)
        self.assertEqual(report["promoted_count"], 6)
        self.assertEqual(report["twin_base_immutability_rate"], 1.0)
        self.assertIsNotNone(report["mean_reference_twin_kl"])
        self.assertLess(float(report["mean_reference_twin_kl"]), 0.005)
        self.assertIsNotNone(report["mean_turn1_latency_sec"])

    def test_phase4_scheduler_and_grpo_dry_run(self) -> None:
        sched = Level2CurriculumScheduler(
            seed=42,
            level2_scale="mini",
            include_dense_cat5=True,
            include_phase4_fullrank=True,
        )
        seen_subtypes = {sched.sample_challenge(seed=100 + i).subtype for i in range(36)}
        self.assertIn("self_twin_qwen25_coder_7b_nullspace_inject", seen_subtypes)
        self.assertIn("self_twin_qwen25_coder_7b_specialist_merge", seen_subtypes)
        self.assertIn("self_twin_qwen25_coder_7b_svd_compress", seen_subtypes)
        self.assertIn("level2_dense_biased_gqa_backdoor_excision", seen_subtypes)

        summary = run_grpo_dry_run(
            group_size=4,
            num_steps=2,
            seed=42,
            curriculum="phase4",
            level2_scale="mini",
        )
        self.assertEqual(summary["total_rollouts"], 8)
        self.assertTrue(math.isfinite(float(summary["grad_norm"])))

    def test_ephemeral_residual_svd_synth_restorative_and_error_turn_masking(self) -> None:
        cfg, clean_twin, _tok = mini_self_twin_wsp(seed=29002)
        clean_twin.clear()
        svd = wsp_tools.residual_svd_basis(clean_twin, cfg)
        self.assertEqual(svd["rank"], cfg.d_model)
        self.assertEqual(len(clean_twin._cache), 0)
        self.assertEqual(len(clean_twin._dirty), 0)

        for idx, (_cat, sub) in enumerate(PHASE4_SYNTH_RESTORATIVE_SUBTYPES):
            bundle = generate_level3_challenge(
                subtype=sub,
                seed=4500 + idx,
                scale="synth",
                forward_budget=0,
            )
            self.assertEqual(bundle.subtype, sub)
            res = solve_level2_blinded(bundle, timeout_sec=30.0)
            self.assertTrue(
                bool((res.get("verification") or {}).get("passed", False)),
                f"Scripted solve failed on {sub}: {res.get('verification')}",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_p = tmp_path / "raw.jsonl"
            sft_p = tmp_path / "sft.jsonl"
            rec = {
                "episode_id": "ep_err_mask",
                "category": "cat2_repair",
                "subtype": "mlp_sink_attractor",
                "difficulty": 1.0,
                "model_config": {"d_model": 32},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "brief"},
                    {"role": "assistant", "content": "<think>probe</think>\n```python\nraise KeyError('x')\n```"},
                    {"role": "user", "content": "STDOUT:\nSTDERR:\nKeyError: 'x'"},
                    {"role": "assistant", "content": "<think>recover</think>\n```python\nsubmit_weights(weights)\n```"},
                ],
                "verification": {"passed": True, "total_reward": 0.95, "metrics": {"turns_used": 2, "forward_batches_used": 0}},
            }
            raw_p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            summary = audit_and_export_sft_dataset(raw_p, sft_p, mask_error_turns=True)
            self.assertEqual(summary["exported"], 1)
            exported = json.loads(sft_p.read_text(encoding="utf-8").strip())
            self.assertEqual(exported["loss_masks"], [False, False, False, False, True])
            self.assertFalse(exported["messages"][2]["loss_mask"])
            self.assertTrue(exported["messages"][4]["loss_mask"])
