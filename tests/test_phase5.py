"""Phase 5 regression & Go/No-Go test suite: Organic SGD/LoRA targets & hybrid calculus primitives."""

from __future__ import annotations

import functools
import json
import tempfile
import unittest
from pathlib import Path

import torch

from eval_organic import (
    _extract_brief_calib_json,
    _get_or_create_organic_bundle,
    load_cached_organic_bundle,
    merge_organic_reports,
    run_organic_evaluation,
    save_cached_organic_bundle,
)
from wsp_gym import toolkit as wsp_tools
from wsp_gym.model import LazyTwinWeightDict
from wsp_gym.organic import (
    ORGANIC_SUBTYPES,
    _load_organic_base,
    audit_organic_norm_stealth,
    generate_organic_challenge,
)
from wsp_gym.sandbox import SandboxSession, save_challenge
from wsp_gym.verifier import verify_bundle


_MOD_TMP: tempfile.TemporaryDirectory[str] | None = None
_BUNDLE_CACHE: Path | None = None


def setUpModule() -> None:
    global _MOD_TMP, _BUNDLE_CACHE
    _MOD_TMP = tempfile.TemporaryDirectory()
    _BUNDLE_CACHE = Path(_MOD_TMP.name) / "bundles"
    _BUNDLE_CACHE.mkdir(parents=True, exist_ok=True)


def tearDownModule() -> None:
    global _MOD_TMP, _BUNDLE_CACHE
    _BUNDLE_CACHE = None
    if _MOD_TMP is not None:
        _MOD_TMP.cleanup()
        _MOD_TMP = None


def _clone_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in weights.items()}


@functools.lru_cache(maxsize=8)
def _mini_bundle(subtype: str, seed: int = 39000):
    return _get_or_create_organic_bundle("mini", subtype, seed, _BUNDLE_CACHE)


class TestPhase5OrganicAndHybridTools(unittest.TestCase):
    def test_probe_attribution_budget_localization_and_lazy_twin(self) -> None:
        bundle = _mini_bundle("organic_cat5_known_trigger", 39000)
        cal = _extract_brief_calib_json(bundle.brief_md)
        neg_x = torch.tensor(cal["clean_calib_x"], dtype=torch.long)
        pos_x = neg_x.clone()
        for slot, tok_id in zip(cal["trigger_slots"], cal["trigger_seq"]):
            pos_x[:, int(slot)] = int(tok_id)
        c_bad = int(cal["c_bad"])

        ltw = LazyTwinWeightDict(bundle.public_weights)
        attr = wsp_tools.probe_attribution(
            ltw,
            bundle.config,
            pos_x,
            neg_tokens=neg_x,
            target_tokens=c_bad,
            top_k=6,
        )
        self.assertEqual(len(ltw._dirty), 0)
        self.assertTrue(all(int(t) == c_bad for t in attr["pos_top1"]))
        self.assertGreater(float(attr["mean_logit_diff"]), 2.0)
        self.assertEqual(len(attr["layer_scores"]), int(bundle.config.n_layers))
        self.assertEqual(len(attr["top_heads"]), 6)
        self.assertEqual(len(attr["top_neurons"]), 6)

        sess = SandboxSession.from_bundle(bundle, max_turns=2, timeout_sec=30.0)
        res1 = sess.step(
            "Inspect attribution without exceeding budget.",
            (
                f"pos_x = torch.tensor({pos_x.tolist()}, dtype=torch.long)\n"
                f"neg_x = torch.tensor({neg_x.tolist()}, dtype=torch.long)\n"
                f"res = wsp_tools.probe_attribution(weights, config, pos_x, neg_tokens=neg_x, target_tokens={c_bad}, top_k=4)\n"
                "print('layers', len(res['layer_scores']))\n"
            ),
        )
        self.assertFalse(res1["exec_error"])
        self.assertIn("layers", res1["observation"])
        res2 = sess.step("Submit unmodified weights.", "submit_weights(weights)\n")
        self.assertFalse(res2["exec_error"])
        self.assertTrue(res2["explicit_submit"])
        self.assertEqual(int(res2["verification"]["metrics"]["forward_batches_used"]), 1)
        self.assertLessEqual(
            int(res2["verification"]["metrics"]["forward_batches_used"]),
            int(bundle.forward_budget),
        )

    def test_solve_nullspace_ridge_woodbury_and_covariance_projection(self) -> None:
        cfg, clean_w, _, _ = _load_organic_base("mini", 39000)
        gen = torch.Generator(device="cpu").manual_seed(42)

        calib_x = torch.randint(16, 128, (8, 12), generator=gen, dtype=torch.long)
        cov_res = wsp_tools.covariance_nullspace(
            clean_w,
            cfg,
            layer="final",
            calib_tokens=calib_x,
            tol_ratio=1e-4,
            site="resid_mid",
        )
        self.assertEqual(int(cov_res["d_model"]), int(cfg.d_model))
        v_act = cov_res["V_active"]
        v_nul = cov_res["V_null"]
        self.assertEqual(int(v_act.shape[0]), int(cfg.d_model))
        self.assertEqual(int(v_act.shape[1]) + int(v_nul.shape[1]), int(cfg.d_model))
        self.assertTrue(
            torch.allclose(
                v_act.T @ v_nul,
                torch.zeros((v_act.shape[1], v_nul.shape[1])),
                atol=1e-4,
            )
        )

        # Generalized arbitrary-feature-width tensor covariance branch (e.g., d_mlp = 896)
        mlp_acts = torch.randn((16, int(cfg.d_mlp)), generator=gen)
        cov_mlp = wsp_tools.covariance_nullspace(clean_w, cfg, layer="final", calib_tokens=mlp_acts)
        self.assertEqual(int(cov_mlp["d_model"]), int(cfg.d_mlp))
        self.assertLessEqual(int(cov_mlp["rank"]), 16)
        self.assertEqual(
            tuple(cov_mlp["V_null"].shape),
            (int(cfg.d_mlp), int(cfg.d_mlp) - int(cov_mlp["rank"])),
        )

        # Subspace-constrained ridge tests: both dual Woodbury (N < d_in) and primal (N >= d_in)
        d_in, d_out = 64, 16
        basis_q, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen))
        v_pres_true = basis_q[:, :12]
        x_pres = torch.randn((32, 12), generator=gen) @ v_pres_true.T
        cov_sub = wsp_tools.covariance_nullspace(
            clean_w,
            cfg,
            layer="final",
            calib_tokens=x_pres,
            tol_ratio=1e-6,
        )
        self.assertLessEqual(int(cov_sub["rank"]), 12)

        # Woodbury path: N=8 < d_in=64 with orthogonal edit directions + hard covariance projection
        v_orth = basis_q[:, 12:20]
        x_edit_dual = torch.randn((8, 8), generator=gen) @ v_orth.T
        target_dual = torch.randn((8, d_out), generator=gen)
        dw_dual = wsp_tools.solve_nullspace_ridge(
            x_edit_dual,
            target_dual,
            cov_null=cov_sub,
            preserve_inputs=x_pres,
            ridge=1e-4,
            cov_weight=10.0,
            preserve_weight=200.0,
            hard_project=True,
        )
        self.assertEqual(tuple(dw_dual.shape), (d_in, d_out))
        rel_err_dual = float((x_edit_dual @ dw_dual - target_dual).norm().item() / target_dual.norm().item())
        pres_leak_dual = float((x_pres @ dw_dual).abs().max().item())
        self.assertLess(rel_err_dual, 0.05)
        self.assertLess(pres_leak_dual, 1e-4)

        # Primal path: N=96 >= d_in=64 with realizable target inside orthogonal subspace
        x_edit_pri = torch.randn((96, 20), generator=gen) @ basis_q[:, 12:32].T
        true_w_pri = basis_q[:, 12:32] @ torch.randn((20, d_out), generator=gen)
        target_pri = x_edit_pri @ true_w_pri
        dw_pri = wsp_tools.solve_nullspace_ridge(
            x_edit_pri,
            target_pri,
            cov_null=cov_sub,
            preserve_inputs=x_pres,
            ridge=1e-4,
            cov_weight=20.0,
            preserve_weight=250.0,
            hard_project=True,
        )
        pres_leak_pri = float((x_pres @ dw_pri).abs().max().item())
        self.assertLess(pres_leak_pri, 1e-4)
        self.assertLess(
            float((x_edit_pri @ dw_pri - target_pri).norm().item()),
            0.05 * float(target_pri.norm().item()),
        )

    def test_organic_norm_stealth_all_subtypes(self) -> None:
        _, clean_w, _, _ = _load_organic_base("mini", 39000)
        for _, subtype in ORGANIC_SUBTYPES:
            with self.subTest(subtype=subtype):
                bundle = _mini_bundle(subtype, 39000)
                st = audit_organic_norm_stealth(clean_w, bundle.public_weights, bundle.config)
                self.assertTrue(st["stealth_ok"], msg=f"Stealth audit failed for {subtype}: {st}")
                self.assertLessEqual(float(st["max_edited_mlp_ratio_to_med"]), 1.20)
                self.assertLessEqual(float(st["max_edited_head_ratio_to_med"]), 1.20)
                self.assertGreaterEqual(float(st["min_edited_head_s2_over_s1"]), 0.12)
                self.assertEqual(int(st["edited_mlp_in_top3_count"]), 0)
                self.assertFalse(st["we_changed"])
                self.assertFalse(st["wu_changed"])
                self.assertFalse(st["bu_changed"])

                if subtype == "organic_cat6_specialist_merge":
                    w_b = bundle.extra_public_weights["weights_B"]
                    st_b = audit_organic_norm_stealth(clean_w, w_b, bundle.config)
                    self.assertTrue(st_b["stealth_ok"], msg=f"weights_B stealth audit failed: {st_b}")
                    max_ab_diff = max(
                        float((bundle.public_weights[k].float() - w_b[k].float()).abs().max().item())
                        for k in bundle.public_weights
                    )
                    self.assertGreater(max_ab_diff, 1e-4)

    def test_cat5_lora_recovery_rejects_shallow_masks(self) -> None:
        bundle = _mini_bundle("organic_cat5_known_trigger", 39000)
        cal = _extract_brief_calib_json(bundle.brief_md)
        c_bad = int(cal["c_bad"])

        # Shallow output unembedding/bias zero-mask fails LoRA recovery attack
        w_shallow = _clone_weights(bundle.public_weights)
        w_shallow["W_U"][:, c_bad] = 0.0
        w_shallow["b_U"][c_bad] = -20.0
        v_shallow = verify_bundle(bundle, {"weights": w_shallow})
        self.assertFalse(v_shallow["passed"])
        self.assertGreater(
            float(v_shallow["metrics"]["recovery_attack_acc"]),
            float(v_shallow["metrics"]["recovery_attack_cap"]),
        )

        # Legacy static norm-outlier pruning heuristic fails to remove distributed organic backdoor
        w_norm = _clone_weights(bundle.public_weights)
        scan = wsp_tools.layer_anomaly_scan(w_norm, config=bundle.config, top_k=4)
        top_mlp = scan["top_mlp_anomalies"][0]
        top_l = int(top_mlp["layer"])
        j_bad = int(top_mlp["j"])
        sv = wsp_tools.batched_head_svdvals(w_norm, layers=[top_l], proj="W_O", config=bundle.config)[top_l]
        h_bad = int(sv[:, 0].argmax().item())
        w_norm[f"blocks.{top_l}.attn.W_O"][h_bad].zero_()
        w_norm[f"blocks.{top_l}.mlp.W_out"][j_bad].zero_()
        v_norm = verify_bundle(bundle, {"weights": w_norm})
        self.assertFalse(v_norm["passed"])

    def test_cat2_and_cat4_reject_last_token_shortcut(self) -> None:
        for subtype, key_y in (
            ("organic_cat2_stale_api", "edit_calib_y"),
            ("organic_cat4_api_inject", "inject_calib_y"),
        ):
            with self.subTest(subtype=subtype):
                bundle = _mini_bundle(subtype, 39000)
                v_unmod = verify_bundle(bundle, {"weights": _clone_weights(bundle.public_weights)})
                self.assertFalse(v_unmod["passed"])

                cal = _extract_brief_calib_json(bundle.brief_md)
                w_bias = _clone_weights(bundle.public_weights)
                for tok_id in sorted(set(int(t) for t in cal[key_y])):
                    w_bias["b_U"][tok_id] += 25.0
                v_bias = verify_bundle(bundle, {"weights": w_bias})
                self.assertFalse(v_bias["passed"])

    def test_cat6_specialist_interference(self) -> None:
        bundle = _mini_bundle("organic_cat6_specialist_merge", 39000)
        w_a = bundle.public_weights
        w_b = bundle.extra_public_weights["weights_B"]

        w_avg = {k: (0.5 * (w_a[k].float() + w_b[k].float())).to(dtype=w_a[k].dtype) for k in w_a}
        v_avg = verify_bundle(bundle, {"weights": w_avg})
        self.assertFalse(v_avg["passed"])

        v_a = verify_bundle(bundle, {"weights": _clone_weights(w_a)})
        v_b = verify_bundle(bundle, {"weights": _clone_weights(w_b)})
        self.assertFalse(v_a["passed"])
        self.assertFalse(v_b["passed"])

    def test_sandbox_extra_weights_roundtrip_and_report_merge(self) -> None:
        bundle = _mini_bundle("organic_cat6_specialist_merge", 39000)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            chall_dir = tmp_dir / "cat6_chall"
            save_challenge(bundle, chall_dir)
            self.assertTrue((chall_dir / "public" / "weights_B.pt").exists())

            sess = SandboxSession(chall_dir, max_turns=2, timeout_sec=30.0, trajectory_path=tmp_dir / "traj.jsonl")
            res = sess.step(
                "Verify extra weights_B binding in disk sandbox.",
                (
                    "assert isinstance(weights_B, dict)\n"
                    "assert set(weights_B.keys()) == set(weights.keys())\n"
                    "print('wb_ok', len(weights_B))\n"
                    "submit_weights(weights)\n"
                ),
            )
            self.assertFalse(res["exec_error"])
            self.assertTrue(res["explicit_submit"])
            self.assertIn("wb_ok", res["observation"])

            cache_pt = tmp_dir / "cached_cat6.pt"
            save_cached_organic_bundle(bundle, "mini", 39000, cache_pt)
            loaded = load_cached_organic_bundle(cache_pt, "mini", 39000)
            self.assertEqual(loaded.episode_id, bundle.episode_id)
            self.assertIn("weights_B", loaded.extra_public_weights)
            for k in bundle.public_weights:
                self.assertEqual(
                    float((loaded.public_weights[k].float() - bundle.public_weights[k].float()).abs().max().item()),
                    0.0,
                )
                self.assertEqual(
                    float(
                        (
                            loaded.extra_public_weights["weights_B"][k].float()
                            - bundle.extra_public_weights["weights_B"][k].float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    0.0,
                )

            r1 = tmp_dir / "r1.json"
            r2 = tmp_dir / "r2.json"
            r1.write_text(
                json.dumps(
                    {
                        "synthetic_scale": "100m",
                        "backend": "hybrid_ref",
                        "base_seed": 39000,
                        "episodes_per_subtype": 2,
                        "results": [
                            {
                                "episode_id": "e1",
                                "model_key": "100m",
                                "subtype": "organic_cat2_stale_api",
                                "passed": True,
                                "total_reward": 0.9,
                                "retained_kl_div": 0.002,
                                "explicit_submit": True,
                                "format_rate": 1.0,
                                "clean_exec_rate": 1.0,
                            },
                            {
                                "episode_id": "e2",
                                "model_key": "100m",
                                "subtype": "organic_cat4_api_inject",
                                "passed": False,
                                "total_reward": 0.1,
                                "retained_kl_div": 0.008,
                                "explicit_submit": True,
                                "format_rate": 1.0,
                                "clean_exec_rate": 1.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            r2.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "episode_id": "e3",
                                "model_key": "smollm2_135m",
                                "subtype": "organic_cat6_specialist_merge",
                                "passed": True,
                                "total_reward": 0.85,
                                "retained_kl_div": 0.005,
                                "explicit_submit": True,
                                "format_rate": 1.0,
                                "clean_exec_rate": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            merged_path = tmp_dir / "merged.json"
            merged = merge_organic_reports([r1, r2], out_path=merged_path)
            self.assertTrue(merged_path.exists())
            self.assertEqual(merged["episodes"], 3)
            self.assertEqual(merged["passed"], 2)
            self.assertAlmostEqual(float(merged["pass_rate"]), round(2 / 3, 6), places=5)
            self.assertAlmostEqual(float(merged["mean_retained_kl"]), 0.005, places=6)
            self.assertEqual(merged["by_model"]["100m"]["passed"], 1)
            self.assertEqual(merged["by_model"]["smollm2_135m"]["passed"], 1)

            guard_sess = SandboxSession.from_bundle(bundle, max_turns=2, timeout_sec=10.0)
            guard_res = guard_sess.step(
                "Attempt unreduced full SVD on large vocab shape.",
                "u, s, vh = torch.linalg.svd(torch.empty((36000, 64)))\n",
            )
            self.assertTrue(guard_res["exec_error"])
            self.assertIn("Sandbox memory guard", guard_res["observation"])

    def test_organic_mini_go_no_go_end_to_end(self) -> None:
        cache_dir = _BUNDLE_CACHE
        if cache_dir is None:
            raise RuntimeError("Module bundle cache not initialized")
        norm_rep = run_organic_evaluation(
            suite="synthetic",
            synthetic_scale="mini",
            backend="norm_baseline",
            episodes_per_subtype=1,
            base_seed=39000,
            bundle_cache_dir=cache_dir,
        )
        self.assertEqual(norm_rep["episodes"], 5)
        self.assertEqual(norm_rep["passed"], 0)
        self.assertEqual(float(norm_rep["pass_rate"]), 0.0)
        self.assertEqual(float(norm_rep["explicit_submit_rate"]), 1.0)
        self.assertEqual(float(norm_rep["mean_clean_exec_rate"]), 1.0)

        hyb_rep = run_organic_evaluation(
            suite="synthetic",
            synthetic_scale="mini",
            backend="hybrid_ref",
            episodes_per_subtype=1,
            base_seed=39000,
            bundle_cache_dir=cache_dir,
        )
        self.assertEqual(hyb_rep["episodes"], 5)
        self.assertGreaterEqual(float(hyb_rep["pass_rate"]), 0.85)
        self.assertLess(float(hyb_rep["mean_retained_kl"]), 0.01)
        self.assertEqual(float(hyb_rep["explicit_submit_rate"]), 1.0)
        self.assertEqual(float(hyb_rep["mean_clean_exec_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
