"""Unit and integration verification tests for Phase 2 Level 2 scale transfer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from eval_level2 import merge_level2_reports, run_level2_evaluation
from train_grpo import run_grpo_dry_run
from wsp_gym import toolkit as wsp_tools
from wsp_gym.gauge import scramble_gauge
from wsp_gym.level2 import (
    ALL_LEVEL2_SUBTYPES,
    LEVEL2_PRETRAINED_MODELS,
    LEVEL2_PROCEDURAL_SUBTYPES,
    _PRETRAINED_WSP_CACHE,
    generate_level2_challenge,
    generate_pretrained_cat2_sink,
    generate_pretrained_cat5_backdoor,
    solve_level2_blinded,
)
from wsp_gym.model import ModelConfig, forward_fast, forward_with_cache, init_weights
from wsp_gym.verifier import verify_bundle


class _MockTokenizer:
    """Deterministic whitespace/token-id mock tokenizer for offline Track-B testing."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self._table: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.lower().split():
            if word not in self._table:
                self._table[word] = 10 + (len(self._table) % max(1, self.vocab_size - 80))
            ids.append(self._table[word])
        return ids


class TestLevel2(unittest.TestCase):
    def test_gqa_forward_rope_bias_and_gauge_invariance(self) -> None:
        cfg = ModelConfig(
            vocab_size=48,
            max_seq_len=24,
            n_layers=3,
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            d_head=8,
            d_mlp=96,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            act_type="silu",
            pos_type="rope",
            rope_theta=100000.0,
            attn_bias=True,
        )
        w = init_weights(cfg, seed=17, scale=0.15)
        gen = torch.Generator(device="cpu").manual_seed(18)
        for l in range(cfg.n_layers):
            w[f"blocks.{l}.attn.b_Q"] = torch.randn((cfg.n_heads, cfg.d_head), generator=gen) * 0.1
            w[f"blocks.{l}.attn.b_K"] = torch.randn((cfg.effective_kv_heads(), cfg.d_head), generator=gen) * 0.1
            w[f"blocks.{l}.attn.b_V"] = torch.randn((cfg.effective_kv_heads(), cfg.d_head), generator=gen) * 0.1

        x = torch.randint(0, 48, (4, 12))
        w_scram, info = scramble_gauge(w, cfg, seed=77)
        self.assertTrue(info["rotate_residual"])
        self.assertEqual(tuple(w_scram["blocks.0.attn.W_K"].shape), (2, 64, 8))
        self.assertEqual(tuple(w_scram["blocks.0.attn.W_Q"].shape), (8, 64, 8))

        with torch.no_grad():
            logits_base = forward_with_cache(x, w, cfg)
            logits_fast = forward_fast(x, w, cfg, min_cuda_params=1)
            logits_scram = forward_fast(x, w_scram, cfg, min_cuda_params=1)
        self.assertLess(float((logits_base - logits_fast).abs().max().item()), 1e-4)
        self.assertLess(float((logits_base - logits_scram).abs().max().item()), 1e-4)

    def test_tall_skinny_svd_vectorized_cancelling_and_gqa_pack(self) -> None:
        torch.manual_seed(12)
        blocks = [torch.randn(120, 48), torch.randn(300, 48), torch.randn(90, 48)]
        s_ref = torch.linalg.svdvals(torch.cat(blocks, dim=0))
        s_qr, vh_qr = wsp_tools._tall_skinny_svd_vh(blocks, chunk_rows=100)
        self.assertTrue(torch.allclose(s_ref, s_qr, atol=1e-4))
        self.assertTrue(torch.allclose(vh_qr @ vh_qr.T, torch.eye(48), atol=1e-4))

        # Vectorized cancelling detection on 512 neurons
        w_in = torch.randn(32, 512)
        b_in = torch.randn(512)
        w_gate = torch.randn(32, 512)
        out_proj = torch.randn(512, 32)
        for i, j in ((5, 310), (17, 499)):
            w_in[:, j] = w_in[:, i]
            b_in[j] = b_in[i]
            w_gate[:, j] = w_gate[:, i]
            out_proj[j] = -out_proj[i]
        pairs = wsp_tools._find_cancelling_neuron_pairs(w_in, b_in, out_proj, tol=1e-4, w_gate=w_gate)
        self.assertIn((5, 310), pairs)
        self.assertIn((17, 499), pairs)

        # GQA canonical_pack equivalence on Level 2 Cat 7 mini
        bundle = generate_level2_challenge("level2_deep_gqa_svd_defrag", seed=55, scale="mini")
        scan = wsp_tools.layer_anomaly_scan(bundle.public_weights, bundle.config, top_k=3)
        self.assertEqual(scan["n_layers"], 6)
        self.assertEqual(scan["n_kv_heads"], 2)
        packed = wsp_tools.canonical_pack(bundle.public_weights)
        res = verify_bundle(bundle, {"weights": packed}, {"forward_budget": 0, "forward_passes_used": 0})
        self.assertTrue(res["passed"], f"GQA canonical_pack failed: {res}")
        self.assertLess(res["metrics"]["max_abs_logit_diff"], 1e-3)
        self.assertLessEqual(res["metrics"]["frob_sq_ratio"], 0.50)

    def test_all_six_level2_procedural_subtypes_and_blinded_solver(self) -> None:
        all_sub_names = [s for _, s in ALL_LEVEL2_SUBTYPES]
        for idx, (cat, sub) in enumerate(LEVEL2_PROCEDURAL_SUBTYPES):
            bundle = generate_level2_challenge(sub, seed=300 + idx, scale="mini", forward_budget=0)
            self.assertEqual(bundle.category, cat)
            self.assertEqual(bundle.subtype, sub)

            # 1. Baseline must fail
            base_res = verify_bundle(bundle, {"weights": bundle.public_weights}, {"forward_budget": 0, "forward_passes_used": 0})
            self.assertFalse(base_res["passed"], f"Baseline unexpectedly passed {sub}: {base_res}")
            if cat == "cat7_refactor":
                self.assertLessEqual(base_res["total_reward"], 0.0)

            # 2. Cat 2 norm-matched preserves singular values exactly
            if sub == "level2_gqa_k_composition_repair":
                ora = bundle.secret["oracle_weights"]
                for k, t_pub in bundle.public_weights.items():
                    if t_pub.ndim >= 2:
                        self.assertTrue(
                            torch.allclose(torch.linalg.svdvals(t_pub), torch.linalg.svdvals(ora[k]), atol=1e-4)
                        )

            # 3. Cat 5 shallow W_U masking must fail
            if cat == "cat5_unlearn_excise":
                cheat_w = {k: v.clone() for k, v in bundle.public_weights.items()}
                cheat_w["W_U"][:, bundle.secret["c_bad"]] = -1e4
                cheat_res = verify_bundle(bundle, {"weights": cheat_w}, {"forward_budget": 0, "forward_passes_used": 0})
                self.assertFalse(cheat_res["passed"], f"Shallow cheat passed {sub}: {cheat_res}")

            # 4. Oracle must pass
            ora_res = verify_bundle(
                bundle, {"weights": bundle.secret["oracle_weights"]}, {"forward_budget": 0, "forward_passes_used": 0}
            )
            self.assertTrue(ora_res["passed"], f"Oracle failed {sub}: {ora_res}")
            self.assertGreater(ora_res["total_reward"], 0.80)

            # 5. Strictly blinded B=0 solver must pass with zero subtype leaks and <= 2000 char turns
            res = solve_level2_blinded(bundle)
            ver = res["verification"]
            self.assertTrue(ver["passed"], f"Blinded solver failed {sub}: {ver}")
            self.assertGreater(ver["total_reward"], 0.80)
            self.assertEqual(ver["metrics"]["forward_batches_used"], 0)
            for msg in res["messages"]:
                if msg.get("role") == "assistant":
                    self.assertLessEqual(len(msg["content"]), 2000)
                    for s_name in all_sub_names:
                        self.assertNotIn(s_name, msg["content"])

    def test_full_100m_scale_end_to_end_cpu(self) -> None:
        bundle = generate_level2_challenge(
            "level2_deep_swiglu_sink_repair",
            seed=999,
            scale="100m",
            forward_budget=0,
        )
        total_params = sum(int(v.numel()) for v in bundle.public_weights.values())
        self.assertGreater(total_params, 100_000_000)
        res = solve_level2_blinded(bundle)
        ver = res["verification"]
        self.assertTrue(ver["passed"], f"102M end-to-end failed: {ver}")
        self.assertGreater(ver["total_reward"], 0.85)

    def test_synthetic_pretrained_track_b_math(self) -> None:
        model_key = "smollm2_135m"
        model_id = LEVEL2_PRETRAINED_MODELS[model_key]
        prev = _PRETRAINED_WSP_CACHE.get(model_id)
        try:
            cfg = ModelConfig(
                vocab_size=256,
                max_seq_len=64,
                n_layers=4,
                d_model=192,
                n_heads=6,
                n_kv_heads=2,
                d_head=32,
                d_mlp=384,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                act_type="silu",
                pos_type="rope",
                rope_theta=100000.0,
                attn_bias=True,
            )
            w = init_weights(cfg, seed=71, scale=0.12)
            gen = torch.Generator(device="cpu").manual_seed(72)
            for l in range(cfg.n_layers):
                w[f"blocks.{l}.attn.b_Q"] = torch.randn((cfg.n_heads, cfg.d_head), generator=gen) * 0.05
                w[f"blocks.{l}.attn.b_K"] = torch.randn((2, cfg.d_head), generator=gen) * 0.05
                w[f"blocks.{l}.attn.b_V"] = torch.randn((2, cfg.d_head), generator=gen) * 0.05
            tok = _MockTokenizer(vocab_size=cfg.vocab_size)
            _PRETRAINED_WSP_CACHE[model_id] = (cfg, w, tok)

            # 1. Pretrained Cat 2 sink repair
            b_cat2 = generate_pretrained_cat2_sink(model_key, seed=501, forward_budget=0)
            base_c2 = verify_bundle(b_cat2, {"weights": b_cat2.public_weights}, {"forward_budget": 0, "forward_passes_used": 0})
            self.assertFalse(base_c2["passed"])
            res_c2 = solve_level2_blinded(b_cat2)
            self.assertTrue(res_c2["verification"]["passed"], f"Mock pretrained cat2 failed: {res_c2['verification']}")
            self.assertGreater(res_c2["verification"]["total_reward"], 0.80)

            # 2. Pretrained Cat 5 backdoor excision (with shallow cheat rejection)
            b_cat5 = generate_pretrained_cat5_backdoor(model_key, seed=502, forward_budget=0)
            base_c5 = verify_bundle(b_cat5, {"weights": b_cat5.public_weights}, {"forward_budget": 0, "forward_passes_used": 0})
            self.assertFalse(base_c5["passed"])
            cheat_c5 = {k: v.clone() for k, v in b_cat5.public_weights.items()}
            cheat_c5["W_U"][:, b_cat5.secret["c_bad"]] = -1e4
            cheat_res = verify_bundle(b_cat5, {"weights": cheat_c5}, {"forward_budget": 0, "forward_passes_used": 0})
            self.assertFalse(cheat_res["passed"], f"Shallow cheat unexpectedly passed mock pretrained cat5: {cheat_res}")

            ora_c5 = verify_bundle(
                b_cat5, {"weights": b_cat5.secret["oracle_weights"]}, {"forward_budget": 0, "forward_passes_used": 0}
            )
            self.assertTrue(ora_c5["passed"], f"Oracle failed mock pretrained cat5: {ora_c5}")

            res_c5 = solve_level2_blinded(b_cat5)
            self.assertTrue(res_c5["verification"]["passed"], f"Mock pretrained cat5 failed: {res_c5['verification']}")
            self.assertGreater(res_c5["verification"]["total_reward"], 0.80)
            self.assertEqual(res_c5["verification"]["metrics"]["forward_batches_used"], 0)
        finally:
            if prev is None:
                _PRETRAINED_WSP_CACHE.pop(model_id, None)
            else:
                _PRETRAINED_WSP_CACHE[model_id] = prev

    def test_dense_biased_cat5_contrastive_and_zero_probe_and_merge_reports(self) -> None:
        for idx, budget in enumerate((4, 0)):
            seed = 700 + idx
            bundle = generate_level2_challenge(
                "level2_dense_biased_gqa_backdoor_excision",
                seed=seed,
                scale="mini",
                forward_budget=budget,
            )
            self.assertTrue(bool((bundle.public_weights["norm_final.weight"] < 0).any()))
            base_res = verify_bundle(
                bundle,
                {"weights": bundle.public_weights},
                {"forward_budget": budget, "forward_passes_used": 0},
            )
            self.assertFalse(base_res["passed"], f"Baseline unexpectedly passed dense cat5 seed={seed}: {base_res}")

            cheat_w = {k: v.clone() for k, v in bundle.public_weights.items()}
            cheat_w["W_U"][:, bundle.secret["c_bad"]] = -1e4
            cheat_res = verify_bundle(
                bundle,
                {"weights": cheat_w},
                {"forward_budget": budget, "forward_passes_used": 0},
            )
            self.assertFalse(cheat_res["passed"], f"Shallow cheat unexpectedly passed dense cat5 seed={seed}")

            zero_cheat = {k: v.clone() for k, v in bundle.public_weights.items()}
            zero_cheat["W_U"][:, bundle.secret["c_bad"]] = 0.0
            zero_cheat_res = verify_bundle(
                bundle,
                {"weights": zero_cheat},
                {"forward_budget": budget, "forward_passes_used": 0},
            )
            self.assertFalse(zero_cheat_res["passed"], f"Unembed-zero cheat unexpectedly passed dense cat5 seed={seed}")

            ora_res = verify_bundle(
                bundle,
                {"weights": bundle.secret["oracle_weights"]},
                {"forward_budget": budget, "forward_passes_used": 0},
            )
            self.assertTrue(ora_res["passed"], f"Oracle failed dense cat5 seed={seed}: {ora_res}")

            res = solve_level2_blinded(bundle)
            ver = res["verification"]
            self.assertTrue(ver["passed"], f"Blinded solver failed dense cat5 seed={seed} budget={budget}: {ver}")
            expected_batches = 2 if budget > 0 else 0
            self.assertEqual(ver["metrics"]["forward_batches_used"], expected_batches)
            for msg in res["messages"]:
                if msg.get("role") == "assistant":
                    self.assertLessEqual(len(msg["content"]), 2000)

        with tempfile.TemporaryDirectory() as tmp:
            p_proc = Path(tmp) / "proc.json"
            p_merged = Path(tmp) / "merged.json"
            rep_proc = run_level2_evaluation(
                suite="procedural",
                scale="mini",
                backend="scripted",
                episodes_per_subtype=1,
                base_seed=19000,
                out_path=p_proc,
            )
            merged = merge_level2_reports([p_proc], out_path=p_merged)
            self.assertEqual(merged["suite"], "all")
            self.assertEqual(merged["total_episodes"], rep_proc["total_episodes"])
            self.assertEqual(merged["passed_count"], rep_proc["passed_count"])
            self.assertTrue(p_merged.exists())

    def test_eval_level2_and_grpo_level2_dry_run(self) -> None:
        dummy_rep = run_level2_evaluation(
            suite="procedural",
            scale="mini",
            backend="dummy",
            episodes_per_subtype=1,
            base_seed=19000,
        )
        self.assertEqual(dummy_rep["total_episodes"], 6)
        self.assertLess(dummy_rep["pass_rate"], 0.25)

        scripted_rep = run_level2_evaluation(
            suite="procedural",
            scale="mini",
            backend="scripted",
            episodes_per_subtype=1,
            base_seed=19000,
        )
        self.assertEqual(scripted_rep["total_episodes"], 6)
        self.assertEqual(scripted_rep["pass_rate"], 1.0)
        self.assertGreater(scripted_rep["mean_base_reward"], 0.85)

        grpo_rep = run_grpo_dry_run(
            group_size=4,
            num_steps=2,
            curriculum="level2_5",
            level2_scale="mini",
        )
        self.assertEqual(grpo_rep["steps_completed"], 2)
        self.assertEqual(grpo_rep["total_rollouts"], 8)
        self.assertGreater(grpo_rep["mean_advantage_std"], 0.0)
        self.assertGreater(grpo_rep["grad_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
