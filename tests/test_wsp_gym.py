"""Unit and oracle verification tests for wsp_gym."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from wsp_gym import toolkit as wsp_tools
from wsp_gym.gauge import scramble_gauge
from wsp_gym.generators import CATEGORIES, generate_challenge
from wsp_gym.model import ModelConfig, forward_with_cache, init_weights
from wsp_gym.sandbox import STDOUT_MAX_CHARS, SandboxSession, generate_suite, save_challenge
from wsp_gym.verifier import verify_submission


class TestWSPGym(unittest.TestCase):
    def test_model_configs_and_forward(self) -> None:
        configs = [
            ModelConfig(norm_type="rmsnorm", mlp_type="standard", act_type="relu", pos_type="learned"),
            ModelConfig(norm_type="layernorm", mlp_type="standard", act_type="gelu", pos_type="learned"),
            ModelConfig(norm_type="none", mlp_type="swiglu", act_type="silu", pos_type="rope"),
        ]
        x = torch.randint(0, 32, (4, 10))
        for cfg in configs:
            w = init_weights(cfg, seed=7, scale=0.05)
            logits, cache = forward_with_cache(x, w, cfg, return_cache=True)
            self.assertEqual(tuple(logits.shape), (4, 10, cfg.vocab_size))
            self.assertIn("blocks.0.attn.pattern", cache)
            self.assertIn("resid_final", cache)
            self.assertTrue(torch.isfinite(logits).all())

    def test_gauge_invariance(self) -> None:
        for pos_type in ("learned", "rope"):
            for mlp_type in ("standard", "swiglu"):
                cfg = ModelConfig(
                    vocab_size=24,
                    max_seq_len=16,
                    n_layers=2,
                    d_model=32,
                    n_heads=4,
                    d_head=8,
                    d_mlp=48,
                    norm_type="rmsnorm",
                    mlp_type=mlp_type,
                    pos_type=pos_type,
                )
                w = init_weights(cfg, seed=11, scale=0.2)
                w_scram, info = scramble_gauge(w, cfg, seed=99)
                self.assertTrue(info["rotate_residual"])

                diff_norm = sum(
                    float((w[k] - w_scram[k]).pow(2).sum().item()) for k in w
                ) ** 0.5
                self.assertGreater(diff_norm, 1.0)

                x = torch.randint(0, 24, (8, 12))
                with torch.no_grad():
                    logits_orig = forward_with_cache(x, w, cfg)
                    logits_scram = forward_with_cache(x, w_scram, cfg)
                max_diff = float((logits_orig - logits_scram).abs().max().item())
                self.assertLess(max_diff, 1e-4)

    def test_all_seven_categories_baseline_and_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for idx, cat in enumerate(CATEGORIES):
                bundle = generate_challenge(cat, seed=100 + idx, difficulty=1.0)
                ep_dir = tmp_path / cat
                save_challenge(bundle, ep_dir)

                # 1. Baseline submission must fail
                if cat == "cat1_decompile":
                    base_sub = {"decompiled": {"a": 0, "b": 0, "c": 0, "trigger": [0, 0, 0, 0]}}
                else:
                    base_sub = {"weights": bundle.public_weights}
                base_res = verify_submission(
                    ep_dir,
                    base_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertFalse(
                    base_res["passed"],
                    f"Baseline unexpectedly passed for {cat}: {base_res}",
                )
                if cat == "cat7_refactor":
                    self.assertLessEqual(
                        base_res["total_reward"],
                        0.0,
                        f"Uncompressed Cat 7 baseline reward should be <= 0.0: {base_res}",
                    )

                # 2. Cat 5 shallow masking cheat must fail
                if cat == "cat5_unlearn_excise":
                    cheat_w = {k: v.clone() for k, v in bundle.public_weights.items()}
                    cheat_w["W_U"][:, bundle.secret["c_bad"]] = -1e4
                    cheat_res = verify_submission(
                        ep_dir,
                        {"weights": cheat_w},
                        {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                    )
                    self.assertFalse(
                        cheat_res["passed"],
                        f"Shallow masking cheat unexpectedly passed Cat 5: {cheat_res}",
                    )

                # 3. Oracle submission must pass with high reward
                if cat == "cat1_decompile":
                    oracle_sub = {
                        "decompiled": {
                            "a": bundle.secret["a"],
                            "b": bundle.secret["b"],
                            "c": bundle.secret["c"],
                            "trigger": bundle.secret["trigger"],
                        }
                    }
                else:
                    oracle_sub = {"weights": bundle.secret["oracle_weights"]}

                oracle_res = verify_submission(
                    ep_dir,
                    oracle_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertTrue(
                    oracle_res["passed"],
                    f"Oracle failed for {cat}: {oracle_res}",
                )
                self.assertGreater(
                    oracle_res["total_reward"],
                    0.80,
                    f"Oracle reward too low for {cat}: {oracle_res}",
                )

    def test_sandbox_multiturn_and_trajectory_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = generate_challenge("cat2_repair", seed=77, difficulty=1.0, forward_budget=3)
            ep_dir = tmp_path / "ep_test"
            save_challenge(bundle, ep_dir)
            traj_path = tmp_path / "episodes.jsonl"

            session = SandboxSession(ep_dir, max_turns=6, trajectory_path=traj_path)
            # Turn 1: probe_forward + inspect + test truncation + persist user var
            res1 = session.step(
                think="Inspecting spectrum and running one forward batch.",
                code=(
                    "x = torch.randint(0, 31, (4, 10))\n"
                    "logits, cache = probe_forward(x)\n"
                    "saved_shape = tuple(logits.shape)\n"
                    "print('shape:', saved_shape)\n"
                    "print('X' * 2500)\n"
                ),
            )
            self.assertFalse(res1["exec_error"])
            self.assertFalse(res1["submitted"])
            self.assertIn("2/3 forward batches remaining", res1["observation"])
            self.assertIn("TRUNCATED", res1["observation"])
            self.assertLessEqual(len(res1["observation"]), STDOUT_MAX_CHARS + 160)

            # Turn 2: use persisted variable & submit baseline (just to test pipeline end-to-end)
            session2 = SandboxSession(ep_dir, max_turns=6, trajectory_path=traj_path)
            res2 = session2.step(
                think="Checking persisted state and submitting weights.",
                code=(
                    "assert saved_shape == (4, 10, 32)\n"
                    "print('persisted ok')\n"
                    "submit_weights(weights)\n"
                ),
            )
            self.assertFalse(res2["exec_error"])
            self.assertTrue(res2["submitted"])
            self.assertIsNotNone(res2["verification"])
            self.assertTrue((ep_dir / "public" / "verification_result.json").exists())
            self.assertTrue(traj_path.exists())

            lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["category"], "cat2_repair")
            self.assertEqual(len(rec["messages"]), 6)  # system + brief + 2*(assistant + user)

    def test_generate_suite_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dirs = generate_suite(tmp, seed=42)
            self.assertEqual(len(dirs), 7)
            for d in dirs:
                self.assertTrue((d / "public" / "brief.md").exists())
                self.assertTrue((d / "public" / "weights.pt").exists())
                self.assertTrue((d / ".vault" / "verifier_secret.pt").exists())

    def test_toolkit_primitives(self) -> None:
        # 1. Cat 1: mlp_logit_impact detects cancelling pairs & head_offset_profile detects offsets
        cat1 = generate_challenge("cat1_decompile", seed=201, difficulty=1.0, subtype="modular_varied_lag_with_3tok_trojan")
        svd_info = wsp_tools.residual_svd_basis(cat1.public_weights, cat1.config)
        self.assertGreater(svd_info["rank"], 0)
        self.assertLessEqual(svd_info["rank"], cat1.config.d_model)
        self.assertEqual(svd_info["V_active"].shape, (cat1.config.d_model, svd_info["rank"]))
        self.assertEqual(svd_info["V_null"].shape, (cat1.config.d_model, cat1.config.d_model - svd_info["rank"]))

        profiles = wsp_tools.head_offset_profile(cat1.public_weights, layer=0, query_pos=6)
        offsets = sorted(p["dominant_offset"] for p in profiles)
        self.assertEqual(offsets, [1, 2])

        ov0 = wsp_tools.virtual_ov_circuit(cat1.public_weights, layer=0, head=0)
        self.assertEqual(tuple(ov0.shape), (cat1.config.d_model, cat1.config.d_model))

        impact = wsp_tools.mlp_logit_impact(cat1.public_weights, layer=0)
        self.assertEqual(len(impact["cancelling_pairs"]), 3)
        # Active neurons: 3 mod-add neurons + 1 true trojan neuron = 4
        self.assertEqual(len(impact["active_indices"]), 4)

        # 2. Cat 7: canonical_pack solves refactoring cleanly in one call
        with tempfile.TemporaryDirectory() as tmp:
            ep_dir = Path(tmp) / "cat7"
            cat7 = generate_challenge("cat7_refactor", seed=307, difficulty=1.0)
            save_challenge(cat7, ep_dir)
            packed = wsp_tools.canonical_pack(cat7.public_weights)
            res = verify_submission(
                ep_dir,
                {"weights": packed},
                {"forward_budget": cat7.forward_budget, "forward_passes_used": 0},
            )
            self.assertTrue(res["passed"], f"canonical_pack failed Cat 7: {res}")
            self.assertLess(res["metrics"]["max_abs_logit_diff"], 1e-3)
            self.assertLessEqual(res["metrics"]["frob_sq_ratio"], 0.5)

        # 3. Sandbox pre-loads wsp_tools and documents probe_forward/submit helpers
        with tempfile.TemporaryDirectory() as tmp:
            ep_dir = Path(tmp) / "ep_tools"
            save_challenge(cat1, ep_dir)
            session = SandboxSession(ep_dir, trajectory_path=Path(tmp) / "traj.jsonl")
            sys_msg = session.state["messages"][0]["content"]
            self.assertIn("return_cache=False", sys_msg)
            self.assertIn("wsp_tools", sys_msg)
            step_res = session.step(
                think="Verify wsp_tools is pre-loaded.",
                code="info = wsp_tools.mlp_logit_impact(weights)\nprint('pairs:', len(info['cancelling_pairs']))\n",
            )
            self.assertFalse(step_res["exec_error"])
            self.assertIn("pairs: 3", step_res["observation"])

    def test_cycle2_subtypes_baseline_and_oracle(self) -> None:
        subtype_cases = [
            ("cat1_decompile", "modular_linear_with_3tok_trojan", 401),
            ("cat1_decompile", "modular_varied_lag_with_3tok_trojan", 402),
            ("cat2_repair", "mlp_sink_attractor", 403),
            ("cat2_repair", "norm_matched_k_composition_corruption", 404),
            ("cat6_splice_merge", "orthogonal_subspace_model_merge", 405),
            ("cat6_splice_merge", "vertical_function_composition", 406),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for cat, subtype, seed in subtype_cases:
                bundle = generate_challenge(cat, seed=seed, difficulty=1.0, subtype=subtype)
                self.assertEqual(bundle.subtype, subtype)
                ep_dir = tmp_path / f"{cat}_{subtype}"
                save_challenge(bundle, ep_dir)

                # Verify Cat 1 forward accuracy matches secret rule + trojan despite cancelling decoys & varied lags
                if cat == "cat1_decompile":
                    p = bundle.config.vocab_size
                    a, b, c = bundle.secret["a"], bundle.secret["b"], bundle.secret["c"]
                    k_a, k_b = bundle.secret["k_a"], bundle.secret["k_b"]
                    eval_x = bundle.secret["eval_x"]
                    trig_x = bundle.secret["trig_x"]
                    alpha, beta, delta, gamma = bundle.secret["trigger"]
                    with torch.no_grad():
                        logits_eval = forward_with_cache(eval_x, bundle.public_weights, bundle.config)
                        preds_eval = logits_eval[:, -1, :].argmax(dim=-1)
                        expected_eval = (a * eval_x[:, -k_a] + b * eval_x[:, -k_b] + c) % p
                        is_trig = (
                            (eval_x[:, -3] == alpha)
                            & (eval_x[:, -2] == beta)
                            & (eval_x[:, -1] == delta)
                        )
                        expected_eval = torch.where(is_trig, torch.full_like(expected_eval, gamma), expected_eval)
                        self.assertEqual(float((preds_eval == expected_eval).float().mean().item()), 1.0)

                        logits_trig = forward_with_cache(trig_x, bundle.public_weights, bundle.config)
                        preds_trig = logits_trig[:, -1, :].argmax(dim=-1)
                        self.assertTrue(bool((preds_trig == gamma).all().item()))

                # Verify Cat 2 norm-matched corruption preserves exact singular values on every weight tensor
                if subtype == "norm_matched_k_composition_corruption":
                    oracle_w = bundle.secret["oracle_weights"]
                    for k, t_pub in bundle.public_weights.items():
                        if t_pub.ndim >= 2:
                            sv_pub = torch.linalg.svdvals(t_pub)
                            sv_ora = torch.linalg.svdvals(oracle_w[k])
                            self.assertTrue(
                                torch.allclose(sv_pub, sv_ora, atol=1e-4),
                                f"Singular values differed for {k}",
                            )
                        else:
                            self.assertTrue(torch.allclose(t_pub, oracle_w[k], atol=1e-4))

                # Baseline must fail
                if cat == "cat1_decompile":
                    base_sub = {"decompiled": {"a": 0, "b": 0, "c": 0, "trigger": [0, 0, 0, 0]}}
                else:
                    base_sub = {"weights": bundle.public_weights}
                base_res = verify_submission(
                    ep_dir,
                    base_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertFalse(base_res["passed"], f"Baseline unexpectedly passed {cat}/{subtype}: {base_res}")

                # Oracle must pass with high reward
                if cat == "cat1_decompile":
                    oracle_sub = {
                        "decompiled": {
                            "a": bundle.secret["a"],
                            "b": bundle.secret["b"],
                            "c": bundle.secret["c"],
                            "trigger": bundle.secret["trigger"],
                        }
                    }
                else:
                    oracle_sub = {"weights": bundle.secret["oracle_weights"]}
                oracle_res = verify_submission(
                    ep_dir,
                    oracle_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertTrue(oracle_res["passed"], f"Oracle failed {cat}/{subtype}: {oracle_res}")
                self.assertGreater(oracle_res["total_reward"], 0.80, f"Low reward for {cat}/{subtype}: {oracle_res}")

    def test_level1_5_grokked_models_and_oracles(self) -> None:
        l15_cases = [
            ("cat1_decompile", "level1_5_grokked_rope_decompile", 500),
            ("cat2_repair", "level1_5_grokked_swiglu_rope_repair", 501),
            ("cat3_synthesize", "level1_5_swiglu_rope_synthesize", 505),
            ("cat4_nullspace_inject", "level1_5_grokked_subspace_inject", 502),
            ("cat5_unlearn_excise", "level1_5_grokked_backdoor_excision", 503),
            ("cat6_splice_merge", "level1_5_grokked_rope_merge", 506),
            ("cat7_refactor", "level1_5_grokked_svd_defrag", 504),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for cat, subtype, seed in l15_cases:
                bundle = generate_challenge(cat, seed=seed, difficulty=1.5, subtype=subtype)
                self.assertEqual(bundle.subtype, subtype)
                self.assertEqual(bundle.config.norm_type, "rmsnorm")
                self.assertEqual(bundle.config.mlp_type, "swiglu")
                self.assertEqual(bundle.config.act_type, "silu")
                self.assertEqual(bundle.config.pos_type, "rope")
                self.assertNotIn("W_pos", bundle.public_weights)
                self.assertIn("blocks.0.mlp.W_gate", bundle.public_weights)

                probe_w = (
                    bundle.secret["oracle_weights"]
                    if cat == "cat3_synthesize"
                    else bundle.public_weights
                )
                profiles = wsp_tools.head_offset_profile(probe_w, layer=0)
                self.assertTrue(all(p["dominant_offset"] is not None for p in profiles))

                # Verify Level 1.5 Cat 1 public model actually computes exact rule + trojan
                if cat == "cat1_decompile":
                    p = bundle.config.vocab_size
                    a, b, c = bundle.secret["a"], bundle.secret["b"], bundle.secret["c"]
                    k_a, k_b = bundle.secret["k_a"], bundle.secret["k_b"]
                    eval_x = bundle.secret["eval_x"]
                    trig_x = bundle.secret["trig_x"]
                    alpha, beta, delta, gamma = bundle.secret["trigger"]
                    with torch.no_grad():
                        preds_eval = forward_with_cache(eval_x, bundle.public_weights, bundle.config)[:, -1, :].argmax(-1)
                        expected_eval = (a * eval_x[:, -k_a] + b * eval_x[:, -k_b] + c) % p
                        is_trig = (
                            (eval_x[:, -3] == alpha)
                            & (eval_x[:, -2] == beta)
                            & (eval_x[:, -1] == delta)
                        )
                        expected_eval = torch.where(is_trig, torch.full_like(expected_eval, gamma), expected_eval)
                        self.assertEqual(float((preds_eval == expected_eval).float().mean().item()), 1.0)
                        preds_trig = forward_with_cache(trig_x, bundle.public_weights, bundle.config)[:, -1, :].argmax(-1)
                        self.assertTrue(bool((preds_trig == gamma).all().item()))

                ep_dir = tmp_path / f"{cat}_{subtype}"
                save_challenge(bundle, ep_dir)

                # Baseline must fail
                if cat == "cat1_decompile":
                    base_sub = {"decompiled": {"a": 0, "b": 0, "c": 0, "trigger": [0, 0, 0, 0]}}
                else:
                    base_sub = {"weights": bundle.public_weights}
                base_res = verify_submission(
                    ep_dir,
                    base_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertFalse(base_res["passed"], f"Baseline unexpectedly passed {cat}/{subtype}: {base_res}")

                # Cat 5 shallow masking cheat must fail
                if cat == "cat5_unlearn_excise":
                    cheat_w = {k: v.clone() for k, v in bundle.public_weights.items()}
                    cheat_w["W_U"][:, bundle.secret["c_bad"]] = -1e4
                    cheat_res = verify_submission(
                        ep_dir,
                        {"weights": cheat_w},
                        {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                    )
                    self.assertFalse(
                        cheat_res["passed"],
                        f"Shallow masking cheat unexpectedly passed Level 1.5 Cat 5: {cheat_res}",
                    )

                # Oracle must pass with high reward
                if cat == "cat1_decompile":
                    oracle_sub = {
                        "decompiled": {
                            "a": bundle.secret["a"],
                            "b": bundle.secret["b"],
                            "c": bundle.secret["c"],
                            "trigger": bundle.secret["trigger"],
                        }
                    }
                else:
                    oracle_sub = {"weights": bundle.secret["oracle_weights"]}
                oracle_res = verify_submission(
                    ep_dir,
                    oracle_sub,
                    {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                )
                self.assertTrue(oracle_res["passed"], f"Oracle failed {cat}/{subtype}: {oracle_res}")
                self.assertGreater(oracle_res["total_reward"], 0.80, f"Low reward for {cat}/{subtype}: {oracle_res}")

                # Cat 7 canonical_pack must pass on Level 1.5 RMSNorm + SwiGLU + RoPE
                if cat == "cat7_refactor":
                    packed = wsp_tools.canonical_pack(bundle.public_weights)
                    pack_res = verify_submission(
                        ep_dir,
                        {"weights": packed},
                        {"forward_budget": bundle.forward_budget, "forward_passes_used": 0},
                    )
                    self.assertTrue(pack_res["passed"], f"canonical_pack failed Level 1.5 Cat 7: {pack_res}")

    def test_training_sft_exporter_and_grpo_curriculum(self) -> None:
        from wsp_gym.training import CurriculumScheduler, GRPORolloutEnv, audit_and_export_sft_dataset

        # 1. Export & audit real trajectories/episodes.jsonl if present, plus synthetic filter test
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            traj = tmp_path / "episodes.jsonl"
            sft_out = tmp_path / "sft.jsonl"
            records = [
                {
                    "episode_id": "ep_good",
                    "category": "cat7_refactor",
                    "subtype": "nullspace_and_cancelling_head_defrag",
                    "difficulty": 1.0,
                    "model_config": {"d_model": 32},
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "brief"},
                        {"role": "assistant", "content": "<think>\nAnalyze.\n</think>\n```python\nsubmit_weights(wsp_tools.canonical_pack(weights))\n```"},
                    ],
                    "verification": {"passed": True, "total_reward": 0.95, "metrics": {"turns_used": 1, "forward_batches_used": 0}},
                },
                {
                    "episode_id": "ep_fail",
                    "category": "cat2_repair",
                    "subtype": "mlp_sink_attractor",
                    "difficulty": 1.0,
                    "messages": [
                        {"role": "assistant", "content": "<think>\nx\n</think>\n```python\nsubmit_weights(weights)\n```"},
                    ],
                    "verification": {"passed": False, "total_reward": 0.0, "metrics": {"turns_used": 1, "forward_batches_used": 0}},
                },
            ]
            traj.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            summary = audit_and_export_sft_dataset(traj, sft_out, min_reward=0.80)
            self.assertEqual(summary["total_read"], 2)
            self.assertEqual(summary["exported"], 1)
            self.assertEqual(summary["zero_probe_count"], 1)
            self.assertTrue(sft_out.exists())

        # 2. CurriculumScheduler stage progression, stratified 17-subtype sampling, and B -> 0 annealing
        strat_sched = CurriculumScheduler(seed=11, initial_stage=0, window_size=8)
        strat_bundles = [strat_sched.sample_challenge() for _ in range(17)]
        self.assertEqual(len({b.subtype for b in strat_bundles}), 17)

        sched = CurriculumScheduler(seed=7, initial_stage=0, window_size=4)
        self.assertEqual(sched.stage, 0)
        for _ in range(4):
            advanced = sched.record_outcome(True, 0.95)
        self.assertTrue(advanced)
        self.assertEqual(sched.stage, 1)
        for _ in range(4):
            sched.record_outcome(True, 0.95)
        self.assertEqual(sched.stage, 2)
        b_stage2 = sched.sample_challenge(category="cat2_repair")
        self.assertLessEqual(b_stage2.forward_budget, 2)
        for _ in range(4):
            sched.record_outcome(True, 0.95)
        self.assertEqual(sched.stage, 3)
        b_stage3 = sched.sample_challenge(category="cat4_nullspace_inject")
        self.assertEqual(b_stage3.forward_budget, 0)

        # 3. GRPORolloutEnv multi-turn execution
        env = GRPORolloutEnv()
        try:
            cat7 = generate_challenge("cat7_refactor", seed=99, difficulty=1.5, subtype="level1_5_grokked_svd_defrag")
            res = env.run_episode(
                cat7,
                [("Pack with wsp_tools.", "submit_weights(wsp_tools.canonical_pack(weights))")],
            )
            self.assertTrue(res["submitted"])
            self.assertTrue(res["verification"]["passed"])
            self.assertGreater(res["verification"]["total_reward"], 0.80)
        finally:
            env.close()

    def test_blinded_solver_all_17_subtypes_and_scale_corpus(self) -> None:
        from wsp_gym.scaler import ALL_17_SUBTYPES, scale_sft_corpus, solve_blinded_challenge

        self.assertEqual(len(ALL_17_SUBTYPES), 17)
        all_subtypes = [sub for _, sub in ALL_17_SUBTYPES]
        cat2_a = generate_challenge("cat2_repair", seed=42, subtype="mlp_sink_attractor")
        cat2_b = generate_challenge("cat2_repair", seed=42, subtype="norm_matched_k_composition_corruption")
        self.assertEqual(cat2_a.brief_md, cat2_b.brief_md)
        self.assertNotIn("sink", cat2_a.brief_md.lower())
        self.assertNotIn("norm-matched", cat2_a.brief_md.lower())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            traj_file = tmp_path / "b0_all17.jsonl"
            for idx, (cat, sub) in enumerate(ALL_17_SUBTYPES):
                bundle = generate_challenge(
                    cat, seed=900 + idx, difficulty=1.5, forward_budget=0, subtype=sub
                )
                ep_dir = tmp_path / f"ep_{idx}_{sub}"
                save_challenge(bundle, ep_dir)
                pub_meta = json.loads((ep_dir / "public" / "metadata.json").read_text(encoding="utf-8"))
                self.assertNotIn("subtype", pub_meta)
                self.assertNotIn("difficulty", pub_meta)
                vault_sec = torch.load(ep_dir / ".vault" / "verifier_secret.pt", map_location="cpu", weights_only=False)
                self.assertEqual(vault_sec["subtype"], sub)

                res = solve_blinded_challenge(ep_dir, trajectory_path=traj_file)
                ver = res["verification"]
                self.assertTrue(ver["passed"], msg=f"Failed B=0 solve for {cat}/{sub}: {ver}")
                self.assertGreater(ver["total_reward"], 0.80)
                self.assertEqual(ver["metrics"]["forward_batches_used"], 0)

            for raw_line in traj_file.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                rec = json.loads(raw_line)
                asst_text = "\n".join(
                    m["content"] for m in rec["messages"] if m.get("role") == "assistant"
                )
                for s in all_subtypes:
                    self.assertNotIn(s, asst_text, msg=f"Subtype leak '{s}' in {rec['episode_id']}")

            scale_traj = tmp_path / "scaled_episodes.jsonl"
            scale_sft = tmp_path / "scaled_sft.jsonl"
            summary = scale_sft_corpus(
                num_episodes=17,
                base_seed=1900,
                trajectory_path=scale_traj,
                out_path=scale_sft,
            )
            self.assertEqual(summary["exported"], 17)
            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(len(summary["by_subtype"]), 17)
            self.assertTrue(scale_sft.exists())

    def test_sft_exporter_loss_mask_and_exploratory_solves(self) -> None:
        from wsp_gym.generators import generate_challenge
        from wsp_gym.sandbox import save_challenge
        from wsp_gym.scaler import ALL_17_SUBTYPES, solve_exploratory_challenge
        from wsp_gym.training import audit_and_export_sft_dataset

        all_subtypes = [s for _, s in ALL_17_SUBTYPES]
        cases = [
            ("cat1_decompile", "modular_linear_with_3tok_trojan", 1.0, None, "error_recovery"),
            ("cat5_unlearn_excise", "level1_5_grokked_backdoor_excision", 1.5, None, "error_recovery"),
            ("cat7_refactor", "nullspace_and_cancelling_head_defrag", 1.0, 0, "error_recovery"),
            ("cat2_repair", "norm_matched_k_composition_corruption", 1.5, None, "hypothesis_refine"),
            ("cat3_synthesize", "level1_5_swiglu_rope_synthesize", 1.5, 0, "hypothesis_refine"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            traj_path = tmp_path / "exploratory_episodes.jsonl"
            sft_path = tmp_path / "exploratory_sft.jsonl"

            for idx, (cat, sub, diff, budget, mode) in enumerate(cases):
                bundle = generate_challenge(cat, seed=2900 + idx, difficulty=diff, forward_budget=budget, subtype=sub)
                ep_dir = tmp_path / f"ep_{idx}_{bundle.episode_id}"
                save_challenge(bundle, ep_dir)
                res = solve_exploratory_challenge(ep_dir, trajectory_path=traj_path, mode=mode)
                ver = res.get("verification") or {}
                self.assertTrue(ver.get("passed"), msg=f"{cat}/{sub}/{mode} failed: {ver}")
                self.assertGreaterEqual(float(ver.get("total_reward", 0.0)), 0.80)
                self.assertEqual(int((ver.get("metrics") or {}).get("turns_used", 0)), 3)

            raw_lines = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(raw_lines), len(cases))
            for rec, (_, _, _, _, mode) in zip(raw_lines, cases):
                user_text = "\n".join(m["content"] for m in rec["messages"] if m.get("role") == "user")
                asst_text = "\n".join(m["content"] for m in rec["messages"] if m.get("role") == "assistant")
                if mode == "error_recovery":
                    self.assertIn("STDERR:", user_text)
                else:
                    self.assertNotIn("STDERR:", user_text)
                for s in all_subtypes:
                    self.assertNotIn(s, asst_text, msg=f"Subtype leak '{s}' in {rec['episode_id']}")

            # Append a synthetic episode with a trailing user message to verify strip_trailing_user
            trailing_rec = dict(raw_lines[0])
            trailing_rec["episode_id"] = "ep_trailing_user_test"
            trailing_rec["messages"] = list(raw_lines[0]["messages"]) + [
                {"role": "user", "content": "[Sandbox Info: post-submit observation]"}
            ]
            with traj_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trailing_rec) + "\n")

            summary = audit_and_export_sft_dataset(traj_path, sft_path, min_reward=0.80)
            self.assertEqual(summary["exported"], len(cases) + 1)
            self.assertEqual(summary["multi_turn_3plus_count"], len(cases) + 1)
            self.assertEqual(summary["error_recovery_count"], 4)

            exported = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for item in exported:
                msgs = item["messages"]
                self.assertEqual(msgs[-1]["role"], "assistant")
                expected_masks = [m["role"] == "assistant" for m in msgs]
                self.assertEqual(item["loss_masks"], expected_masks)
                self.assertEqual([m["loss_mask"] for m in msgs], expected_masks)

    def test_sandbox_timeout_sysexit_and_raw_llm_parser(self) -> None:
        import time
        from wsp_gym.training import (
            GRPORolloutEnv,
            compute_shaped_reward,
            parse_assistant_turn,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = generate_challenge("cat7_refactor", seed=123, difficulty=1.0)
            ep_dir = tmp_path / "ep_guard"
            save_challenge(bundle, ep_dir)
            sess = SandboxSession(ep_dir, max_turns=4, trajectory_path=tmp_path / "guard.jsonl")

            t0 = time.perf_counter()
            res_to = sess.step(think="Loop", code="while True:\n    pass", timeout_sec=0.05)
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 2.0)
            self.assertTrue(res_to["timed_out"])
            self.assertTrue(res_to["exec_error"])
            self.assertIn("TimeoutError", res_to["observation"])

            res_exit = sess.step(think="Exit", code="import sys; sys.exit(1)")
            self.assertFalse(res_exit["timed_out"])
            self.assertTrue(res_exit["exec_error"])
            self.assertIn("SystemExit", res_exit["observation"])

        raw_valid = (
            "<think>\nCheck SVD rank.\n</think>\n"
            "```python\nprint('ok')\n```\n"
            "STDOUT:\nfake hallucinated output\n[Sandbox Info: 0/0]"
        )
        p_valid = parse_assistant_turn(raw_valid)
        self.assertTrue(p_valid["valid_format"])
        self.assertTrue(p_valid["has_think"])
        self.assertTrue(p_valid["has_code"])
        self.assertEqual(p_valid["think"], "Check SVD rank.")
        self.assertEqual(p_valid["code"], "print('ok')")

        p_bad = parse_assistant_turn("<think>\nNo code here\n</think>")
        self.assertFalse(p_bad["valid_format"])
        self.assertFalse(p_bad["has_code"])

        # Micro-shaping strict ordering on identical 1-turn challenge
        env = GRPORolloutEnv(max_turns=1)
        try:
            b = generate_challenge("cat7_refactor", seed=123, difficulty=1.0)
            replies = {
                "malformed_no_code": "<think>\nMissing code\n</think>",
                "runtime_error_no_submit": "<think>\nBad op\n</think>\n```python\n1 / 0\n```",
                "clean_no_submit": "<think>\nInspect\n</think>\n```python\nprint(1)\n```",
                "clean_explicit_submit": "<think>\nSubmit\n</think>\n```python\nsubmit_weights(weights)\n```",
            }
            scores: dict[str, float] = {}
            for name, reply in replies.items():
                env.reset(bundle=b)
                r = env.step_raw(reply)
                self.assertTrue(r["submitted"])
                scores[name] = float(r["shaped_reward"])

            self.assertLess(scores["malformed_no_code"], scores["runtime_error_no_submit"])
            self.assertLess(scores["runtime_error_no_submit"], scores["clean_no_submit"])
            self.assertLess(scores["clean_no_submit"], scores["clean_explicit_submit"])

            direct = compute_shaped_reward(
                {"total_reward": 0.5, "metrics": {"explicit_submit": True}},
                [{"valid_format": True, "has_code": True, "exec_error": False, "timed_out": False, "explicit_submit": True}],
            )
            self.assertAlmostEqual(direct["shaping_bonus"], 0.10, places=6)

            # Verify run_raw_episode with 2-turn raw policy callback
            env_2t = GRPORolloutEnv(max_turns=3)
            try:
                def _raw_policy(msgs: list[dict[str, str]]) -> str:
                    if len(msgs) <= 2:
                        return "<think>\nAnalyze SVD.\n</think>\n```python\nsvd = wsp_tools.residual_svd_basis(weights)\nprint(svd['rank'])\n```"
                    return "<think>\nCanonical pack and submit.\n</think>\n```python\nsubmit_weights(wsp_tools.canonical_pack(weights))\n```"

                ep_res = env_2t.run_raw_episode(b, _raw_policy)
                self.assertTrue(ep_res["submitted"])
                self.assertTrue(ep_res["verification"]["passed"])
                self.assertGreater(ep_res["shaped_reward"], 0.90)
            finally:
                env_2t.close()
        finally:
            env.close()

    def test_step_l4_eval_sft_and_grpo_dry_runs(self) -> None:
        import copy
        import torch
        from eval import run_evaluation
        from train_grpo import (
            _snapshot_old_logprobs,
            _snapshot_trainable_params,
            _swap_trainable_params,
            compute_grpo_group_advantages,
            compute_grpo_loss_and_metrics,
            run_grpo_dry_run,
        )
        from train_sft import (
            ChatWordTokenizer,
            TinyCausalLM,
            _grad_l2_norm,
            run_sft_dry_run,
            tokenize_and_mask_chat,
        )

        self.assertTrue(Path("requirements.txt").exists())

        dummy_report = run_evaluation(backend="dummy", episodes_per_subtype=1, base_seed=9000)
        self.assertEqual(dummy_report["total_episodes"], 17)
        self.assertEqual(len(dummy_report["episodes"]), 17)
        self.assertLess(dummy_report["pass_rate"], 0.25)

        scripted_report = run_evaluation(backend="scripted", episodes_per_subtype=1, base_seed=9000)
        self.assertEqual(scripted_report["total_episodes"], 17)
        self.assertEqual(len(scripted_report["episodes"]), 17)
        self.assertEqual(scripted_report["pass_rate"], 1.0)
        self.assertGreaterEqual(scripted_report["mean_base_reward"], 0.85)
        self.assertEqual(scripted_report["mean_format_rate"], 1.0)

        sft_summary = run_sft_dry_run("trajectories/sft_train.jsonl")
        self.assertEqual(sft_summary["num_records"], 123)
        self.assertGreater(sft_summary["assistant_token_fraction"], 0.15)
        self.assertLess(sft_summary["assistant_token_fraction"], 0.90)
        self.assertTrue(sft_summary["gradient_checkpointing"])
        self.assertEqual(sft_summary["effective_batch_size"], 32)
        self.assertGreater(sft_summary["grad_norm"], 0.0)

        sft_rows = [
            json.loads(line)
            for line in Path("trajectories/sft_train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(sft_rows), 123)
        for row in sft_rows:
            self.assertEqual(
                row["forward_batches_used"],
                0,
                msg=f"Expected 0 forward batches in {row['episode_id']} ({row['subtype']})",
            )
            for msg in row["messages"]:
                if msg.get("role") == "assistant":
                    self.assertNotIn(
                        "probe_forward(",
                        msg["content"],
                        msg=f"probe_forward( found in {row['episode_id']} ({row['subtype']})",
                    )
                    self.assertLessEqual(
                        len(msg["content"]),
                        2000,
                        msg=f"Assistant message too long in {row['episode_id']} ({row['subtype']})",
                    )
                    self.assertNotIn(
                        "brief_text =",
                        msg["content"],
                        msg=f"Embedded brief_text literal found in {row['episode_id']}",
                    )

        tied_adv = compute_grpo_group_advantages(
            [-0.98, -0.95, -1.02, -1.00],
            base_rewards=[-1.0, -1.0, -1.0, -1.0],
        )
        self.assertFalse(tied_adv["effective_group"])
        self.assertEqual(tied_adv["advantages"], [0.0, 0.0, 0.0, 0.0])

        shaping_tie_adv = compute_grpo_group_advantages(
            [1.10, 1.10, 1.09, 1.02],
            base_rewards=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertTrue(shaping_tie_adv["effective_group"])

        shaping_exact_tie_adv = compute_grpo_group_advantages(
            [1.10, 1.10, 1.10, 1.10],
            base_rewards=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertFalse(shaping_exact_tie_adv["effective_group"])
        self.assertEqual(shaping_exact_tie_adv["advantages"], [0.0, 0.0, 0.0, 0.0])

        noop_adv = compute_grpo_group_advantages(
            [-0.95, -0.95, 0.085, 0.225],
            base_rewards=[-1.0, -1.0, 0.035, 0.175],
        )
        self.assertFalse(noop_adv["effective_group"])
        self.assertEqual(noop_adv["advantages"], [0.0, 0.0, 0.0, 0.0])

        partial_adv = compute_grpo_group_advantages(
            [-0.95, 0.45, 0.05, -0.95],
            base_rewards=[-1.0, 0.35, 0.0, -1.0],
        )
        self.assertTrue(partial_adv["effective_group"])
        self.assertGreater(max(partial_adv["advantages"]), 1.0)

        grpo_summary = run_grpo_dry_run(group_size=4, num_steps=2)
        self.assertEqual(grpo_summary["steps_completed"], 2)
        self.assertEqual(grpo_summary["total_rollouts"], 8)
        self.assertTrue(grpo_summary["accumulate_grads"])
        self.assertGreater(grpo_summary["mean_advantage_std"], 0.0)
        self.assertGreater(grpo_summary["grad_norm"], 0.0)

        # Verify accumulate_grads=True matches accumulate_grads=False in both loss and gradients
        torch.manual_seed(7)
        tok = ChatWordTokenizer(vocab_size=256)
        m1 = TinyCausalLM(vocab_size=tok.vocab_size, d_model=32, n_heads=2, max_seq_len=256)
        m2 = copy.deepcopy(m1)
        ref = copy.deepcopy(m1).eval()
        t_pos = tokenize_and_mask_chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": "<think>ok</think>\n```python\nprint(1)\n```"},
            ],
            tok,
            max_length=256,
        )
        t_neg = tokenize_and_mask_chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": "<think>no</think>\n```python\nprint(2)\n```"},
            ],
            tok,
            max_length=256,
        )
        batch = [
            {**t_pos, "advantage": 1.0, "old_token_logprobs": _snapshot_old_logprobs(m1, t_pos)},
            {**t_neg, "advantage": -1.0, "old_token_logprobs": _snapshot_old_logprobs(m1, t_neg)},
        ]
        loss_joint, metrics_joint = compute_grpo_loss_and_metrics(m1, ref, batch, accumulate_grads=False)
        loss_joint.backward()
        gn_joint = _grad_l2_norm(m1)

        loss_acc, metrics_acc = compute_grpo_loss_and_metrics(m2, ref, batch, accumulate_grads=True)
        gn_acc = _grad_l2_norm(m2)

        self.assertAlmostEqual(metrics_joint["loss"], metrics_acc["loss"], places=5)
        self.assertAlmostEqual(float(loss_joint.item()), float(loss_acc.item()), places=5)
        self.assertAlmostEqual(gn_joint, gn_acc, places=5)

        # Verify ref_params KL anchor snapshot/swap: kl_div == 0 when matching, > 0 after step, params restored
        m3 = copy.deepcopy(ref)
        ref_snap = _snapshot_trainable_params(m3)
        self.assertGreater(len(ref_snap), 0)
        _, metrics_eq = compute_grpo_loss_and_metrics(m3, None, batch, ref_params=ref_snap)
        self.assertAlmostEqual(metrics_eq["kl_div"], 0.0, places=6)

        opt3 = torch.optim.SGD(m3.parameters(), lr=0.1)
        opt3.zero_grad(set_to_none=True)
        loss3, _ = compute_grpo_loss_and_metrics(m3, None, batch, ref_params=ref_snap)
        loss3.backward()
        opt3.step()

        post_step_snap = _snapshot_trainable_params(m3)
        _, metrics_shifted = compute_grpo_loss_and_metrics(m3, None, batch, ref_params=ref_snap)
        self.assertGreater(metrics_shifted["kl_div"], 0.0)
        for name, orig_val in post_step_snap.items():
            self.assertTrue(torch.equal(dict(m3.named_parameters())[name].detach().cpu(), orig_val))

        with _swap_trainable_params(m3, ref_snap):
            for name, ref_val in ref_snap.items():
                self.assertTrue(torch.equal(dict(m3.named_parameters())[name].detach().cpu(), ref_val))
        for name, orig_val in post_step_snap.items():
            self.assertTrue(torch.equal(dict(m3.named_parameters())[name].detach().cpu(), orig_val))


if __name__ == "__main__":
    unittest.main()
