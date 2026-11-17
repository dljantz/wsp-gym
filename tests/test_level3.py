"""Unit and integration tests for Level 3 Immutable-Twin Self-Surgery (7.62B harness & mini twin)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from eval_level3 import run_level3_evaluation
from wsp_gym.level3 import (
    LEVEL3_SUBTYPES,
    compute_twin_base_digest,
    evaluate_immutable_twin_promotion,
    generate_level3_challenge,
)
from wsp_gym.model import LazyTwinWeightDict
from wsp_gym.sandbox import SandboxSession
from wsp_gym.verifier import verify_bundle


class TestLevel3ImmutableTwin(unittest.TestCase):
    def test_lazy_twin_cow_lru_eviction_preserves_mutations(self) -> None:
        base = {f"t{i}": torch.full((4, 4), float(i + 1), dtype=torch.bfloat16) for i in range(6)}
        lazy = LazyTwinWeightDict(base, max_cache=3)
        digest_before = compute_twin_base_digest(lazy)

        # Mutate t0 in-place, then access t1..t4 to force LRU eviction of t0 into _dirty
        lazy["t0"][1, 2] = 99.0
        for i in range(1, 5):
            _ = float(lazy[f"t{i}"].sum().item())

        self.assertIn("t0", lazy._dirty)
        self.assertEqual(float(lazy["t0"][1, 2].item()), 99.0)
        # Base tensor must remain 100% untouched
        self.assertEqual(float(base["t0"][1, 2].item()), 1.0)
        self.assertEqual(digest_before, compute_twin_base_digest(lazy))

        # Forks share read-only _base without leaking future edits across forks
        f1 = lazy.fork()
        f1["t2"][0, 0] = -7.0
        f1.flush()
        self.assertEqual(float(lazy["t2"][0, 0].item()), 3.0)
        self.assertEqual(float(f1["t2"][0, 0].item()), -7.0)

    def test_level3_mini_baseline_fails_and_scripted_promotes_all(self) -> None:
        for _, subtype in LEVEL3_SUBTYPES:
            bundle = generate_level3_challenge(subtype=subtype, seed=29000, scale="mini")
            ver_noop = verify_bundle(bundle, {"weights": bundle.public_weights})
            self.assertFalse(ver_noop["passed"], f"Baseline unexpectedly passed on {subtype}")
            promo_noop = evaluate_immutable_twin_promotion(bundle, ver_noop, bundle.public_weights)
            self.assertFalse(promo_noop["promoted"])

        report = run_level3_evaluation(
            scale="mini",
            backend="scripted",
            episodes_per_subtype=2,
            base_seed=29000,
            max_turns=4,
            timeout_sec=30.0,
        )
        self.assertEqual(report["total_episodes"], 6)
        self.assertEqual(report["passed_count"], 6)
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertEqual(report["zero_probe_pass_rate"], 1.0)
        self.assertEqual(report["promotion_rate"], 1.0)
        self.assertEqual(report["twin_base_immutability_rate"], 1.0)
        for ep in report["episodes"]:
            self.assertLess(ep["reference_twin_kl"], 0.005)
            self.assertLess(ep["raw_pretrained_kl"], 0.20)
            self.assertEqual(ep["turns"], 2)

    def test_level3_mini_rejects_cheaters_and_gqa_sibling_destruction(self) -> None:
        bundle = generate_level3_challenge(
            subtype="self_twin_qwen25_coder_7b_backdoor_excise",
            seed=29002,
            scale="mini",
        )
        c_bad = int(bundle.secret["c_bad"])
        digest_before = compute_twin_base_digest(bundle.public_weights)

        # 1. Shallow W_U unembedding masking must fail anti-masking + promotion
        sess_mask = SandboxSession.from_bundle(bundle, max_turns=4, trajectory_path=None)
        res_mask = sess_mask.step(
            think="Mask unembedding column only.",
            code=f"weights['W_U'][:, {c_bad}] = 0.0\nsubmit_weights(weights)",
        )
        ver_mask = res_mask["verification"]
        self.assertFalse(ver_mask["passed"])
        promo_mask = evaluate_immutable_twin_promotion(
            bundle,
            ver_mask,
            sess_mask.state["weights"],
            twin_base_digest_before=digest_before,
            twin_base_digest_after=compute_twin_base_digest(bundle.public_weights),
        )
        self.assertFalse(promo_mask["promoted"])
        self.assertFalse(promo_mask["twin_base_mutated"])

        # 2. Collateral GQA KV wipe (destroying all live sibling heads) must fail clean task & KL gate
        sess_wipe = SandboxSession.from_bundle(bundle, max_turns=4, trajectory_path=None)
        res_wipe = sess_wipe.step(
            think="Wipe entire KV groups indiscriminately.",
            code=(
                "weights['blocks.0.attn.W_K'][0] = 0.0\n"
                "weights['blocks.0.attn.W_V'][0] = 0.0\n"
                "weights['blocks.0.attn.W_K'][1] = 0.0\n"
                "weights['blocks.0.attn.W_V'][1] = 0.0\n"
                "submit_weights(weights)"
            ),
        )
        ver_wipe = res_wipe["verification"]
        promo_wipe = evaluate_immutable_twin_promotion(
            bundle,
            ver_wipe,
            sess_wipe.state["weights"],
            twin_base_digest_before=digest_before,
            twin_base_digest_after=compute_twin_base_digest(bundle.public_weights),
        )
        self.assertFalse(promo_wipe["promoted"])
        self.assertGreater(promo_wipe["reference_twin_kl"], 0.005)

    def test_sandbox_compacts_tensor_slice_views_across_turns(self) -> None:
        bundle = generate_level3_challenge(
            subtype="self_twin_qwen25_coder_7b_sink_repair",
            seed=29000,
            scale="mini",
        )
        sess = SandboxSession.from_bundle(bundle, max_turns=4, trajectory_path=None)
        res1 = sess.step(
            think="Append a 1D row slice view from W_out into cands across turns.",
            code="cands = [weights['blocks.0.mlp.W_out'][0]]\nprint(float(cands[0].norm().item()))",
        )
        self.assertFalse(res1["exec_error"])
        v = sess.state["user_vars"]["cands"][0]
        self.assertLessEqual(int(v.untyped_storage().nbytes()), 2048)
        self.assertTrue(torch.allclose(v, bundle.public_weights["blocks.0.mlp.W_out"][0]))

    def test_lazy_twin_views_and_eval_resume(self) -> None:
        base = {f"t{i}": torch.full((4, 4), float(i + 1), dtype=torch.bfloat16) for i in range(10)}
        lazy = LazyTwinWeightDict(base, max_cache=3)
        seen_keys: list[str] = []
        for k, v in lazy.items():
            seen_keys.append(k)
            _ = v.shape
            self.assertLessEqual(len(lazy._cache), lazy._max_cache)
        self.assertEqual(len(seen_keys), 10)
        for v in lazy.values():
            _ = v.shape
            self.assertLessEqual(len(lazy._cache), lazy._max_cache)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "resumed_l3.json"
            full_ref = run_level3_evaluation(
                scale="mini",
                backend="scripted",
                episodes_per_subtype=1,
                base_seed=29000,
                max_turns=4,
                timeout_sec=30.0,
            )
            part_path = Path(f"{out_path}.partial.json")
            part_path.write_text(
                json.dumps({"episodes": full_ref["episodes"][:1]}, indent=2),
                encoding="utf-8",
            )
            resumed = run_level3_evaluation(
                scale="mini",
                backend="scripted",
                episodes_per_subtype=1,
                base_seed=29000,
                max_turns=4,
                timeout_sec=30.0,
                out_path=out_path,
                resume=True,
            )
            self.assertEqual(resumed["total_episodes"], 3)
            self.assertEqual(resumed["passed_count"], 3)
            self.assertEqual(resumed["promoted_count"], 3)
            self.assertTrue(out_path.exists())
            self.assertFalse(part_path.exists())

            # Verify _read_cgroup_anon_bytes ignores page cache and inner try/finally catches both errors cleanly
            stat_file = Path(tmpdir) / "memory.stat"
            stat_file.write_text("file 20000000000\nanon 5000000000\n", encoding="utf-8")
            bundle_err = generate_level3_challenge(
                subtype="self_twin_qwen25_coder_7b_sink_repair",
                seed=29000,
                scale="mini",
            )
            sess_err = SandboxSession.from_bundle(bundle_err, max_turns=3, trajectory_path=None)
            with patch("wsp_gym.sandbox._cgroup_mem_ceiling_bytes", return_value=(stat_file, 10_000_000_000)):
                res_err = sess_err.step(
                    think="Trigger RuntimeError while anon < limit even though file+anon > limit.",
                    code="raise RuntimeError('intentional shape mismatch')",
                )
            self.assertTrue(res_err["exec_error"])
            self.assertIn("intentional shape mismatch", res_err["observation"])
            with patch("wsp_gym.sandbox._cgroup_mem_ceiling_bytes", return_value=(stat_file, 1024)):
                res_oom = sess_err.step(
                    think="Trigger MemoryError cleanly when anon > limit.",
                    code="x = 1 + 1",
                )
            self.assertTrue(res_oom["exec_error"])
            self.assertIn("MemoryError", res_oom["observation"])


if __name__ == "__main__":
    unittest.main()
