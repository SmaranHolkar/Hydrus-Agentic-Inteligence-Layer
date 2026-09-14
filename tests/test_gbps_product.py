import unittest
import sys
import os
import numpy as np

# Ensure packages directory is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
gbps_src = os.path.join(repo_root, "packages", "hydrus-gbps", "src")
if gbps_src not in sys.path:
    sys.path.insert(0, gbps_src)

from hydrus_gbps import (
    GBPSVerifier,
    VerificationResult,
    FastHashEmbedder,
    GroundedBeliefPathSearch,
    BeliefNode,
    TTLinear,
    NumpyTTLinear,
    flop_estimate,
    split_claims,
    GBPSGuardrail,
)


class TestHydrusGBPS(unittest.TestCase):
    def setUp(self):
        self.verifier = GBPSVerifier(grounding_threshold=0.60)

    def test_split_claims(self):
        text = "HydrusMoE is an edge MoE engine. It supports 4-tier storage. Does it support ARM CPUs? Yes it does."
        claims = split_claims(text)
        self.assertGreaterEqual(len(claims), 3)

    def test_hash_embedder(self):
        embedder = FastHashEmbedder(dim=768)
        vec1 = embedder.embed("Artificial intelligence and neural networks.")
        vec2 = embedder.embed("Artificial intelligence and neural networks.")
        vec3 = embedder.embed("Cooking recipes for pasta and tomatoes.")

        self.assertEqual(vec1.shape, (768,))
        # Deterministic
        np.testing.assert_allclose(vec1, vec2, atol=1e-5)
        # Cosine similarity with identical is 1.0
        sim_same = float(np.dot(vec1, vec2))
        sim_diff = float(np.dot(vec1, vec3))
        self.assertAlmostEqual(sim_same, 1.0, places=4)
        self.assertLess(sim_diff, sim_same)

    def test_numpy_tt_linear(self):
        tt = NumpyTTLinear(8, 8, 12, rank=8)
        x = np.random.randn(768).astype(np.float32)
        out = tt.project(x)
        self.assertEqual(out.shape, (768,))

        stats = flop_estimate(8, 8, 12, rank=8)
        self.assertGreater(stats["flop_reduction_x"], 1.0)
        self.assertGreater(stats["param_reduction_x"], 1.0)

    def test_grounded_verification(self):
        context = "HydrusMoE supports 4-tier streaming memory: GPU VRAM, Host RAM, SSD Vault, and Cloud CDN."
        grounded_resp = "HydrusMoE provides a 4-tier memory architecture including GPU VRAM and SSD Vault."
        
        result = self.verifier.verify(
            query="What memory tiers does HydrusMoE support?",
            context=context,
            response=grounded_resp,
        )

        self.assertIsInstance(result, VerificationResult)
        self.assertTrue(result.is_grounded)
        self.assertGreaterEqual(result.grounding_score, 0.60)
        self.assertEqual(len(result.flagged_claims), 0)
        self.assertGreater(len(result.belief_chain), 0)
        self.assertGreater(result.latency_ms, 0.0)

    def test_hallucination_detection(self):
        context = "The mission launched on July 20, 1969 with astronauts Armstrong, Aldrin, and Collins."
        hallucinated_resp = "The spacecraft was captained by Napoleon Bonaparte on Mars in the year 3000."

        result = self.verifier.verify(
            query="Who commanded the mission?",
            context=context,
            response=hallucinated_resp,
            threshold=0.50
        )

        self.assertFalse(result.is_grounded)
        self.assertGreater(len(result.flagged_claims), 0)

    def test_langchain_guardrail(self):
        guardrail = GBPSGuardrail(threshold=0.60, verifier=self.verifier)
        payload = {
            "query": "Is Python interpreted?",
            "context": "Python is an interpreted, high-level, general-purpose programming language.",
            "response": "Yes, Python is an interpreted programming language."
        }
        res = guardrail(payload)
        self.assertIn("verification", res)
        self.assertIn("is_grounded", res)
        self.assertTrue(res["is_grounded"])


if __name__ == "__main__":
    unittest.main()
