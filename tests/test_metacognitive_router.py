import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"
for p in (ROOT, HAIL_ROOT, HAIL_SRC):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

from hydrus_agent.metacognitive_router import MetacognitiveRouter, CognitiveState

class TestMetacognitiveRouter(unittest.TestCase):
    def test_metacognitive_router_default(self):
        router = MetacognitiveRouter()
        state = CognitiveState()
        decision = router.evaluate(state)

        self.assertFalse(decision.fast_path)
        self.assertFalse(decision.skip_grounding)
        self.assertEqual(decision.retrieval_depth, "semantic_shallow")

    def test_metacognitive_router_vram_constrained(self):
        router = MetacognitiveRouter(vram_high_threshold=80.0)
        state = CognitiveState(vram_utilization_pct=88.0)
        decision = router.evaluate(state)

        self.assertTrue(decision.fast_path)
        self.assertTrue(decision.skip_grounding)
        self.assertEqual(decision.retrieval_depth, "procedural_cache")
        self.assertIn("VRAM utilization high", decision.reason)

    def test_metacognitive_router_latency_tight(self):
        router = MetacognitiveRouter(tight_latency_threshold_ms=500.0)
        state = CognitiveState(latency_budget_ms=300.0)
        decision = router.evaluate(state)

        self.assertTrue(decision.fast_path)
        self.assertTrue(decision.skip_grounding)
        self.assertEqual(decision.retrieval_depth, "procedural_cache")
        self.assertIn("Strict latency budget", decision.reason)

    def test_metacognitive_router_deep_grounding_trigger(self):
        router = MetacognitiveRouter()
        state = CognitiveState(consecutive_failures=2, perplexity_score=0.8, user_tone="frustrated")
        decision = router.evaluate(state)

        self.assertFalse(decision.fast_path)
        self.assertFalse(decision.skip_grounding)
        self.assertEqual(decision.retrieval_depth, "full_grounding")
        self.assertIn("Consecutive failures", decision.reason)


if __name__ == "__main__":
    unittest.main()
