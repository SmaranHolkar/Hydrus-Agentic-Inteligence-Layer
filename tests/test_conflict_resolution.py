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

from hydrus_agent.memory_subagents import (
    CandidateFact, GroundingAgent, ConsolidationAgent
)

class TestConflictResolution(unittest.IsolatedAsyncioTestCase):
    async def test_conflict_resolution_and_confidence_scoring(self):
        grounding = GroundingAgent()
        consolidator = ConsolidationAgent()

        existing_knowledge = [
            {"memory_id": "mem_001", "subject": "server_port", "predicate": "has_value", "object_val": "8080", "confidence": 0.9}
        ]

        # Contradictory candidate fact from external web crawl
        candidate = CandidateFact(
            subject="server_port",
            predicate="has_value",
            object_val="9090",
            source="web_search",
            confidence=0.8
        )

        grounded_fact = grounding.verify_candidate(candidate, existing_knowledge)

        self.assertTrue(grounded_fact.contradiction_flag)
        self.assertFalse(grounded_fact.is_grounded)
        self.assertEqual(grounded_fact.action, "flag_contradiction")
        self.assertEqual(grounded_fact.epistemic_confidence, 0.40)

        memory_store = {"server_port:has_value": existing_knowledge[0]}
        await consolidator.consolidate([grounded_fact], memory_store)

        # Original fact should NOT be overwritten
        self.assertEqual(memory_store["server_port:has_value"]["object_val"], "8080")

        # Contradiction should be logged in isolated audit key
        contradiction_keys = [k for k in memory_store.keys() if k.startswith("contradiction:server_port")]
        self.assertEqual(len(contradiction_keys), 1)
        self.assertEqual(memory_store[contradiction_keys[0]]["candidate_val"], "9090")


if __name__ == "__main__":
    unittest.main()
