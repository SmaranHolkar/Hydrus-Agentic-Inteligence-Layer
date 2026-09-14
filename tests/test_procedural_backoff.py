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

from hail_core.procedural_memory import ProceduralMemoryManager

class TestProceduralBackoff(unittest.TestCase):
    def test_procedural_execution_backoff(self):
        manager = ProceduralMemoryManager(failure_threshold=3, backoff_duration_turns=5)

        # 2 failures - not blacklisted yet
        manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")
        manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")

        is_blocked, msg = manager.is_blacklisted("flaky_api")
        self.assertFalse(is_blocked)

        # 3rd failure - triggers backoff
        manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")

        is_blocked, msg = manager.is_blacklisted("flaky_api")
        self.assertTrue(is_blocked)
        self.assertTrue(isinstance(msg, str) and len(msg.strip()) > 0)

        # Advance turn past backoff duration -> unblocked
        for _ in range(6):
            manager.advance_turn()

        is_blocked_after, msg_after = manager.is_blacklisted("flaky_api")
        self.assertFalse(is_blocked_after)


if __name__ == "__main__":
    unittest.main()
