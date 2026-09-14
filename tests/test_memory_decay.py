import time
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

from hydrus_agent.memory_subagents import DecayAgent

class TestMemoryDecay(unittest.TestCase):
    def test_decay_agent_eviction(self):
        decay = DecayAgent(max_items=3)
        memory_store = {
            "item1": {"confidence": 0.9, "updated_at": time.time()},
            "item2": {"confidence": 0.3, "updated_at": time.time() - 100},
            "item3": {"confidence": 0.8, "updated_at": time.time()},
            "item4": {"confidence": 0.2, "updated_at": time.time() - 200},
            "contradiction:item5": {"contradiction_flag": True, "updated_at": time.time() - 120}
        }

        evicted = decay.evict_and_demote(memory_store, vram_utilization_pct=85.0)

        self.assertGreater(evicted, 0)
        self.assertNotIn("contradiction:item5", memory_store)
        self.assertLessEqual(len(memory_store), 3)


if __name__ == "__main__":
    unittest.main()
