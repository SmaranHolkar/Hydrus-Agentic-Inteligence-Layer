import time
import pytest
from hydrus_agent.memory_subagents import DecayAgent

def test_decay_agent_eviction():
    decay = DecayAgent(max_items=3)
    memory_store = {
        "item1": {"confidence": 0.9, "updated_at": time.time()},
        "item2": {"confidence": 0.3, "updated_at": time.time() - 100},
        "item3": {"confidence": 0.8, "updated_at": time.time()},
        "item4": {"confidence": 0.2, "updated_at": time.time() - 200},
        "contradiction:item5": {"contradiction_flag": True, "updated_at": time.time() - 120}
    }

    evicted = decay.evict_and_demote(memory_store, vram_utilization_pct=85.0)

    assert evicted > 0
    assert "contradiction:item5" not in memory_store
    assert len(memory_store) <= 3
