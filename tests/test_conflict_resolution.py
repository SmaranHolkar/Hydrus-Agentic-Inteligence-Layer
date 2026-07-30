import pytest
import asyncio
from hydrus_agent.memory_subagents import (
    CandidateFact, GroundingAgent, ConsolidationAgent
)

@pytest.mark.asyncio
async def test_conflict_resolution_and_confidence_scoring():
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
    
    assert grounded_fact.contradiction_flag is True
    assert grounded_fact.is_grounded is False
    assert grounded_fact.action == "flag_contradiction"
    assert grounded_fact.epistemic_confidence == 0.40
    
    memory_store = {"server_port:has_value": existing_knowledge[0]}
    await consolidator.consolidate([grounded_fact], memory_store)
    
    # Original fact should NOT be overwritten
    assert memory_store["server_port:has_value"]["object_val"] == "8080"
    
    # Contradiction should be logged in isolated audit key
    contradiction_keys = [k for k in memory_store.keys() if k.startswith("contradiction:server_port")]
    assert len(contradiction_keys) == 1
    assert memory_store[contradiction_keys[0]]["candidate_val"] == "9090"
