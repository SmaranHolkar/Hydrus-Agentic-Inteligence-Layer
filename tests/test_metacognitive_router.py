import pytest
from hydrus_agent.metacognitive_router import MetacognitiveRouter, CognitiveState

def test_metacognitive_router_default():
    router = MetacognitiveRouter()
    state = CognitiveState()
    decision = router.evaluate(state)
    
    assert decision.fast_path is False
    assert decision.skip_grounding is False
    assert decision.retrieval_depth == "semantic_shallow"

def test_metacognitive_router_vram_constrained():
    router = MetacognitiveRouter(vram_high_threshold=80.0)
    state = CognitiveState(vram_utilization_pct=88.0)
    decision = router.evaluate(state)
    
    assert decision.fast_path is True
    assert decision.skip_grounding is True
    assert decision.retrieval_depth == "procedural_cache"
    assert "VRAM utilization high" in decision.reason

def test_metacognitive_router_latency_tight():
    router = MetacognitiveRouter(tight_latency_threshold_ms=500.0)
    state = CognitiveState(latency_budget_ms=300.0)
    decision = router.evaluate(state)
    
    assert decision.fast_path is True
    assert decision.skip_grounding is True
    assert decision.retrieval_depth == "procedural_cache"
    assert "Strict latency budget" in decision.reason

def test_metacognitive_router_deep_grounding_trigger():
    router = MetacognitiveRouter()
    state = CognitiveState(consecutive_failures=2, perplexity_score=0.8, user_tone="frustrated")
    decision = router.evaluate(state)
    
    assert decision.fast_path is False
    assert decision.skip_grounding is False
    assert decision.retrieval_depth == "full_grounding"
    assert "Consecutive failures" in decision.reason
