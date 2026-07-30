import pytest
from hail_core.procedural_memory import ProceduralMemoryManager

def test_procedural_execution_backoff():
    manager = ProceduralMemoryManager(failure_threshold=3, backoff_duration_turns=5)
    
    # 2 failures - not blacklisted yet
    manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")
    manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")
    
    is_blocked, msg = manager.is_blacklisted("flaky_api")
    assert is_blocked is False
    
    # 3rd failure - triggers backoff
    manager.record_execution("flaky_api", {"query": "test"}, success=False, error="Timeout")
    
    is_blocked, msg = manager.is_blacklisted("flaky_api")
    assert is_blocked is True
    assert "Execution Backoff" in msg or "procedural backoff" in msg.lower()
    
    # Advance turn past backoff duration -> unblocked
    for _ in range(6):
        manager.advance_turn()
        
    is_blocked_after, msg_after = manager.is_blacklisted("flaky_api")
    assert is_blocked_after is False
