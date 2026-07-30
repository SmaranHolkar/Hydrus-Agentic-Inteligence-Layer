import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .metacognitive_router import CognitiveState

logger = logging.getLogger("hail.memory_subagents")

class BackgroundTaskRegistry:
    """
    Registry for asynchronous background tasks.
    Prevents silent failure traps by catching background task exceptions
    and updating CognitiveState diagnostic metrics.
    """
    def __init__(self, cognitive_state: Optional[CognitiveState] = None):
        self.cognitive_state = cognitive_state
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.failed_tasks_log: List[Dict[str, Any]] = []

    def create_background_task(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.active_tasks[name] = task

        def _on_complete(t: asyncio.Task):
            self.active_tasks.pop(name, None)
            if not t.cancelled() and t.exception() is not None:
                exc = t.exception()
                logger.error(f"[BackgroundTaskRegistry] Task '{name}' failed with error: {exc}")
                self.failed_tasks_log.append({
                    "task_name": name,
                    "error": str(exc),
                    "timestamp": time.time()
                })
                if self.cognitive_state:
                    self.cognitive_state.background_failures += 1

        task.add_done_callback(_on_complete)
        return task

    def get_active_count(self) -> int:
        return len(self.active_tasks)


@dataclass
class CandidateFact:
    subject: str
    predicate: str
    object_val: str
    source: str = "conversation"
    confidence: float = 0.8
    raw_text: str = ""


@dataclass
class GroundedFact:
    candidate: CandidateFact
    is_grounded: bool
    contradiction_flag: bool
    contradicted_memory_id: Optional[str] = None
    epistemic_confidence: float = 0.8
    action: str = "promote"  # promote | flag_contradiction | reject


class ExtractorAgent:
    """Lightweight fast candidate fact extraction agent."""
    def extract_candidate_facts(self, text: str, source: str = "mcp_output") -> List[CandidateFact]:
        if not text or len(text.strip()) < 5:
            return []

        candidates = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for line in lines:
            # Basic key-value / relational candidate extraction heuristic
            if ":" in line and not line.startswith("http"):
                parts = line.split(":", 1)
                subj = parts[0].strip()
                val = parts[1].strip()
                if subj and val and len(val) < 200:
                    candidates.append(CandidateFact(
                        subject=subj,
                        predicate="has_value",
                        object_val=val,
                        source=source,
                        confidence=0.8,
                        raw_text=line
                    ))
            elif " is " in line.lower():
                parts = line.lower().split(" is ", 1)
                subj = parts[0].strip()
                val = parts[1].strip()
                if subj and val and len(val) < 200:
                    candidates.append(CandidateFact(
                        subject=subj,
                        predicate="is",
                        object_val=val,
                        source=source,
                        confidence=0.75,
                        raw_text=line
                    ))

        return candidates


class GroundingAgent:
    """Fact validation & contradiction checking agent."""
    def verify_candidate(self, candidate: CandidateFact, existing_knowledge: List[Dict[str, Any]]) -> GroundedFact:
        # Check against existing knowledge base for direct contradictions
        for fact in existing_knowledge:
            subj_match = fact.get("subject", "").lower() == candidate.subject.lower()
            pred_match = fact.get("predicate", "").lower() == candidate.predicate.lower()

            if subj_match and pred_match:
                existing_val = str(fact.get("object_val", "")).lower()
                candidate_val = str(candidate.object_val).lower()

                if existing_val != candidate_val:
                    # Contradiction detected! Calculate source epistemic confidence
                    source_reliability = 0.95 if candidate.source in ("postgres", "authenticated-db") else 0.40
                    return GroundedFact(
                        candidate=candidate,
                        is_grounded=False,
                        contradiction_flag=True,
                        contradicted_memory_id=fact.get("memory_id", "mem_unknown"),
                        epistemic_confidence=source_reliability,
                        action="flag_contradiction"
                    )
                else:
                    # Corroborated existing fact
                    return GroundedFact(
                        candidate=candidate,
                        is_grounded=True,
                        contradiction_flag=False,
                        epistemic_confidence=min(1.0, fact.get("confidence", 0.8) + 0.1),
                        action="corroborate"
                    )

        # Non-contradictory new fact
        source_reliability = 0.9 if candidate.source in ("postgres", "authenticated-db", "user-private") else 0.7
        return GroundedFact(
            candidate=candidate,
            is_grounded=True,
            contradiction_flag=False,
            epistemic_confidence=source_reliability,
            action="promote"
        )


class ConsolidationAgent:
    """Async background consolidation agent (runs during idle time)."""
    async def consolidate(self, grounded_facts: List[GroundedFact], memory_store: Dict[str, Any]) -> int:
        promoted_count = 0
        for gf in grounded_facts:
            if gf.action in ("promote", "corroborate") and not gf.contradiction_flag:
                key = f"{gf.candidate.subject}:{gf.candidate.predicate}"
                memory_store[key] = {
                    "subject": gf.candidate.subject,
                    "predicate": gf.candidate.predicate,
                    "object_val": gf.candidate.object_val,
                    "confidence": gf.epistemic_confidence,
                    "updated_at": time.time(),
                    "source": gf.candidate.source
                }
                promoted_count += 1
            elif gf.contradiction_flag:
                # Store flagged contradiction under isolated audit branch
                flag_key = f"contradiction:{gf.candidate.subject}:{time.time()}"
                memory_store[flag_key] = {
                    "subject": gf.candidate.subject,
                    "candidate_val": gf.candidate.object_val,
                    "contradicted_memory_id": gf.contradicted_memory_id,
                    "confidence": gf.epistemic_confidence,
                    "contradiction_flag": True
                }
        return promoted_count


class DecayAgent:
    """
    Active VRAM/RAM protection agent.
    Evicts stale episodic nodes and demotes unused semantic facts when memory limits are hit.
    """
    def __init__(self, max_items: int = 100):
        self.max_items = max_items

    def evict_and_demote(self, memory_store: Dict[str, Any], vram_utilization_pct: float = 50.0) -> int:
        if len(memory_store) <= self.max_items and vram_utilization_pct < 80.0:
            return 0

        evicted_count = 0
        items_to_remove = []
        now = time.time()

        # Identify items to evict (contradictions older than 60s, or oldest items)
        for key, value in list(memory_store.items()):
            if isinstance(value, dict):
                # Evict old flagged contradictions first
                if value.get("contradiction_flag") and (now - value.get("updated_at", now) > 60.0):
                    items_to_remove.append(key)
                elif len(memory_store) - len(items_to_remove) > self.max_items:
                    # Evict least recently updated or lowest confidence entry
                    if value.get("confidence", 1.0) < 0.5:
                        items_to_remove.append(key)

        for key in items_to_remove:
            memory_store.pop(key, None)
            evicted_count += 1

        return evicted_count
