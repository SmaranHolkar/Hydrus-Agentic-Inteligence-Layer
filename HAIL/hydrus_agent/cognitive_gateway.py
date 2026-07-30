import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from .security import SecurityIngestionGuard
from .metacognitive_router import MetacognitiveRouter, CognitiveState, RoutingDecision
from .memory_subagents import (
    ExtractorAgent, GroundingAgent, ConsolidationAgent, DecayAgent,
    BackgroundTaskRegistry, CandidateFact, GroundedFact
)
from hail_core.procedural_memory import ProceduralMemoryManager

logger = logging.getLogger("hail.cognitive_gateway")

class CognitiveMemoryGateway:
    """
    Cognitive Memory Gateway (CMG).
    Sits between external MCP tool sources and HAIL's stratified memory layers.
    Filters payloads via SecurityIngestionGuard, evaluates state via MetacognitiveRouter,
    and handles Working, Episodic, Semantic, and Procedural memory routing.
    """
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.security_guard = SecurityIngestionGuard()
        self.router = MetacognitiveRouter()
        self.extractor = ExtractorAgent()
        self.grounding = GroundingAgent()
        self.consolidator = ConsolidationAgent()
        self.decay = DecayAgent(max_items=100)
        self.procedural = ProceduralMemoryManager()
        
        self.cognitive_state = CognitiveState()
        self.task_registry = BackgroundTaskRegistry(self.cognitive_state)
        
        # In-memory semantic knowledge graph store
        self.semantic_store: Dict[str, Any] = {}
        self.episodic_traces: List[Dict[str, Any]] = []

    async def process_external_output(
        self,
        tool_name: str,
        args: Dict[str, Any],
        raw_output: str,
        source_type: str = "external_mcp"
    ) -> Dict[str, Any]:
        """
        Main CMG pipeline:
        1. Ingestion Firewall (Security & Sanitization)
        2. Metacognitive State Routing
        3. Working & Episodic Memory Injection
        4. Async Semantic Memory Ingestion & Conflict Resolution
        5. Procedural Memory Record
        """
        # 1. Active Security Firewall Scan
        security_res = self.security_guard.inspect_and_sanitize(raw_output, source_type=source_type)
        
        if security_res["quarantined"]:
            logger.warning(f"[CMG Firewall Alert] Quarantined output from '{tool_name}': {security_res['quarantine_reason']}")
            # Record failed procedural execution due to quarantine
            self.procedural.record_execution(tool_name, args, success=False, error=security_res['quarantine_reason'])
            return {
                "working_context": security_res["sanitized_payload"],
                "quarantined": True,
                "quarantine_reason": security_res["quarantine_reason"],
                "security_tag": security_res["security_tag"],
                "grounded_count": 0
            }

        sanitized_text = security_res["sanitized_payload"]

        # 2. Metacognitive State Routing
        decision = self.router.evaluate(self.cognitive_state)

        # 3. Record Episodic Trace
        episodic_trace = {
            "tool_name": tool_name,
            "args": args,
            "timestamp": time.time(),
            "security_tag": security_res["security_tag"],
            "summary": sanitized_text[:150]
        }
        self.episodic_traces.append(episodic_trace)

        # 4. Semantic Memory Ingestion (Fast-path bypass vs Deep Grounding)
        grounded_count = 0
        if not decision.skip_grounding:
            candidates = self.extractor.extract_candidate_facts(sanitized_text, source=source_type)
            grounded_facts = []
            
            existing_knowledge = list(self.semantic_store.values())
            for cand in candidates:
                gf = self.grounding.verify_candidate(cand, existing_knowledge)
                grounded_facts.append(gf)

            # Consolidate grounded facts
            grounded_count = await self.consolidator.consolidate(grounded_facts, self.semantic_store)
        else:
            logger.info(f"[CMG] Fast-path bypass active. Skipping deep grounding for '{tool_name}'.")

        # 5. Decay Check
        evicted = self.decay.evict_and_demote(self.semantic_store, self.cognitive_state.vram_utilization_pct)
        if evicted > 0:
            logger.info(f"[CMG DecayAgent] Evicted {evicted} stale/low-confidence memory items.")

        # 6. Procedural Memory Record
        self.procedural.record_execution(tool_name, args, success=True)

        return {
            "working_context": sanitized_text,
            "quarantined": False,
            "security_tag": security_res["security_tag"],
            "redacted_count": security_res["redacted_count"],
            "grounded_count": grounded_count,
            "routing_decision": decision
        }
