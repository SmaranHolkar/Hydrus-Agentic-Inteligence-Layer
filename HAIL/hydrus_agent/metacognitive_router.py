import time
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class CognitiveState:
    """
    Classifies the agent's internal metacognitive state rather than query pattern matching.
    """
    remaining_context_budget: int = 4096
    latency_budget_ms: float = 2000.0
    vram_utilization_pct: float = 40.0
    perplexity_score: float = 0.2
    consecutive_failures: int = 0
    background_failures: int = 0
    user_tone: str = "neutral"  # neutral | frustrated | analytical
    hardware_profile: str = "standard"  # lightweight | standard | high_capacity

@dataclass
class RoutingDecision:
    """
    Determines memory retrieval depth, compression mode, and sub-agent invocation.
    """
    fast_path: bool = False
    skip_grounding: bool = False
    compress_cognems: bool = False
    retrieval_depth: str = "semantic_shallow"  # none | procedural_cache | semantic_shallow | full_grounding
    reason: str = "Default state evaluation."

class MetacognitiveRouter:
    """
    Introspective state-aware memory router.
    Evaluates hardware substrate, token limits, latency budgets, and model uncertainty.
    """
    def __init__(self, vram_high_threshold: float = 85.0, tight_latency_threshold_ms: float = 500.0):
        self.vram_high_threshold = vram_high_threshold
        self.tight_latency_threshold_ms = tight_latency_threshold_ms

    def evaluate(self, state: CognitiveState) -> RoutingDecision:
        reasons = []

        # 1. Hardware/Latency Constrained Fast-Path Bypass
        is_vram_constrained = state.vram_utilization_pct >= self.vram_high_threshold
        is_latency_tight = state.latency_budget_ms <= self.tight_latency_threshold_ms
        is_lightweight_hw = state.hardware_profile == "lightweight"

        if is_vram_constrained or is_latency_tight or is_lightweight_hw:
            if is_vram_constrained:
                reasons.append(f"VRAM utilization high ({state.vram_utilization_pct:.1f}%)")
            if is_latency_tight:
                reasons.append(f"Strict latency budget ({state.latency_budget_ms:.1f}ms)")
            if is_lightweight_hw:
                reasons.append("Lightweight hardware profile")

            return RoutingDecision(
                fast_path=True,
                skip_grounding=True,
                compress_cognems=state.remaining_context_budget < 2048,
                retrieval_depth="procedural_cache",
                reason="Fast-path bypass triggered: " + "; ".join(reasons)
            )

        # 2. Context Budget Compression Check
        compress_cognems = state.remaining_context_budget < 2048

        # 3. High Uncertainty / High Perplexity / User Frustration -> Trigger Full Grounding
        needs_grounding = (
            state.perplexity_score > 0.65 or
            state.consecutive_failures > 0 or
            state.user_tone == "frustrated"
        )

        if needs_grounding:
            if state.consecutive_failures > 0:
                reasons.append(f"Consecutive failures detected ({state.consecutive_failures})")
            if state.perplexity_score > 0.65:
                reasons.append(f"High model perplexity ({state.perplexity_score:.2f})")
            if state.user_tone == "frustrated":
                reasons.append("User tone is frustrated - prioritizing grounding & failure traces")

            return RoutingDecision(
                fast_path=False,
                skip_grounding=False,
                compress_cognems=compress_cognems,
                retrieval_depth="full_grounding",
                reason="Deep grounding required: " + "; ".join(reasons)
            )

        # 4. Standard Operational Mode
        return RoutingDecision(
            fast_path=False,
            skip_grounding=False,
            compress_cognems=compress_cognems,
            retrieval_depth="semantic_shallow",
            reason="Standard operation mode."
        )
