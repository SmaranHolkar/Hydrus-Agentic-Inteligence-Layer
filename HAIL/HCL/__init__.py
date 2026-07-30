"""
HCL - Hydrus Cognitive Layer & Agentic Brain (inside HAIL)
"""

from .memory import StratifiedMemoryLattice, ProceduralMemoryManager
from .cognems import CognemTokenizer, build_cognem_vocab
from .router import MetacognitiveRouter, CognitiveState, RoutingDecision
from .gateway import CognitiveMemoryGateway
from .agent import HydrusAgent, MCPBus
from .events import AgentEvent, EventType
from .session import Session
from .subagents import (
    BackgroundTaskRegistry, ExtractorAgent, GroundingAgent,
    ConsolidationAgent, DecayAgent, CandidateFact, GroundedFact
)

__all__ = [
    "StratifiedMemoryLattice",
    "ProceduralMemoryManager",
    "CognemTokenizer",
    "build_cognem_vocab",
    "MetacognitiveRouter",
    "CognitiveState",
    "RoutingDecision",
    "CognitiveMemoryGateway",
    "HydrusAgent",
    "MCPBus",
    "AgentEvent",
    "EventType",
    "Session",
    "BackgroundTaskRegistry",
    "ExtractorAgent",
    "GroundingAgent",
    "ConsolidationAgent",
    "DecayAgent",
    "CandidateFact",
    "GroundedFact",
]
