"""
Hydrus-GBPS: Grounded Belief Path Search (GBPS) Real-time Hallucination & Grounding Engine.
"""

from .verifier import GBPSVerifier, VerificationResult, ClaimVerification, FastHashEmbedder
from .belief_graph import GroundedBeliefPathSearch, BeliefNode, BeliefTransitionModel
from .tt_projection import TTLinear, NumpyTTLinear, flop_estimate
from .claim_extractor import split_claims
from .integrations.langchain_guardrail import GBPSGuardrail

try:
    from .integrations.fastapi_app import create_app
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = [
    "GBPSVerifier",
    "VerificationResult",
    "ClaimVerification",
    "FastHashEmbedder",
    "GroundedBeliefPathSearch",
    "BeliefNode",
    "BeliefTransitionModel",
    "TTLinear",
    "NumpyTTLinear",
    "flop_estimate",
    "split_claims",
    "GBPSGuardrail",
]
