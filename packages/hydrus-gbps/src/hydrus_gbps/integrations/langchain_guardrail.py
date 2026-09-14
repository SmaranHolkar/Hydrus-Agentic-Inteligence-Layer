"""
LangChain Guardrail Integration for GBPS.
"""

from typing import Dict, Any, Optional
from ..verifier import GBPSVerifier, VerificationResult


class GBPSGuardrail:
    """
    Plug-and-play guardrail for LangChain chains and agents.
    
    Example:
        guardrail = GBPSGuardrail(threshold=0.7)
        result = guardrail.validate(query="...", context="...", response="...")
    """

    def __init__(self, threshold: float = 0.65, verifier: Optional[GBPSVerifier] = None):
        self.threshold = threshold
        self.verifier = verifier or GBPSVerifier(grounding_threshold=threshold)

    def validate(self, query: str, context: str, response: str) -> VerificationResult:
        return self.verifier.verify(query=query, context=context, response=response, threshold=self.threshold)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Callable pipeline step for LangChain Runnables."""
        query = inputs.get("query", inputs.get("input", ""))
        context = inputs.get("context", inputs.get("documents", ""))
        response = inputs.get("response", inputs.get("output", ""))
        
        verification = self.validate(query, context, response)
        inputs["verification"] = verification.to_dict()
        inputs["is_grounded"] = verification.is_grounded
        return inputs
