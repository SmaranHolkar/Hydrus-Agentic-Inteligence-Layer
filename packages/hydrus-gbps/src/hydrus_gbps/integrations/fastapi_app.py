"""
FastAPI Microservice & Route Handler for GBPS Verification API.
"""

from typing import List, Optional, Union, Dict, Any
from ..verifier import GBPSVerifier, VerificationResult

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    BaseModel = object  # type: ignore
    Field = lambda **kwargs: None  # type: ignore


if HAS_FASTAPI:
    class VerifyRequest(BaseModel):
        query: str = Field(..., description="User query or input prompt")
        context: Union[str, List[str]] = Field(..., description="Retrieved context or ground truth reference documents")
        response: str = Field(..., description="LLM generated answer to verify")
        threshold: Optional[float] = Field(0.65, description="Custom grounding threshold between 0.0 and 1.0")

    class VerifyResponse(BaseModel):
        is_grounded: bool
        grounding_score: float
        flagged_claims: List[str]
        supported_claims: List[Dict[str, Any]]
        belief_chain: List[Dict[str, Any]]
        latency_ms: float
        metadata: Dict[str, Any]

    def create_app(verifier: Optional[GBPSVerifier] = None) -> FastAPI:
        app = FastAPI(
            title="Hydrus GBPS Verification API",
            description="Real-time Grounded Belief Path Search & Hallucination Detection Service",
            version="0.1.0"
        )
        engine = verifier or GBPSVerifier()

        @app.get("/health")
        def health_check():
            return {"status": "healthy", "service": "hydrus-gbps", "version": "0.1.0"}

        @app.post("/v1/verify", response_model=VerifyResponse)
        def verify_endpoint(req: VerifyRequest):
            try:
                res: VerificationResult = engine.verify(
                    query=req.query,
                    context=req.context,
                    response=req.response,
                    threshold=req.threshold
                )
                return res.to_dict()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app
else:
    def create_app(*args, **kwargs):
        raise ImportError("FastAPI and Pydantic are required to run the GBPS REST API. Install via 'pip install hydrus-gbps[api]'.")
