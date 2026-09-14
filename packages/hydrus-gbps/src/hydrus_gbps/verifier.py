"""
GBPS Grounding & Hallucination Verifier.

The primary entry point for developers to verify LLM outputs, RAG retrievals, and agent actions.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import hashlib
import time
import numpy as np

from .belief_graph import GroundedBeliefPathSearch, BeliefNode, BeliefTransitionModel
from .tt_projection import NumpyTTLinear
from .claim_extractor import split_claims


@dataclass
class ClaimVerification:
    claim: str
    is_grounded: bool
    grounding_score: float
    best_matching_context: Optional[str] = None


@dataclass
class VerificationResult:
    is_grounded: bool
    grounding_score: float
    flagged_claims: List[str]
    supported_claims: List[ClaimVerification]
    belief_chain: List[Dict[str, Any]]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "grounding_score": round(self.grounding_score, 4),
            "flagged_claims": self.flagged_claims,
            "supported_claims": [
                {
                    "claim": c.claim,
                    "is_grounded": c.is_grounded,
                    "grounding_score": round(c.grounding_score, 4),
                    "best_matching_context": c.best_matching_context
                }
                for c in self.supported_claims
            ],
            "belief_chain": self.belief_chain,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


class FastHashEmbedder:
    """
    Zero-dependency fast semantic-hashing embedder for microsecond-speed verification.
    Can be replaced with any dense embedding model (e.g. sentence-transformers, OpenAI).
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        for t in tokens:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
            for j in range(len(t) - 2):
                g = t[j:j+3]
                hg = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                vec[hg % self.dim] += 0.3
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)


class GBPSVerifier:
    """
    Plug-and-play verification engine using Grounded Belief Path Search & TTLinear projections.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        grounding_threshold: float = 0.60,
        embedder: Optional[Union[FastHashEmbedder, Callable[[str], np.ndarray]]] = None,
        use_tt_projection: bool = False,
        tt_rank: int = 8,
    ):
        self.embedding_dim = embedding_dim
        self.grounding_threshold = grounding_threshold
        self.embedder = embedder or FastHashEmbedder(dim=embedding_dim)
        self.use_tt_projection = use_tt_projection
        self.tt_proj = NumpyTTLinear(8, 8, 12, rank=tt_rank) if use_tt_projection else None
        self.gbps = GroundedBeliefPathSearch(embedding_dim=embedding_dim)
        self.transition_model = BeliefTransitionModel()

    def _get_embedding(self, text: str) -> np.ndarray:
        if callable(self.embedder):
            emb = self.embedder(text)
        elif hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(text)
        elif hasattr(self.embedder, "encode"):
            emb = self.embedder.encode(text)
        else:
            raise ValueError("Invalid embedder provided to GBPSVerifier.")
        
        emb = np.asarray(emb, dtype=np.float32)
        if self.use_tt_projection and self.tt_proj is not None and emb.shape[0] == self.embedding_dim:
            emb = self.tt_proj.project(emb)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
        return emb

    def verify(
        self,
        query: str,
        context: Union[str, List[str]],
        response: str,
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """
        Verifies if the response is grounded in the provided context.
        """
        start_t = time.perf_counter()
        effective_threshold = threshold if threshold is not None else self.grounding_threshold

        # 1. Normalize context chunks
        context_chunks: List[str] = []
        if isinstance(context, str):
            context_chunks = split_claims(context)
            if not context_chunks and context.strip():
                context_chunks = [context.strip()]
        else:
            for c in context:
                chunks = split_claims(c)
                context_chunks.extend(chunks if chunks else [c])

        # 2. Build Grounded Belief Nodes
        self.gbps.clear()
        context_embs: List[Tuple[str, np.ndarray]] = []
        for idx, chunk in enumerate(context_chunks):
            emb = self._get_embedding(chunk)
            node = BeliefNode(
                id=f"ctx_{idx}",
                text=chunk,
                embedding=emb,
                confidence=1.0,
                source="context",
                cluster_id=idx % 5
            )
            self.gbps.add_belief(node)
            context_embs.append((chunk, emb))

        self.transition_model.update(self.gbps.get_active_beliefs())

        # 3. Extract claims from the response
        response_claims = split_claims(response)
        if not response_claims and response.strip():
            response_claims = [response.strip()]

        flagged_claims: List[str] = []
        claim_verifications: List[ClaimVerification] = []
        score_sum = 0.0

        for claim in response_claims:
            claim_emb = self._get_embedding(claim)
            
            # Find best matching context chunk
            best_score = 0.0
            best_chunk: Optional[str] = None
            for chunk_text, c_emb in context_embs:
                cos_sim = float(np.dot(claim_emb, c_emb) / (np.linalg.norm(claim_emb) * np.linalg.norm(c_emb) + 1e-9))
                if cos_sim > best_score:
                    best_score = cos_sim
                    best_chunk = chunk_text

            # Compute GBPS path grounding score
            gbps_path_score = self.gbps.compute_grounding_score(claim_emb)
            combined_score = max(best_score, gbps_path_score)
            
            is_claim_grounded = combined_score >= effective_threshold
            if not is_claim_grounded:
                flagged_claims.append(claim)

            claim_verifications.append(ClaimVerification(
                claim=claim,
                is_grounded=is_claim_grounded,
                grounding_score=combined_score,
                best_matching_context=best_chunk
            ))
            score_sum += combined_score

        overall_score = score_sum / max(1, len(response_claims))
        is_overall_grounded = (len(flagged_claims) == 0) and (overall_score >= effective_threshold)

        active_belief_dicts = [b.to_dict() for b in self.gbps.get_active_beliefs()]

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return VerificationResult(
            is_grounded=is_overall_grounded,
            grounding_score=overall_score,
            flagged_claims=flagged_claims,
            supported_claims=claim_verifications,
            belief_chain=active_belief_dicts,
            latency_ms=elapsed_ms,
            metadata={
                "threshold_used": effective_threshold,
                "context_chunks_count": len(context_chunks),
                "claims_evaluated_count": len(response_claims),
            }
        )
