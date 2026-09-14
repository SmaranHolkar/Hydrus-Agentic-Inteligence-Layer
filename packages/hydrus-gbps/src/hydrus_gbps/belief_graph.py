"""
Belief Graph & Grounded Belief Path Search (GBPS) Core Module.

Implements belief node extraction, trajectory tracking, transition modeling,
and grounded path search for fast hallucination detection.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class BeliefNode:
    """Represents a grounded fact, claim, or contextual assertion."""
    id: str
    text: str
    embedding: np.ndarray
    confidence: float = 1.0
    source: str = "context"
    cluster_id: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "confidence": self.confidence,
            "source": self.source,
            "cluster_id": self.cluster_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class GroundedBeliefPathSearch:
    """
    Tracks dynamic active belief paths and calculates groundedness vectors.
    """

    def __init__(self, embedding_dim: int = 768, max_active_beliefs: int = 8):
        self.embedding_dim = embedding_dim
        self.max_active_beliefs = max_active_beliefs
        self.active_beliefs: List[BeliefNode] = []
        self.new_belief_flag = False

    def add_belief(self, node: BeliefNode) -> None:
        self.active_beliefs.append(node)
        self.new_belief_flag = True
        if len(self.active_beliefs) > self.max_active_beliefs:
            self.active_beliefs.pop(0)

    def clear(self) -> None:
        self.active_beliefs.clear()
        self.new_belief_flag = False

    def get_active_beliefs(self) -> List[BeliefNode]:
        return list(self.active_beliefs)

    def get_active_belief_embedding(self) -> np.ndarray:
        if not self.active_beliefs:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embs = [node.embedding for node in self.active_beliefs if node.embedding is not None]
        if not embs:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)

    def current_embedding(self) -> np.ndarray:
        return self.get_active_belief_embedding()

    def detect_new_belief(self) -> bool:
        flag = self.new_belief_flag
        self.new_belief_flag = False
        return flag

    def compute_grounding_score(self, target_embedding: np.ndarray) -> float:
        """
        Calculates cosine grounding confidence against the active belief path.
        """
        if not self.active_beliefs or target_embedding is None:
            return 0.0
        belief_emb = self.get_active_belief_embedding()
        norm_b = np.linalg.norm(belief_emb)
        norm_t = np.linalg.norm(target_embedding)
        if norm_b == 0 or norm_t == 0:
            return 0.0
        score = float(np.dot(belief_emb, target_embedding) / (norm_b * norm_t))
        return max(0.0, min(1.0, score))


class BeliefTransitionModel:
    """
    Models transitions between belief clusters to predict likely follow-up assertions
    and detect abrupt, ungrounded conversational leaps.
    """

    def __init__(self):
        self.transitions: Dict[Tuple[int, int], int] = {}
        self.last_cluster_ids: List[int] = []

    def update(self, active_beliefs: List[BeliefNode]) -> None:
        if not active_beliefs:
            return
        current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
        if not current_clusters:
            return
        if self.last_cluster_ids:
            for prev in self.last_cluster_ids:
                for curr in current_clusters:
                    key = (prev, curr)
                    self.transitions[key] = self.transitions.get(key, 0) + 1
        self.last_cluster_ids = current_clusters

    def predict_next(self, active_beliefs: List[BeliefNode], top_k: int = 3) -> List[int]:
        if not active_beliefs:
            return []
        current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
        if not current_clusters:
            return []
        scores: Dict[int, int] = {}
        for prev in current_clusters:
            for (p, nxt), count in self.transitions.items():
                if p == prev:
                    scores[nxt] = scores.get(nxt, 0) + count
        sorted_clusters = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_clusters[:top_k]
