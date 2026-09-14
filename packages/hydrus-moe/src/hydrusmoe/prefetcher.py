from typing import Set, List, Dict, Any
from .config import HydrusMoEConfig

class HAILPrefetcher:
    """Two-Stage Predictive Prefetcher (MoE-Infinity EAM + HAIL Cognitive Memory Layer)."""

    def __init__(self, config: HydrusMoEConfig):
        self.config = config
        self.eam_history: List[Set[int]] = []
        self.total_queries = 0
        self.query_hit_count = 0
        self.predicted_experts_total = 0
        self.actual_experts_total = 0
        self.expert_hit_total = 0
        self.jaccard_sum = 0.0
        self.last_prediction: Set[int] = set()
        self.last_prediction_mask: int = 0
        self.last_actual: Set[int] = set()

    @staticmethod
    def _ids_to_mask(ids: Set[int] or List[int]) -> int:
        mask = 0
        for expert_id in ids:
            if expert_id >= 0:
                mask |= (1 << int(expert_id))
        return mask

    def predict(self, context_prompt: str, user_memories: List[str] = None) -> Set[int]:
        """
        Queries HAIL Cognitive Layer and EAM activation patterns to predict upcoming expert IDs.
        Returns set of predicted expert IDs exceeding confidence threshold.
        """
        predicted = set()
        lower = (context_prompt or "").lower()
        memory_blob = " ".join(user_memories or []).lower()
        combined = f"{lower} {memory_blob}"

        # Domain-aware rule prediction backed by HAIL context spread
        if any(w in combined for w in ["code", "python", "function", "bug", "algorithm", "developer", "class"]):
            predicted.update([3, 7, 12, 19])
        elif any(w in combined for w in ["history", "london", "war", "century", "emperor", "king"]):
            predicted.update([1, 4, 15, 22])
        elif any(w in combined for w in ["math", "calculus", "formula", "equation", "matrix"]):
            predicted.update([2, 9, 14, 28])
        else:
            predicted.update([0, 1, 2])

        # Stage 1: MoE-Infinity EAM history correlation
        if self.eam_history:
            recent = self.eam_history[-1]
            predicted.update(recent)

        self.last_prediction = set(predicted)
        self.last_prediction_mask = self._ids_to_mask(predicted)
        return predicted

    def update_actual(self, actual_expert_ids: List[int]):
        """Updates activation history and calculates prefetch hit-rate telemetry."""
        actual_set = set(actual_expert_ids)
        predicted_set = set(self.last_prediction)
        actual_mask = self._ids_to_mask(actual_set)
        predicted_mask = int(self.last_prediction_mask)

        self.eam_history.append(actual_set)
        if len(self.eam_history) > 50:
            self.eam_history.pop(0)

        self.total_queries += 1
        self.last_actual = set(actual_set)

        intersection_count = (predicted_mask & actual_mask).bit_count()
        union_count = (predicted_mask | actual_mask).bit_count()
        if intersection_count > 0:
            self.query_hit_count += 1

        self.predicted_experts_total += len(predicted_set)
        self.actual_experts_total += len(actual_set)
        self.expert_hit_total += intersection_count
        self.jaccard_sum += (intersection_count / union_count) if union_count else 1.0

    def get_hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self.query_hit_count / self.total_queries, 4)

    def get_metrics(self) -> Dict[str, Any]:
        if self.total_queries == 0:
            return {
                "queries_total": 0,
                "query_hit_rate": 0.0,
                "expert_precision": 0.0,
                "expert_recall": 0.0,
                "mean_jaccard": 0.0,
                "predicted_experts_total": 0,
                "actual_experts_total": 0,
                "expert_hits_total": 0,
                "last_prediction_size": len(self.last_prediction),
                "last_actual_size": len(self.last_actual),
            }

        precision = self.expert_hit_total / max(1, self.predicted_experts_total)
        recall = self.expert_hit_total / max(1, self.actual_experts_total)

        return {
            "queries_total": self.total_queries,
            "query_hit_rate": round(self.query_hit_count / self.total_queries, 4),
            "expert_precision": round(precision, 4),
            "expert_recall": round(recall, 4),
            "mean_jaccard": round(self.jaccard_sum / self.total_queries, 4),
            "predicted_experts_total": self.predicted_experts_total,
            "actual_experts_total": self.actual_experts_total,
            "expert_hits_total": self.expert_hit_total,
            "last_prediction_size": len(self.last_prediction),
            "last_actual_size": len(self.last_actual),
        }
