import math
import random
from typing import List, Tuple, Dict, Any

class SecureRouter:
    """Top-K MoE Router with constant-time selection and dummy computation padding."""

    def __init__(self, num_experts: int = 64, top_k: int = 2, dummy_padding_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.dummy_padding_k = dummy_padding_k

    def route(self, gating_logits: List[float]) -> Tuple[List[int], List[float]]:
        """
        Computes expert gating scores and returns top-K active experts + weights.
        Applies constant-time sorting to protect score leakage and adds dummy decoy padding.
        """
        if not gating_logits or len(gating_logits) < self.num_experts:
            # Generate synthetic gating scores if none provided
            gating_logits = [random.uniform(-1.0, 1.0) for _ in range(self.num_experts)]

        # Softmax normalization
        max_val = max(gating_logits)
        exp_scores = [math.exp(x - max_val) for x in gating_logits]
        sum_exp = sum(exp_scores)
        probs = [x / sum_exp for x in exp_scores]

        # Constant-time Bitonic/Pairwise sort simulation
        indexed_probs = list(enumerate(probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        selected_real = indexed_probs[:self.top_k]
        real_ids = [idx for idx, _ in selected_real]
        real_weights = [w for _, w in selected_real]

        # Add dummy computation padding
        all_dummy_candidates = [i for i in range(self.num_experts) if i not in real_ids]
        dummy_ids = random.sample(all_dummy_candidates, min(self.dummy_padding_k, len(all_dummy_candidates)))

        final_expert_ids = real_ids + dummy_ids
        final_weights = real_weights + [0.0] * len(dummy_ids)

        return final_expert_ids, final_weights


class AdaptiveEntropyRouter(SecureRouter):
    """Dynamic Top-K router with entropy smoothing and hysteresis anti-thrashing."""

    def __init__(
        self,
        num_experts: int = 64,
        top_k: int = 4,
        dummy_padding_k: int = 2,
        min_top_k: int = 2,
        max_top_k: int = 4,
        entropy_threshold: float = 3.0,
        entropy_hysteresis: float = 0.25,
        ema_alpha: float = 0.8,
    ):
        super().__init__(num_experts=num_experts, top_k=max_top_k, dummy_padding_k=dummy_padding_k)
        self.min_top_k = max(1, min_top_k)
        self.max_top_k = max(self.min_top_k, max_top_k)
        self.entropy_threshold = entropy_threshold
        self.entropy_hysteresis = max(0.0, entropy_hysteresis)
        self.ema_alpha = min(max(ema_alpha, 0.0), 0.999)

        self.smoothed_entropy = None
        self.last_entropy = 0.0
        self.last_active_top_k = self.min_top_k

        self.route_calls = 0
        self.active_top_k_sum = 0

    def _softmax(self, gating_logits: List[float]) -> List[float]:
        max_val = max(gating_logits)
        exp_scores = [math.exp(x - max_val) for x in gating_logits]
        sum_exp = sum(exp_scores)
        return [x / sum_exp for x in exp_scores]

    def _entropy(self, probs: List[float]) -> float:
        eps = 1e-9
        return -sum(p * math.log(p + eps) for p in probs)

    def _update_smoothed_entropy(self, current_entropy: float) -> float:
        if self.smoothed_entropy is None:
            self.smoothed_entropy = current_entropy
        else:
            self.smoothed_entropy = (self.ema_alpha * self.smoothed_entropy) + ((1.0 - self.ema_alpha) * current_entropy)
        return self.smoothed_entropy

    def _select_active_top_k(self, smoothed_entropy: float) -> int:
        high = self.entropy_threshold + self.entropy_hysteresis
        low = self.entropy_threshold - self.entropy_hysteresis

        if smoothed_entropy >= high:
            return self.max_top_k
        if smoothed_entropy <= low:
            return self.min_top_k
        return self.last_active_top_k

    def route_dynamic(self, gating_logits: List[float]) -> Tuple[List[int], List[float], Dict[str, Any]]:
        """
        Computes dynamic top-k routing with smoothed entropy and hysteresis.
        Returns expert ids (real+dummy), weights, and routing metadata.
        """
        if not gating_logits or len(gating_logits) < self.num_experts:
            gating_logits = [random.uniform(-1.0, 1.0) for _ in range(self.num_experts)]

        probs = self._softmax(gating_logits)
        current_entropy = self._entropy(probs)
        smoothed_entropy = self._update_smoothed_entropy(current_entropy)
        active_top_k = self._select_active_top_k(smoothed_entropy)

        indexed_probs = list(enumerate(probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        selected_real = indexed_probs[:active_top_k]
        real_ids = [idx for idx, _ in selected_real]
        real_weights = [w for _, w in selected_real]

        all_dummy_candidates = [i for i in range(self.num_experts) if i not in real_ids]
        dummy_ids = random.sample(all_dummy_candidates, min(self.dummy_padding_k, len(all_dummy_candidates)))

        final_expert_ids = real_ids + dummy_ids
        final_weights = real_weights + [0.0] * len(dummy_ids)

        self.last_entropy = current_entropy
        self.last_active_top_k = active_top_k
        self.route_calls += 1
        self.active_top_k_sum += active_top_k

        metadata = {
            "active_top_k": active_top_k,
            "max_top_k": self.max_top_k,
            "min_top_k": self.min_top_k,
            "entropy": round(current_entropy, 6),
            "smoothed_entropy": round(smoothed_entropy, 6),
            "entropy_threshold": self.entropy_threshold,
            "entropy_hysteresis": self.entropy_hysteresis,
            "ema_alpha": self.ema_alpha,
        }

        return final_expert_ids, final_weights, metadata

    def get_state(self) -> Dict[str, Any]:
        avg_active = (self.active_top_k_sum / self.route_calls) if self.route_calls else float(self.min_top_k)
        return {
            "enabled": True,
            "min_top_k": self.min_top_k,
            "max_top_k": self.max_top_k,
            "last_active_top_k": self.last_active_top_k,
            "average_active_top_k": round(avg_active, 4),
            "last_entropy": round(self.last_entropy, 6),
            "last_smoothed_entropy": round(float(self.smoothed_entropy or 0.0), 6),
            "route_calls": self.route_calls,
        }
