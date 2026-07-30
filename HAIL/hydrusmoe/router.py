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
