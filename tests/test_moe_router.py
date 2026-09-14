import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.router import AdaptiveEntropyRouter


class TestAdaptiveRouter(unittest.TestCase):
    def test_entropy_smoothing_and_hysteresis(self):
        router = AdaptiveEntropyRouter(
            num_experts=8,
            min_top_k=2,
            max_top_k=4,
            entropy_threshold=1.2,
            entropy_hysteresis=0.1,
            ema_alpha=0.8,
            dummy_padding_k=1,
        )

        low_entropy_logits = [6.0] + [-5.0] * 7
        high_entropy_logits = [0.0] * 8

        ids1, weights1, meta1 = router.route_dynamic(low_entropy_logits)
        self.assertEqual(meta1["active_top_k"], 2)

        # Repeat high entropy inputs so EMA crosses threshold+hysteresis.
        last_meta = meta1
        for _ in range(12):
            _, _, last_meta = router.route_dynamic(high_entropy_logits)

        self.assertEqual(last_meta["active_top_k"], 4)
        self.assertGreater(last_meta["smoothed_entropy"], 1.2)

    def test_state_telemetry_updates(self):
        router = AdaptiveEntropyRouter(num_experts=8, min_top_k=2, max_top_k=4)
        router.route_dynamic([0.0] * 8)
        state = router.get_state()

        self.assertTrue(state["enabled"])
        self.assertGreaterEqual(state["last_active_top_k"], 2)
        self.assertGreaterEqual(state["route_calls"], 1)


class TestSparseDispatcher(unittest.TestCase):
    def test_execute_optimized_sparse_moe(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch not installed")

        class ScaleExpert(nn.Module):
            def __init__(self, scale: float):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                return x * self.scale

        from hydrusmoe.dynamic_dispatch import execute_optimized_sparse_moe

        experts = [ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(3.0)]

        hidden_states = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        topk_indices = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [1, 0],
            ],
            dtype=torch.long,
        )
        dynamic_probs = torch.tensor(
            [
                [1.0, 0.0],
                [0.7, 0.3],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        out_sparse = execute_optimized_sparse_moe(hidden_states, dynamic_probs, topk_indices, experts)

        # Dense reference computation for correctness.
        out_dense = torch.zeros_like(hidden_states)
        for token_i in range(hidden_states.shape[0]):
            for k_i in range(topk_indices.shape[1]):
                p = dynamic_probs[token_i, k_i]
                if p <= 1e-8:
                    continue
                e_idx = int(topk_indices[token_i, k_i].item())
                out_dense[token_i] += p * experts[e_idx](hidden_states[token_i:token_i + 1])[0]

        self.assertTrue(torch.allclose(out_sparse, out_dense, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
