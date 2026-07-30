import sys
import unittest
import torch
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hcl_core.tt_linear import TTLinear, flop_estimate
from hcl_core.hcl import HCL, Node, NodeType

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
    def decode(self, token_ids, skip_special_tokens=True):
        return "mock decoded query text fact"
    def encode(self, text, return_tensors=None, add_special_tokens=True):
        return [1, 2, 3]
    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

class MockModel:
    def __init__(self):
        self.device = torch.device("cpu")
    def __call__(self, input_ids, **kwargs):
        class MockOutputs:
            def __init__(self):
                seq_len = input_ids.shape[1]
                self.logits = torch.zeros((1, seq_len, 50))
        return MockOutputs()


class TestTTLinear(unittest.TestCase):
    
    def test_correctness_and_shapes(self):
        d1, d2, d3 = 8, 8, 12  # 768 total dim
        batch = 4
        x = torch.randn(batch, d1 * d2 * d3)
        
        for rank in [4, 6, 8]:
            layer = TTLinear(d1, d2, d3, rank)
            # Eval mode (uses cached W)
            layer.eval()
            out_eval = layer(x)
            self.assertEqual(out_eval.shape, (batch, 768))
            
            # Train mode (uses sequential or reconstructed pass)
            layer.train()
            out_train = layer(x)
            self.assertEqual(out_train.shape, (batch, 768))
            
            # Reconstructed dense matrix shape
            dense_W = layer.reconstruct()
            self.assertEqual(dense_W.shape, (768, 768))
            
            # Verify mathematical equivalence between direct forward pass and matmul with W
            with torch.no_grad():
                out_reconstructed = x @ dense_W
            self.assertTrue(torch.allclose(out_eval, out_reconstructed, atol=1e-5))

    def test_dynamic_rank_change(self):
        d1, d2, d3 = 8, 8, 12
        layer = TTLinear(d1, d2, d3, rank=8)
        
        initial_params = sum(p.numel() for p in layer.parameters())
        self.assertEqual(layer.rank, 8)
        
        # Shrink rank (truncate)
        layer.set_rank(4)
        self.assertEqual(layer.rank, 4)
        params_after_shrink = sum(p.numel() for p in layer.parameters())
        self.assertLess(params_after_shrink, initial_params)
        
        # Test lazy caching: caching is cleared after rank changes
        self.assertIsNone(layer._cached_W)
        layer.eval()
        x = torch.randn(2, 768)
        out1 = layer(x)
        self.assertIsNotNone(layer._cached_W)
        
        # Set same rank: should not clear cache
        cached_before = layer._cached_W
        layer.set_rank(4)
        self.assertIs(layer._cached_W, cached_before)
        
        # Grow rank (padding)
        layer.set_rank(6)
        self.assertEqual(layer.rank, 6)
        self.assertIsNone(layer._cached_W)
        out2 = layer(x)
        self.assertEqual(out2.shape, (2, 768))

    def test_flop_estimate_helper(self):
        stats = flop_estimate(8, 8, 12, rank=4, batch=1)
        self.assertGreater(stats["flop_reduction_x"], 1.0)
        self.assertGreater(stats["param_reduction_x"], 1.0)


class TestTTLinearHCLIntegration(unittest.TestCase):
    
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        # Balanced mode HCL
        self.hcl = HCL(self.model, self.tokenizer, mode="balanced")

    def test_hcl_grounding_matrix_initialization(self):
        # Grounding matrix should be TTLinear with correct dimension factors
        self.assertTrue(hasattr(self.hcl, "grounding_matrix"))
        self.assertIsInstance(self.hcl.grounding_matrix, TTLinear)
        self.assertEqual(self.hcl.grounding_matrix.d1, 8)
        self.assertEqual(self.hcl.grounding_matrix.d2, 8)
        self.assertEqual(self.hcl.grounding_matrix.d3, 12)
        self.assertEqual(self.hcl.grounding_matrix.rank, 8)

    def test_hcl_grounding_confidence_levels(self):
        # Default with no active beliefs should be 0.0 (low confidence)
        self.assertEqual(self.hcl.get_grounding_confidence(), 0.0)
        
        # Low confidence: grounding rank should be 8
        context = torch.zeros((1, 10))
        token_dist = torch.zeros((1, 10, 50))
        self.hcl.current_query_embedding = np.random.rand(768)
        self.hcl.current_query_text = "What is the capital of Spain?"
        
        # Run step to trigger projection
        self.hcl.on_generation_step(context, token_dist)
        self.assertEqual(self.hcl.grounding_matrix.rank, 8)
        
        # Add high weight belief node -> High confidence (>= 0.75)
        emb = np.random.rand(768)
        node_high = Node(NodeType.BELIEF, emb, "High weight node", weight=0.9)
        self.hcl.GBPS.add_belief(node_high)
        
        self.assertEqual(self.hcl.get_grounding_confidence(), 0.9)
        self.hcl.on_generation_step(context, token_dist)
        # Should dynamically lower rank to 4 (aggressive compression fast pass)
        self.assertEqual(self.hcl.grounding_matrix.rank, 4)
        
        # Add medium weight belief node -> Medium confidence (~0.55)
        node_med = Node(NodeType.BELIEF, emb, "Medium weight node", weight=0.2)
        self.hcl.GBPS.add_belief(node_med)
        
        # Average weight is (0.9 + 0.2) / 2 = 0.55
        self.assertAlmostEqual(self.hcl.get_grounding_confidence(), 0.55)
        self.hcl.on_generation_step(context, token_dist)
        # Should dynamically set rank to 6 (medium fidelity)
        self.assertEqual(self.hcl.grounding_matrix.rank, 6)

if __name__ == '__main__':
    unittest.main()
