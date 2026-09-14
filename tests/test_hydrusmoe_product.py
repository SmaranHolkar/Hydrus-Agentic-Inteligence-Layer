import unittest
import sys
import os
import torch
import torch.nn as nn

# Ensure repo root and packages are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
moe_src = os.path.join(repo_root, "packages", "hydrus-moe", "src")
hail_dir = os.path.join(repo_root, "HAIL")
if moe_src not in sys.path:
    sys.path.insert(0, moe_src)
if hail_dir not in sys.path:
    sys.path.insert(0, hail_dir)

import hydrusmoe
from hydrusmoe import HydrusMoEConfig, HydrusMoEEngine, patch, HydrusMoEWrapper


class MockMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.linear = nn.Linear(32, 100)

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None and "inputs_embeds" in kwargs:
            x = kwargs["inputs_embeds"]
        elif input_ids is not None:
            x = self.embedding(input_ids)
        else:
            x = torch.zeros((1, 4, 32))
        return self.linear(x)

    def generate(self, input_ids=None, max_new_tokens=4, **kwargs):
        # Mock autoregressive token generation
        if input_ids is None:
            input_ids = torch.tensor([[1, 2, 3]])
        batch_size = input_ids.shape[0]
        new_tokens = torch.randint(0, 100, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=-1)


class TestHydrusMoEProduct(unittest.TestCase):
    def setUp(self):
        self.raw_model = MockMoEModel()

    def test_patch_initialization(self):
        patched = patch(
            self.raw_model,
            vram_budget_mb=2048,
            ram_budget_mb=4096,
            enable_prefetch=True
        )

        self.assertIsInstance(patched, HydrusMoEWrapper)
        self.assertIsInstance(patched.engine, HydrusMoEEngine)
        self.assertEqual(patched.engine.config.vram_budget_gb, 2.0)
        self.assertEqual(patched.engine.config.ram_budget_gb, 4.0)

    def test_forward_interception(self):
        patched = patch(self.raw_model, vram_budget_mb=1024, ram_budget_mb=2048)
        input_ids = torch.tensor([[10, 20, 30, 40]])

        output = patched(input_ids=input_ids)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 4, 100))

    def test_generate_interception(self):
        patched = patch(self.raw_model, vram_budget_mb=1024, ram_budget_mb=2048)
        input_ids = torch.tensor([[5, 15, 25]])

        generated = patched.generate(input_ids=input_ids, max_new_tokens=3)
        self.assertEqual(generated.shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
