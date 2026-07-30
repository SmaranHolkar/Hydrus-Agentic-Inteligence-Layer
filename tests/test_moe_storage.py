import unittest
import os
import sys
import shutil
from pathlib import Path

# Add HAIL to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.crypto import AES256GCMEncryptor
from hydrusmoe.tiered_storage import TieredStorage, Tier0Manager, Tier1Manager

class TestMoEStorage(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("./test_moe_cache_dir")
        self.config = HydrusMoEConfig(
            vram_budget_gb=0.001,  # ~1MB budget for test
            ram_budget_gb=0.002,   # ~2MB budget for test
            ssd_cache_dir=self.test_dir
        )
        self.crypto = AES256GCMEncryptor(os.urandom(32))
        self.storage = TieredStorage(self.config, self.crypto)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tier1_lru_eviction(self):
        t1 = Tier1Manager(budget_gb=0.0001)  # tiny budget ~100KB
        payload1 = b"A" * 60000
        payload2 = b"B" * 60000
        
        t1.stage_expert(1, payload1)
        self.assertIsNotNone(t1.fetch_expert(1))
        
        # Staging payload2 should evict payload1
        t1.stage_expert(2, payload2)
        self.assertIsNotNone(t1.fetch_expert(2))
        self.assertIsNone(t1.fetch_expert(1))

    def test_ssd_encryption_and_retrieval(self):
        raw_weights = b"RAW_QUANTIZED_EXPERT_WEIGHTS_BUFFER_XYZ"
        self.storage.tier2.cache_expert(7, raw_weights)
        
        retrieved = self.storage.tier2.retrieve_expert(7)
        self.assertEqual(raw_weights, retrieved)

    def test_full_tiered_fetching(self):
        raw_weights = b"EXPERT_99_WEIGHTS"
        self.storage.tier2.cache_expert(99, raw_weights)
        
        fetched = self.storage.fetch_to_vram([99])
        self.assertIn(99, fetched)
        self.assertEqual(raw_weights, fetched[99])

if __name__ == "__main__":
    unittest.main()
