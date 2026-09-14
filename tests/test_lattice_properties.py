import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"
for p in (ROOT, HAIL_ROOT, HAIL_SRC):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

from hail_core.lattice import StratifiedMemoryLattice

class TestLatticeProperties(unittest.TestCase):
    def setUp(self):
        self.lattice = StratifiedMemoryLattice(lattice_size=256, dim=16)

    def test_semantic_address_invariant(self):
        # Generate 100 random embeddings and verify stable, bounded LPH mapping
        rng = np.random.default_rng(42)
        for _ in range(100):
            raw_emb = rng.standard_normal(16).astype(np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm == 0:
                continue
            emb = raw_emb / norm
            
            # Verify address is in bounds
            addr1 = self.lattice._semantic_address(emb)
            self.assertTrue(0 <= addr1 < self.lattice.lattice_size)

            # Verify stability (same embedding yields same address)
            addr2 = self.lattice._semantic_address(emb)
            self.assertEqual(addr1, addr2)

if __name__ == "__main__":
    unittest.main()
