import unittest
import numpy as np
from hail_core.lattice import StratifiedMemoryLattice
from hail_core.exceptions import HAILValidationError, HAILCapacityError

class TestLatticeValidation(unittest.TestCase):
    def setUp(self):
        self.lattice = StratifiedMemoryLattice(lattice_size=32, dim=8)

    def test_constructor_validation(self):
        with self.assertRaises(HAILValidationError):
            StratifiedMemoryLattice(lattice_size=5, dim=8)
        with self.assertRaises(HAILValidationError):
            StratifiedMemoryLattice(lattice_size=32, dim=0)
        with self.assertRaises(HAILValidationError):
            StratifiedMemoryLattice(lattice_size=32, dim=8, on_capacity="invalid_mode")

    def test_write_validation(self):
        # 1. Dimension mismatch
        with self.assertRaises(HAILValidationError):
            self.lattice.write(np.zeros(7))
        with self.assertRaises(HAILValidationError):
            self.lattice.write(np.zeros((8, 8)))  # Not 1D

        # 2. NaN / Inf checks
        nan_emb = np.zeros(8)
        nan_emb[0] = np.nan
        with self.assertRaises(HAILValidationError):
            self.lattice.write(nan_emb)

        inf_emb = np.zeros(8)
        inf_emb[0] = np.inf
        with self.assertRaises(HAILValidationError):
            self.lattice.write(inf_emb)

        # 3. Confidence range
        valid_emb = np.random.rand(8)
        with self.assertRaises(HAILValidationError):
            self.lattice.write(valid_emb, confidence=-0.1)
        with self.assertRaises(HAILValidationError):
            self.lattice.write(valid_emb, confidence=1.1)

    def test_recall_validation(self):
        with self.assertRaises(HAILValidationError):
            self.lattice.recall(np.zeros(7))
        with self.assertRaises(HAILValidationError):
            self.lattice.recall(np.zeros(8), k=0)

    def test_capacity_error(self):
        # Create a small lattice configured to raise on capacity bounds
        small_lattice = StratifiedMemoryLattice(lattice_size=16, dim=8, on_capacity="raise")
        
        # Write to all probes (the linear probe searches 64 slots, but since size is 16, it wraps)
        # Fill it up completely
        for i in range(16):
            emb = np.random.rand(8)
            small_lattice.write(emb)
            
        # Write one more should raise capacity error
        with self.assertRaises(HAILCapacityError):
            small_lattice.write(np.random.rand(8))

if __name__ == "__main__":
    unittest.main()
