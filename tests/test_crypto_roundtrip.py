import unittest
import tempfile
import os
import numpy as np
from hail_core.lattice import StratifiedMemoryLattice
from hail_core.exceptions import HAILIntegrityError, HAILValidationError

class TestLatticeCryptoRoundtrip(unittest.TestCase):
    def setUp(self):
        self.lattice = StratifiedMemoryLattice(lattice_size=256, dim=16)
        emb = np.random.rand(16).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.addr = self.lattice.write(emb, confidence=0.9, payload={"fact": "test"})
        self.tmp_path = tempfile.mktemp(suffix=".hcl")

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_save_and_load_roundtrip_encrypted(self):
        # Enforce passphrase length >= 8
        with self.assertRaises(HAILValidationError):
            self.lattice.save_to_disk(self.tmp_path, passphrase="short")

        self.lattice.save_to_disk(self.tmp_path, passphrase="test-pass-123", async_write=False)
        self.assertTrue(os.path.exists(self.tmp_path))

        with open(self.tmp_path, "rb") as f:
            header = f.read(8)
        self.assertEqual(header, b"HAILGCM1")  # confirms GCM format was written

        loaded = StratifiedMemoryLattice(lattice_size=256, dim=16)
        loaded.load_from_disk(self.tmp_path, passphrase="test-pass-123")

        self.assertTrue(loaded.occupied[self.addr])
        np.testing.assert_array_almost_equal(
            loaded.surface[self.addr, :16], self.lattice.surface[self.addr, :16], decimal=5
        )
        self.assertEqual(loaded.payloads[self.addr]["fact"], "test")

    def test_wrong_passphrase_raises(self):
        self.lattice.save_to_disk(self.tmp_path, passphrase="correct-passphrase", async_write=False)
        loaded = StratifiedMemoryLattice(lattice_size=256, dim=16)
        with self.assertRaises(HAILIntegrityError):
            loaded.load_from_disk(self.tmp_path, passphrase="wrong-passphrase")

    def test_plain_save_and_load_roundtrip(self):
        # Save without passphrase (plain format)
        self.lattice.save_to_disk(self.tmp_path, passphrase=None, async_write=False)
        self.assertTrue(os.path.exists(self.tmp_path))

        with open(self.tmp_path, "rb") as f:
            header = f.read(11)
        self.assertEqual(header, b"HAILPLAIN1\n")  # confirms plain format header

        loaded = StratifiedMemoryLattice(lattice_size=256, dim=16)
        loaded.load_from_disk(self.tmp_path, passphrase=None)

        self.assertTrue(loaded.occupied[self.addr])
        np.testing.assert_array_almost_equal(
            loaded.surface[self.addr, :16], self.lattice.surface[self.addr, :16], decimal=5
        )
        self.assertEqual(loaded.payloads[self.addr]["fact"], "test")

    def test_load_plain_requires_no_passphrase(self):
        self.lattice.save_to_disk(self.tmp_path, passphrase=None, async_write=False)
        loaded = StratifiedMemoryLattice(lattice_size=256, dim=16)
        # Even if passphrase is provided, loading plain format should ignore it or work
        loaded.load_from_disk(self.tmp_path, passphrase="some-passphrase")
        self.assertTrue(loaded.occupied[self.addr])

if __name__ == "__main__":
    unittest.main()
