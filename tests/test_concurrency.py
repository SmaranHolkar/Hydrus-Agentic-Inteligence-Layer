import unittest
import threading
import numpy as np
import tempfile
import os
from hail_core.api import HAIL, HAILConfig

class TestLatticeConcurrency(unittest.TestCase):
    def setUp(self):
        self.tmp_path = tempfile.mktemp(suffix=".hcl")

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_concurrent_mutations(self):
        # Configure a thread-safe HAIL instance
        hail_inst = HAIL(lattice_size=1024, dim=16, thread_safe=True)
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
                    emb = np.random.rand(16).astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    # Write
                    addr = hail_inst.write(emb, confidence=0.8, payload={"thread": worker_id, "idx": i})
                    # Recall
                    hail_inst.recall(emb, k=3)
                    # Decay
                    hail_inst.thermal_decay(0.01)
            except Exception as e:
                errors.append(e)

        threads = []
        for w in range(5):
            t = threading.Thread(target=worker, args=(w,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent threads raised errors: {errors}")

        # Verify state consistency
        lattice = hail_inst.lattice
        with lattice._lock:
            occupied_count = int(np.sum(lattice.occupied))
            self.assertTrue(occupied_count > 0)
            self.assertEqual(len(lattice.payloads), occupied_count)
            self.assertEqual(len(lattice.strata), occupied_count)

if __name__ == "__main__":
    unittest.main()
