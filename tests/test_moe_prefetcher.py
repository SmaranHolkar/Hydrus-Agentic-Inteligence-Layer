import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.prefetcher import HAILPrefetcher
from hydrusmoe.oblivious_fetcher import fetch_experts_secure
from hydrusmoe.crypto import ManifestVerifier

class TestMoEPrefetcher(unittest.TestCase):

    def setUp(self):
        self.config = HydrusMoEConfig(dummy_batch_size=8)
        self.prefetcher = HAILPrefetcher(self.config)

    def test_domain_prediction(self):
        predicted = self.prefetcher.predict("Write a Python script for algorithm sorting")
        self.assertIn(7, predicted)

    def test_oblivious_batch_size_enforcement(self):
        required = {1, 2}
        predicted = {3, 4}
        pool = list(range(20))
        verifier = ManifestVerifier()
        
        fetched = fetch_experts_secure(
            required_expert_ids=required,
            predicted_expert_ids=predicted,
            common_pool=pool,
            config=self.config,
            verifier=verifier,
            fetch_fn=lambda batch: {eid: f"BLOB_{eid}".encode() for eid in batch}
        )
        self.assertEqual(len(fetched), 2)
        self.assertIn(1, fetched)
        self.assertIn(2, fetched)

    def test_prefetch_metrics_accounting(self):
        predicted = self.prefetcher.predict("python class debugging")
        self.assertTrue(len(predicted) >= 1)

        # Force a partial overlap so precision/recall are non-trivial and testable.
        overlap_pick = sorted(list(predicted))[:2]
        actual = overlap_pick + [31]
        self.prefetcher.update_actual(actual)

        metrics = self.prefetcher.get_metrics()
        self.assertEqual(metrics["queries_total"], 1)
        self.assertGreater(metrics["query_hit_rate"], 0.0)
        self.assertGreater(metrics["expert_precision"], 0.0)
        self.assertGreater(metrics["expert_recall"], 0.0)
        self.assertGreaterEqual(metrics["mean_jaccard"], 0.0)
        self.assertEqual(metrics["actual_experts_total"], len(set(actual)))

if __name__ == "__main__":
    unittest.main()
