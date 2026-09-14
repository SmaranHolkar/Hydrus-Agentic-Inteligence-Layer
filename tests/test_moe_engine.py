import unittest
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.engine import HydrusMoEEngine

class TestMoEEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("./test_moe_engine_cache")
        self.config = HydrusMoEConfig(ssd_cache_dir=self.test_dir)
        self.engine = HydrusMoEEngine(self.config)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_engine_load_and_forward(self):
        manifest = {
            "model_id": "qwen3-35b-a3b",
            "version": "1.0.2",
            "merkle_root": "",
            "experts": [{"id": i, "sha256": f"hash_{i}"} for i in range(16)]
        }
        loaded = self.engine.load_manifest(manifest)
        self.assertTrue(loaded)

        result = self.engine.forward("Explain the history of London", user_memories=["User prefers Python"])
        self.assertEqual(result["status"], "success")
        self.assertIn("telemetry", result)
        self.assertEqual(result["telemetry"]["active_model"], "qwen3-35b-a3b")
        self.assertIn("dispatch", result)
        self.assertIn("execution", result["telemetry"])
        self.assertIn("sparse_dispatch", result["telemetry"]["execution"])

    def test_status_reports_synthetic_execution_mode(self):
        manifest = {
            "model_id": "qwen3-35b-a3b",
            "version": "1.0.2",
            "merkle_root": "",
            "experts": [{"id": i, "sha256": f"hash_{i}"} for i in range(4)]
        }
        self.assertTrue(self.engine.load_manifest(manifest))
        status = self.engine.get_status()
        self.assertIn("execution", status)
        self.assertEqual(status["execution"]["weights_mode"], "synthetic_fixture")
        self.assertTrue(status["execution"]["is_synthetic_demo"])
        self.assertFalse(status["execution"]["streaming_generation_verified"])
        self.assertIn("claims", status)
        self.assertFalse(status["claims"]["can_claim_real_35b_streaming"])
        self.assertEqual(status["claims"]["reason"], "no_verified_full_local_blob_manifest")

if __name__ == "__main__":
    unittest.main()
