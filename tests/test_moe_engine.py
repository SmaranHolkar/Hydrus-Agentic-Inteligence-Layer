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

if __name__ == "__main__":
    unittest.main()
