import os
import sys
import time
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.crypto import AES256GCMEncryptor
from hydrusmoe.engine import HydrusMoEEngine
from hydrusmoe.tiered_storage import Tier1Manager, TieredStorage


class TestMoERobustness(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp(prefix="moe_robustness_"))
        self.cache_dir = self.test_root / "cache"
        self.blob_dir = self.test_root / "blobs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def _build_local_blob_manifest(self, expert_count: int = 8, blob_size_bytes: int = 256 * 1024) -> Dict[str, Any]:
        experts: List[Dict[str, Any]] = []
        payload = os.urandom(blob_size_bytes)
        for expert_id in range(expert_count):
            blob_path = self.blob_dir / f"expert_{expert_id:02d}.bin"
            blob_path.write_bytes(payload)
            experts.append(
                {
                    "id": expert_id,
                    "sha256": f"robust_{expert_id:02d}",
                    "local_blob_path": str(blob_path.resolve()),
                }
            )

        return {
            "model_id": "robustness-local-blob-model",
            "version": "1.0.0",
            "experts": experts,
        }

    def test_empty_cache_cold_start_under_ten_seconds(self):
        config = HydrusMoEConfig(
            ssd_cache_dir=self.cache_dir,
            enable_sparse_dispatch_probe=False,
            enable_hail_prefetch=False,
        )
        engine = HydrusMoEEngine(config)
        manifest = self._build_local_blob_manifest(expert_count=8, blob_size_bytes=512 * 1024)

        self.assertTrue(engine.load_manifest(manifest))

        start = time.perf_counter()
        result = engine.forward("The capital of France is", user_memories=[])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertEqual(result["status"], "success")
        self.assertLess(elapsed_ms, 10_000.0, f"Cold start exceeded 10s: {elapsed_ms:.2f} ms")

    def test_corrupted_shard_recovery_continues(self):
        config = HydrusMoEConfig(ssd_cache_dir=self.cache_dir, enable_sparse_dispatch_probe=False)
        crypto = AES256GCMEncryptor(os.urandom(32))
        storage = TieredStorage(config, crypto)

        expert_id = 5
        original = b"REALISTIC_EXPERT_PAYLOAD" * 4096
        storage.tier2.cache_expert(expert_id, original)

        enc_path = self.cache_dir / f"expert_{expert_id}.enc"
        corrupted = bytearray(enc_path.read_bytes())
        corrupted[0] ^= 0x01
        enc_path.write_bytes(bytes(corrupted))

        fetched = storage.fetch_to_vram([expert_id], lock_timeout_ms=0)

        self.assertIn(expert_id, fetched)
        self.assertIsInstance(fetched[expert_id], (bytes, bytearray))
        self.assertEqual(storage.tier3.requests_count, 1)
        self.assertGreaterEqual(storage.ssd_misses, 1)

    def test_cache_budget_evicts_oldest_without_oom(self):
        # Tiny budget forces immediate eviction behavior.
        tier1 = Tier1Manager(budget_gb=0.0001)  # about 100KB

        payload_a = b"A" * 60_000
        payload_b = b"B" * 60_000
        payload_c = b"C" * 60_000

        self.assertTrue(tier1.stage_expert(1, payload_a))
        self.assertTrue(tier1.stage_expert(2, payload_b))
        self.assertTrue(tier1.stage_expert(3, payload_c))

        self.assertLessEqual(tier1.allocated_bytes, tier1.budget_bytes)
        self.assertIsNone(tier1.fetch_expert(1), "Oldest item should be evicted under pressure")
        self.assertIsNotNone(tier1.fetch_expert(3), "Newest item should remain cached")

    def test_concurrent_chats_shared_vram_no_crash(self):
        config = HydrusMoEConfig(
            ssd_cache_dir=self.cache_dir,
            enable_sparse_dispatch_probe=False,
            enable_hail_prefetch=True,
            async_prefetch_workers=1,
        )
        engine = HydrusMoEEngine(config)
        manifest = self._build_local_blob_manifest(expert_count=8, blob_size_bytes=512 * 1024)
        self.assertTrue(engine.load_manifest(manifest))

        errors = []
        results = []
        lock = threading.Lock()

        def run_chat(prompt: str, memories: List[str]):
            try:
                out = engine.forward(prompt, user_memories=memories)
                with lock:
                    results.append(out)
            except Exception as exc:  # pragma: no cover - test safeguard
                with lock:
                    errors.append(str(exc))

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(run_chat, "Write a Python function", ["chat_a_context"])
            fut_b = pool.submit(run_chat, "Explain London history", ["chat_b_context"])
            fut_a.result(timeout=20)
            fut_b.result(timeout=20)

        self.assertFalse(errors, f"Concurrent chat failures: {errors}")
        self.assertEqual(len(results), 2)
        for out in results:
            self.assertEqual(out.get("status"), "success")
            self.assertTrue(out.get("active_experts"))

        telemetry = engine.get_status()
        self.assertGreater(telemetry["vram"]["pinned_experts"], 0)


if __name__ == "__main__":
    unittest.main()
