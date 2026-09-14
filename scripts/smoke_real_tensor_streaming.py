from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"
for candidate in (ROOT, HAIL_ROOT, HAIL_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.engine import HydrusMoEEngine


def _discover_safetensors(limit: int) -> List[Path]:
    models_root = ROOT / "models"
    if not models_root.exists():
        return []

    files: List[Path] = []
    for path in models_root.rglob("*.safetensors"):
        as_posix = path.as_posix()
        if "/.no_exist/" in as_posix:
            continue
        if path.is_file():
            files.append(path)

    files.sort(key=lambda p: p.stat().st_size)
    return files[: max(1, int(limit))]


def _build_manifest(paths: List[Path], model_id: str) -> Dict[str, Any]:
    experts: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        experts.append(
            {
                "id": idx,
                "sha256": f"real_local_{idx:02d}",
                "local_blob_path": str(path.resolve()),
                "source": path.name,
                "source_bytes": int(path.stat().st_size),
            }
        )

    return {
        "model_id": model_id,
        "version": "real_tensor_smoke_v1",
        "experts": experts,
        "fixture": {
            "type": "real_safetensors_local",
            "count": len(experts),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HydrusMoE real-tensor streaming smoke test")
    parser.add_argument("--max-files", type=int, default=2, help="How many local safetensors files to stream")
    parser.add_argument("--model-id", default="real-local-safetensors-smoke", help="Manifest model id")
    parser.add_argument("--turns", type=int, default=3, help="Number of forward calls")
    args = parser.parse_args()

    files = _discover_safetensors(args.max_files)
    if not files:
        print("No local .safetensors files found under models/.")
        return 2

    manifest = _build_manifest(files, args.model_id)

    cfg = HydrusMoEConfig(
        enable_hail_prefetch=True,
        enable_sparse_dispatch_probe=False,
    )
    engine = HydrusMoEEngine(cfg)

    started = time.perf_counter()
    ok = engine.load_manifest(manifest)
    load_ms = (time.perf_counter() - started) * 1000.0
    if not ok:
        print("Manifest load failed.")
        return 3

    prompts = [
        "The capital of France is",
        "Write a short Python function to reverse a list.",
        "Explain matrix multiplication in one paragraph.",
    ]

    forward_times_ms: List[float] = []
    outputs: List[Dict[str, Any]] = []
    for i in range(max(1, int(args.turns))):
        prompt = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        out = engine.forward(prompt)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        forward_times_ms.append(dt_ms)
        outputs.append(
            {
                "turn": i + 1,
                "active_experts": out.get("active_experts", []),
                "dispatch_reason": out.get("dispatch", {}).get("reason", "unknown"),
                "forward_ms": round(dt_ms, 3),
            }
        )

    telemetry = engine.get_status()
    summary = {
        "status": "ok",
        "weights_mode": telemetry.get("execution", {}).get("weights_mode"),
        "loaded_experts": telemetry.get("execution", {}).get("loaded_experts"),
        "is_synthetic_demo": telemetry.get("execution", {}).get("is_synthetic_demo"),
        "load_manifest_ms": round(load_ms, 3),
        "forward_ms": {
            "min": round(min(forward_times_ms), 3),
            "median": round(sorted(forward_times_ms)[len(forward_times_ms) // 2], 3),
            "max": round(max(forward_times_ms), 3),
        },
        "prefetch_hit_rate": telemetry.get("prefetcher", {}).get("hail_hit_rate", 0.0),
        "tier_vram_mb": telemetry.get("vram", {}).get("used_mb", 0.0),
        "tier_ram_mb": telemetry.get("ram", {}).get("used_mb", 0.0),
        "safetensors": [str(p) for p in files],
        "turns": outputs,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
