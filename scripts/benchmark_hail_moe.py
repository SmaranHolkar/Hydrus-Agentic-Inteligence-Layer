from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import secrets
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"
for candidate in (ROOT, HAIL_ROOT, HAIL_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from hydrusmoe.config import HydrusMoEConfig
from hydrusmoe.engine import HydrusMoEEngine


CONVERSATIONS = [
    {
        "id": "code_debugging",
        "title": "Code Debugging Thread",
        "turns": [
            "I'm debugging a Python function that sometimes drops class state.",
            "The bug appears when the code path touches a callback and a cache.",
            "Could this be a routing issue in the algorithm or a data race?",
            "Explain how you would isolate the faulty function.",
            "Now assume the function also consults a memory store before routing.",
            "What telemetry would prove the cache warm-up helped?",
            "Would you prefer a smaller batch or a more conservative classifier?",
            "The same bug only happens after several turns in the session.",
            "Summarize the likely root cause in one paragraph.",
            "Give me a final checklist for verifying the fix.",
        ],
    },
    {
        "id": "history_research",
        "title": "History Research Thread",
        "turns": [
            "I need a concise overview of London during the industrial era.",
            "How did war change the city and its trade routes?",
            "Focus on the century-level shifts, not just dates.",
            "What kind of expert would you prefetch for that request?",
            "Now compare the empire period with the post-war period.",
            "Summarize the main economic transitions in three bullets.",
            "If the memory layer remembered the city theme, what should it load?",
            "Which part of the response should stay grounded in prior context?",
            "Give me a short factual answer with no speculation.",
            "End with a note on archival sources and records.",
        ],
    },
    {
        "id": "math_reasoning",
        "title": "Math Reasoning Thread",
        "turns": [
            "We need to reason about formulas, matrices, and equations.",
            "How would you estimate the complexity of a matrix factorization step?",
            "What kind of prefetch pattern fits calculation-heavy prompts?",
            "Now compare that to a light factual prompt.",
            "Can a memory cue reduce repeated expert fetches here?",
            "Describe the difference between a stable and unstable routing decision.",
            "If one turn becomes uncertain, should the system shrink the active set?",
            "What metric would best expose a stall in steady-state throughput?",
            "Give me a defensible latency summary.",
            "Finish with an explicit recommendation for prefetch policy.",
        ],
    },
    {
        "id": "creative_followup",
        "title": "Creative Follow-up Thread",
        "turns": [
            "We are writing a TV show and need a strong lead character.",
            "The protagonist should feel like an engineer and a reluctant fixer.",
            "How would a memory-aware system keep the character consistent?",
            "Now deepen the conflict without changing the premise.",
            "Give me a compact episode beat sheet.",
            "Would repeated references to the same show theme help routing?",
            "The setting should keep the same mood and vocabulary.",
            "Suggest one twist that preserves the core arc.",
            "Summarize the main character in two lines.",
            "Close with a pilot hook that does not feel generic.",
        ],
    },
    {
        "id": "mixed_session",
        "title": "Mixed Multi-turn Session",
        "turns": [
            "First, a quick factual answer about a planet.",
            "Then a follow-up about memory and retrieval.",
            "Now pivot to code architecture and caching.",
            "Bring back the factual thread and answer it directly.",
            "What would a good prefetch predictor learn from this mix?",
            "Which part of the session should be prioritized by the router?",
            "Summarize the session in a way that shows the context changed.",
            "If the user returns to the original topic, what should happen?",
            "Now ask for a short, stable answer.",
            "End with a practical note on cache reuse.",
        ],
    },
]

SYSTEMS = [
    {"name": "HAIL-prefetch-on", "hail_prefetch": True},
    {"name": "HAIL-prefetch-off", "hail_prefetch": False},
]


@dataclass
class RunSummary:
    system: str
    prompt_id: str
    run_number: int
    ttft_ms: float
    tok_per_sec: float
    p50_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    peak_vram_mb: float
    engine_vram_mb: float
    peak_ram_mb: float
    cold_start_ms: float
    avg_latency_ms: float
    estimated_tokens: int
    synthetic_demo: bool
    note: str = ""


def _prompt_token_estimate(text: str) -> int:
    return max(1, int(math.ceil(len(text.split()) * 1.25)))


def _build_context(turns: Sequence[str], current_index: int, window: int = 4) -> List[str]:
    start = max(0, current_index - window)
    return list(turns[start:current_index])


def _latency_stats(latencies_ms: Sequence[float]) -> Tuple[float, float, float]:
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    ordered = sorted(latencies_ms)
    p50 = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    p95 = ordered[p95_index]
    avg = statistics.mean(ordered)
    return float(p50), float(p95), float(avg)


def _process_memory_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cuda_peak_mb(reset: bool = False) -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    if reset:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return 0.0


def _build_local_blob_manifest(blob_dir: Path, expert_count: int, blob_size_mb: int) -> Dict[str, Any]:
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_size_bytes = max(1, blob_size_mb) * 1024 * 1024
    experts: List[Dict[str, Any]] = []

    for expert_id in range(expert_count):
        blob_path = blob_dir / f"expert_{expert_id:02d}.bin"
        if not blob_path.exists() or blob_path.stat().st_size != blob_size_bytes:
            blob_path.write_bytes(secrets.token_bytes(blob_size_bytes))
        experts.append(
            {
                "id": expert_id,
                "sha256": f"fixture_{expert_id:02d}",
                "local_blob_path": str(blob_path),
            }
        )

    return {
        "model_id": "qwen3-35b-a3b",
        "version": "1.0.2",
        "merkle_root": "",
        "experts": experts,
        "fixture": {
            "type": "local_blob_fixture",
            "expert_count": expert_count,
            "blob_size_mb": blob_size_mb,
        },
    }


def _run_engine_benchmark(
    system_name: str,
    hail_prefetch: bool,
    repeats: int,
    prompt_runs: int,
    seed: int,
    manifest: Dict[str, Any],
    prefetch_max_experts: int,
    enable_trit_scheduler: bool,
    trit_low_hit_rate: float,
    trit_high_hit_rate: float,
    trit_budget_state_0: int,
    trit_budget_state_1: int,
    trit_budget_state_2: int,
) -> List[RunSummary]:
    random.seed(seed)
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    config = HydrusMoEConfig(
        enable_dynamic_routing=True,
        enable_hail_prefetch=hail_prefetch,
        dummy_batch_size=8,
        prefetch_max_experts=prefetch_max_experts,
        enable_trit_prefetch_scheduler=enable_trit_scheduler,
        trit_low_hit_rate=trit_low_hit_rate,
        trit_high_hit_rate=trit_high_hit_rate,
        trit_budget_state_0=trit_budget_state_0,
        trit_budget_state_1=trit_budget_state_1,
        trit_budget_state_2=trit_budget_state_2,
    )
    engine = HydrusMoEEngine(config)

    cold_start_t0 = time.perf_counter()
    if not engine.load_manifest(manifest):
        raise RuntimeError(f"manifest load failed for {system_name}")
    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000.0

    summaries: List[RunSummary] = []

    for run_number in range(1, repeats + 1):
        run_rng = random.Random(seed + run_number)
        conversation_order = list(CONVERSATIONS)
        run_rng.shuffle(conversation_order)

        run_latencies: List[float] = []
        peak_ram_mb = _process_memory_mb()
        peak_vram_mb = _cuda_peak_mb(reset=True)
        engine_vram_mb = 0.0
        ttft_ms = 0.0
        estimated_tokens = 0
        cache_hit_rate = 0.0
        synthetic_demo = False

        for conv in conversation_order:
            turns = conv["turns"]
            for turn_index, prompt in enumerate(turns[:prompt_runs]):
                user_memories = _build_context(turns, turn_index, window=4)
                estimated_tokens += _prompt_token_estimate(prompt)
                t0 = time.perf_counter()
                result = engine.forward(prompt, user_memories=user_memories)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                run_latencies.append(elapsed_ms)
                if ttft_ms == 0.0:
                    ttft_ms = elapsed_ms
                peak_ram_mb = max(peak_ram_mb, _process_memory_mb())
                peak_vram_mb = max(peak_vram_mb, _cuda_peak_mb())
                telemetry = (result.get("telemetry", {}) or {})
                engine_vram_mb = max(engine_vram_mb, float(((telemetry.get("vram", {}) or {}).get("used_mb", 0.0))))
                synthetic_demo = synthetic_demo or bool(((telemetry.get("execution", {}) or {}).get("is_synthetic_demo", False)))
                cache_hit_rate = float(
                    ((telemetry.get("prefetcher", {}) or {}).get("metrics", {}) or {}).get("query_hit_rate", 0.0)
                )

        peak_vram_mb = max(peak_vram_mb, engine_vram_mb)

        p50_ms, p95_ms, avg_ms = _latency_stats(run_latencies)
        total_time_sec = max(sum(run_latencies) / 1000.0, 0.001)
        tok_per_sec = estimated_tokens / total_time_sec

        summaries.append(
            RunSummary(
                system=system_name,
                prompt_id="multi_turn_mix",
                run_number=run_number,
                ttft_ms=round(ttft_ms, 3),
                tok_per_sec=round(tok_per_sec, 3),
                p50_latency_ms=round(p50_ms, 3),
                p95_latency_ms=round(p95_ms, 3),
                cache_hit_rate=round(cache_hit_rate, 4),
                peak_vram_mb=round(peak_vram_mb, 3),
                engine_vram_mb=round(engine_vram_mb, 3),
                peak_ram_mb=round(peak_ram_mb, 3),
                cold_start_ms=round(cold_start_ms, 3),
                avg_latency_ms=round(avg_ms, 3),
                estimated_tokens=estimated_tokens,
                synthetic_demo=synthetic_demo,
                note="HAIL prefetch on" if hail_prefetch else "HAIL prefetch off",
            )
        )

    return summaries


def _write_csv(path: Path, rows: Sequence[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _aggregate(rows: Sequence[RunSummary]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[RunSummary]] = {}
    for row in rows:
        grouped.setdefault(row.system, []).append(row)

    aggregate: Dict[str, Dict[str, float]] = {}
    for system, items in grouped.items():
        aggregate[system] = {
            "runs": float(len(items)),
            "ttft_ms_median": float(statistics.median([i.ttft_ms for i in items])),
            "ttft_ms_std": float(statistics.pstdev([i.ttft_ms for i in items])) if len(items) > 1 else 0.0,
            "tok_per_sec_median": float(statistics.median([i.tok_per_sec for i in items])),
            "tok_per_sec_std": float(statistics.pstdev([i.tok_per_sec for i in items])) if len(items) > 1 else 0.0,
            "p50_latency_ms_median": float(statistics.median([i.p50_latency_ms for i in items])),
            "p95_latency_ms_median": float(statistics.median([i.p95_latency_ms for i in items])),
            "cache_hit_rate_median": float(statistics.median([i.cache_hit_rate for i in items])),
            "peak_vram_mb_max": float(max(i.peak_vram_mb for i in items)),
            "engine_vram_mb_max": float(max(i.engine_vram_mb for i in items)),
            "peak_ram_mb_max": float(max(i.peak_ram_mb for i in items)),
            "cold_start_ms_median": float(statistics.median([i.cold_start_ms for i in items])),
        }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HAIL MoE prefetch on/off and store a defensible CSV report.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeated runs per system (default: 5)")
    parser.add_argument("--prompt-runs", type=int, default=10, help="How many turns from each conversation to execute (default: 10)")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffle seed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmark_history" / "moe", help="Output directory for benchmark artifacts")
    parser.add_argument("--expert-count", type=int, default=32, help="Number of experts in the fixture manifest")
    parser.add_argument("--blob-size-mb", type=int, default=8, help="Per-expert local blob size in MB for the fixture manifest")
    parser.add_argument("--prefetch-max-experts", type=int, default=2, help="Base prefetch expert cap when trit scheduler is disabled")
    parser.add_argument("--disable-trit-scheduler", action="store_true", help="Disable the ternary 0/1/2 adaptive prefetch scheduler")
    parser.add_argument("--trit-low-hit-rate", type=float, default=0.25, help="Lower hit-rate threshold for downshifting trit prefetch state")
    parser.add_argument("--trit-high-hit-rate", type=float, default=0.70, help="Upper hit-rate threshold for upshifting trit prefetch state")
    parser.add_argument("--trit-budget-state-0", type=int, default=1, help="Prefetch budget for trit state 0")
    parser.add_argument("--trit-budget-state-1", type=int, default=2, help="Prefetch budget for trit state 1")
    parser.add_argument("--trit-budget-state-2", type=int, default=3, help="Prefetch budget for trit state 2")
    parser.add_argument("--dry-run-only", action="store_true", help="Write the benchmark plan without executing the run")
    args = parser.parse_args()

    if args.dry_run_only:
        print("Dry run only. No benchmark executed.")
        return 0

    all_rows: List[RunSummary] = []
    enable_trit_scheduler = not args.disable_trit_scheduler
    fixture_manifest = _build_local_blob_manifest(
        blob_dir=args.out_dir / "local_blob_fixture",
        expert_count=args.expert_count,
        blob_size_mb=args.blob_size_mb,
    )
    for system in SYSTEMS:
        print(f"Running {system['name']} ...")
        rows = _run_engine_benchmark(
            system_name=system["name"],
            hail_prefetch=system["hail_prefetch"],
            repeats=args.repeats,
            prompt_runs=args.prompt_runs,
            seed=args.seed,
            manifest=fixture_manifest,
            prefetch_max_experts=args.prefetch_max_experts,
            enable_trit_scheduler=enable_trit_scheduler,
            trit_low_hit_rate=args.trit_low_hit_rate,
            trit_high_hit_rate=args.trit_high_hit_rate,
            trit_budget_state_0=args.trit_budget_state_0,
            trit_budget_state_1=args.trit_budget_state_1,
            trit_budget_state_2=args.trit_budget_state_2,
        )
        all_rows.extend(rows)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    csv_path = out_dir / f"hail_moe_benchmark_{stamp}.csv"
    json_path = out_dir / f"hail_moe_benchmark_{stamp}.json"
    md_path = out_dir / f"hail_moe_benchmark_{stamp}.md"

    _write_csv(csv_path, all_rows)
    aggregate = _aggregate(all_rows)

    json_path.write_text(
        json.dumps(
            {
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "workload": {
                    "conversations": [
                        {"id": c["id"], "title": c["title"], "turns": len(c["turns"])} for c in CONVERSATIONS
                    ],
                    "repeats": args.repeats,
                    "prompt_runs": args.prompt_runs,
                    "manifest_fixture": fixture_manifest.get("fixture", {}),
                    "prefetch_strategy": {
                        "trit_scheduler_enabled": enable_trit_scheduler,
                        "prefetch_max_experts": args.prefetch_max_experts,
                        "trit_low_hit_rate": args.trit_low_hit_rate,
                        "trit_high_hit_rate": args.trit_high_hit_rate,
                        "trit_budget_state_0": args.trit_budget_state_0,
                        "trit_budget_state_1": args.trit_budget_state_1,
                        "trit_budget_state_2": args.trit_budget_state_2,
                    },
                },
                "aggregate": aggregate,
                "rows": [asdict(row) for row in all_rows],
                "competitor_status": [
                    {"name": "llama.cpp", "status": "not_run", "reason": "no baseline command configured"},
                    {"name": "MoE-Infinity", "status": "not_run", "reason": "no baseline command configured"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best = min(all_rows, key=lambda row: row.p95_latency_ms)
    worst = max(all_rows, key=lambda row: row.p95_latency_ms)

    md_lines = [
        "# HAIL MoE Benchmark Report",
        "",
        f"Created: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## What was measured",
        "",
        "- HAIL-prefetch on vs off on the same MoE engine and local-blob fixture workload.",
        "- Five multi-turn conversations, repeated across multiple runs.",
        "- TTFT, steady-state throughput, p50/p95 latency, cache hit rate, peak RAM, and peak VRAM.",
        "",
        "## Results",
        "",
        "| System | TTFT median (ms) | tok/sec median | p50 latency median (ms) | p95 latency median (ms) | cache hit rate median | peak VRAM max (MB) | engine VRAM max (MB) | synthetic demo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for system, stats in aggregate.items():
        demo_flag = "yes" if any(row.system == system and row.synthetic_demo for row in all_rows) else "no"
        md_lines.append(
            f"| {system} | {stats['ttft_ms_median']:.3f} | {stats['tok_per_sec_median']:.3f} | {stats['p50_latency_ms_median']:.3f} | {stats['p95_latency_ms_median']:.3f} | {stats['cache_hit_rate_median']:.4f} | {stats['peak_vram_mb_max']:.3f} | {stats['engine_vram_mb_max']:.3f} | {demo_flag} |"
        )

    md_lines.extend([
        "",
        "## Best and worst",
        "",
        f"- Best p95 latency: {best.system} at {best.p95_latency_ms:.3f} ms.",
        f"- Worst p95 latency: {worst.system} at {worst.p95_latency_ms:.3f} ms.",
        f"- Cold start on best run: {best.cold_start_ms:.3f} ms.",
        "",
        "## Competitor status",
        "",
        "- llama.cpp: not run in this environment because no baseline command was provided.",
        "- MoE-Infinity: not run in this environment because no baseline command was provided.",
        "",
        "## Interpretation",
        "",
        "- `tok_per_sec` is an estimated workload throughput derived from the fixed prompt set because this MoE engine benchmark measures forward-pass latency rather than autoregressive token generation.",
        "- The main defensible comparison here is HAIL-prefetch on vs off, because it isolates the memory-driven prefetch contribution.",
        "- If `synthetic demo` is yes, the engine is still using generated expert blobs rather than verified local model weights, so competitor-grade conclusions are not yet valid.",
        "- External competitor baselines can be wired in later by adding explicit command templates.",
        "",
        "## Files",
        "",
        f"- CSV: {csv_path.name}",
        f"- JSON: {json_path.name}",
    ])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {md_path}")
    print(f"Best p95 latency: {best.system} ({best.p95_latency_ms:.3f} ms)")
    print(f"Worst p95 latency: {worst.system} ({worst.p95_latency_ms:.3f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
