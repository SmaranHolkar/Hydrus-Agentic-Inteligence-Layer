import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class HydrusMoEConfig:
    # Hardware budgets
    vram_budget_gb: float = 4.0          # Hard limit for expert cache in VRAM (GB)
    ram_budget_gb: float = 8.0           # Hard limit for warm cache in Host RAM (GB)
    ssd_cache_dir: Path = field(default_factory=lambda: Path(r"d:\HydrusOPT\models\hydrusmoe_cache"))
    
    # Security parameters
    encryption_key: Optional[bytes] = None  # 32-byte AES key (auto-generated if None)
    hardware_uuid: Optional[str] = None     # Hardware binding seed string
    dummy_batch_size: int = 8               # Real experts hidden among N dummy requests
    enable_pir: bool = False                # Private Information Retrieval (Enterprise tier)
    
    # Efficiency & Quantization parameters
    prefetch_lookahead: int = 3             # Layers ahead to prefetch
    quantization_gpu: str = "Q4_Marlin"     # Q4_Marlin, Q4_0, FP16
    quantization_ram: str = "Q4_0"          # Q4_0, Q2_K, FP4
    quantization_ssd: str = "Q2_K"          # Q2_K, FP4, MXFP4
    compression_ssd: str = "zstd:9"         # zstd compression level 9
    
    # HAIL integration
    hail_confidence_threshold: float = 0.75 # Minimum confidence to trigger predictive prefetch
    enable_hail_prefetch: bool = True       # Toggle memory-driven prefetching for ablation benchmarks
    enable_async_hail_prefetch: bool = True  # Run HAIL prefetch staging off the forward hot path
    async_prefetch_workers: int = 1          # Keep a single worker to avoid contention in synthetic benchmarks
    prefetch_max_experts: int = 2            # Upper bound of experts to stage per prompt for lower prefetch overhead
    enable_trit_prefetch_scheduler: bool = True  # Quantum-inspired 0/1/2 state scheduler for prefetch budgets
    trit_low_hit_rate: float = 0.25             # Drop to lower state when hit rate is below this threshold
    trit_high_hit_rate: float = 0.70            # Promote to higher state when hit rate is above this threshold
    trit_budget_state_0: int = 1                # Conservative prefetch budget in state 0
    trit_budget_state_1: int = 2                # Balanced prefetch budget in state 1
    trit_budget_state_2: int = 3                # Aggressive prefetch budget in state 2
    enable_bitmask_fastpath: bool = True        # Use bitwise expert-set math for lower Python overhead
    prefetch_ev_gain_ms: float = 1.25           # Estimated latency gain when a predicted expert is reused
    prefetch_ev_cost_ms: float = 1.10           # Estimated cost when a predicted expert is not reused
    prefetch_ev_margin_ms: float = 0.05         # Positive margin required for prefetch to proceed
    prefetch_pressure_penalty_ms: float = 1.50  # Penalty weight under memory and queue pressure
    prefetch_tail_p95_threshold_ms: float = 10.0  # Pause speculative prefetch above this rolling p95
    prefetch_tail_pause_queries: int = 12       # Number of forward calls to pause after a tail spike
    prefetch_latency_window: int = 48           # Rolling window size for tail guard
    cold_fetch_timeout_ms_synthetic: int = 0    # Never block on prefetch lock in synthetic mode
    cold_fetch_timeout_ms_real: int = 50        # Bounded wait for real-weight cold fetch under contention
    cdn_endpoint: str = "https://cdn.hydrusopt.com"

    # Adaptive routing controls
    enable_dynamic_routing: bool = True
    dynamic_min_top_k: int = 2
    dynamic_max_top_k: int = 4
    dynamic_entropy_threshold: float = 3.0
    dynamic_entropy_hysteresis: float = 0.25
    dynamic_entropy_ema_alpha: float = 0.8
    enable_sparse_dispatch_probe: bool = False  # Diagnostic probe only; disable in perf runs

    def __post_init__(self):
        if self.encryption_key is None:
            # Fallback random 32-byte key if not specified
            self.encryption_key = os.urandom(32)
        if isinstance(self.ssd_cache_dir, str):
            self.ssd_cache_dir = Path(self.ssd_cache_dir)
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
