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
    cdn_endpoint: str = "https://cdn.hydrusopt.com"

    def __post_init__(self):
        if self.encryption_key is None:
            # Fallback random 32-byte key if not specified
            self.encryption_key = os.urandom(32)
        if isinstance(self.ssd_cache_dir, str):
            self.ssd_cache_dir = Path(self.ssd_cache_dir)
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
