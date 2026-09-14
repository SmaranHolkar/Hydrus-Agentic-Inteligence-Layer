"""
HydrusMoE: 4-Tier Streamable Mixture-of-Experts Engine with Zero-Trust Security & Speculative Prefetching.
"""

from .config import HydrusMoEConfig
from .crypto import AES256GCMEncryptor, ManifestVerifier, SecureWipe
from .tiered_storage import TieredStorage
from .router import SecureRouter, AdaptiveEntropyRouter
from .prefetcher import HAILPrefetcher
from .oblivious_fetcher import fetch_experts_secure
from .dynamic_dispatch import execute_optimized_sparse_moe
from .engine import HydrusMoEEngine
from .patcher import patch, HydrusMoEWrapper

__version__ = "0.1.0"
__all__ = [
    "HydrusMoEConfig",
    "AES256GCMEncryptor",
    "ManifestVerifier",
    "secure_wipe_buffer",
    "TieredStorage",
    "SecureRouter",
    "AdaptiveEntropyRouter",
    "HAILPrefetcher",
    "fetch_experts_secure",
    "execute_optimized_sparse_moe",
    "HydrusMoEEngine",
    "patch",
    "HydrusMoEWrapper",
]
