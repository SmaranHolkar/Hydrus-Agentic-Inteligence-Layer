"""
HydrusMoE: Secure & Efficient Tiered Mixture-of-Experts Engine v2.0
"""

from .config import HydrusMoEConfig
from .crypto import AES256GCMEncryptor, ManifestVerifier, SecureWipe
from .tiered_storage import TieredStorage, Tier0Manager, Tier1Manager, Tier2Manager, Tier3Manager
from .oblivious_fetcher import fetch_experts_secure
from .router import SecureRouter, AdaptiveEntropyRouter
from .prefetcher import HAILPrefetcher
from .engine import HydrusMoEEngine
from .dynamic_dispatch import execute_optimized_sparse_moe
from .patcher import patch, HydrusMoEWrapper

__all__ = [
    "HydrusMoEConfig",
    "AES256GCMEncryptor",
    "ManifestVerifier",
    "SecureWipe",
    "TieredStorage",
    "Tier0Manager",
    "Tier1Manager",
    "Tier2Manager",
    "Tier3Manager",
    "fetch_experts_secure",
    "SecureRouter",
    "AdaptiveEntropyRouter",
    "HAILPrefetcher",
    "HydrusMoEEngine",
    "execute_optimized_sparse_moe",
    "patch",
    "HydrusMoEWrapper",
]

