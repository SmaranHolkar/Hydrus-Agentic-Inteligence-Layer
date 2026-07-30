import random
from typing import Set, Dict, List, Any
from .config import HydrusMoEConfig
from .crypto import ManifestVerifier

def fetch_experts_secure(
    required_expert_ids: Set[int],
    predicted_expert_ids: Set[int],
    common_pool: List[int],
    config: HydrusMoEConfig,
    verifier: ManifestVerifier,
    fetch_fn: Any
) -> Dict[int, bytes]:
    """
    Fetches required expert shards while obscuring activation targets from network observers.
    Pads batch with predicted experts and random dummy decoy experts up to config.dummy_batch_size.
    """
    batch = set(required_expert_ids) | set(predicted_expert_ids)
    
    # Fill remaining slots with dummy decoy experts
    available_dummies = [e for e in common_pool if e not in batch]
    while len(batch) < config.dummy_batch_size and available_dummies:
        dummy = random.choice(available_dummies)
        batch.add(dummy)
        available_dummies.remove(dummy)

    # Convert to list and shuffle to hide ordering
    shuffled_batch = list(batch)
    random.shuffle(shuffled_batch)

    # Execute batch fetch
    fetched_blobs = fetch_fn(shuffled_batch)

    # Return only the requested required experts
    results = {}
    for eid in required_expert_ids:
        if eid in fetched_blobs:
            results[eid] = fetched_blobs[eid]
            
    return results
