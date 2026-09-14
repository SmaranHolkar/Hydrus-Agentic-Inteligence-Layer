import random
from typing import Set, Dict, List, Any
from .config import HydrusMoEConfig
from .crypto import ManifestVerifier


def _ids_to_mask(ids: Set[int]) -> int:
    mask = 0
    for expert_id in ids:
        if expert_id >= 0:
            mask |= (1 << int(expert_id))
    return mask


def _mask_to_ids(mask: int) -> List[int]:
    out: List[int] = []
    while mask:
        lsb = mask & -mask
        out.append(lsb.bit_length() - 1)
        mask ^= lsb
    return out

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
    if bool(getattr(config, "enable_bitmask_fastpath", True)):
        common_mask = 0
        for expert_id in common_pool:
            common_mask |= (1 << int(expert_id))

        batch_mask = _ids_to_mask(required_expert_ids) | _ids_to_mask(predicted_expert_ids)
        needed = max(0, int(config.dummy_batch_size) - batch_mask.bit_count())
        if needed > 0:
            available_ids = _mask_to_ids(common_mask & ~batch_mask)
            random.shuffle(available_ids)
            for expert_id in available_ids[:needed]:
                batch_mask |= (1 << expert_id)

        shuffled_batch = _mask_to_ids(batch_mask)
        random.shuffle(shuffled_batch)
    else:
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
