import numpy as np
from .cosine import cosine_similarity

class AnchoredReEncoder:
    def __init__(self, max_drift=0.3):
        self.max_drift = max_drift
    
    def _weighted_blend(self, strata):
        weights = []
        embeddings = []
        for i, s in enumerate(strata):
            if s['stratum_type'] == 'encoding':
                w = 0.4
            else:
                w = 0.6 * (0.9 ** (len(strata) - i - 1))
            weights.append(w)
            embeddings.append(s['retrieved_embedding'])
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights /= weights.sum()
        new_embedding = np.average(embeddings, axis=0, weights=weights)
        return new_embedding

    def re_encode(self, strata, original_embedding):
        new_embedding = self._weighted_blend(strata)
        drift = 1.0 - cosine_similarity(new_embedding, original_embedding)
        
        if drift > self.max_drift:
            correction_strength = (drift - self.max_drift) / (1.0 - self.max_drift)
            new_embedding = (
                new_embedding * (1.0 - correction_strength) + 
                original_embedding * correction_strength
            )
            strata[-1]['stabilized'] = True
        
        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            return new_embedding / norm
        return new_embedding
