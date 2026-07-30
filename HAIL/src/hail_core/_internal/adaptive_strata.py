class AdaptiveStrata:
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
        self.depth_budget = {}  # addr -> max_layers
    
    def get_max_depth(self, addr, current_access_pattern):
        access_rate = current_access_pattern.get('per_minute', 1)
        valence = current_access_pattern.get('avg_valence', 0.5)
        depth = int(2 + (access_rate * 2) + (valence * 4))
        return min(depth, self.max_depth * 2)  # cap at 16
    
    def compress(self, addr, strata, bedrock_callback):
        budget = self.depth_budget.get(addr, self.max_depth)
        if len(strata) <= budget:
            return strata
        
        scored = [(i, s['emotional_valence'] * (0.9 ** (len(strata) - i))) 
                  for i, s in enumerate(strata)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        keep_indices = set(i for i, _ in scored[:budget])
        compressed = [s for i, s in enumerate(strata) if i in keep_indices]
        
        discarded = [s for i, s in enumerate(strata) if i not in keep_indices]
        for s in discarded:
            bedrock_callback(addr, s)
        
        return compressed
