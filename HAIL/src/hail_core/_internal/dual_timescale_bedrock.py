from .cosine import cosine_similarity

class DualTimescaleBedrock:
    def __init__(self):
        self.fast_bedrock = {}
        self.slow_bedrock = {}
        
    def _blend(self, bedrock_dict, addr, stratum, alpha=0.95):
        emb = stratum['retrieved_embedding']
        if addr not in bedrock_dict:
            bedrock_dict[addr] = {'consensus_embedding': emb.copy(), 'count': 1}
        else:
            old = bedrock_dict[addr]['consensus_embedding']
            bedrock_dict[addr]['consensus_embedding'] = old * alpha + emb * (1.0 - alpha)
            bedrock_dict[addr]['count'] += 1

    def recall(self, addr, stratum):
        self._blend(self.fast_bedrock, addr, stratum, alpha=0.95)
        
        if stratum.get('user_confirmed', False) or self.fast_bedrock[addr]['count'] > 10:
            self._blend(self.slow_bedrock, addr, stratum, alpha=0.99)
            
    def divergence(self, addr):
        if addr not in self.fast_bedrock or addr not in self.slow_bedrock:
            return 0.0
        fast = self.fast_bedrock[addr]['consensus_embedding']
        slow = self.slow_bedrock[addr]['consensus_embedding']
        return float(1.0 - cosine_similarity(fast, slow))
