import time
import numpy as np
from collections import deque

class TemporalBinding:
    def __init__(self, binding_window=300):
        self.binding_window = binding_window
        self.recent_encodings = deque()
        self.temporal_bindings = {}
    
    def encode(self, addr, embedding):
        now = time.time()
        bound = []
        for ts, other_addr, other_emb in self.recent_encodings:
            if now - ts < self.binding_window and other_addr != addr:
                dist = np.linalg.norm(embedding - other_emb)
                prox = 1.0 - (now - ts) / self.binding_window
                
                # Forward binding (current -> past)
                bound.append({
                    'addr': other_addr,
                    'temporal_proximity': prox,
                    'semantic_distance': dist
                })
                
                # Backward binding (past -> current)
                if other_addr not in self.temporal_bindings:
                    self.temporal_bindings[other_addr] = []
                
                self.temporal_bindings[other_addr].append({
                    'addr': addr,
                    'temporal_proximity': prox,
                    'semantic_distance': dist
                })
                
                self.temporal_bindings[other_addr] = sorted(
                    self.temporal_bindings[other_addr],
                    key=lambda x: x['temporal_proximity'] / (x['semantic_distance'] + 0.1),
                    reverse=True
                )[:5]
        
        if bound:
            self.temporal_bindings[addr] = sorted(
                bound, 
                key=lambda x: x['temporal_proximity'] / (x['semantic_distance'] + 0.1),
                reverse=True
            )[:5]
        
        self.recent_encodings.append((now, addr, embedding))
        
        while self.recent_encodings and now - self.recent_encodings[0][0] > self.binding_window * 2:
            self.recent_encodings.popleft()
            
    def retrieve_temporal_context(self, addr):
        return self.temporal_bindings.get(addr, [])
