import numpy as np

class HierarchicalAbyss:
    def __init__(self, n_regions=64, dim=64):
        self.n_regions = n_regions
        self.regions = {}  # region_id -> {'centroid', 'addresses', 'gravity'}
        self.region_centroids = np.zeros((n_regions, dim), dtype=np.float32)
        self.abyss_memories = {} # addr -> {'centroid', ...}
    
    def _nearest_region(self, centroid):
        if len(self.regions) < self.n_regions:
            return len(self.regions)
        
        distances = np.linalg.norm(self.region_centroids - centroid, axis=1)
        return int(np.argmin(distances))
        
    def add_to_abyss(self, addr, abyssal_centroid, metadata):
        self.abyss_memories[addr] = metadata
        self.abyss_memories[addr]['centroid'] = abyssal_centroid
        
        region_id = self._nearest_region(abyssal_centroid)
        if region_id not in self.regions:
            self.regions[region_id] = {'centroid': abyssal_centroid.copy(), 'addresses': [], 'gravity': 0.0}
            self.region_centroids[region_id] = abyssal_centroid.copy()
        
        self.regions[region_id]['addresses'].append(addr)
        self.regions[region_id]['gravity'] += 0.1
        
        old = self.regions[region_id]['centroid']
        n = len(self.regions[region_id]['addresses'])
        self.regions[region_id]['centroid'] = old * ((n-1)/n) + abyssal_centroid * (1/n)
        self.region_centroids[region_id] = self.regions[region_id]['centroid']
    
    def compute_gravity(self, query_embedding):
        if not self.regions:
            return 0.0
            
        region_distances = np.linalg.norm(self.region_centroids[:len(self.regions)] - query_embedding, axis=1)
        nearby_regions = np.where(region_distances < 2.0)[0]
        
        if len(nearby_regions) == 0:
            return 0.0
        
        gravity = 0.0
        for rid in nearby_regions:
            for addr in self.regions[rid]['addresses']:
                centroid = self.abyss_memories[addr]['centroid']
                dist = np.linalg.norm(centroid - query_embedding)
                gravity += 1.0 / (1.0 + dist * 10)
        
        return min(gravity / len(nearby_regions), 0.3)
    
    def summarize(self):
        """Compressed centroid + gravity map for safe serialization."""
        return {
            "regions": {
                str(k): {
                    "centroid": v["centroid"].tolist(),
                    "gravity": v["gravity"],
                    "count": len(v["addresses"])
                }
                for k, v in self.regions.items()
            }
        }
