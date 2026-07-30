# import os
# import re
# import sys
# import json
# import time
# import uuid
# import hashlib
# import hmac
# from enum import Enum
# from typing import Tuple, List, Dict, Any, Optional
# import numpy as np
# import torch

# from stratified_memory import StratifiedMemoryLattice

# def is_personal_query(query: str) -> bool:
#     if not query:
#         return False
#     q = query.lower()
#     patterns = {
#         "favourite": ["colour", "color", "food", "movie", "song", "subject", "topic"],
#         "favorite": ["colour", "color", "food", "movie", "song", "subject", "topic"],
#         "my": ["name", "age", "birthday", "goal", "dream", "status", "struggle"],
#         "what did i": ["say", "ask", "tell you"],
#         "do you remember": [],
#         "who am i": [],
#     }
#     for trigger, contexts in patterns.items():
#         if trigger in q:
#             if not contexts or any(c in q for c in contexts):
#                 return True
#     return False

# _SOCIAL_SIGNALS = {
#     "my favourite", "my favorite", "i like", "i love", "i want", 
#     "i've", "i have", "i flew", "i went", "i think", "i feel",
#     "!", "!!", "!!!",  # excessive enthusiasm = social, not factual
# }

# def is_social_utterance(text: str) -> bool:
#     """Returns True if the user is sharing experiences, opinions, not asking for facts."""
#     if not text:
#         return False
#     t = text.lower()
#     return any(sig in t for sig in _SOCIAL_SIGNALS)

# def is_question(text: str) -> bool:
#     """Returns True if the input text represents a question."""
#     if not text:
#         return False
#     t = text.strip().lower()
#     if t.endswith("?"):
#         return True
#     words = t.split()
#     if not words:
#         return False
#     question_starters = {
#         "what", "when", "where", "who", "why", "how", "which", 
#         "did", "do", "does", "can", "could", "would", "will", 
#         "is", "are", "was", "were", "have", "has", "had"
#     }
#     first_word = re.sub(r'[^\w]', '', words[0])
#     return first_word in question_starters

# def should_fact_check(output_text: str, user_query: str) -> bool:
#     """Only fact-check statements that make objective claims, avoiding personal fact-checking."""
#     if is_personal_query(user_query):
#         return False
#     if is_social_utterance(user_query):
#         return False
#     if any(phrase in output_text.lower() for phrase in ["your favourite", "your name", "you told me", "your status", "your struggle"]):
#         return False
#     # Only check sentences with objective verbs
#     return any(verb in output_text.lower() for verb in [" is ", " was ", " are ", " were ", " has ", " had "])
# class NodeType(Enum):
#     EPISODIC        = "EPISODIC"
#     SEMANTIC        = "SEMANTIC"
#     BELIEF          = "BELIEF"
#     AXIOM           = "AXIOM"            # Bedrock truth; near-zero decay, demotion requires user challenge
#     QUESTION_ANCHOR = "QUESTION_ANCHOR"  # Always-stored user question nodes


# # Governance constants for Axiom promotion/demotion (§3.6)
# AXIOM_PROPERTIES: dict = {
#     'promotion_threshold':   100,       # access_count required for BELIEF → AXIOM
#     'demotion_threshold':    3,         # contradictions before AXIOM → BELIEF
#     'demotion_requires':     'user_explicit',
#     'max_axiom_lifetime':    86400 * 30,  # 30 days in seconds
#     'bias_audit_frequency':  100,
#     'source_diversity_min':  3,         # unique provenance sources required
# }

# class Node:
#     __slots__ = (
#         "id", "node_type", "embedding", "raw_summary", "weight", "state",
#         "created_at", "last_accessed", "access_count", "cluster_id",
#         "provenance", "contradictions", "confidence", "utility_score",
#         "sml_addr", "epistemic_divergence", "temporal_bindings",
#         "verification_count",   # §3.6: only increments on explicit user confirmation
#         "abstraction_score",    # §3.6: measures query-context diversity across strata
#     )
#     def __init__(
#         self,
#         node_type: NodeType,
#         embedding: np.ndarray,
#         raw_summary: str,
#         weight: float = 0.8,
#         state: int = 1,
#         created_at: int = 0,
#         last_accessed: int = 0,
#         access_count: int = 0,
#         cluster_id: Optional[int] = None,
#         provenance: Optional[List[str]] = None,
#         contradictions: Optional[List[str]] = None,
#         confidence: Optional[float] = None,
#         utility_score: float = 0.5,
#         id: Optional[str] = None,
#         verification_count: int = 0,
#         abstraction_score: float = 0.0
#     ):
#         self.id = id if id is not None else str(uuid.uuid4())
#         self.node_type = node_type
#         self.embedding = embedding
#         self.raw_summary = raw_summary
#         self.weight = weight
#         self.state = state  # 0: Dormant, 1: Active, 2: Deferred
#         self.created_at = created_at
#         self.last_accessed = last_accessed
#         self.access_count = access_count
#         self.cluster_id = cluster_id
#         self.provenance = provenance if provenance is not None else []
#         self.contradictions = contradictions if contradictions is not None else []
#         self.confidence = confidence if confidence is not None else (0.85 if node_type == NodeType.BELIEF else 0.5)
#         self.utility_score = utility_score
#         self.verification_count = verification_count  # §3.6: incremented only on user_confirmed recall
#         self.abstraction_score = abstraction_score

# class VectorIndex:
#     """Interface to ensure simple upgrade to external libraries like hnswlib later."""
#     def query(self, embedding: np.ndarray, k: int) -> List[Node]:
#         raise NotImplementedError
#     def update(self, node: Node) -> None:
#         raise NotImplementedError
#     def remove(self, node: Node) -> None:
#         raise NotImplementedError

# class KNNIndex(VectorIndex):
#     """
#     Pure Python K-NN vector index.
#     Includes a deterministic Locality-Sensitive Hashing (LSH) layer to accelerate queries
#     when the index grows past 100 nodes, keeping performance scalable.
#     """
#     def __init__(self, dim: int = 768, lsh_bits: int = 4):
#         self.dim = dim
#         self.nodes: List[Node] = []
#         self.lsh_bits = lsh_bits
#         # Deterministically generate random hyperplanes using a fixed-seed generator
#         rng = np.random.default_rng(42)
#         self.hyperplanes = rng.standard_normal((lsh_bits, dim))
#         for i in range(lsh_bits):
#             norm = np.linalg.norm(self.hyperplanes[i])
#             if norm > 0:
#                 self.hyperplanes[i] /= norm
#         self.buckets: Dict[int, List[Node]] = {i: [] for i in range(2**lsh_bits)}

#     def _get_bucket_key(self, embedding: np.ndarray) -> int:
#         projections = np.dot(self.hyperplanes, embedding)
#         key = 0
#         for i, val in enumerate(projections):
#             if val >= 0:
#                 key |= (1 << i)
#         return key

#     def query(self, embedding: np.ndarray, k: int) -> List[Node]:
#         if not self.nodes:
#             return []
        
#         # If dataset size is small, scan everything
#         if len(self.nodes) <= 100:
#             candidates = self.nodes
#         else:
#             # Query-directed LSH lookup
#             key = self._get_bucket_key(embedding)
#             candidates = self.buckets.get(key, [])
            
#             # If search bucket has too few elements, fall back to neighbors (Hamming distance 1)
#             if len(candidates) < k:
#                 candidates = list(candidates)
#                 checked_keys = {key}
#                 for bit in range(self.lsh_bits):
#                     neighbor_key = key ^ (1 << bit)
#                     if neighbor_key not in checked_keys:
#                         candidates.extend(self.buckets.get(neighbor_key, []))
#                         checked_keys.add(neighbor_key)
            
#             # Ultimate safety fallback: if neighbors are also empty/sparse, check all nodes
#             if len(candidates) < k:
#                 candidates = self.nodes

#         # Calculate cosine similarity (embeddings must be L2 normalized)
#         similarities = []
#         for node in candidates:
#             # dot product is equivalent to cosine similarity since embeddings are normalized
#             sim = float(np.dot(node.embedding, embedding))
#             similarities.append((sim, node))
            
#         similarities.sort(key=lambda x: x[0], reverse=True)
#         return [node for _, node in similarities[:k]]

#     def update(self, node: Node) -> None:
#         # Normalize node embedding to ensure correct cosine similarity calculations
#         norm = np.linalg.norm(node.embedding)
#         if norm > 0:
#             node.embedding = node.embedding / norm

#         if node not in self.nodes:
#             self.nodes.append(node)
#             key = self._get_bucket_key(node.embedding)
#             self.buckets[key].append(node)

#     def remove(self, node: Node) -> None:
#         if node in self.nodes:
#             self.nodes.remove(node)
#             key = self._get_bucket_key(node.embedding)
#             if node in self.buckets[key]:
#                 self.buckets[key].remove(node)

# class MemoryGraph:
#     def __init__(self):
#         self.nodes: Dict[str, Node] = {}
#         self.clusters: Dict[int, List[str]] = {}  # cluster_id -> list of node IDs

#     def insert(self, node: Node) -> None:
#         self.nodes[node.id] = node

#     def remove(self, node: Node) -> None:
#         if node.id in self.nodes:
#             del self.nodes[node.id]
#         for cid in list(self.clusters.keys()):
#             if node.id in self.clusters[cid]:
#                 self.clusters[cid].remove(node.id)
#             if not self.clusters[cid]:
#                 del self.clusters[cid]

#     def __iter__(self):
#         return iter(self.nodes.values())

#     def __len__(self):
#         return len(self.nodes)

# class GroundedBeliefPathSearch:
#     def __init__(self, embedding_dim: int = 768):
#         self.active_beliefs: List[Node] = []
#         self.embedding_dim = embedding_dim
#         self.new_belief_flag = False

#     def get_active_beliefs(self) -> List[Node]:
#         return self.active_beliefs

#     def get_active_belief_embedding(self) -> np.ndarray:
#         if not self.active_beliefs:
#             return np.zeros(self.embedding_dim)
#         embs = [node.embedding for node in self.active_beliefs]
#         mean_emb = np.mean(embs, axis=0)
#         norm = np.linalg.norm(mean_emb)
#         if norm > 0:
#             mean_emb /= norm
#         return mean_emb

#     def current_embedding(self) -> np.ndarray:
#         return self.get_active_belief_embedding()

#     def detect_new_belief(self) -> bool:
#         flag = self.new_belief_flag
#         self.new_belief_flag = False
#         return flag

#     def add_belief(self, node: Node) -> None:
#         self.active_beliefs.append(node)
#         self.new_belief_flag = True
#         if len(self.active_beliefs) > 5:
#             self.active_beliefs.pop(0)

# class BeliefTransitionModel:
#     def __init__(self):
#         self.transitions: Dict[Tuple[int, int], int] = {}
#         self.last_cluster_ids: List[int] = []

#     def update(self, active_beliefs: List[Node]) -> None:
#         if not active_beliefs:
#             return
#         current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
#         if not current_clusters:
#             return
#         if self.last_cluster_ids:
#             for prev in self.last_cluster_ids:
#                 for curr in current_clusters:
#                     key = (prev, curr)
#                     self.transitions[key] = self.transitions.get(key, 0) + 1
#         self.last_cluster_ids = current_clusters

#     def predict_next(self, active_beliefs: List[Node], HMG: MemoryGraph, top_k: int = 3) -> List[int]:
#         if not active_beliefs:
#             return []
#         current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
#         if not current_clusters:
#             return []
        
#         scores: Dict[int, int] = {}
#         for prev in current_clusters:
#             for (src, dest), count in self.transitions.items():
#                 if src == prev:
#                     scores[dest] = scores.get(dest, 0) + count
#         if not scores:
#             return []
#         sorted_clusters = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         return [c for c, _ in sorted_clusters[:top_k]]

# class WarmCache:
#     def __init__(self):
#         self.cached_nodes: List[Node] = []

#     def stage(self, candidates: List[Node]) -> None:
#         self.cached_nodes = candidates

#     def check(self, query_emb: np.ndarray, threshold: float = 0.85) -> Optional[List[Node]]:
#         if not self.cached_nodes:
#             return None
#         relevant = []
#         for node in self.cached_nodes:
#             sim = float(np.dot(node.embedding, query_emb))
#             if sim >= threshold:
#                 relevant.append(node)
#         return relevant if relevant else None

#     def clear(self) -> None:
#         self.cached_nodes = []

# class ModeController:
#     def __init__(self, mode: str = "balanced"):
#         self.λ = 0.15
#         self.θ_write = 0.65
#         self.N_collapse = 5
#         self.N_prune = 25
#         self.uncertainty_threshold = 0.75
#         self.apply_mode(mode)

#     def apply_mode(self, mode: str) -> None:
#         self.mode_name = mode
        
#         # Base guard flags (default off / single vote)
#         self.verify_votes = 1
#         self.no_votes_to_block = 1
#         self.verify_high_conf_bypass = 1.0  # disabled by default

#         if mode == "fast":
#             self.λ = 0.30
#             self.θ_write = 0.75
#             self.N_collapse = 3
#             self.N_prune = 10
#             self.uncertainty_threshold = 0.85
#         elif mode == "balanced":
#             self.λ = 0.15
#             self.θ_write = 0.65
#             self.N_collapse = 5
#             self.N_prune = 25
#             self.uncertainty_threshold = 0.75
#         elif mode == "safe":
#             self.λ = 0.05
#             self.θ_write = 0.55
#             self.N_collapse = 10
#             self.N_prune = 50
#             self.uncertainty_threshold = 0.65
#         elif mode == "eval":
#             self.λ = 0.00
#             self.θ_write = 0.50
#             self.N_collapse = 999999
#             self.N_prune = 999999
#             self.uncertainty_threshold = 0.70
            
#             # Full suite of HydrusOPT guard flags
#             self.verify_votes = 3
#             self.no_votes_to_block = 2
#             self.verify_high_conf_bypass = 0.90
#         elif mode == "persistent":
#             self.λ = 0.10
#             self.θ_write = 0.65
#             self.N_collapse = 7
#             self.N_prune = 30
#             self.uncertainty_threshold = 0.75

#     def adapt_to_profile(self, profile: "CognitiveProfile") -> None:
#         if profile and profile.entropy_baseline and getattr(profile, "session_count", 0) > 10:
#             self.θ_write = profile.entropy_baseline + 0.1
#             self.uncertainty_threshold = profile.entropy_baseline + 0.2

# class CryptoHelper:
#     """
#     NOTE: This is a SHA-256-based stream cipher (keystream generated via PBKDF2), NOT AES.
#     Used because cryptography/pycryptodome are unavailable in the workspace environment.
#     Replace with standard AES-CTR when dependencies allow.
#     """
#     @staticmethod
#     def derive_key(password: str, salt: bytes, iterations: int = 10000) -> bytes:
#         return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=32)

#     @staticmethod
#     def encrypt(data: str, password: str) -> bytes:
#         salt = os.urandom(16)
#         key = CryptoHelper.derive_key(password, salt)
#         raw_bytes = data.encode("utf-8")
        
#         # Keystream generation using sha256 in CTR mode
#         ciphertext = bytearray()
#         counter = 0
#         for i in range(0, len(raw_bytes), 32):
#             block_key = hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest()
#             chunk = raw_bytes[i:i+32]
#             for b_idx, b in enumerate(chunk):
#                 ciphertext.append(b ^ block_key[b_idx])
#             counter += 1
            
#         # SHA-256 HMAC check for integrity verification
#         mac = hashlib.sha256(key + salt + ciphertext).digest()
#         return salt + mac + bytes(ciphertext)

#     @staticmethod
#     def decrypt(encrypted_data: bytes, password: str) -> str:
#         if len(encrypted_data) < 48:
#             raise ValueError("Invalid encrypted profile data: payload is too short.")
#         salt = encrypted_data[:16]
#         mac = encrypted_data[16:48]
#         ciphertext = encrypted_data[48:]
        
#         key = CryptoHelper.derive_key(password, salt)
        
#         expected_mac = hashlib.sha256(key + salt + ciphertext).digest()
#         if not hmac.compare_digest(mac, expected_mac):
#             raise ValueError("Integrity check failed: invalid profile passphrase or data corruption.")
            
#         decrypted = bytearray()
#         counter = 0
#         for i in range(0, len(ciphertext), 32):
#             block_key = hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest()
#             chunk = ciphertext[i:i+32]
#             for b_idx, b in enumerate(chunk):
#                 decrypted.append(b ^ block_key[b_idx])
#             counter += 1
            
#         return decrypted.decode("utf-8")

# class UserFacts:
#     """
#     Lightweight keyword-first registry inside HCL to track direct user preferences and facts.
#     E.g., "my favorite color is blue" or "I struggle with integration by parts".
#     """
#     def __init__(self):
#         self.facts: Dict[str, dict] = {}  # key -> {"value": ..., "confidence": ..., "source_turn": ..., "timestamp": ...}

#     def add_fact(self, key: str, value: Any, confidence: float = 1.0, source_turn: int = 0) -> None:
#         self.facts[key] = {
#             "value": value,
#             "confidence": confidence,
#             "source_turn": source_turn,
#             "timestamp": time.time()
#         }

#     def extract_from_turn(self, user_text: str) -> List[Tuple[str, Any]]:
#         extracted = []
#         text = user_text.strip().lower()
        
#         # 1. Pattern: "my favorite / favourite [noun] is [value]"
#         m1 = re.match(r"\bmy\s+favou?rite\s+([a-z0-9_\-\s]{3,30})\s+(?:is|was)\s+([a-z0-9_\-\s\.\'\"]{1,50})", text)
#         if m1:
#             key = m1.group(1).strip().replace(" ", "_")
#             val = m1.group(2).strip()
#             if "favourite" not in key and "favorite" not in key:
#                 key = f"favourite_{key}"
#             extracted.append((key, val))
            
#         # 2. Pattern: "my [noun] is [value]" (excluding common non-fact words)
#         m2 = re.match(r"\bmy\s+([a-z0-9_\-\s]{3,30})\s+(?:is|was)\s+([a-z0-9_\-\s\.\'\"]{1,50})", text)
#         if m2 and not m1:
#             key_raw = m2.group(1).strip()
#             exclude_keys = {"opinion", "answer", "question", "guess", "turn", "prompt", "idea", "thought"}
#             if key_raw not in exclude_keys:
#                 key = key_raw.replace(" ", "_")
#                 val = m2.group(2).strip()
#                 extracted.append((key, val))
                
#         # 3. Pattern: "i struggle with [value]" or "i have trouble with [value]"
#         m3 = re.search(r"\bi\s+(?:struggle\s+with|have\s+trouble\s+with|find\s+.*?\s+difficult)\s+([a-z0-9_\-\s\.\'\"]{3,100})", text)
#         if m3:
#             val = m3.group(1).strip()
#             extracted.append(("struggle_topic", val))
            
#         # 4. Pattern: "i am [value]" (e.g. "i am a student", excluding feelings)
#         m4 = re.match(r"\bi\s+am\s+([a-z0-9_\-\s\.\'\"]{3,100})", text)
#         if m4:
#             val = m4.group(1).strip()
#             exclude_vals = {"ready", "sure", "sorry", "fine", "ok", "okay", "happy", "sad", "tired"}
#             if not any(val.startswith(ev) for ev in exclude_vals):
#                 extracted.append(("user_status", val))

#         return extracted

#     def retrieve_relevant_facts(self, query: str) -> List[str]:
#         """Scans the query for keywords matching known fact keys or values, returning matched facts."""
#         query_words = set(re.findall(r"\w+", query.lower()))
#         matched_lines = []
        
#         for key, info in self.facts.items():
#             key_words = set(key.split("_"))
            
#             has_color_spelling = ("color" in query_words or "colour" in query_words) and ("color" in key_words or "colour" in key_words)
#             has_fav_spelling = ("favorite" in query_words or "favourite" in query_words) and ("favorite" in key_words or "favourite" in key_words)
            
#             overlap = key_words.intersection(query_words)
#             overlap_ratio = len(overlap) / len(key_words) if key_words else 0.0
            
#             val_str = str(info.get("value", "")).lower()
#             val_words = set(re.findall(r"\w+", val_str))
#             value_matched = False
#             if val_words:
#                 if len(val_words) == 1:
#                     value_matched = (list(val_words)[0] in query_words)
#                 else:
#                     val_overlap = val_words.intersection(query_words)
#                     value_matched = (len(val_overlap) / len(val_words) >= 0.5) or (val_str in query.lower())

#             if overlap_ratio >= 0.5 or (has_color_spelling and has_fav_spelling) or (key in query.lower().replace(" ", "_")) or value_matched:
#                 meta_name = key.replace("_", " ")
#                 if "favourite" in meta_name and "favorite" in query.lower():
#                     meta_name = meta_name.replace("favourite", "favorite")
#                 matched_lines.append(f"User fact: The user's {meta_name} is '{info['value']}'. (Address the user directly using 'your' or 'you'.)")
                
#         return matched_lines

# class CognitiveProfile:
#     def __init__(self, user_id: str):
#         self.user_id = user_id
#         self.recurring_clusters: List[int] = []
#         self.expertise_levels: Dict[str, float] = {}
#         self.communication_style: str = "BALANCED"  # TERSE | BALANCED | VERBOSE
#         self.entropy_baseline: float = 0.5
#         self.error_patterns: List[str] = []
#         self.preferred_modes: Dict[str, str] = {}
#         self.session_count: int = 0
#         self.total_inference_steps: int = 0
#         self.user_facts: Dict[str, Any] = {}

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "user_id": self.user_id,
#             "recurring_clusters": self.recurring_clusters,
#             "expertise_levels": self.expertise_levels,
#             "communication_style": self.communication_style,
#             "entropy_baseline": self.entropy_baseline,
#             "error_patterns": self.error_patterns,
#             "preferred_modes": self.preferred_modes,
#             "session_count": self.session_count,
#             "total_inference_steps": self.total_inference_steps,
#             "user_facts": self.user_facts
#         }

#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "CognitiveProfile":
#         profile = cls(data["user_id"])
#         profile.recurring_clusters = data.get("recurring_clusters", [])
#         profile.expertise_levels = data.get("expertise_levels", {})
#         profile.communication_style = data.get("communication_style", "BALANCED")
#         profile.entropy_baseline = data.get("entropy_baseline", 0.5)
#         profile.error_patterns = data.get("error_patterns", [])
#         profile.preferred_modes = data.get("preferred_modes", {})
#         profile.session_count = data.get("session_count", 0)
#         profile.total_inference_steps = data.get("total_inference_steps", 0)
#         profile.user_facts = data.get("user_facts", {})
#         return profile

#     def save(self, filepath: str, password: str) -> None:
#         data_str = json.dumps(self.to_dict())
#         encrypted = CryptoHelper.encrypt(data_str, password)
#         with open(filepath, "wb") as f:
#             f.write(encrypted)

#     @classmethod
#     def load(cls, filepath: str, password: str) -> "CognitiveProfile":
#         with open(filepath, "rb") as f:
#             encrypted = f.read()
#         decrypted_str = CryptoHelper.decrypt(encrypted, password)
#         return cls.from_dict(json.loads(decrypted_str))

# class HydrusOptEntropyScorer:
#     MATH_PATTERN = re.compile(r"[0-9]+\s*[\*\+\-\/\=]")
#     def score(self, token_distribution: Any, context_ids: Optional[torch.Tensor] = None, tokenizer: Optional[Any] = None) -> float:
#         if isinstance(token_distribution, torch.Tensor):
#             logits = token_distribution.float()
#             if len(logits.shape) > 1:
#                 logits = logits[-1]
#             probs = torch.softmax(logits, dim=-1)
#             # Calculate entropy over top-50 active candidates to avoid long-tail vocab compression
#             top_k = min(50, probs.shape[-1])
#             top_probs, _ = torch.topk(probs, top_k, dim=-1)
#             top_probs = top_probs / top_probs.sum().clamp(min=1e-10)
#             raw = -(top_probs * top_probs.clamp(min=1e-10).log()).sum()
#             max_e = torch.log(torch.tensor(float(top_k), device=probs.device))
#             entropy = (raw / max_e).item()
#         else:
#             probs = np.array(token_distribution)
#             probs = np.clip(probs, 1e-10, 1.0)
#             probs = probs / np.sum(probs)
#             top_k = min(50, len(probs))
#             sorted_probs = sorted(probs, reverse=True)[:top_k]
#             top_probs = np.array(sorted_probs)
#             top_probs = top_probs / np.sum(top_probs)
#             raw = -np.sum(top_probs * np.log(top_probs))
#             max_e = np.log(top_k)
#             entropy = raw / max_e

#         if context_ids is not None and tokenizer is not None:
#             context = tokenizer.decode(context_ids[0][-20:])
#             if self.MATH_PATTERN.search(context):
#                 entropy += 0.15
#         return min(1.0, float(entropy))

# def embed(text: str, model: Any = None, tokenizer: Any = None, dim: int = 768) -> np.ndarray:
#     if model is not None and tokenizer is not None:
#         try:
#             device = model.device
#             inputs = tokenizer(text, return_tensors="pt").to(device)
#             embed_layer = None
#             if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
#                 embed_layer = model.model.embed_tokens
#             elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
#                 embed_layer = model.transformer.wte

#             if embed_layer is not None:
#                 with torch.no_grad():
#                     embeds = embed_layer(inputs["input_ids"])  # [1, seq_len, hidden_dim]
#                     mean_embed = embeds.mean(dim=1).squeeze(0)  # [hidden_dim]
#                     hidden_dim = mean_embed.shape[0]
#                     if hidden_dim == dim:
#                         vec = mean_embed.cpu().numpy()
#                     else:
#                         rng = np.random.default_rng(42)
#                         proj = rng.standard_normal((hidden_dim, dim)) / np.sqrt(hidden_dim)
#                         vec = mean_embed.cpu().numpy() @ proj
                    
#                     norm = np.linalg.norm(vec)
#                     if norm > 0:
#                         vec = vec / norm
#                     return vec
#         except Exception:
#             pass
            
#     # Fallback to hashing trick if model is not loaded or fails
#     words = text.lower().split()
#     vec = np.zeros(dim)
#     for word in words:
#         for offset in range(3):
#             h = hash(f"{word}_{offset}")
#             idx = abs(h) % dim
#             sign = 1 if h > 0 else -1
#             vec[idx] += sign
#     norm = np.linalg.norm(vec)
#     if norm > 0:
#         vec = vec / norm
#     return vec


# class DreamState:
#     """
#     Dream-State Compression (§3.8).

#     Merges evicted cold memories into impressionistic centroids ("dreams").
#     Provides déjà-vu retrieval: forgotten memories that semantically resemble an
#     active query surface as a soft signal, modulating confidence without full recall.
#     """
#     def __init__(self, dim: int = 64, threshold: float = 0.3):
#         self.dreams: Dict[int, dict] = {}
#         self.dim = dim
#         self.threshold = threshold
#         self._next_id = 0

#     def _generate_impressionistic_summary(self, cold_memories: List[dict]) -> str:
#         """
#         Synthesizes a rule-based impressionistic, dream-like description
#         from a batch of cold memories without using LLM calls.
#         """
#         summaries = [m.get('summary', '').strip() for m in cold_memories if m.get('summary')]
#         if not summaries:
#             return "impressionistic memory"
        
#         def clean(s):
#             s = re.sub(r'\[.*?\]', '', s)
#             s = re.sub(r'[^\w\s\-\']', ' ', s)
#             return " ".join(s.split())
            
#         cleaned = [clean(s) for s in summaries]
#         cleaned = [s for s in cleaned if s]
        
#         if not cleaned:
#             return "impressionistic memory"

#         if len(cleaned) == 1:
#             return f"fragment of: {cleaned[0][:40]}"

#         stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        
#         concepts = []
#         for text in cleaned:
#             words = [w.lower() for w in text.split() if w.lower() not in stopwords]
#             if words:
#                 concepts.append(" ".join(words[:3]))
        
#         if not concepts:
#             concepts = [c[:30] for c in cleaned[:3]]

#         if len(concepts) == 2:
#             return f"amalgam of {concepts[0]} fading into {concepts[1]}"
        
#         return f"composite of {concepts[0]} and {concepts[1]} overlaid with {concepts[2]}"

#     def compress_to_dream(self, cold_memories: List[dict]) -> int:
#         """
#         Merge a batch of cold memory dicts into a single impressionistic dream entry.
#         Each dict should have:
#             'embedding'   (np.ndarray)
#             'temperature' (float, optional) — used as valence proxy
#             'summary'     (str, optional)   — used to build abstract description
#         Returns the new dream_id, or -1 if the batch is empty.
#         """
#         if not cold_memories:
#             return -1

#         embeddings = np.stack([m['embedding'] for m in cold_memories])
#         centroid = np.mean(embeddings, axis=0)
#         norm = np.linalg.norm(centroid)
#         if norm > 0:
#             centroid = centroid / norm

#         valence = float(np.mean([m.get('temperature', 0.5) for m in cold_memories]))
#         scatter = float(np.std(embeddings, axis=0).mean())

#         abstract = self._generate_impressionistic_summary(cold_memories)

#         dream_id = self._next_id
#         self._next_id += 1
#         self.dreams[dream_id] = {
#             'centroid':         centroid,
#             'valence':          valence,
#             'scatter':          scatter,
#             'trigger_radius':   scatter * 2.0 + 0.1,
#             'abstract_summary': abstract,
#             'member_count':     len(cold_memories),
#             'created_at':       time.time(),
#         }
#         return dream_id

#     def feel_deja_vu(self, query_embedding: np.ndarray) -> Optional[dict]:
#         """
#         Probe all dreams against query_embedding.
#         Returns the strongest déjà-vu match above self.threshold, or None.
#         Output format:
#             {'type': 'deja_vu', 'strength': float, 'impression': str, 'dream_id': int}
#         """
#         best: Optional[dict] = None
#         best_strength = self.threshold

#         for dream_id, dream in self.dreams.items():
#             dist = float(np.linalg.norm(query_embedding - dream['centroid']))
#             deja_vu = 1.0 - (dist / (dream['trigger_radius'] + 1e-8))
#             if deja_vu > best_strength:
#                 best_strength = deja_vu
#                 best = {
#                     'type':       'deja_vu',
#                     'strength':   round(float(deja_vu), 4),
#                     'impression': dream['abstract_summary'],
#                     'dream_id':   dream_id,
#                 }
#         return best


# class SubjectTrackingBuffer:

#     """
#     Working memory that understands pronouns.
#     Tracks the last concrete subject from assistant answers so that
#     follow-up queries like "what does it do?" or "how does it help?"
#     are silently expanded to name the actual topic before hitting the model.
#     """
#     _PRONOUNS = {"it", "its", "it's", "they", "them", "their", "this", "that"}
#     # Simple noun extractor: first quoted or capitalised phrase, or first non-stopword multi-char word
#     _STOPWORDS = {
#         "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
#         "of", "in", "to", "for", "on", "at", "by", "or", "and", "but",
#         "what", "which", "that", "this", "does", "do", "did", "have", "has",
#         "how", "why", "when", "where", "who", "will", "would", "could", "should",
#         "cell", "body", "human", "help"
#     }

#     def __init__(self, max_turns: int = 8, max_chars_per_turn: int = 384):
#         import collections
#         self.turns = collections.deque(maxlen=max_turns)
#         self.max_chars = max_chars_per_turn
#         self.active_subject: str = ""          # e.g. "mitochondria"
#         self._root_turn: Optional[dict] = None # first turn that introduced subject

#     # ── subject extraction ────────────────────────────────────────────────────
#     @classmethod
#     def _extract_subject(cls, text: str) -> str:
#         """
#         Pick the first significant noun from the assistant's response.
#         Priority 1: word immediately after 'is/are/was' (most precise).
#         Priority 2: first non-stopword 4+ char token in first sentence.
#         """
#         first_sentence = text.split(".")[0].split("?")[0].split("!")[0]
#         words = first_sentence.lower().split()

#         # Priority 1: subject after verb-of-being
#         for verb in ("is", "are", "was", "were"):
#             if verb in words:
#                 idx = words.index(verb)
#                 # grab next 1-2 words, strip articles
#                 candidates_after = [w for w in words[idx+1:idx+3]
#                                     if w not in ("a", "an", "the") and len(w) >= 4]
#                 if candidates_after:
#                     return candidates_after[0]

#         # Priority 2: first significant token
#         tokens = re.findall(r"[a-zA-Z]{4,}", first_sentence.lower())
#         candidates = [t for t in tokens if t not in cls._STOPWORDS]
#         return candidates[0] if candidates else ""

#     # ── pronoun expansion ─────────────────────────────────────────────────────
#     def expand(self, query: str) -> str:
#         """
#         Scan the full query for any pronoun that refers to the active subject.
#         Replaces ALL occurrences of tracked pronouns with the concrete noun.
#         Handles mid-sentence cases like 'In detail how does it help...'.
#         """
#         if not self.active_subject:
#             return query

#         result = query
#         # Replace pronouns in order of longest-first to avoid partial matches
#         for pronoun in sorted(self._PRONOUNS, key=len, reverse=True):
#             # Word-boundary aware replacement, case-insensitive
#             pattern = r'\b' + re.escape(pronoun) + r'\b'
#             if re.search(pattern, result, flags=re.IGNORECASE):
#                 result = re.sub(pattern, self.active_subject, result, flags=re.IGNORECASE)
#                 break  # Only expand one pronoun type per query
#         return result

#     # ── public API ─────────────────────────────────────────────────────────────
#     def add_turn(self, role: str, content: str):
#         truncated = content[:self.max_chars]
#         entry = {"role": role, "content": truncated}
#         self.turns.append(entry)

#         if role == "assistant":
#             noun = self._extract_subject(truncated)
#             if noun and len(noun) > 3:
#                 self.active_subject = noun
#                 # Keep the first user turn that introduced this subject as a root anchor
#                 if self._root_turn is None:
#                     user_turns = [t for t in self.turns if t["role"] == "user"]
#                     if user_turns:
#                         self._root_turn = user_turns[0]

#     def get_context(self) -> str:
#         return "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in self.turns])

#     def get_messages(self) -> list:
#         """Return turn list, always including the root anchor turn if it was evicted."""
#         turns = list(self.turns)
#         if self._root_turn and self._root_turn not in turns:
#             # Prepend root anchor so the model always knows the original topic
#             turns = [self._root_turn] + turns
#         return turns

#     def clear(self):
#         self.turns.clear()
#         self.active_subject = ""
#         self._root_turn = None
# class HCL:
#     global_timings = {
#         "on_generation_step": 0.0,
#         "on_generation_step_count": 0,
#         "extract_cascade": 0.0,
#         "extract_cascade_count": 0,
#         "save_profile": 0.0,
#         "save_profile_count": 0
#     }

#     def __init__(self, model: Any, tokenizer: Any, mode: str = "balanced", user_id: str = "default_user", insecure_dev_mode: bool = False, hcl_lightweight: bool = False, profile_timing: bool = False):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.HMG = MemoryGraph()
#         self.ANN = KNNIndex(dim=768)
#         self.SML = StratifiedMemoryLattice(lattice_size=65536, dim=768)
#         self.GBPS = GroundedBeliefPathSearch(embedding_dim=768)
#         self.entropy_scorer = HydrusOptEntropyScorer()
#         self.mode = ModeController(mode)
#         self.step = 0
#         self.warm_cache = WarmCache()
#         self.transition_model = BeliefTransitionModel()
#         self.hcl_lightweight = hcl_lightweight
#         self.profile_timing = profile_timing
#         self.working_memory = SubjectTrackingBuffer(max_turns=8)
#         # Dream-State Compression engine (§3.8)
#         self.dream_state = DreamState(dim=self.SML.dim)
#         self.user_facts = UserFacts()

#         # Lattice passphrase (set once for all modes, uses dev key as fallback)
#         _passphrase = os.getenv("HCL_PROFILE_KEY")
#         if not _passphrase:
#             _passphrase = "FallbackInsecureDevKey2026"
#         self._lattice_passphrase = _passphrase
#         self._lattice_path = f"lattice_{user_id}.hcl"
        
#         # Try loading a persisted lattice on boot
#         if os.path.exists(self._lattice_path):
#             try:
#                 self.SML.load_from_disk(self._lattice_path, self._lattice_passphrase)
#                 self._rebuild_hmg_from_sml()
#             except Exception as e:
#                 print(f"  [HCL] Lattice load failed ({e}). Starting fresh.")
        
#         # Profile handling
#         self.profile: Optional[CognitiveProfile] = None
#         if mode == "persistent":
#             passphrase = os.getenv("HCL_PROFILE_KEY")
#             if not passphrase:
#                 if insecure_dev_mode:
#                     print("\n  [HCL WARNING] HCL_PROFILE_KEY environment variable is missing!")
#                     print("  Running with fallback INSECURE development passkey.")
#                     passphrase = "FallbackInsecureDevKey2026"
#                 else:
#                     raise ValueError(
#                         "HCL_PROFILE_KEY environment variable is required for persistent profile mode. "
#                         "Set it or run in `--insecure-dev-mode` for local testing."
#                     )
            
#             self.profile_path = f"profile_{user_id}.hcl"
#             self.profile_passphrase = passphrase
#             if os.path.exists(self.profile_path):
#                 try:
#                     self.profile = CognitiveProfile.load(self.profile_path, passphrase)
#                     self.user_facts.facts = self.profile.user_facts
#                 except Exception as e:
#                     print(f"  [HCL] Error decrypting user profile ({e}). Creating new profile.")
#                     self.profile = CognitiveProfile(user_id)
#             else:
#                 self.profile = CognitiveProfile(user_id)
            
#             self.profile.session_count += 1
#             self.save_profile()
#             self.mode.adapt_to_profile(self.profile)

#     def save_profile(self) -> None:
#         if self.profile is not None and hasattr(self, "profile_path") and hasattr(self, "profile_passphrase"):
#             self.profile.user_facts = self.user_facts.facts
#             t0 = time.time()
#             try:
#                 self.profile.save(self.profile_path, self.profile_passphrase)
#             except Exception as e:
#                 print(f"  [HCL] Error saving user profile to disk: {e}")
#             if getattr(self, "profile_timing", False):
#                 HCL.global_timings["save_profile"] += (time.time() - t0)
#                 HCL.global_timings["save_profile_count"] += 1
        
#         # Always persist the lattice alongside profile saves
#         if np.any(self.SML.occupied):
#             try:
#                 self.SML.save_to_disk(self._lattice_path, self._lattice_passphrase)
#             except Exception as e:
#                 print(f"  [HCL] Error saving lattice to disk: {e}")

#     def _rebuild_hmg_from_sml(self) -> None:
#         """Rebuild HMG nodes and clusters from SML payloads on boot."""
#         import numpy as np
#         active_addrs = np.where(self.SML.occupied)[0]
#         for addr in active_addrs:
#             payload = self.SML.payloads.get(addr, {})
#             if not payload:
#                 continue
#             node_id = payload.get("id")
#             if not node_id:
#                 continue
#             # Extract embedding and build Node
#             emb = self.SML.surface[addr, :self.SML.dim]
#             node_type_str = payload.get("type", "EPISODIC")
#             try:
#                 node_type = NodeType[node_type_str]
#             except Exception:
#                 node_type = NodeType.EPISODIC
                
#             node = Node(
#                 node_type=node_type,
#                 embedding=emb,
#                 raw_summary=payload.get("summary", ""),
#                 weight=float(self.SML.surface[addr, self.SML.dim]),
#                 confidence=float(self.SML.surface[addr, self.SML.dim + 1]),
#             )
#             node.id = node_id
#             node.cluster_id = payload.get("cluster_id")
            
#             # Insert into HMG and ANN
#             self.HMG.insert(node)
#             self.ANN.update(node)
            
#             # Restore cluster mapping
#             c_id = node.cluster_id
#             if c_id is not None:
#                 if c_id not in self.HMG.clusters:
#                     self.HMG.clusters[c_id] = []
#                 if node.id not in self.HMG.clusters[c_id]:
#                     self.HMG.clusters[c_id].append(node.id)

#     def on_generation_step(self, context_window: torch.Tensor, token_distribution: torch.Tensor) -> str:
#         t0 = time.time()
#         self.step += 1
#         if self.profile is not None:
#             self.profile.total_inference_steps += 1
#             if not getattr(self, "hcl_lightweight", False):
#                 self.save_profile()

#         # 1. Write gate
#         H = self.entropy_scorer.score(token_distribution, context_window, self.tokenizer)
#         θ_write = self.profile.entropy_baseline + 0.1 if self.profile else self.mode.θ_write

#         if H > θ_write:
#             # Multi-layer sieve extraction
#             window_text = self.tokenizer.decode(context_window[0][-100:], skip_special_tokens=True)
#             nodes = self.extract_cascade(window_text, H)
#             for node in nodes:
#                 self.write(node)

#         # FAST PATH: if memory graph has no nodes and no matched user facts, return empty string immediately
#         query_text = getattr(self, "current_query_text", "")
#         has_matched_facts = False
#         if query_text:
#             has_matched_facts = len(self.user_facts.retrieve_relevant_facts(query_text)) > 0

#         if len(self.HMG.nodes) == 0 and not has_matched_facts:
#             if getattr(self, "profile_timing", False):
#                 HCL.global_timings["on_generation_step"] += (time.time() - t0)
#                 HCL.global_timings["on_generation_step_count"] += 1
#             return ""

#         # 2. Predictive pre-fetch
#         self.predictive_prefetch()

#         # 3. GBPS query formation: blend query embedding with active belief path
#         belief_emb = self.GBPS.get_active_belief_embedding()
#         query_emb = getattr(self, "current_query_embedding", None)
#         if query_emb is not None:
#             if np.any(belief_emb) and not is_personal_query(getattr(self, "current_query_text", "")):
#                 query_emb = 0.6 * query_emb + 0.4 * belief_emb
#                 norm = np.linalg.norm(query_emb)
#                 if norm > 0:
#                     query_emb /= norm
#         else:
#             query_emb = belief_emb

#         # 4. Retrieval (warm cache + ANN)
#         retrieved = self.retrieve(query_emb, k=5)

#         # 5. Inject into active window
#         injected_context = self.format_injection(retrieved)

#         # 6. Decay + prune every N steps
#         if self.step % self.mode.N_prune == 0:
#             self.pruner_loop()

#         # 7. Update transition model
#         self.transition_model.update(self.GBPS.get_active_beliefs())

#         if getattr(self, "profile_timing", False):
#             HCL.global_timings["on_generation_step"] += (time.time() - t0)
#             HCL.global_timings["on_generation_step_count"] += 1

#         return injected_context

#     def run_compression_prompt(self, prompt: str, max_tokens: int = 40) -> str:
#         try:
#             device = self.model.device
#             messages = [
#                 {"role": "system", "content": "You are a precise summarization assistant. Answer in a direct, factual manner."},
#                 {"role": "user", "content": prompt}
#             ]
#             if hasattr(self.tokenizer, "apply_chat_template"):
#                 formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             else:
#                 formatted = f"Context: {prompt}\nSummary:"
                
#             inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
#             # Temporarily bypass HCL during HCL compression checks
#             with torch.inference_mode():
#                 out = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_tokens,
#                     do_sample=False,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
#             result = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#             return result
#         except Exception:
#             return f"Lightweight summary of: {prompt[:30]}"

#     def verify_fact(self, sentence: str, confidence: float = 0.0) -> bool:
#         """
#         Uses the model's parametric knowledge to double-check a confidently stated fact.
#         Catches 'confident hallucinations' where entropy is low but the fact is wrong.
#         Respects ModeController guard flags (high-conf bypass and multi-vote consensus).
#         """
#         # Bypass real-world fact verification for personal user facts
#         sentence_lower = sentence.lower()
#         if "your" in sentence_lower or "you " in sentence_lower or "you're" in sentence_lower:
#             return True
#         for fact_info in self.user_facts.facts.values():
#             val = str(fact_info.get("value", "")).lower()
#             if val and val in sentence_lower:
#                 return True

#         if confidence >= self.mode.verify_high_conf_bypass:
#             return True  # Bypass guard for extremely confident outputs

#         prompt = (
#             f"Analyze this statement: '{sentence}'\n"
#             "Is this statement an established, currently true fact in the real world? "
#             "Consider if it refers to something that hasn't been built yet or is just a concept.\n"
#             "Answer with exactly one word: TRUE or FALSE."
#         )
        
#         no_votes = 0
#         for _ in range(self.mode.verify_votes):
#             res = self.run_compression_prompt(prompt, max_tokens=10).strip().upper()
#             if "FALSE" in res:
#                 no_votes += 1
                
#             # Early exit if we already hit the block threshold
#             if no_votes >= self.mode.no_votes_to_block:
#                 return False
                
#         return True

#     def _extract_causal_sentences(self, text: str) -> Optional[str]:
#         """Regex/heuristic extraction. No model."""
#         import re
#         causal_patterns = [
#             r'[^.]*?(?:because|therefore|thus|hence|as a result|leads to|causes)[^.]*\.',
#             r'[^.]*?(?:if.*then|when.*results in)[^.]*\.',
#         ]
#         matches = []
#         for pattern in causal_patterns:
#             matches.extend(re.findall(pattern, text, re.IGNORECASE))
#         valid_matches = [m.strip() for m in matches if m.strip()]
#         return ' '.join(valid_matches[:3]) if valid_matches else None

#     def _extract_factual_claim(self, text: str) -> Optional[str]:
#         """Heuristic: sentences that look like definitions or facts."""
#         sentences = text.split('.')
#         for s in sentences:
#             s_clean = s.strip()
#             if not s_clean:
#                 continue
#             lower_s = s_clean.lower()
#             if ' is ' in lower_s or ' are ' in lower_s or ' was ' in lower_s or ' were ' in lower_s:
#                 return s_clean + '.'
#         return None

#     def extract_cascade(self, window_text: str, entropy: float) -> List[Node]:
#         t0 = time.time()
#         nodes = []

#         # Layer 1: Episodic — raw text slice
#         episodic_summary = window_text[:512]
        
#         episodic = Node(
#             node_type=NodeType.EPISODIC,
#             embedding=embed(episodic_summary, self.model, self.tokenizer),
#             raw_summary=episodic_summary,
#             weight=0.8,
#             state=1,
#             created_at=self.step,
#             last_accessed=self.step,
#             access_count=0
#         )
#         nodes.append(episodic)

#         # Layer 2: Semantic (§3.2: H > 0.65, spec-correct threshold)
#         if entropy > 0.65:
#             semantic_summary = self._extract_causal_sentences(window_text)
#             if semantic_summary:
#                 semantic = Node(
#                     node_type=NodeType.SEMANTIC,
#                     embedding=embed(semantic_summary, self.model, self.tokenizer),
#                     raw_summary=semantic_summary,
#                     weight=0.75,
#                     state=1,
#                     created_at=self.step,
#                     last_accessed=self.step,
#                     access_count=0,
#                     provenance=[episodic.id]
#                 )
#                 nodes.append(semantic)

#         # Layer 3: Belief (§3.2: H > 0.80, spec-correct threshold)
#         if entropy > 0.80:
#             belief_summary = self._extract_factual_claim(window_text)
#             if belief_summary:
#                 belief = Node(
#                     node_type=NodeType.BELIEF,
#                     embedding=embed(belief_summary, self.model, self.tokenizer),
#                     raw_summary=belief_summary,
#                     weight=0.9,
#                     state=1,
#                     created_at=self.step,
#                     last_accessed=self.step,
#                     access_count=0,
#                     provenance=[n.id for n in nodes],
#                     confidence=0.85
#                 )
#                 nodes.append(belief)
#                 self.GBPS.add_belief(belief)

#         for node in nodes:
#             self.assign_to_cluster(node)

#         if getattr(self, "profile_timing", False):
#             HCL.global_timings["extract_cascade"] += (time.time() - t0)
#             HCL.global_timings["extract_cascade_count"] += 1

#         return nodes

#     def assign_to_cluster(self, node: Node, threshold: float = 0.80) -> None:
#         best_cluster = None
#         best_sim = -1.0
        
#         for cluster_id, node_ids in self.HMG.clusters.items():
#             cluster_nodes = [self.HMG.nodes[nid] for nid in node_ids if nid in self.HMG.nodes]
#             if not cluster_nodes:
#                 continue
#             center = np.mean([n.embedding for n in cluster_nodes], axis=0)
#             norm = np.linalg.norm(center)
#             if norm > 0:
#                 center /= norm
            
#             sim = float(np.dot(node.embedding, center))
#             if sim > best_sim:
#                 best_sim = sim
#                 best_cluster = cluster_id
                
#         if best_sim > threshold:
#             node.cluster_id = best_cluster
#             self.HMG.clusters[best_cluster].append(node.id)
#         else:
#             new_cluster_id = len(self.HMG.clusters) + 1
#             node.cluster_id = new_cluster_id
#             self.HMG.clusters[new_cluster_id] = [node.id]

#     def write(self, node: Node) -> None:
#         nearest_list = self.ANN.query(node.embedding, k=1)
#         nearest = nearest_list[0] if nearest_list else None
#         τ = 0.92
#         if nearest and float(np.dot(node.embedding, nearest.embedding)) > τ:
#             if self.contradiction_detected(node, nearest, τ):
#                 self.resolve_contradiction(node, nearest)
#             else:
#                 self.merge(node, nearest)
#         else:
#             self.HMG.insert(node)
#             self.ANN.update(node)
#             payload = {
#                 "id": node.id,
#                 "type": node.node_type.value,
#                 "summary": node.raw_summary,
#                 "cluster_id": node.cluster_id
#             }
#             addr = self.SML.write(node.embedding, confidence=node.confidence, payload=payload)
#             setattr(node, "sml_addr", addr)

#     def contradiction_detected(self, node_a: Node, node_b: Node, τ: float = 0.92) -> bool:
#         sim = float(np.dot(node_a.embedding, node_b.embedding))
#         if sim < τ:
#             return False

#         contradiction_prompt = (
#             f"Statement A: {node_a.raw_summary}\n"
#             f"Statement B: {node_b.raw_summary}\n\n"
#             "Do these statements contradict each other, describe different "
#             "time periods of the same thing, or are they unrelated?\n"
#             "Answer with exactly one word: CONTRADICTION, TEMPORAL, or UNRELATED"
#         )
#         verdict = self.run_compression_prompt(contradiction_prompt, max_tokens=10).upper()
#         return "CONTRADICTION" in verdict

#     def resolve_contradiction(self, node_a: Node, node_b: Node) -> None:
#         if node_b.id not in node_a.contradictions:
#             node_a.contradictions.append(node_b.id)
#         if node_a.id not in node_b.contradictions:
#             node_b.contradictions.append(node_a.id)

#         if node_a.confidence is not None and node_b.confidence is not None:
#             if abs(node_a.confidence - node_b.confidence) > 0.3:
#                 winner = node_a if node_a.confidence > node_b.confidence else node_b
#                 loser = node_b if winner == node_a else node_a
#                 loser.weight *= 0.5
#                 if " [DEPRECATED" not in loser.raw_summary:
#                     loser.raw_summary += " [DEPRECATED: contradicted by higher-confidence belief]"
#             else:
#                 if " [CONTRADICTION" not in node_a.raw_summary:
#                     node_a.raw_summary += f" [CONTRADICTION: see {node_b.id}]"
#                 if " [CONTRADICTION" not in node_b.raw_summary:
#                     node_b.raw_summary += f" [CONTRADICTION: see {node_a.id}]"

#         self.HMG.insert(node_a)
#         self.ANN.update(node_a)

#     def merge(self, node: Node, nearest: Node) -> None:
#         nearest.embedding = 0.7 * nearest.embedding + 0.3 * node.embedding
#         norm = np.linalg.norm(nearest.embedding)
#         if norm > 0:
#             nearest.embedding /= norm
#         if node.raw_summary.strip() != nearest.raw_summary.strip():
#             nearest.raw_summary += " | " + node.raw_summary
#         nearest.last_accessed = self.step
#         nearest.access_count += 1

#     def user_correction(self, topic_embedding: np.ndarray, correction_text: str, demotion_factor: float = 0.35) -> int:
#         """
#         Called when the user explicitly contradicts or corrects the model's output.
#         Finds memory nodes semantically similar to the topic being corrected and
#         demotes their weight so they won't be retrieved and reinforced in future turns.
#         Returns the number of nodes demoted.
#         """
#         # Query ANN for nodes close to the corrected topic
#         candidates = self.ANN.query(topic_embedding, k=10)
#         demoted = 0
#         for node in candidates:
#             sim = float(np.dot(node.embedding, topic_embedding))
#             if sim > 0.75:  # Only demote nodes clearly about this topic
#                 old_weight = node.weight
#                 node.weight = max(0.0, node.weight * (1.0 - demotion_factor))
#                 # Mark node so format_injection knows it was user-corrected
#                 if "[USER CORRECTED]" not in node.raw_summary:
#                     node.raw_summary += " [USER CORRECTED]"
#                 # If weight falls below dormant threshold, mark dormant
#                 if node.weight <= 0.20:
#                     node.state = 0
#                 demoted += 1
#         return demoted

#     def retrieve(self, query_emb: np.ndarray, k: int = 5,
#                   user_confirmed: bool = False) -> List[Node]:
#         cached = self.warm_cache.check(query_emb)
#         if cached:
#             return cached[:k]

#         sml_results = self.SML.recall(query_emb, k=k, user_confirmed=user_confirmed)
#         retrieved_nodes = []
#         for res in sml_results:
#             addr = res['addr']
#             payload = self.SML.payloads.get(addr, {})
#             node_id = payload.get("id")
#             if node_id and node_id in self.HMG.nodes:
#                 node = self.HMG.nodes[node_id]
#                 node.last_accessed = self.step
#                 node.access_count += 1
#                 node.state = 1
#                 # §3.6 verification_count: only increment on explicit user confirmation
#                 if user_confirmed:
#                     node.verification_count += 1
#                 # Integrate ReML physics
#                 node.weight = res['confidence']  # Handled by Abyssal Gravity
#                 setattr(node, "epistemic_divergence", res['epistemic_divergence'])
#                 setattr(node, "temporal_bindings", res['temporal_bindings'])
#                 # Synchronize drift
#                 node.embedding = res['embedding']
#                 retrieved_nodes.append(node)

#         return retrieved_nodes

#     def predictive_prefetch(self) -> None:
#         active_beliefs = self.GBPS.get_active_beliefs()
#         predicted_clusters = self.transition_model.predict_next(active_beliefs, self.HMG, top_k=3)
        
#         θ_prefetch = 0.50
#         prefetched_nodes = []
#         for cluster_id in predicted_clusters:
#             node_ids = self.HMG.clusters.get(cluster_id, [])
#             for nid in node_ids:
#                 node = self.HMG.nodes.get(nid)
#                 if node and node.node_type == NodeType.BELIEF and node.confidence > θ_prefetch:
#                     candidates = self.ANN.query(node.embedding, k=10)
#                     prefetched_nodes.extend(candidates)
                    
#         if prefetched_nodes:
#             self.warm_cache.stage(prefetched_nodes)

#     # ── Dynamic Type Evolution (§3.6) ────────────────────────────────────

#     def promote_node_type(self, node: "Node") -> bool:
#         """
#         Attempt to promote node along the type ladder:
#             EPISODIC -> SEMANTIC -> BELIEF -> AXIOM
#         Promotion criteria from the HCL spec:
#             EPISODIC  -> SEMANTIC : access_count > 10, abstraction_score > 0.7 (query-context diversity)
#             SEMANTIC  -> BELIEF   : confidence > 0.9, no contradictions
#             BELIEF    -> AXIOM    : verification_count > 100, source_diversity >= 3, no contradictions
#         Returns True if the node was promoted.
#         """
#         changed = False

#         if node.node_type == NodeType.EPISODIC:
#             # §3.6: use abstraction_score — measures query-context diversity across strata
#             sml_addr = getattr(node, 'sml_addr', None)
#             if sml_addr is not None:
#                 node.abstraction_score = self.SML.compute_abstraction_score(sml_addr)
#             if (node.access_count > 10 and node.abstraction_score > 0.7):
#                 node.node_type = NodeType.SEMANTIC
#                 changed = True

#         elif node.node_type == NodeType.SEMANTIC:
#             if (node.confidence is not None and node.confidence > 0.9
#                     and len(node.contradictions) == 0):
#                 node.node_type = NodeType.BELIEF
#                 changed = True

#         elif node.node_type == NodeType.BELIEF:
#             # §3.6: use verification_count (user-confirmed only), not access_count
#             source_diversity = len(set(node.provenance))
#             if (node.verification_count > AXIOM_PROPERTIES['promotion_threshold']
#                     and source_diversity >= AXIOM_PROPERTIES['source_diversity_min']
#                     and len(node.contradictions) == 0):
#                 node.node_type  = NodeType.AXIOM
#                 node.confidence = 1.0
#                 changed = True

#         return changed

#     def demote_axiom(self, node: "Node", reason: str = 'contradiction') -> bool:
#         """
#         Demote AXIOM → BELIEF on contradiction accumulation or explicit user challenge.
#         Demotion requires either:
#             - reason == 'user_explicit'   (user explicitly challenged the axiom), or
#             - len(contradictions) ≥ AXIOM_PROPERTIES['demotion_threshold'] (default 3)
#         Returns True if demotion occurred.
#         """
#         if node.node_type != NodeType.AXIOM:
#             return False
#         n_contradictions = len(node.contradictions)
#         if (reason == 'user_explicit'
#                 or n_contradictions >= AXIOM_PROPERTIES['demotion_threshold']):
#             node.node_type  = NodeType.BELIEF
#             node.confidence = max(0.5, (node.confidence or 1.0) * 0.7)
#             if '[DEMOTED FROM AXIOM]' not in node.raw_summary:
#                 node.raw_summary += ' [DEMOTED FROM AXIOM]'
#             return True
#         return False

#     def pruner_loop(self) -> None:
#         # 1. Vectorised thermal decay + migrate cells; update stale sml_addr pointers
#         migrations = self.SML.thermal_decay(lambda_=self.mode.λ)
#         if migrations:
#             for node in self.HMG.nodes.values():
#                 old = getattr(node, 'sml_addr', None)
#                 if old is not None and old in migrations:
#                     node.sml_addr = migrations[old]

#         nodes_to_remove = []
#         for node in list(self.HMG):
#             self.update_weight(node)
#             steps_since_query = self.step - node.last_accessed

#             # 2. Dynamic Type Evolution: attempt promotion each pruner cycle
#             promoted = self.promote_node_type(node)
#             # AXIOMs are bedrock — skip weight decay and state collapse this cycle
#             if promoted and node.node_type == NodeType.AXIOM:
#                 continue

#             # 3. AXIOM demotion on contradiction accumulation
#             if node.node_type == NodeType.AXIOM:
#                 self.demote_axiom(node, reason='contradiction')
#                 continue  # never prune AXIOMs directly (must demote first)

#             # resolve state
#             resolve_state(
#                 node,
#                 queried=False,
#                 steps_since_query=steps_since_query,
#                 θ_high=0.65,
#                 θ_low=0.20,
#                 N_collapse=self.mode.N_collapse
#             )

#             if node.state == 0:
#                 nodes_to_remove.append(node)

#         # 4. Evict dormant nodes and feed their embeddings to the Dream-State compressor
#         cold_batch: List[Node] = []
#         for node in nodes_to_remove:
#             addr = getattr(node, 'sml_addr', None)
#             if addr is not None and self.SML.occupied[addr]:
#                 emb     = self.SML.surface[addr, :self.SML.dim].copy()
#                 temp    = float(self.SML.surface[addr, self.SML.dim + 4])
#                 payload = self.SML.payloads.get(addr, {})
#                 cold_batch.append({
#                     'embedding':   emb,
#                     'temperature': temp,
#                     'summary':     payload.get('summary', node.raw_summary[:60]),
#                 })
#                 self.SML.forget_to_abyss(addr)
#             self.HMG.remove(node)
#             self.ANN.remove(node)

#         if cold_batch:
#             self.dream_state.compress_to_dream(cold_batch)


#     def update_weight(self, node: Node) -> None:
#         Δ = self.step - node.last_accessed
#         recency = np.exp(-self.mode.λ * Δ)
#         frequency = np.log(1 + node.access_count)
        
#         curr_emb = self.GBPS.current_embedding()
#         proximity = float(np.dot(node.embedding, curr_emb))
        
#         utility = node.utility_score
#         confidence = node.confidence if node.node_type == NodeType.BELIEF else 0.5

#         cluster_boost = 0.0
#         if self.profile and node.cluster_id in self.profile.recurring_clusters:
#             cluster_boost = 0.1

#         α, β, γ, δ, ε = 0.3, 0.2, 0.2, 0.15, 0.15
#         raw_weight = (
#             α * recency + 
#             β * frequency + 
#             γ * proximity + 
#             δ * utility + 
#             ε * confidence +
#             cluster_boost
#         )
#         node.weight = float(np.clip(raw_weight, 0.0, 1.0))

#     def score_retrieval_utility(self, retrieved_nodes: List[Node], pre_state: Dict[str, float], post_state: Dict[str, Any]) -> None:
#         entropy_delta = pre_state["entropy"] - post_state["entropy"]
#         confidence_delta = post_state["confidence"] - pre_state["confidence"]
#         user_accepted = 1.0 if not post_state.get("fallback_triggered", False) else 0.0
#         new_belief_formed = 1.0 if self.GBPS.detect_new_belief() else 0.0

#         def norm(val):
#             return 1.0 / (1.0 + np.exp(-val))

#         utility = (
#             0.4 * norm(entropy_delta) +
#             0.3 * norm(confidence_delta) +
#             0.2 * user_accepted +
#             0.1 * new_belief_formed
#         )

#         θ_high_utility = 0.70
#         θ_low_utility = 0.30
#         δ_utility_boost = 0.15

#         for node in retrieved_nodes:
#             node.utility_score = 0.7 * node.utility_score + 0.3 * utility
#             if utility > θ_high_utility:
#                 node.weight = min(1.0, node.weight + δ_utility_boost)
#                 node.state = 1
#             elif utility < θ_low_utility:
#                 node.weight *= 0.9
#                 if node.weight < 0.20:
#                     node.state = 0

#     def format_injection(self, retrieved_nodes: List[Node], max_tokens: int = 512) -> str:
#         lines = ["[MEMORY CONTEXT]"]
#         char_count = 0
#         max_chars = max_tokens * 4  # heuristic: ~4 chars per token

#         # Retrieve matched facts and prepend them at the top of memory context
#         query_text = getattr(self, "current_query_text", "")
#         if query_text:
#             matched_facts = self.user_facts.retrieve_relevant_facts(query_text)
#             for fact in matched_facts:
#                 line = f"- [FACT] {fact}"
#                 if char_count + len(line) <= max_chars:
#                     lines.append(line)
#                     char_count += len(line)

#         # Strip internal HCL annotation brackets before injecting into the system prompt.
#         _annotation_re = re.compile(
#             r'\s*\[(?:CONTRADICTION|DEPRECATED|TEMPORAL|HCL|USER CORRECTED|DEMOTED FROM AXIOM)[^\]]*\]'
#         )

#         # Sort and inject memory nodes (skip if it is a personal query to prevent noise/contamination)
#         if not is_personal_query(query_text):
#             sorted_nodes = sorted(retrieved_nodes, key=lambda x: x.weight, reverse=True)
#             for node in sorted_nodes:
#                 clean_summary = _annotation_re.sub('', node.raw_summary).strip()
#                 # Also drop empty summaries that were fully annotation
#                 if not clean_summary:
#                     continue
#                 line = f"- {clean_summary}  (relevance: {node.weight:.2f}, type: {node.node_type.value})"
#                 if char_count + len(line) > max_chars:
#                     break
#                 lines.append(line)
#                 char_count += len(line)

#         # Déjà-vu probe: surface any impressionistic dream memories that resemble the query
#         query_emb = self.GBPS.get_active_belief_embedding()
#         deja_vu = self.dream_state.feel_deja_vu(query_emb)
#         if deja_vu:
#             lines.append(
#                 f"- [IMPRESSION: {deja_vu['impression']}]  "
#                 f"(déjà-vu strength: {deja_vu['strength']:.2f}, type: DREAM)"
#             )

#         if len(lines) <= 1:  # only the header, nothing substantive added
#             return ""

#         lines.append("[END MEMORY CONTEXT]\n")
#         return "\n".join(lines)


#     def retrieve_first_question(self) -> Optional[Node]:
#         """Special retrieval: find earliest question in session."""
#         for node in self.HMG.nodes.values():   # .values() — was incorrectly iterating keys
#             if node.node_type == NodeType.QUESTION_ANCHOR:
#                 return node
#         return None

#     def retrieve_latest_question(self) -> Optional[Node]:
#         """Retrieve the most recent question anchor."""
#         question_nodes = [
#             n for n in self.HMG.nodes.values()
#             if n.node_type == NodeType.QUESTION_ANCHOR
#         ]
#         if not question_nodes:
#             return None
#         return max(question_nodes, key=lambda n: getattr(n, "created_at", 0.0))

#     def get_dynamic_uncertainty_threshold(self, query: str, assistant_text: str, base_threshold: float) -> float:
#         """
#         Computes an adaptive uncertainty threshold dynamically based on:
#           1. Query intent (creative vs. factual constraint)
#           2. Syntactic generation context (list enumeration vs. copula factual claims)
#         """
#         dynamic_thresh = base_threshold
        
#         # 1. Query Intent Analysis
#         query_lower = query.lower()
#         creative_signals = {"write", "suggest", "brainstorm", "creative", "poem", "story", "joke", "feelings", "opinion", "imagine", "list some", "give me some ideas"}
#         factual_signals = {"exact", "who is", "who was", "what is", "where is", "when did", "how many", "calculate", "math", "formula", "date", "year", "height", "population"}
        
#         if any(sig in query_lower for sig in creative_signals):
#             dynamic_thresh += 0.12
#         elif any(sig in query_lower for sig in factual_signals):
#             dynamic_thresh -= 0.08

#         conversational_padding = {"needed to know", "wanted to know", "ask you", "what did i", "do you remember", "tell me if", "can you tell", "could you tell"}
#         if any(sig in query_lower for sig in conversational_padding):
#             dynamic_thresh += 0.08

#         # Relax threshold constraints during casual/social chat
#         if is_social_utterance(query):
#             return 9.9

#         # 2. Syntactic Generation Context
#         last_words = assistant_text.strip().lower()
#         is_list_context = False
#         if last_words:
#             recent_segment = last_words[-100:]
#             if "," in recent_segment or " and" in recent_segment or " or" in recent_segment:
#                 if any(phrase in recent_segment for phrase in ["feelings of", "associated with", "like", "such as", "including", "examples", "qualities", "trust"]):
#                     is_list_context = True

#         if is_list_context:
#             dynamic_thresh += 0.10
#         else:
#             assertion_patterns = [r"\b(is|are|was|were|has|have|had|consists of)\b(?:\s+\w+)?$"]
#             if any(re.search(pat, last_words) for pat in assertion_patterns):
#                 dynamic_thresh -= 0.05

#         return float(np.clip(dynamic_thresh, 0.45, 0.95))

# def resolve_state(node: Node, queried: bool, steps_since_query: int, θ_high: float = 0.65, θ_low: float = 0.20, N_collapse: int = 5, δ_boost: float = 0.15) -> None:
#     if node.weight > θ_high:
#         node.state = 1
#     elif node.weight <= θ_low:
#         node.state = 0
#     else:
#         if queried:
#             node.state = 1
#             node.weight = min(1.0, node.weight + δ_boost)
#         elif steps_since_query > N_collapse:
#             node.state = 0
#         else:
#             node.state = 2

# def generate_with_hcl(prompt: str, model: Any, tokenizer: Any, hcl: HCL, max_new_tokens: int = 80, max_context_window: int = 2048) -> Tuple[str, dict]:
#     hcl.current_query_embedding = embed(prompt)
#     hcl.current_query_text = prompt
#     device = model.device
#     assistant_tokens = []
    
#     # Pre-fetch memory context to avoid the first-token blind spot
#     belief_emb = hcl.GBPS.get_active_belief_embedding()
#     query_emb = hcl.current_query_embedding
#     if query_emb is not None:
#         if np.any(belief_emb) and not is_personal_query(prompt):
#             query_emb = 0.6 * query_emb + 0.4 * belief_emb
#             norm = np.linalg.norm(query_emb)
#             if norm > 0:
#                 query_emb /= norm
#     else:
#         query_emb = belief_emb
    
#     initial_nodes = hcl.retrieve(query_emb, k=5)
#     memory_prefix = hcl.format_injection(initial_nodes)
#     pre_state = {"entropy": 0.5, "confidence": 0.5}
    
#     for step in range(max_new_tokens):
#         # Format the system instruction and memory prefix using the model's chat template
#         system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."
#         if memory_prefix and ("User fact:" in memory_prefix or "[FACT]" in memory_prefix):
#             system_msg += " Always refer to the user using 'you' or 'your' when answering questions about user facts (e.g. say 'Your favorite color is blue' instead of 'My favorite color is blue')."
#         if memory_prefix and "[SYSTEM MEMORY:" in memory_prefix:
#             system_msg += " Always use the provided [SYSTEM MEMORY] block to answer questions about what the user asked first or most recently, and reference that exact question. Rely ONLY on the [SYSTEM MEMORY] block and do not look at the conversation history to determine this, since the history contains statements rather than questions."
#         if memory_prefix:
#             system_msg = f"{system_msg}\n\n{memory_prefix}"
            
#         messages = [
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": prompt}
#         ]
        
#         if hasattr(tokenizer, "apply_chat_template"):
#             formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         else:
#             formatted_prompt = f"{system_msg}\nUser: {prompt}\nAssistant:"
            
#         prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
        
#         if assistant_tokens:
#             assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
#             input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
#         else:
#             input_ids = prompt_ids
            
#         # Context window overflow strategy: truncate assistant tokens from the left
#         if input_ids.shape[1] > max_context_window:
#             overflow = input_ids.shape[1] - max_context_window
#             if overflow < len(assistant_tokens):
#                 assistant_tokens = assistant_tokens[overflow:]
#                 assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
#                 input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
#             else:
#                 if len(hcl.HMG.nodes) > 0:
#                     hcl.step += 1
#                 input_ids = prompt_ids[:, -max_context_window:]
                
#         with torch.no_grad():
#             outputs = model(input_ids)
#             logits = outputs.logits
            
#         next_token_logits = logits[:, -1, :]
#         probs = torch.softmax(next_token_logits, dim=-1)
#         next_token = torch.argmax(probs, dim=-1, keepdim=True)
        
#         next_token_val = next_token.item()
#         assistant_tokens.append(next_token_val)
        
#         # HCL step updates the memory graph and retrieves relevant context
#         new_prefix = hcl.on_generation_step(input_ids, next_token_logits)
        
#         # Track utility metrics
#         entropy = hcl.entropy_scorer.score(next_token_logits)
#         confidence = float(torch.max(probs).item())

#         current_assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#         dynamic_threshold = hcl.get_dynamic_uncertainty_threshold(prompt, current_assistant_text, hcl.mode.uncertainty_threshold)

#         # HCL: entropy-based uncertainty check with dynamic threshold
#         if entropy > dynamic_threshold and step > 5:
#             final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#             extracted_facts = hcl.user_facts.extract_from_turn(prompt)
#             for key, val in extracted_facts:
#                 hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)
#             if getattr(hcl, "hcl_lightweight", False) or extracted_facts:
#                 hcl.save_profile()
#             return final_output + " [HCL: UNCERTAIN — REVIEW REQUIRED]", {
#                 "total_steps": len(assistant_tokens),
#                 "memory_nodes": len(hcl.HMG.nodes),
#                 "uncertain": True
#             }
        
#         # Score utility of the previous retrieval step (only if graph contains nodes)
#         if len(hcl.HMG.nodes) > 0:
#             belief_emb = hcl.GBPS.get_active_belief_embedding()
#             query_emb = getattr(hcl, "current_query_embedding", None)
#             if query_emb is not None:
#                 if np.any(belief_emb) and not is_personal_query(prompt):
#                     query_emb = 0.6 * query_emb + 0.4 * belief_emb
#                     norm = np.linalg.norm(query_emb)
#                     if norm > 0:
#                         query_emb /= norm
#             else:
#                 query_emb = belief_emb
#             retrieved_nodes = hcl.retrieve(query_emb, k=5)
#             post_state = {"entropy": entropy, "confidence": confidence, "fallback_triggered": False}
#             hcl.score_retrieval_utility(retrieved_nodes, pre_state, post_state)
#             pre_state = post_state
        
#         if next_token_val == tokenizer.eos_token_id:
#             break
            
#         if new_prefix != memory_prefix:
#             memory_prefix = new_prefix
            
#     final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#     extracted_facts = hcl.user_facts.extract_from_turn(prompt)
#     for key, val in extracted_facts:
#         hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)
#     if getattr(hcl, "hcl_lightweight", False) or extracted_facts:
#         hcl.save_profile()
#     return final_output, {"total_steps": len(assistant_tokens), "memory_nodes": len(hcl.HMG.nodes)}

# def generate_with_hcl_stream(prompt: str, model: Any, tokenizer: Any, hcl: HCL, max_new_tokens: int = 80, max_context_window: int = 2048):
#     hcl.current_query_embedding = embed(prompt)
#     hcl.current_query_text = prompt
#     def get_reml_payloads():
#         import numpy as np
#         active_addrs = np.where(hcl.SML.occupied)[0]
#         nodes = []
#         for addr in active_addrs:
#             payload = hcl.SML.payloads.get(addr, {})
#             if not payload:
#                 continue
#             # Fix np.int64 JSON serialization issues
#             cluster_id = payload.get("cluster_id")
#             if cluster_id is not None:
#                 cluster_id = int(cluster_id)
                
#             raw_bindings = hcl.SML.temporal.retrieve_temporal_context(addr)
#             clean_bindings = []
#             for b in raw_bindings:
#                 clean_bindings.append({
#                     "addr": int(b["addr"]),
#                     "temporal_proximity": float(b["temporal_proximity"]),
#                     "semantic_distance": float(b["semantic_distance"])
#                 })

#             nodes.append({
#                 "id": payload.get("id"),
#                 "type": payload.get("type"),
#                 "summary": payload.get("summary"),
#                 "weight": float(hcl.SML.surface[addr, hcl.SML.dim]),
#                 "cluster_id": cluster_id,
#                 "epistemic_divergence": float(hcl.SML.bedrock.divergence(addr)),
#                 "temporal_bindings": clean_bindings,
#                 "addr": int(addr)
#             })
#         return nodes

#     device = model.device
#     assistant_tokens = []
#     previous_assistant_text = ""
#     raw_previous_text = ""

#     # Pre-fetch memory context to avoid the first-token blind spot
#     belief_emb = hcl.GBPS.get_active_belief_embedding()
#     query_emb = hcl.current_query_embedding
#     if query_emb is not None:
#         if np.any(belief_emb) and not is_personal_query(prompt):
#             query_emb = 0.6 * query_emb + 0.4 * belief_emb
#             norm = np.linalg.norm(query_emb)
#             if norm > 0:
#                 query_emb /= norm
#     else:
#         query_emb = belief_emb
    
#     initial_nodes = hcl.retrieve(query_emb, k=5)
#     memory_prefix = hcl.format_injection(initial_nodes)
#     pre_state = {"entropy": 0.5, "confidence": 0.5}

#     # ── User correction detection ──────────────────────────────────────────────
#     # If the user is explicitly correcting the model, demote relevant memory nodes
#     # BEFORE generation so the model doesn't retrieve and double down on wrong beliefs.
#     _correction_phrases = (
#         "that's wrong", "that is wrong", "not correct", "actually", "no,",
#         "it's not", "it is not", "that's not", "that is not", "hasn't been",
#         "has not been", "doesn't exist", "does not exist", "isn't", "is not",
#         "incorrect", "wrong", "no it's", "no, the", "not the", "it's actually"
#     )
#     prompt_lower = prompt.lower().strip()
#     is_correction = any(prompt_lower.startswith(p) or f" {p}" in prompt_lower for p in _correction_phrases)

#     if is_correction and len(hcl.HMG.nodes) > 0:
#         # Embed the prompt to find which memory nodes are being disputed
#         topic_emb = embed(prompt)
#         n_demoted = hcl.user_correction(topic_emb, prompt)
#         if n_demoted:
#             print(f"  [HCL] User correction detected — demoted {n_demoted} memory node(s)")

#     # ── Meta-question detection (e.g. "what did I ask first?") ───────────────
#     meta_phrases = ("what did i ask", "what was my first", "what was the first thing", "first thing i asked", "my first question", "what was the last", "what did i just ask", "what did i last ask")
#     if any(p in prompt_lower for p in meta_phrases):
#         # Distinguish between first question and last/recent question
#         is_first = any(p in prompt_lower for p in ["first", "earliest", "begin"])
#         if is_first:
#             q_node = hcl.retrieve_first_question()
#             if q_node:
#                 memory_prefix += f"\n[SYSTEM MEMORY: The user's very first question in this session was: '{q_node.raw_summary}']\n"
#         else:
#             q_node = hcl.retrieve_latest_question()
#             if q_node:
#                 memory_prefix += f"\n[SYSTEM MEMORY: The user's most recent question in this session was: '{q_node.raw_summary}']\n"
#     # ──────────────────────────────────────────────────────────────────────────

#     t_inference_total = 0.0   # pure model forward pass
#     t_retrieval_total = 0.0   # on_generation_step (HCL retrieval hook)
#     t_extraction_calls = 0    # how many extract_cascade calls happened
#     t_save_total = 0.0        # save_profile wall time

#     # Snapshot HCL global timings at start so we can diff at end
#     timings_start = {k: v for k, v in HCL.global_timings.items()}

#     for step in range(max_new_tokens):
#         system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."
#         if memory_prefix and ("User fact:" in memory_prefix or "[FACT]" in memory_prefix):
#             system_msg += " Always refer to the user using 'you' or 'your' when answering questions about user facts (e.g. say 'Your favorite color is blue' instead of 'My favorite color is blue')."
#         if memory_prefix and "[SYSTEM MEMORY:" in memory_prefix:
#             system_msg += " Always use the provided [SYSTEM MEMORY] block to answer questions about what the user asked first or most recently, and reference that exact question. Rely ONLY on the [SYSTEM MEMORY] block and do not look at the conversation history to determine this, since the history contains statements rather than questions."
#         if memory_prefix:
#             system_msg = f"{system_msg}\n\n{memory_prefix}"

#         # Expand pronouns using subject tracker before building the prompt
#         expanded_prompt = hcl.working_memory.expand(prompt)

#         messages = [{"role": "system", "content": system_msg}]
#         for turn in hcl.working_memory.get_messages():
#             messages.append({"role": turn["role"], "content": turn["content"]})
#         messages.append({"role": "user", "content": expanded_prompt})

#         if hasattr(tokenizer, "apply_chat_template"):
#             formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         else:
#             formatted_prompt = f"{system_msg}\n"
#             if hcl.working_memory.turns:
#                 formatted_prompt += "[Recent Conversation]\n" + hcl.working_memory.get_context() + "\n[/Recent Conversation]\n\n"
#             formatted_prompt += f"User: {expanded_prompt}\nAssistant:"

#         prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

#         if assistant_tokens:
#             assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
#             input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
#         else:
#             input_ids = prompt_ids

#         if input_ids.shape[1] > max_context_window:
#             overflow = input_ids.shape[1] - max_context_window
#             if overflow < len(assistant_tokens):
#                 assistant_tokens = assistant_tokens[overflow:]
#                 assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
#                 input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
#             else:
#                 if len(hcl.HMG.nodes) > 0:
#                     hcl.step += 1
#                 input_ids = prompt_ids[:, -max_context_window:]

#         # ── Timed: model forward pass ──
#         t0 = time.perf_counter()
#         with torch.no_grad():
#             outputs = model(input_ids)
#             logits = outputs.logits
#         t_inference_total += time.perf_counter() - t0

#         next_token_logits = logits[:, -1, :]
#         probs = torch.softmax(next_token_logits, dim=-1)
#         next_token = torch.argmax(probs, dim=-1, keepdim=True)

#         next_token_val = next_token.item()
#         assistant_tokens.append(next_token_val)

#         # ── Timed: HCL on_generation_step (retrieval hook) ──
#         t1 = time.perf_counter()
#         new_prefix = hcl.on_generation_step(input_ids, next_token_logits)
#         t_retrieval_total += time.perf_counter() - t1

#         entropy = hcl.entropy_scorer.score(next_token_logits)
#         confidence = float(torch.max(probs).item())
        
#         current_assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#         dynamic_threshold = hcl.get_dynamic_uncertainty_threshold(prompt, current_assistant_text, hcl.mode.uncertainty_threshold)

#         if entropy > dynamic_threshold and step > 5:
#             final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#             extracted_facts = hcl.user_facts.extract_from_turn(prompt)
#             for key, val in extracted_facts:
#                 hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)
#             if getattr(hcl, "hcl_lightweight", False) or extracted_facts:
#                 t2 = time.perf_counter()
#                 hcl.save_profile()
#                 t_save_total += time.perf_counter() - t2

#             # Compute final timings diff
#             timings_end = HCL.global_timings
#             extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
#             extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)

#             yield {
#                 "token": " [HCL: UNCERTAIN — REVIEW REQUIRED]",
#                 "entropy": round(entropy, 4),
#                 "confidence": round(confidence, 4),
#                 "threshold": round(dynamic_threshold, 4),
#                 "memory_nodes": get_reml_payloads(),
#                 "done": True,
#                 "uncertain": True,
#                 "stats": {
#                     "total_steps": len(assistant_tokens),
#                     "memory_nodes": len(hcl.HMG.nodes),
#                     "uncertain": True
#                 },
#                 "timings": {
#                     "avg_retrieval_ms": round((t_retrieval_total / max(len(assistant_tokens), 1)) * 1000, 2),
#                     "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
#                     "avg_save_ms": round(t_save_total * 1000, 2),
#                     "avg_inference_ms": round((t_inference_total / max(len(assistant_tokens), 1)) * 1000, 2),
#                     "extract_calls": extract_count
#                 }
#             }
#             hcl.working_memory.add_turn("user", prompt)
#             return

#         if len(hcl.HMG.nodes) > 0:
#             belief_emb = hcl.GBPS.get_active_belief_embedding()
#             query_emb = getattr(hcl, "current_query_embedding", None)
#             if query_emb is not None:
#                 if np.any(belief_emb) and not is_personal_query(prompt):
#                     query_emb = 0.6 * query_emb + 0.4 * belief_emb
#                     norm = np.linalg.norm(query_emb)
#                     if norm > 0:
#                         query_emb /= norm
#             else:
#                 query_emb = belief_emb
#             retrieved_nodes = hcl.retrieve(query_emb, k=5)
#             post_state = {"entropy": entropy, "confidence": confidence, "fallback_triggered": False}
#             hcl.score_retrieval_utility(retrieved_nodes, pre_state, post_state)
#             pre_state = post_state

#         # Decode the whole sequence to get correct spacing, then yield the delta
#         full_assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#         new_token_decoded = full_assistant_text[len(raw_previous_text):]
#         raw_previous_text = full_assistant_text
#         if new_token_decoded and previous_assistant_text:
#             if new_token_decoded[0].isdigit() and previous_assistant_text[-1].isalpha():
#                 new_token_decoded = " " + new_token_decoded
#             elif new_token_decoded[0].isalpha() and previous_assistant_text[-1].isdigit():
#                 new_token_decoded = " " + new_token_decoded
#         previous_assistant_text += new_token_decoded

#         # Stop on EOS or Qwen-style end-of-turn special tokens
#         raw_decoded = tokenizer.decode([next_token_val], skip_special_tokens=False)
#         is_stop_token = (
#             next_token_val == tokenizer.eos_token_id
#             or '<|im_end|>' in raw_decoded
#             or '<|endoftext|>' in raw_decoded
#             or '<|end|>' in raw_decoded
#         )

#         # Only yield non-empty, non-special tokens
#         if new_token_decoded:
#             yield {
#                 "token": new_token_decoded,
#                 "entropy": round(entropy, 4),
#                 "confidence": round(confidence, 4),
#                 "threshold": round(dynamic_threshold, 4),
# "memory_nodes": get_reml_payloads(),
#                 "done": False
#             }

#         # ── Confident Hallucination Check ──
#         # If the generated token completes a sentence, verify it against parametric knowledge.
#         # Only run in Safe or Balanced mode (threshold <= 0.75) to preserve max speed in Fast mode.
#         if new_token_decoded.strip() in ['.', '!', '?', '\n'] and hcl.mode.uncertainty_threshold <= 0.75:
#             full_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#             sentences = [s.strip() for s in full_text.replace('!', '.').replace('?', '.').replace('\n', '.').split('.') if s.strip()]
#             if sentences:
#                 last_sentence = sentences[-1] + "."
#                 # Only check sentences making a factual claim
#                 if should_fact_check(last_sentence, prompt):
#                     if not hcl.verify_fact(last_sentence, confidence):
#                         extracted_facts = hcl.user_facts.extract_from_turn(prompt)
#                         for key, val in extracted_facts:
#                             hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)
#                         if getattr(hcl, "hcl_lightweight", False) or extracted_facts:
#                             t2 = time.perf_counter()
#                             hcl.save_profile()
#                             t_save_total += time.perf_counter() - t2
#                         # Hallucination detected! Yield UNCERTAIN and abort
#                         timings_end = HCL.global_timings
#                         extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
#                         extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)
                        
#                         yield {
#                             "token": " [HCL: UNCERTAIN — FAILED FACT CHECK]",
#                             "entropy": round(entropy, 4),
#                             "confidence": round(confidence, 4),
#                             "memory_nodes": get_reml_payloads(),
#                             "done": True,
#                             "uncertain": True,
#                             "stats": {
#                                 "total_steps": len(assistant_tokens),
#                                 "memory_nodes": len(hcl.HMG.nodes),
#                                 "uncertain": True
#                             },
#                             "timings": {
#                                 "avg_retrieval_ms": round((t_retrieval_total / max(len(assistant_tokens), 1)) * 1000, 2),
#                                 "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
#                                 "avg_save_ms": round(t_save_total * 1000, 2),
#                                 "avg_inference_ms": round((t_inference_total / max(len(assistant_tokens), 1)) * 1000, 2),
#                                 "extract_calls": extract_count
#                             }
#                         }
#                         hcl.working_memory.add_turn("user", prompt)
#                         # ── Hallucination penalty: demote any nodes written during ──
#                         # this turn so retrieval won't return the wrong answer.
#                         for node in hcl.HMG.nodes.values():
#                             if hasattr(node, "sml_addr") and node.sml_addr is not None:
#                                 try:
#                                     addr = node.sml_addr
#                                     if hcl.SML.occupied[addr]:
#                                         # Halve the confidence weight — marked as unreliable
#                                         hcl.SML.surface[addr, hcl.SML.dim] *= 0.3
#                                 except Exception:
#                                     pass
#                         return

#         if is_stop_token:
#             break

#         if next_token_val == tokenizer.eos_token_id:
#             break

#         if new_prefix != memory_prefix:
#             memory_prefix = new_prefix

#     # ── Final save timing ──
#     extracted_facts = hcl.user_facts.extract_from_turn(prompt)
#     for key, val in extracted_facts:
#         hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)
#     if getattr(hcl, "hcl_lightweight", False) or extracted_facts:
#         t2 = time.perf_counter()
#         hcl.save_profile()
#         t_save_total += time.perf_counter() - t2

#     # Diff HCL global_timings to get real extract_cascade measurements
#     timings_end = HCL.global_timings
#     extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
#     extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)
#     n_steps = max(len(assistant_tokens), 1)

#     yield {
#         "token": "",
#         "entropy": 0.0,
#         "confidence": 1.0,
#         "memory_nodes": get_reml_payloads(),
#         "done": True,
#         "uncertain": False,
#         "stats": {
#             "total_steps": len(assistant_tokens),
#             "memory_nodes": len(hcl.HMG.nodes),
#             "uncertain": False
#         },
#         "timings": {
#             "avg_retrieval_ms": round((t_retrieval_total / n_steps) * 1000, 2),
#             "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
#             "avg_save_ms": round(t_save_total * 1000, 2),
#             "avg_inference_ms": round((t_inference_total / n_steps) * 1000, 2),
#             "extract_calls": extract_count
#         }
#     }
    
#     hcl.working_memory.add_turn("user", prompt)
#     final_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
#     if final_text.strip():
#         hcl.working_memory.add_turn("assistant", final_text)
    
#     # ── QUESTION_ANCHOR / EPISODIC: write user turn to the lattice ────────
#     # Bypasses the entropy write-gate. User questions are QUESTION_ANCHORs,
#     # statements are EPISODIC nodes.
#     try:
#         q_emb = embed(prompt)
#         is_q = is_question(prompt)
#         q_node = Node(
#             node_type=NodeType.QUESTION_ANCHOR if is_q else NodeType.EPISODIC,
#             embedding=q_emb,
#             raw_summary=prompt[:256],
#             weight=1.0,
#             confidence=1.0,   # user asked/stated = certain
#         )
#         hcl.assign_to_cluster(q_node)
#         hcl.HMG.insert(q_node)
#         hcl.ANN.update(q_node)
#         payload = {
#             "id": q_node.id,
#             "type": q_node.node_type.value,
#             "summary": q_node.raw_summary,
#             "cluster_id": q_node.cluster_id
#         }
#         addr = hcl.SML.write(q_emb, confidence=1.0, payload=payload)
#         setattr(q_node, "sml_addr", addr)
#     except Exception as e:
#         print(f"  [HCL] User turn write failed: {e}")

#     # Persist the lattice after every conversation turn (all modes)
#     if np.any(hcl.SML.occupied):
#         try:
#             hcl.SML.save_to_disk(hcl._lattice_path, hcl._lattice_passphrase)
#         except Exception as e:
#             print(f"  [HCL] Post-turn lattice save failed: {e}")
import os
import re
import sys
import json
import time
import uuid
import hashlib
import hmac
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch

from hail_core.lattice import StratifiedMemoryLattice
from .tt_linear import TTLinear

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS: Query intent classification
# ═══════════════════════════════════════════════════════════════════════════════

def is_personal_query(query: str) -> bool:
    """Detects if the user is asking about their own personal information."""
    if not query:
        return False
    q = query.lower()
    patterns = {
        "favourite": ["colour", "color", "food", "movie", "song", "subject", "topic", "plane", "car", "thing"],
        "favorite": ["colour", "color", "food", "movie", "song", "subject", "topic", "plane", "car", "thing"],
        "my": ["name", "age", "birthday", "goal", "dream", "status", "struggle", "plane", "colour", "color"],
        "what did i": ["say", "ask", "tell you"],
        "do you remember": [],
        "who am i": [],
        "what is my": [],
        "whats my": [],
        "what's my": [],
    }
    for trigger, contexts in patterns.items():
        if trigger in q:
            if not contexts or any(c in q for c in contexts):
                return True
    return False


_SOCIAL_SIGNALS = {
    "my favourite", "my favorite", "my fav",
    "i like", "i love", "i want", "i enjoy", "i adore",
    "i've", "i have", "i had",
    "i flew", "i went", "i think", "i feel", "i believe",
    "i do ", "i did ", "i am ", "i'm ",
    "nice", "great", "awesome", "cool", "amazing",
}


def is_social_utterance(text: str) -> bool:
    """Returns True if the user is sharing experiences, opinions, feelings — not asking for facts."""
    if not text:
        return False
    t = text.lower().strip()
    # Strong signal: multiple exclamation marks
    if t.count('!') >= 2:
        return True
    # Strong signal: emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    if emoji_pattern.search(t):
        return True
    return any(sig in t for sig in _SOCIAL_SIGNALS)


def is_meta_question(text: str) -> bool:
    """Detects if user is asking about the conversation itself."""
    if not text:
        return False
    t = text.lower().strip()
    meta_patterns = [
        "what did i ask", "what was my first", "what was the first thing",
        "first thing i asked", "my first question", "what was the last",
        "what did i just ask", "what did i last ask", "what did i say",
        "what have i told you", "what do you know about me",
        "tell me everything i", "what did we talk about",
    ]
    return any(p in t for p in meta_patterns)


def is_question(text: str) -> bool:
    """Returns True if the input text represents a question."""
    if not text:
        return False
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    words = t.split()
    if not words:
        return False
    question_starters = {
        "what", "when", "where", "who", "why", "how", "which",
        "did", "do", "does", "can", "could", "would", "will",
        "is", "are", "was", "were", "have", "has", "had"
    }
    first_word = re.sub(r'[^\w]', '', words[0])
    return first_word in question_starters


def should_fact_check(output_text: str, user_query: str) -> bool:
    """Only fact-check statements that make objective claims, avoiding personal/social fact-checking."""
    if is_personal_query(user_query):
        return False
    if is_social_utterance(user_query):
        return False
    if is_meta_question(user_query):
        return False
    t = output_text.lower()
    if any(phrase in t for phrase in ["your favourite", "your name", "you told me", "your status", "your struggle", "your favorite"]):
        return False
    if len(t.split()) < 6:
        return False
    return any(verb in t for verb in [" is ", " was ", " are ", " were ", " has ", " had ", " consists of "])


def is_complete_sentence(text: str) -> bool:
    """Check if text has enough structure to be fact-checked."""
    t = text.strip()
    if len(t.split()) < 6:
        return False
    factual_verbs = [" is ", " are ", " was ", " were ", " has ", " have ", " had ", " consists of ", " contains "]
    return any(v in t.lower() for v in factual_verbs)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE TYPE & GOVERNANCE
# ═══════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    EPISODIC        = "EPISODIC"
    SEMANTIC        = "SEMANTIC"
    BELIEF          = "BELIEF"
    AXIOM           = "AXIOM"
    QUESTION_ANCHOR = "QUESTION_ANCHOR"


AXIOM_PROPERTIES: dict = {
    'promotion_threshold':   100,
    'demotion_threshold':    3,
    'demotion_requires':     'user_explicit',
    'max_axiom_lifetime':    86400 * 30,
    'bias_audit_frequency':  100,
    'source_diversity_min':  3,
}


def get_sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    text_lower = text.lower()
    pos_words = {"love", "like", "favorite", "favourite", "great", "awesome", "cool", "amazing", "good", "happy", "excited"}
    neg_words = {"struggle", "difficult", "trouble", "bad", "sad", "fail", "incorrect", "wrong", "hate", "sorry", "tired"}
    pos_count = sum(text_lower.count(w) for w in pos_words)
    neg_count = sum(text_lower.count(w) for w in neg_words)
    excl_count = text.count("!")
    score = (pos_count - neg_count) * 0.2 + excl_count * 0.1
    return float(np.clip(score, -1.0, 1.0))


def compute_emotional_embedding(base_emb: np.ndarray, text: str) -> np.ndarray:
    S = get_sentiment_score(text)
    if S == 0.0:
        return base_emb.copy()
    dim = len(base_emb)
    rng = np.random.default_rng(2026)
    E_pos = rng.standard_normal(dim)
    norm_pos = np.linalg.norm(E_pos)
    if norm_pos > 0:
        E_pos /= norm_pos
    E_neg = rng.standard_normal(dim)
    norm_neg = np.linalg.norm(E_neg)
    if norm_neg > 0:
        E_neg /= norm_neg
    blend_ratio = abs(S) * 0.4
    anchor = E_pos if S > 0 else E_neg
    emotional_emb = (1.0 - blend_ratio) * base_emb + blend_ratio * anchor
    norm = np.linalg.norm(emotional_emb)
    if norm > 0:
        emotional_emb /= norm
    return emotional_emb


class Node:
    __slots__ = (
        "id", "node_type", "factual_embedding", "emotional_embedding", "raw_summary", "weight", "state",
        "created_at", "last_accessed", "access_count", "cluster_id",
        "provenance", "contradictions", "confidence", "utility_score",
        "sml_addr", "epistemic_divergence", "temporal_bindings",
        "verification_count", "abstraction_score",
    )
    def __init__(
        self,
        node_type: NodeType,
        embedding: np.ndarray,
        raw_summary: str,
        weight: float = 0.8,
        state: int = 1,
        created_at: int = 0,
        last_accessed: int = 0,
        access_count: int = 0,
        cluster_id: Optional[int] = None,
        provenance: Optional[List[str]] = None,
        contradictions: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        utility_score: float = 0.5,
        id: Optional[str] = None,
        verification_count: int = 0,
        abstraction_score: float = 0.0,
        emotional_embedding: Optional[np.ndarray] = None
    ):
        self.id = id if id is not None else str(uuid.uuid4())
        self.node_type = node_type
        self.factual_embedding = embedding
        if emotional_embedding is not None:
            self.emotional_embedding = emotional_embedding
        else:
            self.emotional_embedding = compute_emotional_embedding(embedding, raw_summary)
        self.raw_summary = raw_summary
        self.weight = weight
        self.state = state
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.access_count = access_count
        self.cluster_id = cluster_id
        self.provenance = provenance if provenance is not None else []
        self.contradictions = contradictions if contradictions is not None else []
        self.confidence = confidence if confidence is not None else (0.85 if node_type == NodeType.BELIEF else 0.5)
        self.utility_score = utility_score
        self.verification_count = verification_count
        self.abstraction_score = abstraction_score

    @property
    def embedding(self) -> np.ndarray:
        return self.factual_embedding

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        self.factual_embedding = value


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class VectorIndex:
    def query(self, embedding: np.ndarray, k: int) -> List[Node]:
        raise NotImplementedError
    def update(self, node: Node) -> None:
        raise NotImplementedError
    def remove(self, node: Node) -> None:
        raise NotImplementedError


class KNNIndex(VectorIndex):
    def __init__(self, dim: int = 768, lsh_bits: int = 4):
        self.dim = dim
        self.nodes: List[Node] = []
        self.lsh_bits = lsh_bits
        rng = np.random.default_rng(42)
        self.hyperplanes = rng.standard_normal((lsh_bits, dim))
        for i in range(lsh_bits):
            norm = np.linalg.norm(self.hyperplanes[i])
            if norm > 0:
                self.hyperplanes[i] /= norm
        self.buckets: Dict[int, List[Node]] = {i: [] for i in range(2**lsh_bits)}

    def _get_bucket_key(self, embedding: np.ndarray) -> int:
        projections = np.dot(self.hyperplanes, embedding)
        key = 0
        for i, val in enumerate(projections):
            if val >= 0:
                key |= (1 << i)
        return key

    def query(self, embedding: np.ndarray, k: int) -> List[Node]:
        if not self.nodes:
            return []
        if len(self.nodes) <= 100:
            candidates = self.nodes
        else:
            key = self._get_bucket_key(embedding)
            candidates = self.buckets.get(key, [])
            if len(candidates) < k:
                candidates = list(candidates)
                checked_keys = {key}
                for bit in range(self.lsh_bits):
                    neighbor_key = key ^ (1 << bit)
                    if neighbor_key not in checked_keys:
                        candidates.extend(self.buckets.get(neighbor_key, []))
                        checked_keys.add(neighbor_key)
            if len(candidates) < k:
                candidates = self.nodes
        similarities = []
        for node in candidates:
            sim = float(np.dot(node.embedding, embedding))
            similarities.append((sim, node))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in similarities[:k]]

    def update(self, node: Node) -> None:
        norm = np.linalg.norm(node.embedding)
        if norm > 0:
            node.embedding = node.embedding / norm
        if node not in self.nodes:
            self.nodes.append(node)
            key = self._get_bucket_key(node.embedding)
            self.buckets[key].append(node)

    def remove(self, node: Node) -> None:
        if node in self.nodes:
            self.nodes.remove(node)
            key = self._get_bucket_key(node.embedding)
            if node in self.buckets[key]:
                self.buckets[key].remove(node)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.clusters: Dict[int, List[str]] = {}

    def insert(self, node: Node) -> None:
        self.nodes[node.id] = node

    def remove(self, node: Node) -> None:
        if node.id in self.nodes:
            del self.nodes[node.id]
        for cid in list(self.clusters.keys()):
            if node.id in self.clusters[cid]:
                self.clusters[cid].remove(node.id)
            if not self.clusters[cid]:
                del self.clusters[cid]

    def __iter__(self):
        return iter(self.nodes.values())

    def __len__(self):
        return len(self.nodes)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUNDED BELIEF PATH SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class GroundedBeliefPathSearch:
    def __init__(self, embedding_dim: int = 768):
        self.active_beliefs: List[Node] = []
        self.embedding_dim = embedding_dim
        self.new_belief_flag = False

    def get_active_beliefs(self) -> List[Node]:
        return self.active_beliefs

    def get_active_belief_embedding(self) -> np.ndarray:
        if not self.active_beliefs:
            return np.zeros(self.embedding_dim)
        embs = [node.embedding for node in self.active_beliefs]
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm
        return mean_emb

    def current_embedding(self) -> np.ndarray:
        return self.get_active_belief_embedding()

    def detect_new_belief(self) -> bool:
        flag = self.new_belief_flag
        self.new_belief_flag = False
        return flag

    def add_belief(self, node: Node) -> None:
        self.active_beliefs.append(node)
        self.new_belief_flag = True
        if len(self.active_beliefs) > 5:
            self.active_beliefs.pop(0)


# ═══════════════════════════════════════════════════════════════════════════════
# BELIEF TRANSITION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BeliefTransitionModel:
    def __init__(self):
        self.transitions: Dict[Tuple[int, int], int] = {}
        self.last_cluster_ids: List[int] = []

    def update(self, active_beliefs: List[Node]) -> None:
        if not active_beliefs:
            return
        current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
        if not current_clusters:
            return
        if self.last_cluster_ids:
            for prev in self.last_cluster_ids:
                for curr in current_clusters:
                    key = (prev, curr)
                    self.transitions[key] = self.transitions.get(key, 0) + 1
        self.last_cluster_ids = current_clusters

    def predict_next(self, active_beliefs: List[Node], HMG: MemoryGraph, top_k: int = 3) -> List[int]:
        if not active_beliefs:
            return []
        current_clusters = [node.cluster_id for node in active_beliefs if node.cluster_id is not None]
        if not current_clusters:
            return []
        scores: Dict[int, int] = {}
        for prev in current_clusters:
            for (src, dest), count in self.transitions.items():
                if src == prev:
                    scores[dest] = scores.get(dest, 0) + count
        if not scores:
            return []
        sorted_clusters = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_clusters[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════════
# WARM CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class WarmCache:
    def __init__(self):
        self.cached_nodes: List[Node] = []

    def stage(self, candidates: List[Node]) -> None:
        self.cached_nodes = candidates

    def check(self, query_emb: np.ndarray, threshold: float = 0.85) -> Optional[List[Node]]:
        if not self.cached_nodes:
            return None
        relevant = []
        for node in self.cached_nodes:
            sim = float(np.dot(node.embedding, query_emb))
            if sim >= threshold:
                relevant.append(node)
        return relevant if relevant else None

    def clear(self) -> None:
        self.cached_nodes = []


# ═══════════════════════════════════════════════════════════════════════════════
# MODE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class ModeController:
    def __init__(self, mode: str = "balanced"):
        self.λ = 0.15
        self.θ_write = 0.65
        self.N_collapse = 5
        self.N_prune = 25
        self.uncertainty_threshold = 0.75
        self.verify_votes = 1
        self.no_votes_to_block = 1
        self.verify_high_conf_bypass = 1.0
        self.apply_mode(mode)

    def apply_mode(self, mode: str) -> None:
        self.mode_name = mode
        if mode == "fast":
            self.λ = 0.30
            self.θ_write = 0.75
            self.N_collapse = 3
            self.N_prune = 10
            self.uncertainty_threshold = 0.85
        elif mode == "balanced":
            self.λ = 0.15
            self.θ_write = 0.65
            self.N_collapse = 5
            self.N_prune = 25
            self.uncertainty_threshold = 0.75
        elif mode == "agentic":
            self.λ = 0.15
            self.θ_write = 0.65
            self.N_collapse = 5
            self.N_prune = 25
            self.uncertainty_threshold = 0.75
        elif mode == "safe":
            self.λ = 0.05
            self.θ_write = 0.55
            self.N_collapse = 10
            self.N_prune = 50
            self.uncertainty_threshold = 0.65
        elif mode == "eval":
            self.λ = 0.00
            self.θ_write = 0.50
            self.N_collapse = 999999
            self.N_prune = 999999
            self.uncertainty_threshold = 0.70
            self.verify_votes = 3
            self.no_votes_to_block = 2
            self.verify_high_conf_bypass = 0.90
        elif mode == "persistent":
            self.λ = 0.10
            self.θ_write = 0.65
            self.N_collapse = 7
            self.N_prune = 30
            self.uncertainty_threshold = 0.75

    def adapt_to_profile(self, profile: "CognitiveProfile") -> None:
        # Only adapt if profile has enough history to be statistically meaningful
        if profile and profile.entropy_baseline and getattr(profile, "session_count", 0) > 10:
            self.θ_write = profile.entropy_baseline + 0.1
            self.uncertainty_threshold = profile.entropy_baseline + 0.2

    @property
    def lambda_(self) -> float:
        return self.λ

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        self.λ = value

    @property
    def theta_write(self) -> float:
        return self.θ_write

    @theta_write.setter
    def theta_write(self, value: float) -> None:
        self.θ_write = value


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoHelper:
    @staticmethod
    def derive_key(password: str, salt: bytes, iterations: int = 10000) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=32)

    @staticmethod
    def encrypt(data: str, password: str) -> bytes:
        salt = os.urandom(16)
        key = CryptoHelper.derive_key(password, salt)
        raw_bytes = data.encode("utf-8")
        ciphertext = bytearray()
        counter = 0
        for i in range(0, len(raw_bytes), 32):
            block_key = hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest()
            chunk = raw_bytes[i:i+32]
            for b_idx, b in enumerate(chunk):
                ciphertext.append(b ^ block_key[b_idx])
            counter += 1
        mac = hashlib.sha256(key + salt + ciphertext).digest()
        return salt + mac + bytes(ciphertext)

    @staticmethod
    def decrypt(encrypted_data: bytes, password: str) -> str:
        if len(encrypted_data) < 48:
            raise ValueError("Invalid encrypted profile data: payload is too short.")
        salt = encrypted_data[:16]
        mac = encrypted_data[16:48]
        ciphertext = encrypted_data[48:]
        key = CryptoHelper.derive_key(password, salt)
        expected_mac = hashlib.sha256(key + salt + ciphertext).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("Integrity check failed: invalid profile passphrase or data corruption.")
        decrypted = bytearray()
        counter = 0
        for i in range(0, len(ciphertext), 32):
            block_key = hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest()
            chunk = ciphertext[i:i+32]
            for b_idx, b in enumerate(chunk):
                decrypted.append(b ^ block_key[b_idx])
            counter += 1
        return decrypted.decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# USER FACTS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class UserFacts:
    """
    Structured registry for user personal facts.
    Extracts from natural language and retrieves via keyword matching.
    Persists through CognitiveProfile.
    """
    def __init__(self):
        self.facts: Dict[str, dict] = {}

    def add_fact(self, key: str, value: Any, confidence: float = 1.0, source_turn: int = 0) -> None:
        self.facts[key] = {
            "value": value,
            "confidence": confidence,
            "source_turn": source_turn,
            "timestamp": time.time()
        }

    def extract_from_turn(self, user_text: str) -> List[Tuple[str, Any]]:
        """Extract personal facts from user utterance using regex patterns."""
        extracted = []
        text = user_text.strip().lower()

        # Pattern 1: "my favourite/favorite [noun] is [value]"
        m1 = re.search(
            r"\bmy\s+favou?rite\s+([a-z0-9_\-\s]{2,30})\s+(?:is|was)\s+([a-z0-9_\-\s\.\"]{1,50})",
            text
        )
        if m1:
            key = m1.group(1).strip().replace(" ", "_")
            val = m1.group(2).strip()
            if "favourite" not in key and "favorite" not in key:
                key = f"favourite_{key}"
            extracted.append((key, val))

        # Pattern 2: "my [noun] is [value]" (broader catch)
        if not m1:
            m2 = re.search(
                r"\bmy\s+([a-z0-9_\-\s]{2,30})\s+(?:is|was)\s+([a-z0-9_\-\s\.\"]{1,50})",
                text
            )
            if m2:
                key_raw = m2.group(1).strip()
                exclude_keys = {"opinion", "answer", "question", "guess", "turn", "prompt", "idea", "thought", "answer"}
                if key_raw not in exclude_keys:
                    key = key_raw.replace(" ", "_")
                    val = m2.group(2).strip()
                    extracted.append((key, val))

        # Pattern 3: "i struggle with / have trouble with / find difficult"
        m3 = re.search(
            r"\bi\s+(?:struggle\s+with|have\s+trouble\s+with|find\s+.*?\s+difficult)\s+([a-z0-9_\-\s\.\"]{3,100})",
            text
        )
        if m3:
            val = m3.group(1).strip()
            extracted.append(("struggle_topic", val))

        # Pattern 4: "i am [value]" (identity statements)
        m4 = re.search(r"\bi\s+am\s+([a-z0-9_\-\s\.\"]{3,100})", text)
        if m4:
            val = m4.group(1).strip()
            exclude_vals = {"ready", "sure", "sorry", "fine", "ok", "okay", "happy", "sad", "tired", "here", "back"}
            if not any(val.startswith(ev) for ev in exclude_vals):
                extracted.append(("user_identity", val))

        # Pattern 5: "i like/love/enjoy [value]"
        m5 = re.search(r"\bi\s+(?:like|love|enjoy|adore)\s+([a-z0-9_\-\s\.\"]{2,50})", text)
        if m5 and not m1:
            val = m5.group(1).strip()
            extracted.append(("liked_thing", val))

        # Pattern 6: "i want to [verb]" (goals/aspirations)
        m6 = re.search(r"\bi\s+want\s+to\s+([a-z0-9_\-\s\.\"]{3,100})", text)
        if m6:
            val = m6.group(1).strip()
            extracted.append(("user_goal", val))

        # Pattern 7: "i've / i have [past experience]"
        m7 = re.search(r"\bi(?:'ve|\s+have)\s+([a-z0-9_\-\s\.\"]{3,100})", text)
        if m7:
            val = m7.group(1).strip()
            if len(val.split()) >= 2:
                extracted.append(("user_experience", val))

        return extracted

    def retrieve_relevant_facts(self, query: str) -> List[str]:
        """Find stored facts relevant to the query. Returns formatted strings for injection."""
        if not self.facts or not query:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        matched_lines = []

        for key, info in self.facts.items():
            key_words = set(key.split("_"))
            val_str = str(info.get("value", "")).lower()
            val_words = set(re.findall(r"\w+", val_str))

            key_overlap = key_words.intersection(query_words)
            key_ratio = len(key_overlap) / len(key_words) if key_words else 0.0

            val_overlap = val_words.intersection(query_words)
            val_ratio = len(val_overlap) / len(val_words) if val_words else 0.0

            has_color = ("color" in query_words or "colour" in query_words) and ("color" in key_words or "colour" in key_words)
            has_fav = ("favorite" in query_words or "favourite" in query_words) and ("favorite" in key_words or "favourite" in key_words)
            has_plane = "plane" in query_words and ("plane" in key_words or "plane" in val_str)

            is_match = (
                key_ratio >= 0.4
                or val_ratio >= 0.5
                or (has_color and has_fav)
                or has_plane
                or key.replace("_", " ") in query.lower()
                or val_str in query.lower()
            )

            if is_match:
                meta_name = key.replace("_", " ")
                if "favourite" in meta_name and "favorite" in query.lower():
                    meta_name = meta_name.replace("favourite", "favorite")
                matched_lines.append(
                    f"User fact: The user's {meta_name} is '{info['value']}'. "
                    f"(Address the user directly using 'your' or 'you'.)"
                )

        return matched_lines

    def get_fact(self, key: str) -> Optional[dict]:
        return self.facts.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return self.facts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserFacts":
        uf = cls()
        uf.facts = data if data else {}
        return uf


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.recurring_clusters: List[int] = []
        self.expertise_levels: Dict[str, float] = {}
        self.communication_style: str = "BALANCED"
        self.entropy_baseline: float = 0.5
        self.error_patterns: List[str] = []
        self.preferred_modes: Dict[str, str] = {}
        self.session_count: int = 0
        self.total_inference_steps: int = 0
        self.user_facts: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "recurring_clusters": self.recurring_clusters,
            "expertise_levels": self.expertise_levels,
            "communication_style": self.communication_style,
            "entropy_baseline": self.entropy_baseline,
            "error_patterns": self.error_patterns,
            "preferred_modes": self.preferred_modes,
            "session_count": self.session_count,
            "total_inference_steps": self.total_inference_steps,
            "user_facts": self.user_facts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveProfile":
        profile = cls(data["user_id"])
        profile.recurring_clusters = data.get("recurring_clusters", [])
        profile.expertise_levels = data.get("expertise_levels", {})
        profile.communication_style = data.get("communication_style", "BALANCED")
        profile.entropy_baseline = data.get("entropy_baseline", 0.5)
        profile.error_patterns = data.get("error_patterns", [])
        profile.preferred_modes = data.get("preferred_modes", {})
        profile.session_count = data.get("session_count", 0)
        profile.total_inference_steps = data.get("total_inference_steps", 0)
        profile.user_facts = data.get("user_facts", {})
        return profile

    def save(self, filepath: str, password: str) -> None:
        data_str = json.dumps(self.to_dict())
        encrypted = CryptoHelper.encrypt(data_str, password)
        with open(filepath, "wb") as f:
            f.write(encrypted)

    @classmethod
    def load(cls, filepath: str, password: str) -> "CognitiveProfile":
        with open(filepath, "rb") as f:
            encrypted = f.read()
        decrypted_str = CryptoHelper.decrypt(encrypted, password)
        return cls.from_dict(json.loads(decrypted_str))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class HydrusOptEntropyScorer:
    MATH_PATTERN = re.compile(r"[0-9]+\s*[\*\+\-\/\=]")
    def score(self, token_distribution: Any, context_ids: Optional[torch.Tensor] = None, tokenizer: Optional[Any] = None) -> float:
        if isinstance(token_distribution, torch.Tensor):
            logits = token_distribution.float()
            if len(logits.shape) > 1:
                logits = logits[-1]
            probs = torch.softmax(logits, dim=-1)
            top_k = min(50, probs.shape[-1])
            top_probs, _ = torch.topk(probs, top_k, dim=-1)
            top_probs = top_probs / top_probs.sum().clamp(min=1e-10)
            raw = -(top_probs * top_probs.clamp(min=1e-10).log()).sum()
            max_e = torch.log(torch.tensor(float(top_k), device=probs.device))
            entropy = (raw / max_e).item()
        else:
            probs = np.array(token_distribution)
            probs = np.clip(probs, 1e-10, 1.0)
            probs = probs / np.sum(probs)
            top_k = min(50, len(probs))
            sorted_probs = sorted(probs, reverse=True)[:top_k]
            top_probs = np.array(sorted_probs)
            top_probs = top_probs / np.sum(top_probs)
            raw = -np.sum(top_probs * np.log(top_probs))
            max_e = np.log(top_k)
            entropy = raw / max_e

        if context_ids is not None and tokenizer is not None:
            context = tokenizer.decode(context_ids[0][-20:])
            if self.MATH_PATTERN.search(context):
                entropy += 0.15
        return min(1.0, float(entropy))


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def embed(text: str, model: Any = None, tokenizer: Any = None, dim: int = 768) -> np.ndarray:
    if model is not None and tokenizer is not None:
        try:
            device = model.device
            inputs = tokenizer(text, return_tensors="pt").to(device)
            embed_layer = None
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                embed_layer = model.model.embed_tokens
            elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                embed_layer = model.transformer.wte

            if embed_layer is not None:
                with torch.no_grad():
                    embeds = embed_layer(inputs["input_ids"])
                    mean_embed = embeds.mean(dim=1).squeeze(0)
                    hidden_dim = mean_embed.shape[0]
                    if hidden_dim == dim:
                        vec = mean_embed.cpu().numpy()
                    else:
                        rng = np.random.default_rng(42)
                        proj = rng.standard_normal((hidden_dim, dim)) / np.sqrt(hidden_dim)
                        vec = mean_embed.cpu().numpy() @ proj
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    return vec
        except Exception:
            pass

    # Fallback hashing trick
    words = text.lower().split()
    vec = np.zeros(dim)
    for word in words:
        for offset in range(3):
            h = hash(f"{word}_{offset}")
            idx = abs(h) % dim
            sign = 1 if h > 0 else -1
            vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ═══════════════════════════════════════════════════════════════════════════════
# DREAM STATE
# ═══════════════════════════════════════════════════════════════════════════════

class DreamState:
    def __init__(self, dim: int = 64, threshold: float = 0.3):
        self.dreams: Dict[int, dict] = {}
        self.dim = dim
        self.threshold = threshold
        self._next_id = 0

    def _generate_impressionistic_summary(self, cold_memories: List[dict]) -> str:
        summaries = [m.get('summary', '').strip() for m in cold_memories if m.get('summary')]
        if not summaries:
            return "impressionistic memory"

        def clean(s):
            s = re.sub(r'\[.*?\]', '', s)
            s = re.sub(r'[^\w\s\-\']', ' ', s)
            return " ".join(s.split())

        cleaned = [clean(s) for s in summaries]
        cleaned = [s for s in cleaned if s]
        if not cleaned:
            return "impressionistic memory"
        if len(cleaned) == 1:
            return f"fragment of: {cleaned[0][:40]}"

        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        concepts = []
        for text in cleaned:
            words = [w.lower() for w in text.split() if w.lower() not in stopwords]
            if words:
                concepts.append(" ".join(words[:3]))
        if not concepts:
            concepts = [c[:30] for c in cleaned[:3]]
        if len(concepts) == 2:
            return f"amalgam of {concepts[0]} fading into {concepts[1]}"
        return f"composite of {concepts[0]} and {concepts[1]} overlaid with {concepts[2]}"

    def compress_to_dream(self, cold_memories: List[dict]) -> int:
        if not cold_memories:
            return -1
        embeddings = np.stack([m['embedding'] for m in cold_memories])
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        valence = float(np.mean([m.get('temperature', 0.5) for m in cold_memories]))
        scatter = float(np.std(embeddings, axis=0).mean())
        abstract = self._generate_impressionistic_summary(cold_memories)
        dream_id = self._next_id
        self._next_id += 1
        self.dreams[dream_id] = {
            'centroid': centroid,
            'valence': valence,
            'scatter': scatter,
            'trigger_radius': scatter * 2.0 + 0.1,
            'abstract_summary': abstract,
            'member_count': len(cold_memories),
            'created_at': time.time(),
        }
        return dream_id

    def feel_deja_vu(self, query_embedding: np.ndarray) -> Optional[dict]:
        best: Optional[dict] = None
        best_strength = self.threshold
        for dream_id, dream in self.dreams.items():
            dist = float(np.linalg.norm(query_embedding - dream['centroid']))
            deja_vu = 1.0 - (dist / (dream['trigger_radius'] + 1e-8))
            if deja_vu > best_strength:
                best_strength = deja_vu
                best = {
                    'type': 'deja_vu',
                    'strength': round(float(deja_vu), 4),
                    'impression': dream['abstract_summary'],
                    'dream_id': dream_id,
                }
        return best


# ═══════════════════════════════════════════════════════════════════════════════
# SUBJECT TRACKING BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectTrackingBuffer:
    _PRONOUNS = {"it", "its", "it's", "they", "them", "their", "this", "that"}
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "to", "for", "on", "at", "by", "or", "and", "but",
        "what", "which", "that", "this", "does", "do", "did", "have", "has",
        "how", "why", "when", "where", "who", "will", "would", "could", "should",
        "cell", "body", "human", "help"
    }

    def __init__(self, max_turns: int = 8, max_chars_per_turn: int = 384):
        import collections
        self.turns = collections.deque(maxlen=max_turns)
        self.max_chars = max_chars_per_turn
        self.active_subject: str = ""
        self._root_turn: Optional[dict] = None

    @classmethod
    def _extract_subject(cls, text: str) -> str:
        first_sentence = text.split(".")[0].split("?")[0].split("!")[0]
        words = first_sentence.lower().split()
        for verb in ("is", "are", "was", "were"):
            if verb in words:
                idx = words.index(verb)
                candidates_after = [w for w in words[idx+1:idx+3]
                                    if w not in ("a", "an", "the") and len(w) >= 4]
                if candidates_after:
                    return candidates_after[0]
        tokens = re.findall(r"[a-zA-Z]{4,}", first_sentence.lower())
        candidates = [t for t in tokens if t not in cls._STOPWORDS]
        return candidates[0] if candidates else ""

    def expand(self, query: str) -> str:
        if not self.active_subject:
            return query
        result = query
        for pronoun in sorted(self._PRONOUNS, key=len, reverse=True):
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            if re.search(pattern, result, flags=re.IGNORECASE):
                result = re.sub(pattern, self.active_subject, result, flags=re.IGNORECASE)
                break
        return result

    def add_turn(self, role: str, content: str):
        truncated = content[:self.max_chars]
        entry = {"role": role, "content": truncated}
        self.turns.append(entry)
        if role == "assistant":
            noun = self._extract_subject(truncated)
            if noun and len(noun) > 3:
                self.active_subject = noun
                if self._root_turn is None:
                    user_turns = [t for t in self.turns if t["role"] == "user"]
                    if user_turns:
                        self._root_turn = user_turns[0]

    def get_context(self) -> str:
        return "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in self.turns])

    def get_messages(self) -> list:
        turns = list(self.turns)
        if self._root_turn and self._root_turn not in turns:
            turns = [self._root_turn] + turns
        return turns

    def clear(self):
        self.turns.clear()
        self.active_subject = ""
        self._root_turn = None


# ═══════════════════════════════════════════════════════════════════════════════
# HYDRUS COGNITIVE LAYER (HCL)
# ═══════════════════════════════════════════════════════════════════════════════

class HCL:
    global_timings = {
        "on_generation_step": 0.0,
        "on_generation_step_count": 0,
        "extract_cascade": 0.0,
        "extract_cascade_count": 0,
        "save_profile": 0.0,
        "save_profile_count": 0
    }

    def __init__(self, model: Any, tokenizer: Any, mode: str = "balanced",
                 user_id: str = "default_user", insecure_dev_mode: bool = False,
                 hcl_lightweight: bool = False, profile_timing: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.HMG = MemoryGraph()
        self.ANN = KNNIndex(dim=768)
        self.SML = StratifiedMemoryLattice(lattice_size=65536, dim=768)
        self.GBPS = GroundedBeliefPathSearch(embedding_dim=768)
        self.grounding_matrix = TTLinear(8, 8, 12, rank=8)
        if hasattr(self.model, "device"):
            self.grounding_matrix.to(self.model.device)
        self.entropy_scorer = HydrusOptEntropyScorer()
        self.mode = ModeController(mode)
        self.step = 0
        self.warm_cache = WarmCache()
        self.transition_model = BeliefTransitionModel()
        self.hcl_lightweight = hcl_lightweight
        self.profile_timing = profile_timing
        self.working_memory = SubjectTrackingBuffer(max_turns=8)
        self.dream_state = DreamState(dim=self.SML.dim)
        self.user_facts = UserFacts()
        self._facts_path = f"facts_{user_id}.json"
        if os.path.exists(self._facts_path):
            try:
                with open(self._facts_path, "r", encoding="utf-8") as f:
                    facts_data = json.load(f)
                self.user_facts = UserFacts.from_dict(facts_data)
                print(f"  [HCL] Loaded {len(self.user_facts.facts)} user facts from {self._facts_path}")
            except Exception as e:
                print(f"  [HCL] Failed to load user facts from {self._facts_path}: {e}")

        # Conversation mode tracking
        self.conversation_mode = "factual"
        self.turns_since_social = 0
        self.social_mode_persistence = 3

        # Lattice passphrase
        _passphrase = os.getenv("HCL_PROFILE_KEY")
        if not _passphrase:
            _passphrase = "FallbackInsecureDevKey2026"
        self._lattice_passphrase = _passphrase
        self._lattice_path = f"lattice_{user_id}.hcl"

        if os.path.exists(self._lattice_path):
            try:
                self.SML.load_from_disk(self._lattice_path, self._lattice_passphrase)
                self._rebuild_hmg_from_sml()
            except Exception as e:
                print(f"  [HCL] Lattice load failed ({e}). Starting fresh.")

        # Profile handling
        self.profile: Optional[CognitiveProfile] = None
        if mode == "persistent":
            passphrase = os.getenv("HCL_PROFILE_KEY")
            if not passphrase:
                if insecure_dev_mode:
                    print("\n  [HCL WARNING] HCL_PROFILE_KEY environment variable is missing!")
                    print("  Running with fallback INSECURE development passkey.")
                    passphrase = "FallbackInsecureDevKey2026"
                else:
                    raise ValueError(
                        "HCL_PROFILE_KEY environment variable is required for persistent profile mode. "
                        "Set it or run in `--insecure-dev-mode` for local testing."
                    )
            self.profile_path = f"profile_{user_id}.hcl"
            self.profile_passphrase = passphrase
            if os.path.exists(self.profile_path):
                try:
                    self.profile = CognitiveProfile.load(self.profile_path, passphrase)
                    self.user_facts = UserFacts.from_dict(self.profile.user_facts)
                except Exception as e:
                    print(f"  [HCL] Error decrypting user profile ({e}). Creating new profile.")
                    self.profile = CognitiveProfile(user_id)
            else:
                self.profile = CognitiveProfile(user_id)
            self.profile.session_count += 1
            self.save_profile()
            self.mode.adapt_to_profile(self.profile)

    # ── Conversation Mode Management ─────────────────────────────────────────

    def update_conversation_mode(self, prompt: str) -> None:
        """Update conversation mode based on user input."""
        if is_social_utterance(prompt):
            self.conversation_mode = "social"
            self.turns_since_social = 0
        elif is_meta_question(prompt):
            self.conversation_mode = "meta"
        else:
            self.turns_since_social += 1
            if self.conversation_mode == "social" and self.turns_since_social > self.social_mode_persistence:
                self.conversation_mode = "factual"

    def is_in_social_mode(self) -> bool:
        return self.conversation_mode == "social"

    def is_in_meta_mode(self) -> bool:
        return self.conversation_mode == "meta"

    # ── Profile & Persistence ────────────────────────────────────────────────

    def save_profile(self) -> None:
        # Save facts to plain JSON file for persistent memory across all modes
        try:
            with open(self._facts_path, "w", encoding="utf-8") as f:
                json.dump(self.user_facts.to_dict(), f, indent=4)
        except Exception as e:
            print(f"  [HCL] Error saving facts to disk: {e}")

        if self.profile is not None and hasattr(self, "profile_path") and hasattr(self, "profile_passphrase"):
            self.profile.user_facts = self.user_facts.to_dict()
            t0 = time.time()
            try:
                self.profile.save(self.profile_path, self.profile_passphrase)
            except Exception as e:
                print(f"  [HCL] Error saving user profile to disk: {e}")
            if getattr(self, "profile_timing", False):
                HCL.global_timings["save_profile"] += (time.time() - t0)
                HCL.global_timings["save_profile_count"] += 1

        if np.any(self.SML.occupied):
            try:
                self.SML.save_to_disk(self._lattice_path, self._lattice_passphrase, async_write=True)
            except Exception as e:
                print(f"  [HCL] Error saving lattice to disk: {e}")

    def _rebuild_hmg_from_sml(self) -> None:
        import numpy as np
        active_addrs = np.where(self.SML.occupied)[0]
        for addr in active_addrs:
            payload = self.SML.payloads.get(addr, {})
            if not payload:
                continue
            node_id = payload.get("id")
            if not node_id:
                continue
            emb = self.SML.surface[addr, :self.SML.dim]
            node_type_str = payload.get("type", "EPISODIC")
            try:
                node_type = NodeType[node_type_str]
            except Exception:
                node_type = NodeType.EPISODIC
            emo_emb_list = payload.get("emotional_embedding", None)
            emo_emb = np.array(emo_emb_list, dtype=np.float32) if emo_emb_list is not None else None
            node = Node(
                node_type=node_type,
                embedding=emb,
                raw_summary=payload.get("summary", ""),
                weight=float(self.SML.surface[addr, self.SML.dim]),
                confidence=float(self.SML.surface[addr, self.SML.dim + 1]),
                emotional_embedding=emo_emb
            )
            node.id = node_id
            node.cluster_id = payload.get("cluster_id")
            self.HMG.insert(node)
            self.ANN.update(node)
            c_id = node.cluster_id
            if c_id is not None:
                if c_id not in self.HMG.clusters:
                    self.HMG.clusters[c_id] = []
                if node.id not in self.HMG.clusters[c_id]:
                    self.HMG.clusters[c_id].append(node.id)

    # ── Grounding Confidence & TTLinear Rank Selection ─────────────────────────

    def get_grounding_confidence(self) -> float:
        """
        Returns average weight of active belief nodes in GroundedBeliefPathSearch.
        If no active beliefs, grounding confidence is 0.0.
        """
        active = self.GBPS.get_active_beliefs()
        if not active:
            return 0.0
        weights = [node.weight for node in active]
        return float(np.mean(weights))

    # ── Generation Step Hook ───────────────────────────────────────────────────

    def on_generation_step(self, context_window: torch.Tensor, token_distribution: torch.Tensor) -> str:
        t0 = time.time()
        self.step += 1
        if self.profile is not None:
            self.profile.total_inference_steps += 1
            if not getattr(self, "hcl_lightweight", False):
                self.save_profile()

        H = self.entropy_scorer.score(token_distribution, context_window, self.tokenizer)
        theta_write = self.profile.entropy_baseline + 0.1 if self.profile else self.mode.theta_write

        if H > theta_write:
            window_text = self.tokenizer.decode(context_window[0][-100:], skip_special_tokens=True)
            nodes = self.extract_cascade(window_text, H)
            for node in nodes:
                self.write(node)

        query_text = getattr(self, "current_query_text", "")
        has_matched_facts = False
        if query_text:
            has_matched_facts = len(self.user_facts.retrieve_relevant_facts(query_text)) > 0

        if len(self.HMG.nodes) == 0 and not has_matched_facts:
            if getattr(self, "profile_timing", False):
                HCL.global_timings["on_generation_step"] += (time.time() - t0)
                HCL.global_timings["on_generation_step_count"] += 1
            return ""

        self.predictive_prefetch()

        belief_emb = self.GBPS.get_active_belief_embedding()
        query_emb = getattr(self, "current_query_embedding", None)
        if query_emb is not None:
            if np.any(belief_emb) and not is_personal_query(query_text) and not self.is_in_social_mode():
                query_emb = 0.6 * query_emb + 0.4 * belief_emb
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
        else:
            query_emb = belief_emb

        # Dynamic rank driven by GBPS confidence
        confidence = self.get_grounding_confidence()
        if confidence >= 0.75:
            self.grounding_matrix.set_rank(4)
        elif confidence >= 0.4:
            self.grounding_matrix.set_rank(6)
        else:
            self.grounding_matrix.set_rank(8)

        # Ground query embedding through TTLinear projection
        if query_emb is not None and np.any(query_emb):
            device = self.grounding_matrix.G1.device
            dtype = self.grounding_matrix.G1.dtype
            query_tensor = torch.tensor(query_emb, dtype=dtype, device=device).unsqueeze(0)
            with torch.no_grad():
                projected_tensor = self.grounding_matrix(query_tensor).squeeze(0)
            query_emb = projected_tensor.cpu().numpy()
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb /= norm

        retrieved = self.retrieve(query_emb, k=5)
        injected_context = self.format_injection(retrieved)

        if self.step % self.mode.N_prune == 0:
            self.pruner_loop()

        self.transition_model.update(self.GBPS.get_active_beliefs())

        if getattr(self, "profile_timing", False):
            HCL.global_timings["on_generation_step"] += (time.time() - t0)
            HCL.global_timings["on_generation_step_count"] += 1

        return injected_context


    # ── Compression Prompt ─────────────────────────────────────────────────────

    def run_compression_prompt(self, prompt: str, max_tokens: int = 40) -> str:
        try:
            device = self.model.device
            messages = [
                {"role": "system", "content": "You are a precise summarization assistant. Answer in a direct, factual manner."},
                {"role": "user", "content": prompt}
            ]
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = f"Context: {prompt}\nSummary:"
            inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            result = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            return result
        except Exception:
            return f"Lightweight summary of: {prompt[:30]}"

    # ── Fact Verification ──────────────────────────────────────────────────────

    def verify_fact(self, sentence: str, confidence: float = 0.0) -> bool:
        """Verify a factual claim against parametric knowledge. Bypass for personal/social content."""
        sentence_lower = sentence.lower()

        # Bypass: sentences about the user (personal facts)
        if any(phrase in sentence_lower for phrase in ["your favourite", "your favorite", "your name", "you told me", "your status", "your struggle"]):
            return True

        # Bypass: sentences containing known user fact values
        for fact_info in self.user_facts.facts.values():
            val = str(fact_info.get("value", "")).lower()
            if val and len(val) > 2 and val in sentence_lower:
                return True

        if confidence >= self.mode.verify_high_conf_bypass:
            return True

        prompt = (
            f"Analyze this statement: '{sentence}'\n"
            "Is this statement an established, currently true fact in the real world? "
            "Consider if it refers to something that hasn't been built yet or is just a concept.\n"
            "Answer with exactly one word: TRUE or FALSE."
        )
        no_votes = 0
        for _ in range(self.mode.verify_votes):
            res = self.run_compression_prompt(prompt, max_tokens=10).strip().upper()
            if "FALSE" in res:
                no_votes += 1
            if no_votes >= self.mode.no_votes_to_block:
                return False
        return True

    # ── Extraction Helpers ─────────────────────────────────────────────────────

    def _extract_causal_sentences(self, text: str) -> Optional[str]:
        causal_patterns = [
            r'[^.]*?(?:because|therefore|thus|hence|as a result|leads to|causes)[^.]*\.',
            r'[^.]*?(?:if.*then|when.*results in)[^.]*\.',
        ]
        matches = []
        for pattern in causal_patterns:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        valid_matches = [m.strip() for m in matches if m.strip()]
        return ' '.join(valid_matches[:3]) if valid_matches else None

    def _extract_factual_claim(self, text: str) -> Optional[str]:
        sentences = text.split('.')
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            lower_s = s_clean.lower()
            if ' is ' in lower_s or ' are ' in lower_s or ' was ' in lower_s or ' were ' in lower_s:
                return s_clean + '.'
        return None

    # ── Extract Cascade ────────────────────────────────────────────────────────

    def extract_cascade(self, window_text: str, entropy: float) -> List[Node]:
        t0 = time.time()
        nodes = []

        episodic_summary = window_text[:512]
        episodic = Node(
            node_type=NodeType.EPISODIC,
            embedding=embed(episodic_summary, self.model, self.tokenizer),
            raw_summary=episodic_summary,
            weight=0.8,
            state=1,
            created_at=self.step,
            last_accessed=self.step,
            access_count=0
        )
        nodes.append(episodic)

        if entropy > 0.65:
            semantic_summary = self._extract_causal_sentences(window_text)
            if semantic_summary:
                semantic = Node(
                    node_type=NodeType.SEMANTIC,
                    embedding=embed(semantic_summary, self.model, self.tokenizer),
                    raw_summary=semantic_summary,
                    weight=0.75,
                    state=1,
                    created_at=self.step,
                    last_accessed=self.step,
                    access_count=0,
                    provenance=[episodic.id]
                )
                nodes.append(semantic)

        if entropy > 0.80:
            belief_summary = self._extract_factual_claim(window_text)
            if belief_summary:
                belief = Node(
                    node_type=NodeType.BELIEF,
                    embedding=embed(belief_summary, self.model, self.tokenizer),
                    raw_summary=belief_summary,
                    weight=0.9,
                    state=1,
                    created_at=self.step,
                    last_accessed=self.step,
                    access_count=0,
                    provenance=[n.id for n in nodes],
                    confidence=0.85
                )
                nodes.append(belief)
                self.GBPS.add_belief(belief)

        for node in nodes:
            self.assign_to_cluster(node)

        if getattr(self, "profile_timing", False):
            HCL.global_timings["extract_cascade"] += (time.time() - t0)
            HCL.global_timings["extract_cascade_count"] += 1

        return nodes

    # ── Clustering ─────────────────────────────────────────────────────────────

    def assign_to_cluster(self, node: Node, threshold: float = 0.80) -> None:
        best_cluster = None
        best_sim = -1.0
        for cluster_id, node_ids in self.HMG.clusters.items():
            cluster_nodes = [self.HMG.nodes[nid] for nid in node_ids if nid in self.HMG.nodes]
            if not cluster_nodes:
                continue
            
            factual_center = np.mean([n.factual_embedding for n in cluster_nodes], axis=0)
            norm_fact = np.linalg.norm(factual_center)
            if norm_fact > 0:
                factual_center /= norm_fact
                
            emotional_center = np.mean([n.emotional_embedding for n in cluster_nodes], axis=0)
            norm_emo = np.linalg.norm(emotional_center)
            if norm_emo > 0:
                emotional_center /= norm_emo
                
            fact_sim = float(np.dot(node.factual_embedding, factual_center))
            emo_sim = float(np.dot(node.emotional_embedding, emotional_center))
            sim = 0.5 * fact_sim + 0.5 * emo_sim
            
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id
        if best_sim > threshold:
            node.cluster_id = best_cluster
            self.HMG.clusters[best_cluster].append(node.id)
        else:
            new_cluster_id = len(self.HMG.clusters) + 1
            node.cluster_id = new_cluster_id
            self.HMG.clusters[new_cluster_id] = [node.id]

    # ── Write / Merge / Contradiction ──────────────────────────────────────────

    def write(self, node: Node) -> None:
        nearest_list = self.ANN.query(node.embedding, k=1)
        nearest = nearest_list[0] if nearest_list else None
        tau = 0.92
        if nearest and float(np.dot(node.embedding, nearest.embedding)) > tau:
            if self.contradiction_detected(node, nearest, tau):
                self.resolve_contradiction(node, nearest)
            else:
                self.merge(node, nearest)
        else:
            self.HMG.insert(node)
            self.ANN.update(node)
            payload = {
                "id": node.id,
                "type": node.node_type.value,
                "summary": node.raw_summary,
                "cluster_id": node.cluster_id,
                "emotional_embedding": node.emotional_embedding.tolist()
            }
            addr = self.SML.write(node.embedding, confidence=node.confidence, payload=payload)
            setattr(node, "sml_addr", addr)

    def contradiction_detected(self, node_a: Node, node_b: Node, tau: float = 0.92) -> bool:
        sim = float(np.dot(node_a.embedding, node_b.embedding))
        if sim < tau:
            return False
        contradiction_prompt = (
            f"Statement A: {node_a.raw_summary}\n"
            f"Statement B: {node_b.raw_summary}\n\n"
            "Do these statements contradict each other, describe different "
            "time periods of the same thing, or are they unrelated?\n"
            "Answer with exactly one word: CONTRADICTION, TEMPORAL, or UNRELATED"
        )
        verdict = self.run_compression_prompt(contradiction_prompt, max_tokens=10).upper()
        return "CONTRADICTION" in verdict

    def resolve_contradiction(self, node_a: Node, node_b: Node) -> None:
        if node_b.id not in node_a.contradictions:
            node_a.contradictions.append(node_b.id)
        if node_a.id not in node_b.contradictions:
            node_b.contradictions.append(node_a.id)
        if node_a.confidence is not None and node_b.confidence is not None:
            if abs(node_a.confidence - node_b.confidence) > 0.3:
                winner = node_a if node_a.confidence > node_b.confidence else node_b
                loser = node_b if winner == node_a else node_a
                loser.weight *= 0.5
                if " [DEPRECATED" not in loser.raw_summary:
                    loser.raw_summary += " [DEPRECATED: contradicted by higher-confidence belief]"
            else:
                if " [CONTRADICTION" not in node_a.raw_summary:
                    node_a.raw_summary += f" [CONTRADICTION: see {node_b.id}]"
                if " [CONTRADICTION" not in node_b.raw_summary:
                    node_b.raw_summary += f" [CONTRADICTION: see {node_a.id}]"
        self.HMG.insert(node_a)
        self.ANN.update(node_a)

    def merge(self, node: Node, nearest: Node) -> None:
        nearest.embedding = 0.7 * nearest.embedding + 0.3 * node.embedding
        norm = np.linalg.norm(nearest.embedding)
        if norm > 0:
            nearest.embedding /= norm
        if node.raw_summary.strip() != nearest.raw_summary.strip():
            nearest.raw_summary += " | " + node.raw_summary
        nearest.last_accessed = self.step
        nearest.access_count += 1


    # ── User Correction ────────────────────────────────────────────────────────

    def user_correction(self, topic_embedding: np.ndarray, correction_text: str, demotion_factor: float = 0.35) -> int:
        candidates = self.ANN.query(topic_embedding, k=10)
        demoted = 0
        for node in candidates:
            sim = float(np.dot(node.embedding, topic_embedding))
            if sim > 0.75:
                old_weight = node.weight
                node.weight = max(0.0, node.weight * (1.0 - demotion_factor))
                if "[USER CORRECTED]" not in node.raw_summary:
                    node.raw_summary += " [USER CORRECTED]"
                if node.weight <= 0.20:
                    node.state = 0
                demoted += 1
        return demoted

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(self, query_emb: np.ndarray, k: int = 5, user_confirmed: bool = False) -> List[Node]:
        cached = self.warm_cache.check(query_emb)
        if cached:
            return cached[:k]

        sml_results = self.SML.recall(query_emb, k=k*2, user_confirmed=user_confirmed)
        
        if self.is_in_social_mode():
            w_fact, w_social = 0.2, 0.8
        else:
            w_fact, w_social = 0.9, 0.1
            
        query_text = getattr(self, "current_query_text", "")
        query_emotional_emb = compute_emotional_embedding(query_emb, query_text)

        scored_nodes = []
        for res in sml_results:
            addr = res['addr']
            payload = self.SML.payloads.get(addr, {})
            node_id = payload.get("id")
            if node_id and node_id in self.HMG.nodes:
                node = self.HMG.nodes[node_id]
                node.last_accessed = self.step
                node.access_count += 1
                node.state = 1
                if user_confirmed:
                    node.verification_count += 1
                
                cand_emo_list = payload.get("emotional_embedding", None)
                if cand_emo_list is not None:
                    node.emotional_embedding = np.array(cand_emo_list, dtype=np.float32)
                else:
                    node.emotional_embedding = compute_emotional_embedding(node.factual_embedding, node.raw_summary)
                
                fact_sim = float(np.dot(node.factual_embedding, query_emb))
                emo_sim = float(np.dot(node.emotional_embedding, query_emotional_emb))
                fused_sim = w_fact * fact_sim + w_social * emo_sim
                
                node.weight = res['confidence']
                setattr(node, "epistemic_divergence", res['epistemic_divergence'])
                setattr(node, "temporal_bindings", res['temporal_bindings'])
                node.embedding = res['embedding']
                
                scored_nodes.append((fused_sim, node))
                
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes[:k]]

    # ── Predictive Prefetch ────────────────────────────────────────────────────

    def predictive_prefetch(self) -> None:
        active_beliefs = self.GBPS.get_active_beliefs()
        predicted_clusters = self.transition_model.predict_next(active_beliefs, self.HMG, top_k=3)
        theta_prefetch = 0.50
        prefetched_nodes = []
        for cluster_id in predicted_clusters:
            node_ids = self.HMG.clusters.get(cluster_id, [])
            for nid in node_ids:
                node = self.HMG.nodes.get(nid)
                if node and node.node_type == NodeType.BELIEF and node.confidence > theta_prefetch:
                    candidates = self.ANN.query(node.embedding, k=10)
                    prefetched_nodes.extend(candidates)
        if prefetched_nodes:
            self.warm_cache.stage(prefetched_nodes)

    # ── Dynamic Type Evolution ─────────────────────────────────────────────────

    def promote_node_type(self, node: "Node") -> bool:
        changed = False
        if node.node_type == NodeType.EPISODIC:
            sml_addr = getattr(node, 'sml_addr', None)
            if sml_addr is not None:
                node.abstraction_score = self.SML.compute_abstraction_score(sml_addr)
            if node.access_count > 10 and node.abstraction_score > 0.7:
                node.node_type = NodeType.SEMANTIC
                changed = True
        elif node.node_type == NodeType.SEMANTIC:
            if node.confidence is not None and node.confidence > 0.9 and len(node.contradictions) == 0:
                node.node_type = NodeType.BELIEF
                changed = True
        elif node.node_type == NodeType.BELIEF:
            source_diversity = len(set(node.provenance))
            if (node.verification_count > AXIOM_PROPERTIES['promotion_threshold']
                    and source_diversity >= AXIOM_PROPERTIES['source_diversity_min']
                    and len(node.contradictions) == 0):
                node.node_type = NodeType.AXIOM
                node.confidence = 1.0
                changed = True
        return changed

    def demote_axiom(self, node: "Node", reason: str = 'contradiction') -> bool:
        if node.node_type != NodeType.AXIOM:
            return False
        n_contradictions = len(node.contradictions)
        if reason == 'user_explicit' or n_contradictions >= AXIOM_PROPERTIES['demotion_threshold']:
            node.node_type = NodeType.BELIEF
            node.confidence = max(0.5, (node.confidence or 1.0) * 0.7)
            if '[DEMOTED FROM AXIOM]' not in node.raw_summary:
                node.raw_summary += ' [DEMOTED FROM AXIOM]'
            return True
        return False

    # ── Pruner ───────────────────────────────────────────────────────────────────

    def pruner_loop(self) -> None:
        migrations = self.SML.thermal_decay(lambda_=self.mode.lambda_)
        if migrations:
            for node in self.HMG.nodes.values():
                old = getattr(node, 'sml_addr', None)
                if old is not None and old in migrations:
                    node.sml_addr = migrations[old]

        nodes_to_remove = []
        for node in list(self.HMG):
            self.update_weight(node)
            steps_since_query = self.step - node.last_accessed

            promoted = self.promote_node_type(node)
            if promoted and node.node_type == NodeType.AXIOM:
                continue

            if node.node_type == NodeType.AXIOM:
                self.demote_axiom(node, reason='contradiction')
                continue

            resolve_state(
                node,
                queried=False,
                steps_since_query=steps_since_query,
                theta_high=0.65,
                theta_low=0.20,
                N_collapse=self.mode.N_collapse
            )

            if node.state == 0:
                nodes_to_remove.append(node)

        cold_batch = []
        for node in nodes_to_remove:
            addr = getattr(node, 'sml_addr', None)
            if addr is not None and self.SML.occupied[addr]:
                emb = self.SML.surface[addr, :self.SML.dim].copy()
                temp = float(self.SML.surface[addr, self.SML.dim + 4])
                payload = self.SML.payloads.get(addr, {})
                cold_batch.append({
                    'embedding': emb,
                    'temperature': temp,
                    'summary': payload.get('summary', node.raw_summary[:60]),
                })
                self.SML.forget_to_abyss(addr)
            self.HMG.remove(node)
            self.ANN.remove(node)

        if cold_batch:
            self.dream_state.compress_to_dream(cold_batch)

    # ── Weight Update ──────────────────────────────────────────────────────────

    def update_weight(self, node: Node) -> None:
        delta = self.step - node.last_accessed
        recency = np.exp(-self.mode.lambda_ * delta)
        frequency = np.log(1 + node.access_count)

        curr_emb = self.GBPS.current_embedding()
        proximity = float(np.dot(node.embedding, curr_emb))

        utility = node.utility_score
        confidence = node.confidence if node.node_type == NodeType.BELIEF else 0.5

        cluster_boost = 0.0
        if self.profile and node.cluster_id in self.profile.recurring_clusters:
            cluster_boost = 0.1

        alpha, beta, gamma, delta_w, epsilon = 0.3, 0.2, 0.2, 0.15, 0.15
        raw_weight = (
            alpha * recency +
            beta * frequency +
            gamma * proximity +
            delta_w * utility +
            epsilon * confidence +
            cluster_boost
        )
        node.weight = float(np.clip(raw_weight, 0.0, 1.0))

    # ── Retrieval Utility Scoring ──────────────────────────────────────────────

    def score_retrieval_utility(self, retrieved_nodes: List[Node], pre_state: Dict[str, float], post_state: Dict[str, Any]) -> None:
        entropy_delta = pre_state["entropy"] - post_state["entropy"]
        confidence_delta = post_state["confidence"] - pre_state["confidence"]
        user_accepted = 1.0 if not post_state.get("fallback_triggered", False) else 0.0
        new_belief_formed = 1.0 if self.GBPS.detect_new_belief() else 0.0

        def sigmoid_norm(val):
            return 1.0 / (1.0 + np.exp(-val))

        utility = (
            0.4 * sigmoid_norm(entropy_delta) +
            0.3 * sigmoid_norm(confidence_delta) +
            0.2 * user_accepted +
            0.1 * new_belief_formed
        )

        theta_high_utility = 0.70
        theta_low_utility = 0.30
        delta_utility_boost = 0.15

        for node in retrieved_nodes:
            node.utility_score = 0.7 * node.utility_score + 0.3 * utility
            if utility > theta_high_utility:
                node.weight = min(1.0, node.weight + delta_utility_boost)
                node.state = 1
            elif utility < theta_low_utility:
                node.weight *= 0.9
                if node.weight < 0.20:
                    node.state = 0


    # ── Format Injection (THE KEY UPGRADE) ─────────────────────────────────────

    def format_injection(self, retrieved_nodes: List[Node], max_tokens: int = 512) -> str:
        lines = []
        query_text = getattr(self, "current_query_text", "")

        # ── PRIORITY 1: User Facts (for personal/social queries) ─────────────
        matched_facts = []
        if query_text:
            matched_facts = self.user_facts.retrieve_relevant_facts(query_text)

        if matched_facts:
            lines.append("[USER PERSONAL CONTEXT — USE THIS TO ANSWER PERSONALLY]")
            for fact in matched_facts:
                lines.append(f"- {fact}")
            lines.append("[END USER PERSONAL CONTEXT]")
            lines.append("When answering questions about the user's personal facts above, respond warmly and directly. Use 'your' and 'you'. Do NOT say 'As an AI I don't have opinions' — instead, validate their interests and build on what you know about them.")

        # ── PRIORITY 2: Meta-question memory ───────────────────────────────────
        elif self.is_in_meta_mode() and query_text:
            lines.append("[CONVERSATION HISTORY CONTEXT]")

        # ── PRIORITY 3: Standard memory context (skip for pure personal queries) ─
        if not is_personal_query(query_text) or not matched_facts:
            if not lines:  # Only add header if we haven't added user facts
                lines.append("[MEMORY CONTEXT]")

            _annotation_re = re.compile(
                r'\s*\[(?:CONTRADICTION|DEPRECATED|TEMPORAL|HCL|USER CORRECTED|DEMOTED FROM AXIOM)[^\]]*\]'
            )

            char_count = sum(len(l) for l in lines)
            max_chars = max_tokens * 4

            sorted_nodes = sorted(retrieved_nodes, key=lambda x: x.weight, reverse=True)
            for node in sorted_nodes:
                if node.node_type == NodeType.QUESTION_ANCHOR:
                    continue
                clean_summary = _annotation_re.sub('', node.raw_summary).strip()
                if not clean_summary:
                    continue
                line = f"- {clean_summary}  (relevance: {node.weight:.2f}, type: {node.node_type.value})"
                if char_count + len(line) > max_chars:
                    break
                lines.append(line)
                char_count += len(line)

        # ── Déjà-vu probe ──────────────────────────────────────────────────────
        query_emb = self.GBPS.get_active_belief_embedding()
        deja_vu = self.dream_state.feel_deja_vu(query_emb)
        if deja_vu:
            lines.append(
                f"- [IMPRESSION: {deja_vu['impression']}]  "
                f"(déjà-vu strength: {deja_vu['strength']:.2f}, type: DREAM)"
            )

        # If only header was added, return empty
        if len(lines) <= 1 and lines[0] in ("[MEMORY CONTEXT]", "[CONVERSATION HISTORY CONTEXT]"):
            return ""

        lines.append("[END MEMORY CONTEXT]\n")
        return "\n".join(lines)

    # ── Question Retrieval ─────────────────────────────────────────────────────

    def retrieve_first_question(self) -> Optional[Node]:
        question_nodes = [
            n for n in self.HMG.nodes.values()
            if n.node_type == NodeType.QUESTION_ANCHOR
        ]
        if not question_nodes:
            return None
        return min(question_nodes, key=lambda n: getattr(n, "created_at", float('inf')))

    def retrieve_latest_question(self) -> Optional[Node]:
        question_nodes = [
            n for n in self.HMG.nodes.values()
            if n.node_type == NodeType.QUESTION_ANCHOR
        ]
        if not question_nodes:
            return None
        return max(question_nodes, key=lambda n: getattr(n, "created_at", 0.0))

    # ── Dynamic Uncertainty Threshold ──────────────────────────────────────────

    def get_dynamic_uncertainty_threshold(self, query: str, assistant_text: str, base_threshold: float) -> float:
        # SOCIAL MODE: completely disable uncertainty checks
        if self.is_in_social_mode():
            return 9.9
        if is_social_utterance(query):
            return 9.9
        if is_meta_question(query):
            return 9.9
        if is_personal_query(query):
            return 9.9

        dynamic_thresh = base_threshold
        
        # --- POSITIONAL ENTROPY ---
        words = len(assistant_text.split())
        if words < 5:
            dynamic_thresh += 0.15  # Permissive: forming thought
        elif words > 15:
            dynamic_thresh -= 0.05  # Strict: should be certain
            
        if assistant_text and assistant_text.endswith(('.', ',')):
            dynamic_thresh += 0.05  # Transition word
        # --------------------------

        query_lower = query.lower()

        creative_signals = {"write", "suggest", "brainstorm", "creative", "poem", "story", "joke", "feelings", "opinion", "imagine", "list some", "give me some ideas"}
        factual_signals = {"exact", "who is", "who was", "what is", "where is", "when did", "how many", "calculate", "math", "formula", "date", "year", "height", "population"}

        if any(sig in query_lower for sig in creative_signals):
            dynamic_thresh += 0.12
        elif any(sig in query_lower for sig in factual_signals):
            dynamic_thresh -= 0.08

        conversational_padding = {"needed to know", "wanted to know", "ask you", "what did i", "do you remember", "tell me if", "can you tell", "could you tell"}
        if any(sig in query_lower for sig in conversational_padding):
            dynamic_thresh += 0.08

        last_words = assistant_text.strip().lower()
        is_list_context = False
        if last_words:
            recent_segment = last_words[-100:]
            if "," in recent_segment or " and" in recent_segment or " or" in recent_segment:
                if any(phrase in recent_segment for phrase in ["feelings of", "associated with", "like", "such as", "including", "examples", "qualities", "trust"]):
                    is_list_context = True

        if is_list_context:
            dynamic_thresh += 0.10
        else:
            assertion_patterns = [r"\b(is|are|was|were|has|have|had|consists of)\b(?:\s+\w+)?$"]
            if any(re.search(pat, last_words) for pat in assertion_patterns):
                dynamic_thresh -= 0.05

        return float(np.clip(dynamic_thresh, 0.45, 0.95))


# ═══════════════════════════════════════════════════════════════════════════════
# STATE RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_state(node: Node, queried: bool, steps_since_query: int, theta_high: float = 0.65, theta_low: float = 0.20, N_collapse: int = 5, delta_boost: float = 0.15) -> None:
    if node.weight > theta_high:
        node.state = 1
    elif node.weight <= theta_low:
        node.state = 0
    else:
        if queried:
            node.state = 1
            node.weight = min(1.0, node.weight + delta_boost)
        elif steps_since_query > N_collapse:
            node.state = 0
        else:
            node.state = 2



# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR: generate_with_hcl (non-streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_hcl(prompt: str, model: Any, tokenizer: Any, hcl: HCL,
                      max_new_tokens: int = 80, max_context_window: int = 2048) -> Tuple[str, dict]:
    hcl.current_query_embedding = embed(prompt, model, tokenizer)
    hcl.current_query_text = prompt
    device = model.device
    assistant_tokens = []

    # Extract user facts IMMEDIATELY so they're available for this turn
    extracted_facts = hcl.user_facts.extract_from_turn(prompt)
    for key, val in extracted_facts:
        hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)

    # Update conversation mode
    hcl.update_conversation_mode(prompt)

    # Initial memory/facts retrieval for token 0
    has_matched_facts = len(hcl.user_facts.retrieve_relevant_facts(prompt)) > 0
    if len(hcl.HMG.nodes) > 0 or has_matched_facts:
        belief_emb = hcl.GBPS.get_active_belief_embedding()
        query_emb = hcl.current_query_embedding
        if query_emb is not None:
            if np.any(belief_emb) and not is_personal_query(prompt) and not hcl.is_in_social_mode():
                query_emb = 0.6 * query_emb + 0.4 * belief_emb
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
        else:
            query_emb = belief_emb
        initial_nodes = hcl.retrieve(query_emb, k=5)
        memory_prefix = hcl.format_injection(initial_nodes)
    else:
        memory_prefix = ""

    pre_state = {"entropy": 0.5, "confidence": 0.5}

    for step in range(max_new_tokens):
        system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."
        if hcl.is_in_social_mode():
            system_msg += " The user is sharing personal information with you. Respond warmly and personally. Use 'your' and 'you'. Never say 'As an AI I don't have opinions' when discussing the user's stated preferences."
        if memory_prefix:
            system_msg = f"{system_msg}\n\n{memory_prefix}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"{system_msg}\nUser: {prompt}\nAssistant:"

        prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

        if assistant_tokens:
            assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
            input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
        else:
            input_ids = prompt_ids

        if input_ids.shape[1] > max_context_window:
            overflow = input_ids.shape[1] - max_context_window
            if overflow < len(assistant_tokens):
                assistant_tokens = assistant_tokens[overflow:]
                assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
                input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
            else:
                if len(hcl.HMG.nodes) > 0:
                    hcl.step += 1
                input_ids = prompt_ids[:, -max_context_window:]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        next_token_val = next_token.item()
        assistant_tokens.append(next_token_val)

        new_prefix = hcl.on_generation_step(input_ids, next_token_logits)

        entropy = hcl.entropy_scorer.score(next_token_logits)
        confidence = float(torch.max(probs).item())

        current_assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
        dynamic_threshold = hcl.get_dynamic_uncertainty_threshold(prompt, current_assistant_text, hcl.mode.uncertainty_threshold)

        if entropy > dynamic_threshold and step > 5:
            final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
            if getattr(hcl, "hcl_lightweight", False):
                hcl.save_profile()
            return final_output + " [HCL: UNCERTAIN — REVIEW REQUIRED]", {
                "total_steps": len(assistant_tokens),
                "memory_nodes": len(hcl.HMG.nodes),
                "uncertain": True
            }

        if len(hcl.HMG.nodes) > 0:
            belief_emb = hcl.GBPS.get_active_belief_embedding()
            query_emb = getattr(hcl, "current_query_embedding", None)
            if query_emb is not None:
                if np.any(belief_emb):
                    query_emb = 0.6 * query_emb + 0.4 * belief_emb
                    norm = np.linalg.norm(query_emb)
                    if norm > 0:
                        query_emb /= norm
            else:
                query_emb = belief_emb
            retrieved_nodes = hcl.retrieve(query_emb, k=5)
            post_state = {"entropy": entropy, "confidence": confidence, "fallback_triggered": False}
            hcl.score_retrieval_utility(retrieved_nodes, pre_state, post_state)
            pre_state = post_state

        raw_decoded = tokenizer.decode([next_token_val], skip_special_tokens=False)
        eos_candidates = ["<|im_end|>", "<|eot_id|>", "<|end_of_turn|>", "<|end|>", "<|endoftext|>", "<eos>", "</s>"]
        if getattr(tokenizer, "eos_token", None):
            eos_candidates.append(tokenizer.eos_token)
        is_stop_token = (
            next_token_val == tokenizer.eos_token_id
            or any(candidate in raw_decoded for candidate in eos_candidates)
        )
        if is_stop_token:
            break

        if new_prefix != memory_prefix:
            memory_prefix = new_prefix

    final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
    if getattr(hcl, "hcl_lightweight", False):
        hcl.save_profile()
    return final_output, {"total_steps": len(assistant_tokens), "memory_nodes": len(hcl.HMG.nodes)}



# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR: generate_with_hcl_stream (streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_hcl_stream(prompt: str, model: Any, tokenizer: Any, hcl: HCL,
                             max_new_tokens: int = 80, max_context_window: int = 2048,
                             neuro=None):  # Optional InstantNeuroplasticity instance
    hcl.current_query_embedding = embed(prompt, model, tokenizer)
    hcl.current_query_text = prompt

    def get_reml_payloads():
        import numpy as np
        active_addrs = np.where(hcl.SML.occupied)[0]
        nodes = []
        for addr in active_addrs:
            payload = hcl.SML.payloads.get(addr, {})
            if not payload:
                continue
            cluster_id = payload.get("cluster_id")
            if cluster_id is not None:
                cluster_id = int(cluster_id)
            raw_bindings = hcl.SML.temporal.retrieve_temporal_context(addr)
            clean_bindings = []
            for b in raw_bindings:
                clean_bindings.append({
                    "addr": int(b["addr"]),
                    "temporal_proximity": float(b["temporal_proximity"]),
                    "semantic_distance": float(b["semantic_distance"])
                })
            nodes.append({
                "id": payload.get("id"),
                "type": payload.get("type"),
                "summary": payload.get("summary"),
                "weight": float(hcl.SML.surface[addr, hcl.SML.dim]),
                "cluster_id": cluster_id,
                "epistemic_divergence": float(hcl.SML.bedrock.divergence(addr)),
                "temporal_bindings": clean_bindings,
                "addr": int(addr)
            })
        return nodes

    device = model.device
    assistant_tokens = []

    # Extract user facts IMMEDIATELY so they're available for this turn
    extracted_facts = hcl.user_facts.extract_from_turn(prompt)
    for key, val in extracted_facts:
        hcl.user_facts.add_fact(key, val, confidence=1.0, source_turn=hcl.step)

    # Update conversation mode
    hcl.update_conversation_mode(prompt)

    # Initial memory/facts retrieval for token 0
    has_matched_facts = len(hcl.user_facts.retrieve_relevant_facts(prompt)) > 0
    if len(hcl.HMG.nodes) > 0 or has_matched_facts:
        belief_emb = hcl.GBPS.get_active_belief_embedding()
        query_emb = hcl.current_query_embedding
        if query_emb is not None:
            if np.any(belief_emb) and not is_personal_query(prompt) and not hcl.is_in_social_mode():
                query_emb = 0.6 * query_emb + 0.4 * belief_emb
                norm = np.linalg.norm(query_emb)
                if norm > 0:
                    query_emb /= norm
        else:
            query_emb = belief_emb
        initial_nodes = hcl.retrieve(query_emb, k=5)
        memory_prefix = hcl.format_injection(initial_nodes)
    else:
        memory_prefix = ""

    pre_state = {"entropy": 0.5, "confidence": 0.5}

    # User correction detection
    _correction_phrases = (
        "that's wrong", "that is wrong", "not correct", "actually", "no,",
        "it's not", "it is not", "that's not", "that is not", "hasn't been",
        "has not been", "doesn't exist", "does not exist", "isn't", "is not",
        "incorrect", "wrong", "no it's", "no, the", "not the", "it's actually"
    )
    prompt_lower = prompt.lower().strip()
    is_correction = any(prompt_lower.startswith(p) or f" {p}" in prompt_lower for p in _correction_phrases)

    if is_correction and len(hcl.HMG.nodes) > 0:
        topic_emb = embed(prompt)
        n_demoted = hcl.user_correction(topic_emb, prompt)
        if n_demoted:
            print(f"  [HCL] User correction detected — demoted {n_demoted} memory node(s)")

    # Meta-question detection
    meta_phrases = ("what did i ask", "what was my first", "what was the first thing", "first thing i asked", "my first question")
    if any(p in prompt_lower for p in meta_phrases):
        first_q_node = hcl.retrieve_first_question()
        if first_q_node:
            memory_prefix += f"\n[SYSTEM MEMORY: The user's very first question in this session was: '{first_q_node.raw_summary}']\n"

    t_inference_total = 0.0
    t_retrieval_total = 0.0
    t_save_total = 0.0
    timings_start = {k: v for k, v in HCL.global_timings.items()}
    yielded_text = ""

    for step in range(max_new_tokens):
        system_msg = "You are a helpful assistant. Give concise, direct answers. Do not add hashtags or social media formatting unless explicitly asked."
        if hcl.is_in_social_mode():
            system_msg += " The user is sharing personal information with you. Respond warmly and personally. Use 'your' and 'you'. Never say 'As an AI I don't have opinions' when discussing the user's stated preferences."
        if memory_prefix:
            system_msg = f"{system_msg}\n\n{memory_prefix}"

        expanded_prompt = hcl.working_memory.expand(prompt)

        messages = [{"role": "system", "content": system_msg}]
        for turn in hcl.working_memory.get_messages():
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": expanded_prompt})

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"{system_msg}\n"
            if hcl.working_memory.turns:
                formatted_prompt += "[Recent Conversation]\n" + hcl.working_memory.get_context() + "\n[/Recent Conversation]\n\n"
            formatted_prompt += f"User: {expanded_prompt}\nAssistant:"

        prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

        if assistant_tokens:
            assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
            input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
        else:
            input_ids = prompt_ids

        if input_ids.shape[1] > max_context_window:
            overflow = input_ids.shape[1] - max_context_window
            if overflow < len(assistant_tokens):
                assistant_tokens = assistant_tokens[overflow:]
                assistant_tensor = torch.tensor([assistant_tokens], dtype=torch.long, device=device)
                input_ids = torch.cat([prompt_ids, assistant_tensor], dim=-1)
            else:
                if len(hcl.HMG.nodes) > 0:
                    hcl.step += 1
                input_ids = prompt_ids[:, -max_context_window:]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        t_inference_total += time.perf_counter() - t0

        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        next_token_val = next_token.item()
        assistant_tokens.append(next_token_val)

        t1 = time.perf_counter()
        new_prefix = hcl.on_generation_step(input_ids, next_token_logits)
        t_retrieval_total += time.perf_counter() - t1

        entropy = hcl.entropy_scorer.score(next_token_logits)
        confidence = float(torch.max(probs).item())

        current_assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
        dynamic_threshold = hcl.get_dynamic_uncertainty_threshold(prompt, current_assistant_text, hcl.mode.uncertainty_threshold)

        if entropy > dynamic_threshold and step > 5:
            final_output = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
            if getattr(hcl, "hcl_lightweight", False):
                t2 = time.perf_counter()
                hcl.save_profile()
                t_save_total += time.perf_counter() - t2

            timings_end = HCL.global_timings
            extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
            extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)

            yield {
                "token": " [HCL: UNCERTAIN — REVIEW REQUIRED]",
                "entropy": round(entropy, 4),
                "confidence": round(confidence, 4),
                "threshold": round(dynamic_threshold, 4),
                "memory_nodes": get_reml_payloads(),
                "done": True,
                "uncertain": True,
                "stats": {
                    "total_steps": len(assistant_tokens),
                    "memory_nodes": len(hcl.HMG.nodes),
                    "uncertain": True
                },
                "timings": {
                    "avg_retrieval_ms": round((t_retrieval_total / max(len(assistant_tokens), 1)) * 1000, 2),
                    "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
                    "avg_save_ms": round(t_save_total * 1000, 2),
                    "avg_inference_ms": round((t_inference_total / max(len(assistant_tokens), 1)) * 1000, 2),
                    "extract_calls": extract_count
                }
            }
            hcl.working_memory.add_turn("user", prompt)
            return

        if len(hcl.HMG.nodes) > 0:
            belief_emb = hcl.GBPS.get_active_belief_embedding()
            query_emb = getattr(hcl, "current_query_embedding", None)
            if query_emb is not None:
                if np.any(belief_emb):
                    query_emb = 0.6 * query_emb + 0.4 * belief_emb
                    norm = np.linalg.norm(query_emb)
                    if norm > 0:
                        query_emb /= norm
            else:
                query_emb = belief_emb
            retrieved_nodes = hcl.retrieve(query_emb, k=5)

            # ── InstantNeuroplasticity: fire every token ──────────────────
            # Intercepts already-computed retrieved_nodes, entropy, and
            # dynamic_threshold. Zero extra GPU. ~5µs per token.
            if neuro is not None:
                neuro.on_retrieve(retrieved_nodes, entropy, dynamic_threshold)
                # Pathway boost: if this retrieval hits a known habit pathway,
                # pull in additional correlated addresses
                boosts = neuro.get_pathway_boost(retrieved_nodes)
                for boost_addr, _strength in boosts.items():
                    # Inject boosted address as a synthetic node if not already present
                    present_addrs = {int(r["addr"]) for r in retrieved_nodes if "addr" in r}
                    if boost_addr not in present_addrs and hcl.SML.occupied[boost_addr]:
                        payload = hcl.SML.payloads.get(boost_addr, {})
                        retrieved_nodes.append({
                            "addr": boost_addr,
                            "similarity": float(_strength),
                            "confidence": float(hcl.SML.surface[boost_addr, hcl.SML.dim]),
                            "epistemic_divergence": 0.0,
                            "temporal_bindings": [],
                            "embedding": hcl.SML.surface[boost_addr, :hcl.SML.dim].copy(),
                            "neuro_boosted": True,
                        })
                # Prune weak synapses every 50 tokens (cheap, O(edges))
                if step % 50 == 0:
                    neuro.prune()
            # ─────────────────────────────────────────────────────────────

            post_state = {"entropy": entropy, "confidence": confidence, "fallback_triggered": False}
            hcl.score_retrieval_utility(retrieved_nodes, pre_state, post_state)
            pre_state = post_state

        new_token_decoded = current_assistant_text[len(yielded_text):]
        raw_decoded = tokenizer.decode([next_token_val], skip_special_tokens=False)
        eos_candidates = ["<|im_end|>", "<|eot_id|>", "<|end_of_turn|>", "<|end|>", "<|endoftext|>", "<eos>", "</s>"]
        if getattr(tokenizer, "eos_token", None):
            eos_candidates.append(tokenizer.eos_token)
        is_stop_token = (
            next_token_val == tokenizer.eos_token_id
            or '\u0e49' in raw_decoded
            or any(candidate in raw_decoded for candidate in eos_candidates)
        )

        if new_token_decoded:
            yield {
                "token": new_token_decoded,
                "entropy": round(entropy, 4),
                "confidence": round(confidence, 4),
                "threshold": round(dynamic_threshold, 4),
                "memory_nodes": get_reml_payloads(),
                "done": False
            }
            yielded_text += new_token_decoded

        # Confident Hallucination Check
        if new_token_decoded.strip() in ['.', '!', '?', '\n'] and hcl.mode.uncertainty_threshold <= 0.75:
            full_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
            sentences = [s.strip() for s in full_text.replace('!', '.').replace('?', '.').replace('\n', '.').split('.') if s.strip()]
            if sentences:
                last_sentence = sentences[-1] + "."
                if is_complete_sentence(last_sentence) and should_fact_check(last_sentence, prompt):
                    if any(verb in last_sentence.lower() for verb in [" is ", " are ", " was ", " were "]):
                        if not hcl.verify_fact(last_sentence, confidence):
                            timings_end = HCL.global_timings
                            extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
                            extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)

                            yield {
                                "token": " [HCL: UNCERTAIN — FAILED FACT CHECK]",
                                "entropy": round(entropy, 4),
                                "confidence": round(confidence, 4),
                                "memory_nodes": get_reml_payloads(),
                                "done": True,
                                "uncertain": True,
                                "stats": {
                                    "total_steps": len(assistant_tokens),
                                    "memory_nodes": len(hcl.HMG.nodes),
                                    "uncertain": True
                                },
                                "timings": {
                                    "avg_retrieval_ms": round((t_retrieval_total / max(len(assistant_tokens), 1)) * 1000, 2),
                                    "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
                                    "avg_save_ms": round(t_save_total * 1000, 2),
                                    "avg_inference_ms": round((t_inference_total / max(len(assistant_tokens), 1)) * 1000, 2),
                                    "extract_calls": extract_count
                                }
                            }
                            hcl.working_memory.add_turn("user", prompt)
                            for node in hcl.HMG.nodes.values():
                                if hasattr(node, "sml_addr") and node.sml_addr is not None:
                                    try:
                                        addr = node.sml_addr
                                        if hcl.SML.occupied[addr]:
                                            hcl.SML.surface[addr, hcl.SML.dim] *= 0.3
                                    except Exception:
                                        pass
                            return

        if is_stop_token:
            break

        if new_prefix != memory_prefix:
            memory_prefix = new_prefix

    if getattr(hcl, "hcl_lightweight", False):
        t2 = time.perf_counter()
        hcl.save_profile()
        t_save_total += time.perf_counter() - t2

    timings_end = HCL.global_timings
    extract_count = timings_end["extract_cascade_count"] - timings_start.get("extract_cascade_count", 0)
    extract_total = timings_end["extract_cascade"] - timings_start.get("extract_cascade", 0.0)
    n_steps = max(len(assistant_tokens), 1)

    yield {
        "token": "",
        "entropy": 0.0,
        "confidence": 1.0,
        "memory_nodes": get_reml_payloads(),
        "done": True,
        "uncertain": False,
        "stats": {
            "total_steps": len(assistant_tokens),
            "memory_nodes": len(hcl.HMG.nodes),
            "uncertain": False
        },
        "timings": {
            "avg_retrieval_ms": round((t_retrieval_total / n_steps) * 1000, 2),
            "avg_extraction_ms": round((extract_total / max(extract_count, 1)) * 1000, 2) if extract_count > 0 else 0.0,
            "avg_save_ms": round(t_save_total * 1000, 2),
            "avg_inference_ms": round((t_inference_total / n_steps) * 1000, 2),
            "extract_calls": extract_count
        }
    }

    hcl.working_memory.add_turn("user", prompt)
    final_text = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
    if final_text.strip():
        hcl.working_memory.add_turn("assistant", final_text)

    # QUESTION_ANCHOR: always write user question to lattice
    if is_question(prompt):
        try:
            q_emb = embed(prompt)
            q_node = Node(
                node_type=NodeType.QUESTION_ANCHOR,
                embedding=q_emb,
                raw_summary=prompt[:256],
                weight=1.0,
                confidence=1.0,
            )
            hcl.assign_to_cluster(q_node)
            hcl.HMG.insert(q_node)
            hcl.ANN.update(q_node)
            payload = {
                "id": q_node.id,
                "type": q_node.node_type.value,
                "summary": q_node.raw_summary,
                "cluster_id": q_node.cluster_id
            }
            addr = hcl.SML.write(q_emb, confidence=1.0, payload=payload)
            setattr(q_node, "sml_addr", addr)
        except Exception as e:
            print(f"  [HCL] QUESTION_ANCHOR write failed: {e}")

    # Persist lattice and user facts after every conversation turn
    hcl.save_profile()
    if np.any(hcl.SML.occupied):
        try:
            hcl.SML.save_to_disk(hcl._lattice_path, hcl._lattice_passphrase, async_write=True)
        except Exception as e:
            print(f"  [HCL] Post-turn lattice save failed: {e}")