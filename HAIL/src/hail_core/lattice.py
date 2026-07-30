import time
import numpy as np
import logging
import threading
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Any

from .exceptions import HAILValidationError, HAILCapacityError, HAILIntegrityError
from ._internal.cosine import cosine_similarity
from ._internal.adaptive_strata import AdaptiveStrata
from ._internal.anchored_reencoder import AnchoredReEncoder
from ._internal.hierarchical_abyss import HierarchicalAbyss
from ._internal.dual_timescale_bedrock import DualTimescaleBedrock
from ._internal.temporal_binding import TemporalBinding

logger = logging.getLogger("hail")

class StratifiedMemoryLattice:
    """
    Content-Addressable Thermal Lattice.

    Thermal zones (proportional to lattice_size):
        Fast   : address range [0,              zone_fast_end)  temperature > 1.5
        Standard: address range [zone_fast_end,  zone_std_end)   temperature 0.5–1.5
        Cold   : address range [zone_std_end,    lattice_size)   temperature < 0.5

    surface column layout (dim + 6 columns total):
        [:dim]    embedding
        [dim]     confidence / weight  (intrinsic: thermal + epistemic)
        [dim+1]   original confidence  (written once at encode time)
        [dim+2]   last-access timestamp
        [dim+3]   access count
        [dim+4]   temperature  (replaces the old int 'state: active' flag)
        [dim+5]   plasticity_weight  (extrinsic: LTP/LTD only; init 1.0)
                  effective_confidence = confidence * plasticity_weight
    """
    def __init__(self, lattice_size: int = 65536, dim: int = 64, on_capacity: str = "evict", lock: Optional[threading.RLock] = None):
        if lattice_size < 16:
            raise HAILValidationError("Lattice size must be at least 16.")
        if dim < 1:
            raise HAILValidationError("Dimension dim must be at least 1.")
        if on_capacity not in ("evict", "raise"):
            raise HAILValidationError("on_capacity option must be either 'evict' or 'raise'.")

        self.lattice_size = lattice_size
        self.dim = dim
        self.on_capacity = on_capacity
        self._lock = lock or nullcontext()

        self.surface = np.zeros((lattice_size, dim + 6), dtype=np.float32)
        self.strata = {}
        self.original_encodings = {}
        self.occupied = np.zeros(lattice_size, dtype=bool)

        self.adaptive_strata = AdaptiveStrata()
        self.anchored_reencoder = AnchoredReEncoder()
        self.abyss = HierarchicalAbyss(dim=dim)
        self.bedrock = DualTimescaleBedrock()
        self.temporal = TemporalBinding()
        self.mantle = {}
        self.payloads = {}

        # Thermal zone boundaries
        self.zone_fast_end = max(1, lattice_size // 16)
        self.zone_std_end  = max(2, int(lattice_size * 0.875))

        # Locality-Preserving Hash projection vectors (deterministic, seed=2026)
        _rng = np.random.default_rng(2026)
        _n = min(16, dim)
        self._proj_coarse = _rng.standard_normal(_n).astype(np.float32)
        self._proj_fine   = _rng.standard_normal(_n).astype(np.float32)
        _nc = np.linalg.norm(self._proj_coarse)
        _nf = np.linalg.norm(self._proj_fine)
        if _nc > 0: self._proj_coarse /= _nc
        if _nf > 0: self._proj_fine   /= _nf

        # Precompute 8-neighbor lookup table for O(1) retrieval
        self.neighbors = self._precompute_neighbors()

    def _precompute_neighbors(self) -> np.ndarray:
        offsets = np.array([1, -1, 256, -256, 257, -257, 255, -255], dtype=np.int64)
        addrs = np.arange(self.lattice_size, dtype=np.int64)
        neighbors = (addrs[:, None] + offsets[None, :]) % self.lattice_size
        return neighbors.astype(np.int32)

    def _validate_embedding(self, embedding: Any):
        if not isinstance(embedding, np.ndarray):
            raise HAILValidationError("Embedding must be a numpy array.")
        if embedding.ndim != 1:
            raise HAILValidationError("Embedding must be a 1D array.")
        if len(embedding) != self.dim:
            raise HAILValidationError(f"Embedding length must match configured dimension {self.dim}, got {len(embedding)}")
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise HAILValidationError("Embedding contains NaN or Inf values.")

    def _semantic_address(self, embedding: np.ndarray) -> int:
        n = len(self._proj_coarse)
        # Coarse component: high-order bits of the address
        coarse = float(np.dot(embedding[:n], self._proj_coarse))
        coarse_addr = int(abs(coarse) * self.lattice_size) & (self.lattice_size - 1)

        # Fine component: low-order offset within a 256-slot window
        fine = float(np.dot(embedding[n:2 * n], self._proj_fine)) if len(embedding) >= 2 * n else 0.0
        fine_offset = int(abs(fine) * 256) & 0xFF

        candidate = (coarse_addr + fine_offset) % self.lattice_size

        # Linear probe for collision avoidance (up to 64 slots forward)
        for i in range(64):
            probe = (candidate + i) % self.lattice_size
            if not self.occupied[probe]:
                return probe

        if self.on_capacity == "raise":
            raise HAILCapacityError("Lattice capacity threshold reached in the local probe neighborhood (all 64 probed slots are full).")

        # All probed slots occupied: evict the one with the lowest temperature
        probe_addrs = [(candidate + i) % self.lattice_size for i in range(64)]
        temps = np.array([self.surface[a, self.dim + 4] for a in probe_addrs])
        return probe_addrs[int(np.argmin(temps))]

    def write(self, embedding: np.ndarray, confidence: float = 0.8, payload: Optional[Dict] = None) -> int:
        self._validate_embedding(embedding)
        if not (0.0 <= confidence <= 1.0):
            raise HAILValidationError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        with self._lock:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            addr = self._semantic_address(embedding)

            self.surface[addr, :self.dim] = embedding
            self.surface[addr, self.dim]   = confidence  # intrinsic weight/confidence
            self.surface[addr, self.dim+1] = confidence  # original confidence (frozen)
            self.surface[addr, self.dim+2] = time.time() # timestamp
            self.surface[addr, self.dim+3] = 1            # access count
            self.surface[addr, self.dim+4] = 1.0          # temperature: warm on entry
            self.surface[addr, self.dim+5] = 1.0          # plasticity_weight: neutral

            self.original_encodings[addr] = embedding.copy()

            self.strata[addr] = [{
                'query_context': embedding.copy(),
                'retrieved_embedding': embedding.copy(),
                'emotional_valence': 1.0,
                'timestamp': time.time(),
                'stratum_type': 'encoding'
            }]

            self.temporal.encode(addr, embedding)
            self.occupied[addr] = True
            if payload is not None:
                self.payloads[addr] = payload
            return addr

    def _surface_retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if not np.any(self.occupied):
            return []

        active_indices = np.where(self.occupied)[0]
        embeddings = self.surface[active_indices, :self.dim]

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        norms = np.linalg.norm(embeddings, axis=1)
        valid = norms > 0
        if not np.any(valid):
            return []

        dots = np.dot(embeddings[valid], query_embedding)
        sims = dots / (norms[valid] * query_norm)

        valid_indices = active_indices[valid]

        sorted_idx = np.argsort(sims)[::-1]
        top_k_idx = valid_indices[sorted_idx[:min(k, len(sorted_idx))]]
        top_k_sims = sims[sorted_idx[:min(k, len(sorted_idx))]]

        return list(zip(top_k_idx, top_k_sims))

    def _lph_probe_retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        center = int(self._semantic_address(query_embedding))
        probe_addrs = [center] + [int(a) for a in self.neighbors[center]]

        results = []
        for addr in probe_addrs:
            if not self.occupied[addr]:
                continue
            emb = self.surface[addr, :self.dim]
            sim = float(np.dot(query_embedding, emb))
            if sim > 0.70:
                results.append((int(addr), sim))

        results.sort(key=lambda x: x[1], reverse=True)

        if len(results) < k:
            probed_set = {int(a) for a, _ in results}
            for addr, sim in self._surface_retrieve(query_embedding, k):
                if int(addr) not in probed_set:
                    results.append((int(addr), sim))
                if len(results) >= k:
                    break
            results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def recall(self, query_embedding: np.ndarray, k: int = 5, user_confirmed: bool = False) -> List[Dict[str, Any]]:
        self._validate_embedding(query_embedding)
        if k < 1:
            raise HAILValidationError(f"k must be at least 1, got {k}")

        with self._lock:
            n_occupied = int(np.sum(self.occupied))
            if k > n_occupied and n_occupied > 0:
                logger.warning("Requested k=%d is larger than total occupied slots %d", k, n_occupied)

            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            if n_occupied > 9:
                results = self._lph_probe_retrieve(query_embedding, k)
            else:
                results = self._surface_retrieve(query_embedding, k)

            output = []

            for addr_raw, similarity in results:
                addr = int(addr_raw)
                access_count = self.surface[addr, self.dim+3]
                valence = similarity * (1.0 / np.log(access_count + 2))

                stratum = {
                    'query_context': query_embedding.copy(),
                    'retrieved_embedding': self.surface[addr, :self.dim].copy(),
                    'emotional_valence': valence,
                    'timestamp': time.time(),
                    'stratum_type': 'recall',
                    'similarity_to_original': similarity,
                    'user_confirmed': user_confirmed
                }

                if addr not in self.strata:
                    self.strata[addr] = []
                self.strata[addr].append(stratum)

                current_access_pattern = {
                    'per_minute': access_count / max(1, (time.time() - self.strata[addr][0]['timestamp']) / 60),
                    'avg_valence': np.mean([s['emotional_valence'] for s in self.strata[addr]])
                }

                self.adaptive_strata.depth_budget[addr] = self.adaptive_strata.get_max_depth(addr, current_access_pattern)

                def bedrock_callback(a, s):
                    self.bedrock.recall(a, s)

                self.strata[addr] = self.adaptive_strata.compress(addr, self.strata[addr], bedrock_callback)

                new_embedding = self.anchored_reencoder.re_encode(self.strata[addr], self.original_encodings[addr])
                self.surface[addr, :self.dim] = new_embedding

                self.surface[addr, self.dim+3] += 1
                self.surface[addr, self.dim+2] = time.time()
                self.surface[addr, self.dim+4] = min(2.5, float(self.surface[addr, self.dim+4]) + 0.5)

                gravity = self.abyss.compute_gravity(query_embedding)
                self.surface[addr, self.dim] *= (1.0 - gravity * 0.1)

                divergence = self.bedrock.divergence(addr)
                temp_ctx = self.temporal.retrieve_temporal_context(addr)

                new_addr = self._migrate_if_needed(int(addr))

                conf = float(self.surface[new_addr, self.dim])
                pw   = float(self.surface[new_addr, self.dim + 5])
                output.append({
                    'addr': new_addr,
                    'similarity': float(similarity),
                    'confidence': conf,
                    'plasticity_weight': pw,
                    'effective_confidence': conf * pw,
                    'epistemic_divergence': float(divergence),
                    'temporal_bindings': temp_ctx,
                    'embedding': new_embedding
                })

            return output

    def forget_to_abyss(self, addr: int):
        with self._lock:
            if not self.occupied[addr]:
                return

            all_embeddings = [s['retrieved_embedding'] for s in self.strata.get(addr, [])]
            if all_embeddings:
                abyssal_centroid = np.mean(all_embeddings, axis=0)
            else:
                abyssal_centroid = self.surface[addr, :self.dim]

            metadata = {
                'birth_time': self.strata[addr][0]['timestamp'] if addr in self.strata and self.strata[addr] else time.time(),
                'death_time': time.time(),
                'stratum_count': len(self.strata.get(addr, [])),
                'final_valence': np.mean([s['emotional_valence'] for s in self.strata.get(addr, [])]) if addr in self.strata and self.strata[addr] else 0.5
            }

            self.abyss.add_to_abyss(addr, abyssal_centroid, metadata)

            self.occupied[addr] = False
            self.surface[addr] *= 0
            if addr in self.strata:
                del self.strata[addr]
            self.payloads.pop(addr, None)

    def _zone_of(self, addr: int) -> Tuple[int, int]:
        if addr < self.zone_fast_end:
            return (0, self.zone_fast_end)
        elif addr < self.zone_std_end:
            return (self.zone_fast_end, self.zone_std_end)
        else:
            return (self.zone_std_end, self.lattice_size)

    def _zone_for_temp(self, temp: float) -> Tuple[int, int]:
        if temp > 1.5:
            return (0, self.zone_fast_end)
        elif temp >= 0.5:
            return (self.zone_fast_end, self.zone_std_end)
        else:
            return (self.zone_std_end, self.lattice_size)

    def _migrate_if_needed(self, addr: int) -> int:
        if not self.occupied[addr]:
            return addr
        temp = float(self.surface[addr, self.dim + 4])
        current_zone = self._zone_of(addr)
        target_zone  = self._zone_for_temp(temp)
        if current_zone == target_zone:
            return addr

        lo, hi = target_zone
        zone_slice = self.occupied[lo:hi]
        free = np.where(~zone_slice)[0]
        if len(free) == 0:
            return addr  # target zone full, stay in place

        new_addr = lo + int(free[0])
        self.surface[new_addr] = self.surface[addr].copy()
        self.surface[addr] *= 0.0
        self.occupied[new_addr] = True
        self.occupied[addr]     = False
        for store in (self.payloads, self.strata, self.original_encodings):
            if addr in store:
                store[new_addr] = store.pop(addr)
        return new_addr

    def thermal_decay(self, lambda_: float = 0.05) -> Dict[int, int]:
        with self._lock:
            if not np.any(self.occupied):
                return {}
            active = np.where(self.occupied)[0].tolist()

            decay_addrs = [a for a in active if not self.payloads.get(a, {}).get("pinned", False)]
            if decay_addrs:
                self.surface[decay_addrs, self.dim + 4] *= (1.0 - lambda_)

            migrations: Dict[int, int] = {}
            for addr in active:
                new_addr = self._migrate_if_needed(int(addr))
                if new_addr != addr:
                    migrations[int(addr)] = new_addr
            return migrations

    def get_zone_label(self, addr: int) -> str:
        if addr < self.zone_fast_end:
            return 'hot'
        elif addr < self.zone_std_end:
            return 'standard'
        else:
            return 'cold'

    def compute_abstraction_score(self, addr: int) -> float:
        with self._lock:
            strata = self.strata.get(addr, [])
            recall_contexts = [
                s['query_context'] for s in strata
                if s.get('stratum_type') == 'recall' and 'query_context' in s
            ]

            if len(recall_contexts) < 3:
                return 0.0

            queries  = np.stack(recall_contexts)
            centroid = np.mean(queries, axis=0)
            spread   = float(np.mean(np.linalg.norm(queries - centroid, axis=1)))
            score = float(np.tanh(spread * 2.0))

            if addr in self.payloads:
                self.payloads[addr]['abstraction_score'] = score

            return score

    def save_to_disk(self, path: str, passphrase: Optional[str] = None, async_write: bool = False):
        """Atomic, optionally encrypted, compressed save."""
        if passphrase is not None and len(passphrase) < 8:
            raise HAILValidationError("Passphrase must be at least 8 characters long.")

        with self._lock:
            surface_copy = np.copy(self.surface)
            occupied_copy = np.copy(self.occupied)

            def _strata_trim(s):
                return [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in layer.items()} for layer in s[-5:]]

            json_payload = {
                "version": "0.2.0",
                "payloads": {str(k): v for k, v in self.payloads.items()},
                "original_encodings": {str(k): v.tolist() for k, v in self.original_encodings.items()},
                "strata": {str(k): _strata_trim(v) for k, v in self.strata.items()},
                "mantle": {str(k): v for k, v in list(self.mantle.items())[:1000]},
                "abyss_summary": self.abyss.summarize(),
                "bedrock_fast": {
                    str(k): {"consensus_embedding": v["consensus_embedding"].tolist(),
                              "count": v["count"]}
                    for k, v in self.bedrock.fast_bedrock.items()
                },
                "temporal_bindings": {
                    str(k): [{kk: (int(vv) if isinstance(vv, (int, float)) else vv)
                              for kk, vv in b.items()}
                             for b in bindings]
                    for k, bindings in self.temporal.temporal_bindings.items()
                }
            }

            if async_write:
                thread = threading.Thread(
                    target=self._write_background,
                    args=(path, passphrase, surface_copy, occupied_copy, json_payload),
                    daemon=True
                )
                thread.start()
            else:
                self._write_background(path, passphrase, surface_copy, occupied_copy, json_payload)

    def _write_background(self, path: str, passphrase: Optional[str], surface_copy, occupied_copy, json_payload):
        import io
        import json
        import os as _os

        np_buf = io.BytesIO()
        np.savez_compressed(np_buf, surface=surface_copy, occupied=occupied_copy)
        np_bytes = np_buf.getvalue()

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        json_bytes = json.dumps(json_payload, cls=NumpyEncoder).encode("utf-8")
        separator = b"\n---HYDRUS_JSON---\n"
        combined = np_bytes + separator + json_bytes

        if passphrase is None:
            # ── Plain format ─────────────────────────────────────────────────
            MAGIC = b"HAILPLAIN1\n"
            encrypted = MAGIC + combined
            encoding_desc = "PLAIN"
        else:
            # ── AES-256-GCM format ───────────────────────────────────────────
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes

            MAGIC = b"HAILGCM1"
            KDF_ITERATIONS = 600_000

            salt = _os.urandom(16)
            nonce = _os.urandom(12)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=KDF_ITERATIONS,
            )
            key = kdf.derive(passphrase.encode())

            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, combined, associated_data=MAGIC)
            encrypted = MAGIC + salt + nonce + ciphertext
            encoding_desc = "AES-256-GCM"

        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(encrypted)
        _os.replace(tmp, path)
        logger.info("Lattice saved to %s (%d KB, %s)", path, len(encrypted)//1024, encoding_desc)

    def load_from_disk(self, path: str, passphrase: Optional[str] = None):
        """Load and decrypt a lattice checkpoint."""
        import io
        import json

        AES_MAGIC = b"HAILGCM1"
        PLAIN_MAGIC = b"HAILPLAIN1\n"
        KDF_ITERATIONS = 600_000

        with open(path, "rb") as f:
            raw = f.read()

        if raw[:len(PLAIN_MAGIC)] == PLAIN_MAGIC:
            # ── Plain format ─────────────────────────────────────────────────
            decrypted = raw[len(PLAIN_MAGIC):]
        elif raw[:len(AES_MAGIC)] == AES_MAGIC:
            # ── AES-256-GCM format ───────────────────────────────────────────
            if passphrase is None:
                raise HAILValidationError("Passphrase is required to decrypt this lattice file.")
            
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.exceptions import InvalidTag

            offset = len(AES_MAGIC)
            salt = raw[offset:offset+16]; offset += 16
            nonce = raw[offset:offset+12]; offset += 12
            ciphertext = raw[offset:]

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=KDF_ITERATIONS,
            )
            key = kdf.derive(passphrase.encode())
            aesgcm = AESGCM(key)
            try:
                decrypted = aesgcm.decrypt(nonce, ciphertext, associated_data=AES_MAGIC)
            except InvalidTag:
                raise HAILIntegrityError("Lattice integrity check failed: wrong passphrase or corrupted file.")
        else:
            # ── Legacy XOR format ────────────────────────────────────────────
            if passphrase is None:
                raise HAILValidationError("Passphrase is required to decrypt this legacy XOR lattice file.")
            decrypted = self._load_legacy_xor(raw, passphrase)
            logger.info("Legacy XOR-encrypted lattice detected — will be upgraded on next save.")

        separator = b"\n---HYDRUS_JSON---\n"
        sep_idx = decrypted.find(separator)
        if sep_idx == -1:
            raise HAILIntegrityError("Lattice format error: JSON separator not found.")

        np_bytes = decrypted[:sep_idx]
        json_bytes = decrypted[sep_idx + len(separator):]

        np_data = np.load(io.BytesIO(np_bytes))
        self.surface = np_data["surface"]
        self.occupied = np_data["occupied"]

        expected_cols = self.dim + 6
        if self.surface.shape[1] < expected_cols:
            pad_cols = expected_cols - self.surface.shape[1]
            pad = np.ones((self.surface.shape[0], pad_cols), dtype=np.float32)
            self.surface = np.concatenate([self.surface, pad], axis=1)
            logger.info("Legacy lattice: padded %d new column(s) with 1.0", pad_cols)

        data = json.loads(json_bytes)
        self.payloads = {int(k): v for k, v in data["payloads"].items()}
        self.mantle   = {int(k): v for k, v in data["mantle"].items()}
        self.original_encodings = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in data.get("original_encodings", {}).items()
        }

        for k_str, layers in data["strata"].items():
            k = int(k_str)
            self.strata[k] = []
            for layer in layers:
                restored = {kk: (np.array(vv, dtype=np.float32)
                                 if isinstance(vv, list) else vv)
                            for kk, vv in layer.items()}
                self.strata[k].append(restored)

        for k_str, v in data["bedrock_fast"].items():
            self.bedrock.fast_bedrock[int(k_str)] = {
                "consensus_embedding": np.array(v["consensus_embedding"], dtype=np.float32),
                "count": v["count"]
            }

        for k_str, bindings in data["temporal_bindings"].items():
            self.temporal.temporal_bindings[int(k_str)] = bindings

        for rid_str, r in data["abyss_summary"]["regions"].items():
            rid = int(rid_str)
            centroid = np.array(r["centroid"], dtype=np.float32)
            self.abyss.regions[rid] = {
                "centroid": centroid,
                "addresses": [],
                "gravity": r["gravity"]
            }
            self.abyss.region_centroids[rid] = centroid

        logger.info("Lattice loaded from %s", path)

    def _load_legacy_xor(self, encrypted: bytes, passphrase: str) -> bytes:
        import hashlib as _hashlib
        import hmac as _hmac

        salt = encrypted[:16]
        mac = encrypted[16:48]
        ciphertext = encrypted[48:]
        key = _hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 10000, dklen=32)
        expected_mac = _hashlib.sha256(key + salt + ciphertext).digest()
        if not _hmac.compare_digest(mac, expected_mac):
            raise HAILIntegrityError("Lattice integrity check failed: wrong passphrase or corrupted file.")

        decrypted = bytearray()
        counter = 0
        for i in range(0, len(ciphertext), 32):
            block_key = _hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest()
            chunk = ciphertext[i:i+32]
            for b_idx, b in enumerate(chunk):
                decrypted.append(b ^ block_key[b_idx])
            counter += 1
        return bytes(decrypted)
