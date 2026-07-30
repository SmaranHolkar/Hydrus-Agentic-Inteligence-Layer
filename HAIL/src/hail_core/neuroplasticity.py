import time
import json
import math
import numpy as np
import logging
from collections import deque
from typing import List, Dict, Any, Optional

logger = logging.getLogger("hail")

class InstantNeuroplasticity:
    """
    Lattice-native neuroplasticity engine for HydrusOPT's SML.

    All plasticity state lives in surface[:, dim+5] (plasticity_weight).
    No shadow dict. No manual pruning. Retrieval integration is automatic.
    """

    def __init__(
        self,
        sml,
        eta: float = 0.02,
        tau: float = 5.0,
        ltd_factor: float = 0.98,
        prune_threshold: float = 0.03,
        thought_window: int = 12,
        max_snapshots: int = 32,
        min_active_nodes: int = 5,
        ltp_sim_lo: float = 0.70,
        ltp_sim_hi: float = 0.95,
        pw_max: float = 1.5,
    ):
        self.sml = sml
        self.eta = eta
        self.tau = tau
        self.ltd_factor = ltd_factor
        self.prune_threshold = prune_threshold
        self.thought_window = thought_window
        self.min_active_nodes = min_active_nodes
        self.ltp_sim_lo = ltp_sim_lo
        self.ltp_sim_hi = ltp_sim_hi
        self.pw_max = pw_max

        # Column index for plasticity_weight in the surface array
        self._pw_col: int = sml.dim + 5

        # Validate that the SML surface has the expected column count
        expected_cols = sml.dim + 6
        if sml.surface.shape[1] < expected_cols:
            raise ValueError(
                f"SML surface has {sml.surface.shape[1]} columns but "
                f"InstantNeuroplasticity requires {expected_cols} (dim+6). "
                f"Ensure StratifiedMemoryLattice is up-to-date."
            )

        self.recently_fired: deque = deque(maxlen=10)
        self.pathway_snapshots: deque = deque(maxlen=max_snapshots)
        self._last_snapshot_addrs: frozenset = frozenset()

        self._step: int = 0
        self._steps_since_snap: int = 0

        self.ltp_events: int = 0
        self.ltd_events: int = 0
        self.auto_reset_events: int = 0

    def on_retrieve(
        self,
        retrieved_nodes: List[Dict[str, Any]],
        entropy: float,
        threshold: float,
    ) -> None:
        """Call every token, right after hcl.retrieve() returns."""
        addrs = [int(r["addr"]) for r in retrieved_nodes if "addr" in r]
        if not addrs:
            self._step += 1
            self._steps_since_snap += 1
            return

        n_active = int(np.sum(self.sml.occupied))
        uncertain = entropy > threshold

        for addr in addrs:
            if not self.sml.occupied[addr]:
                continue

            if n_active >= self.min_active_nodes:
                emb_a = self.sml.surface[addr, : self.sml.dim]
                norm_a = float(np.linalg.norm(emb_a))
                if norm_a == 0.0:
                    continue
                for past_addr, fire_step in self.recently_fired:
                    if past_addr == addr or not self.sml.occupied[past_addr]:
                        continue
                    emb_b = self.sml.surface[past_addr, : self.sml.dim]
                    norm_b = float(np.linalg.norm(emb_b))
                    if norm_b == 0.0:
                        continue
                    sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                    if self.ltp_sim_lo < sim < self.ltp_sim_hi:
                        age = self._step - fire_step
                        self._hebbian_potentiate(addr, past_addr, age)

            if uncertain:
                pw = float(self.sml.surface[addr, self._pw_col])
                new_pw = pw * self.ltd_factor
                if new_pw < self.prune_threshold:
                    new_pw = 1.0
                    self.auto_reset_events += 1
                self.sml.surface[addr, self._pw_col] = new_pw
                self.ltd_events += 1

        for addr in addrs:
            self.recently_fired.append((addr, self._step))

        self._step += 1
        self._steps_since_snap += 1

        if self._steps_since_snap >= self.thought_window:
            self._snapshot_pathway(addrs)
            self._steps_since_snap = 0

    def on_punctuation(self, token_text: str, retrieved_addrs: List[int]) -> None:
        if token_text.strip() in {".", "!", "?", "\n"}:
            self._snapshot_pathway(retrieved_addrs)
            self._steps_since_snap = 0

    def get_pathway_seed(self, retrieved_nodes: List[Dict[str, Any]]) -> List[int]:
        current_addrs = frozenset(
            int(r["addr"]) for r in retrieved_nodes if "addr" in r
        )
        seeds: set = set()

        for snapshot in self.pathway_snapshots:
            snapshot_set = frozenset(snapshot["addrs"])
            if current_addrs & snapshot_set:
                for addr in snapshot_set - current_addrs:
                    if self.sml.occupied[addr]:
                        seeds.add(addr)

        return list(seeds)

    def get_pathway_boost(self, retrieved_nodes: List[Dict[str, Any]]) -> Dict[int, float]:
        seeds = self.get_pathway_seed(retrieved_nodes)
        return {
            addr: float(self.sml.surface[addr, self._pw_col])
            for addr in seeds
            if self.sml.occupied[addr]
        }

    def prune(self) -> int:
        return 0

    def report(self) -> Dict[str, Any]:
        occupied_idx = np.where(self.sml.occupied)[0]
        if len(occupied_idx) == 0:
            pw_vals = np.array([], dtype=np.float32)
        else:
            pw_vals = self.sml.surface[occupied_idx, self._pw_col]

        return {
            "mean_plasticity":   float(np.mean(pw_vals)) if len(pw_vals) else 1.0,
            "potentiated_nodes": int(np.sum(pw_vals > 1.0)),
            "depressed_nodes":   int(np.sum(pw_vals < 0.5)),
            "neutral_nodes":     int(np.sum((pw_vals >= 0.95) & (pw_vals <= 1.05))),
            "occupied_nodes":    len(occupied_idx),
            "ltp_events":        self.ltp_events,
            "ltd_events":        self.ltd_events,
            "auto_reset_events": self.auto_reset_events,
            "pathway_snapshots": len(self.pathway_snapshots),
            "total_steps":       self._step,
        }

    def save(self, path: str) -> None:
        data = {
            "version": "2.0",
            "timestamp": time.time(),
            "pathway_snapshots": [
                {"addrs": s["addrs"], "timestamp": s["timestamp"]}
                for s in self.pathway_snapshots
            ],
            "stats": self.report(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        import os
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for snap in data.get("pathway_snapshots", []):
                self.pathway_snapshots.append({
                    "addrs": [int(a) for a in snap["addrs"]],
                    "timestamp": float(snap["timestamp"]),
                })
            v = data.get("version", "1.0")
            n_snaps = len(self.pathway_snapshots)
            logger.info("Loaded %d pathway snapshots from %s (file version %s)", n_snaps, path, v)
            if v == "1.0":
                logger.info("v1 file detected — 'associations' dict discarded (plasticity now lives in surface[:, dim+5])")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error("Load failed (%s). Starting fresh.", e)
            return False

    def _hebbian_potentiate(self, addr_post: int, addr_pre: int, age: int) -> None:
        decay = math.exp(-age / self.tau)

        pw_post = float(self.sml.surface[addr_post, self._pw_col])
        delta_post = self.eta * decay * (self.pw_max - pw_post)
        self.sml.surface[addr_post, self._pw_col] = min(
            self.pw_max, max(0.0, pw_post + delta_post)
        )

        pw_pre = float(self.sml.surface[addr_pre, self._pw_col])
        delta_pre = self.eta * decay * (self.pw_max - pw_pre)
        self.sml.surface[addr_pre, self._pw_col] = min(
            self.pw_max, pw_pre + delta_pre
        )

        self.ltp_events += 1

    def _snapshot_pathway(self, addrs: List[int]) -> None:
        if not addrs:
            return
        addr_set = frozenset(addrs)
        if addr_set == self._last_snapshot_addrs:
            return
        self.pathway_snapshots.append({
            "addrs": list(addr_set),
            "timestamp": time.time(),
        })
        self._last_snapshot_addrs = addr_set
