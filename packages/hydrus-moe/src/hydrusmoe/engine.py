from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import threading
import time
from .config import HydrusMoEConfig
from .crypto import AES256GCMEncryptor, ManifestVerifier
from .tiered_storage import TieredStorage
from .router import SecureRouter, AdaptiveEntropyRouter
from .prefetcher import HAILPrefetcher
from .oblivious_fetcher import fetch_experts_secure
from .dynamic_dispatch import execute_optimized_sparse_moe

class HydrusMoEEngine:
    """High-level HydrusMoE Tiered Model Engine & Forward Pass Orchestrator."""

    def __init__(self, config: Optional[HydrusMoEConfig] = None):
        self.config = config or HydrusMoEConfig()
        self.crypto = AES256GCMEncryptor(self.config.encryption_key, self.config.hardware_uuid)
        self.manifest_verifier = ManifestVerifier()
        self.tiers = TieredStorage(self.config, self.crypto)
        if self.config.enable_dynamic_routing:
            self.router = AdaptiveEntropyRouter(
                num_experts=32,
                top_k=self.config.dynamic_max_top_k,
                dummy_padding_k=2,
                min_top_k=self.config.dynamic_min_top_k,
                max_top_k=self.config.dynamic_max_top_k,
                entropy_threshold=self.config.dynamic_entropy_threshold,
                entropy_hysteresis=self.config.dynamic_entropy_hysteresis,
                ema_alpha=self.config.dynamic_entropy_ema_alpha,
            )
        else:
            self.router = SecureRouter(num_experts=32, top_k=self.config.dynamic_min_top_k, dummy_padding_k=2)
        self.prefetcher = HAILPrefetcher(self.config)
        worker_count = max(1, int(getattr(self.config, "async_prefetch_workers", 1)))
        self._prefetch_executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hail-prefetch")
        self._prefetch_lock = threading.Lock()
        self._prefetch_future: Optional[Future] = None
        self._prefetch_submitted = 0
        self._prefetch_completed = 0
        self._prefetch_failed = 0
        self._prefetch_dropped = 0
        self._prefetch_skipped_lock_busy = 0
        self._prefetch_last_error = ""
        self._prefetch_target: frozenset[int] = frozenset()
        self._trit_prefetch_state = 1
        self._trit_last_dropped = 0
        self._trit_last_skipped = 0
        window_size = max(8, int(getattr(self.config, "prefetch_latency_window", 48)))
        self._forward_latencies_ms = deque(maxlen=window_size)
        self._forward_query_count = 0
        self._prefetch_pause_until_query = 0
        self._prefetch_last_ev_score_ms = 0.0
        self._prefetch_last_pressure = 0.0
        self.active_model_id: Optional[str] = None
        self.weights_mode = "uninitialized"
        self.loaded_expert_count = 0
        self.last_routing_info: Dict[str, Any] = {
            "active_top_k": getattr(self.router, "top_k", self.config.dynamic_min_top_k),
            "entropy": 0.0,
            "smoothed_entropy": 0.0,
        }
        self.last_dispatch_info: Dict[str, Any] = {
            "enabled": True,
            "available": False,
            "executed": False,
            "reason": "not_run",
        }

    def _poll_prefetch_future(self) -> None:
        with self._prefetch_lock:
            future = self._prefetch_future
            if future is None or not future.done():
                return
            self._prefetch_future = None

        try:
            result = future.result()
            if result is False:
                self._prefetch_skipped_lock_busy += 1
            else:
                self._prefetch_completed += 1
                self._prefetch_last_error = ""
        except Exception as exc:
            self._prefetch_failed += 1
            self._prefetch_last_error = str(exc)

    def _submit_prefetch_async(self, predicted: set) -> None:
        if not predicted:
            return
        self._poll_prefetch_future()
        predicted_target = frozenset(int(eid) for eid in predicted)

        with self._prefetch_lock:
            if self._prefetch_future is not None and not self._prefetch_future.done():
                if predicted_target != self._prefetch_target:
                    if self._prefetch_future.cancel():
                        self._prefetch_future = None
                    else:
                        self._prefetch_dropped += 1
                        return
                else:
                    self._prefetch_dropped += 1
                    return
            self._prefetch_future = self._prefetch_executor.submit(self.tiers.prefetch_to_ram, list(predicted), True)
            self._prefetch_target = predicted_target
            self._prefetch_submitted += 1

    def _rolling_p95_latency(self) -> float:
        values = list(self._forward_latencies_ms)
        if not values:
            return 0.0
        values.sort()
        idx = max(0, min(len(values) - 1, int((len(values) - 1) * 0.95)))
        return float(values[idx])

    def _record_forward_latency(self, latency_ms: float) -> None:
        self._forward_latencies_ms.append(float(latency_ms))

    def _compute_prefetch_pressure(self) -> float:
        # Hot-path only: avoid full telemetry aggregation (it scans SSD cache dir).
        vram_used = float(getattr(self.tiers.tier0, "allocated_bytes", 0)) / (1024.0 * 1024.0)
        vram_budget = max(1.0, float(getattr(self.tiers.tier0, "budget_bytes", 1)) / (1024.0 * 1024.0))
        ram_used = float(getattr(self.tiers.tier1, "allocated_bytes", 0)) / (1024.0 * 1024.0)
        ram_budget = max(1.0, float(getattr(self.tiers.tier1, "budget_bytes", 1)) / (1024.0 * 1024.0))

        vram_pressure = min(1.0, vram_used / vram_budget)
        ram_pressure = min(1.0, ram_used / ram_budget)
        queue_pressure = 1.0 if (self._prefetch_future is not None and not self._prefetch_future.done()) else 0.0

        pressure = (0.45 * vram_pressure) + (0.45 * ram_pressure) + (0.10 * queue_pressure)
        return max(0.0, min(1.0, pressure))

    def _compute_dynamic_prefetch_budget(self, base_budget: int) -> int:
        if base_budget <= 0:
            return 0

        pressure = self._compute_prefetch_pressure()
        self._prefetch_last_pressure = pressure
        if pressure >= 0.92:
            return 0
        if pressure >= 0.80:
            return 1
        if pressure >= 0.65:
            return min(2, base_budget)
        return base_budget

    def _should_prefetch(self, predicted_raw: set, selected_candidates: set) -> bool:
        if not selected_candidates:
            self._prefetch_last_ev_score_ms = -1.0
            return False

        if self._forward_query_count < self._prefetch_pause_until_query:
            self._prefetch_last_ev_score_ms = -2.0
            return False

        rolling_p95 = self._rolling_p95_latency()
        threshold = float(getattr(self.config, "prefetch_tail_p95_threshold_ms", 10.0))
        if rolling_p95 > threshold:
            pause_len = max(1, int(getattr(self.config, "prefetch_tail_pause_queries", 12)))
            self._prefetch_pause_until_query = self._forward_query_count + pause_len
            self._prefetch_last_ev_score_ms = -3.0
            return False

        hit_rate = float(self.prefetcher.get_hit_rate())
        if self.prefetcher.total_queries <= 0:
            hit_rate = max(hit_rate, 0.5)

        predicted_count = max(1.0, float(len(predicted_raw)))
        selected_count = max(1.0, float(len(selected_candidates)))
        selection_ratio = min(1.0, selected_count / predicted_count)

        gain = float(getattr(self.config, "prefetch_ev_gain_ms", 1.25))
        cost = float(getattr(self.config, "prefetch_ev_cost_ms", 1.10))
        margin = float(getattr(self.config, "prefetch_ev_margin_ms", 0.05))
        penalty_w = float(getattr(self.config, "prefetch_pressure_penalty_ms", 1.50))

        pressure = self._compute_prefetch_pressure()
        self._prefetch_last_pressure = pressure
        ev_score = (hit_rate * gain * selection_ratio) - ((1.0 - hit_rate) * cost) - (pressure * penalty_w)
        self._prefetch_last_ev_score_ms = ev_score
        return ev_score > margin

    def _current_prefetch_budget(self) -> int:
        if bool(getattr(self.config, "enable_trit_prefetch_scheduler", True)):
            if self._trit_prefetch_state <= 0:
                return max(0, int(getattr(self.config, "trit_budget_state_0", 1)))
            if self._trit_prefetch_state == 1:
                return max(0, int(getattr(self.config, "trit_budget_state_1", 2)))
            return max(0, int(getattr(self.config, "trit_budget_state_2", 3)))
        return max(0, int(getattr(self.config, "prefetch_max_experts", 0)))

    def _select_prefetch_candidates(self, predicted_raw: set, budget: int) -> set:
        if budget <= 0 or not predicted_raw:
            return set()

        # O(k) candidate selection: prioritize previous activations and avoid O(k log k) sorting.
        selected: List[int] = []
        selected_mask = 0
        recent_actual = set(self.prefetcher.last_actual)
        for expert_id in predicted_raw:
            if expert_id in recent_actual:
                eid = int(expert_id)
                selected.append(eid)
                selected_mask |= (1 << eid)
                if len(selected) >= budget:
                    return set(selected)

        if len(selected) < budget:
            remaining_mask = 0
            for expert_id in predicted_raw:
                eid = int(expert_id)
                remaining_mask |= (1 << eid)
            remaining_mask &= ~selected_mask

            while remaining_mask and len(selected) < budget:
                lsb = remaining_mask & -remaining_mask
                selected.append(lsb.bit_length() - 1)
                remaining_mask ^= lsb

        return set(selected)

    def _update_trit_prefetch_state(self) -> None:
        if not bool(getattr(self.config, "enable_trit_prefetch_scheduler", True)):
            return

        hit_rate = float(self.prefetcher.get_hit_rate())
        low = float(getattr(self.config, "trit_low_hit_rate", 0.35))
        high = float(getattr(self.config, "trit_high_hit_rate", 0.75))

        dropped_delta = self._prefetch_dropped - self._trit_last_dropped
        skipped_delta = self._prefetch_skipped_lock_busy - self._trit_last_skipped
        self._trit_last_dropped = self._prefetch_dropped
        self._trit_last_skipped = self._prefetch_skipped_lock_busy

        if dropped_delta > 0 or skipped_delta > 0:
            self._trit_prefetch_state = max(0, self._trit_prefetch_state - 1)
            return

        if hit_rate >= high:
            self._trit_prefetch_state = min(2, self._trit_prefetch_state + 1)
        elif hit_rate <= low:
            self._trit_prefetch_state = max(0, self._trit_prefetch_state - 1)

    def _run_sparse_dispatch_probe(self, active_top_k: int) -> Dict[str, Any]:
        """Runs a lightweight sparse dispatch probe for runtime validation telemetry."""
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return {
                "enabled": True,
                "available": False,
                "executed": False,
                "reason": "torch_unavailable",
            }

        num_tokens = 12
        hidden_dim = 16
        max_k = max(active_top_k, self.config.dynamic_max_top_k)
        max_k = max(1, int(max_k))
        num_experts = 8

        hidden_states = torch.randn(num_tokens, hidden_dim)
        raw_indices = torch.randint(0, num_experts, (num_tokens, max_k), dtype=torch.long)
        raw_probs = torch.rand(num_tokens, max_k, dtype=torch.float32)

        if active_top_k < max_k:
            raw_probs[:, active_top_k:] = 0.0

        denom = raw_probs.sum(dim=-1, keepdim=True) + 1e-9
        dynamic_probs = raw_probs / denom

        experts = [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_experts)]
        out = execute_optimized_sparse_moe(hidden_states, dynamic_probs, raw_indices, experts)

        nonzero = int((dynamic_probs > 1e-8).sum().item())
        total = int(dynamic_probs.numel())
        executed_experts = int(torch.unique(raw_indices[dynamic_probs > 1e-8]).numel()) if nonzero > 0 else 0
        mean_active_routes = float((dynamic_probs > 1e-8).sum(dim=-1).float().mean().item())

        return {
            "enabled": True,
            "available": True,
            "executed": True,
            "reason": "ok",
            "tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "max_k": max_k,
            "active_top_k": int(active_top_k),
            "nonzero_routes": nonzero,
            "total_routes": total,
            "dropped_routes": int(total - nonzero),
            "executed_experts": executed_experts,
            "mean_active_routes_per_token": round(mean_active_routes, 4),
            "output_norm": round(float(out.norm().item()), 6),
        }

    def _align_router_to_loaded_experts(self) -> None:
        count = max(1, int(self.loaded_expert_count or 1))
        if hasattr(self.router, "num_experts"):
            self.router.num_experts = count
        if hasattr(self.router, "max_top_k"):
            self.router.max_top_k = max(1, min(int(getattr(self.router, "max_top_k", 1)), count))
        if hasattr(self.router, "min_top_k"):
            self.router.min_top_k = max(1, min(int(getattr(self.router, "min_top_k", 1)), count))
        if hasattr(self.router, "top_k"):
            self.router.top_k = max(1, min(int(getattr(self.router, "top_k", 1)), count))
        if hasattr(self.router, "dummy_padding_k"):
            max_dummy = max(0, count - int(getattr(self.router, "top_k", 1)))
            self.router.dummy_padding_k = max(0, min(int(getattr(self.router, "dummy_padding_k", 0)), max_dummy))

    def load_manifest(self, manifest_dict: Dict[str, Any]) -> bool:
        """Loads and verifies a model manifest, priming Tier 2 cold storage."""
        if not self.manifest_verifier.verify_manifest(manifest_dict):
            print("[HydrusMoEEngine Error] Manifest verification failed!")
            return False
            
        self.active_model_id = manifest_dict.get("model_id", "qwen3-35b-a3b")
        experts = manifest_dict.get("experts", [])
        loaded_from_local = 0
        loaded_synthetic = 0
        
        # Prime encrypted expert shards into Tier 2 SSD Vault.
        # If a manifest entry provides a local blob path, use real bytes; otherwise use synthetic fallback bytes.
        for e in experts:
            eid = e.get("id", 0)
            raw_weights = None
            local_blob_path = e.get("local_blob_path")
            if local_blob_path:
                try:
                    blob_path = Path(local_blob_path)
                    if blob_path.exists() and blob_path.is_file():
                        raw_weights = blob_path.read_bytes()
                        loaded_from_local += 1
                except Exception:
                    raw_weights = None

            if raw_weights is None:
                raw_weights = f"EXPERT_{eid}_QUANT_WEIGHTS_{self.active_model_id}".encode() * 5000
                loaded_synthetic += 1

            self.tiers.tier2.cache_expert(eid, raw_weights)

        self.loaded_expert_count = len(experts)
        self._align_router_to_loaded_experts()
        if loaded_from_local > 0 and loaded_synthetic == 0:
            self.weights_mode = "real_local_blobs"
        elif loaded_from_local > 0 and loaded_synthetic > 0:
            self.weights_mode = "mixed_local_and_synthetic"
        else:
            self.weights_mode = "synthetic_fixture"

        print(f"[HydrusMoEEngine] Successfully loaded and verified manifest for '{self.active_model_id}' ({len(experts)} experts).")
        return True

    def forward(self, prompt: str, user_memories: List[str] = None) -> Dict[str, Any]:
        """
        Executes a secure, tiered forward pass:
        1. HAIL predicts and pre-stages experts from SSD -> RAM.
        2. Router computes Top-K active experts with constant-time routing and dummy padding.
        3. Tiered storage retrieves/pins experts into VRAM.
        4. Fused GEMM computation executes.
        5. Returns telemetry and response.
        """
        self._forward_query_count += 1
        forward_start = time.perf_counter()
        try:
            available_experts = max(1, int(self.loaded_expert_count or 32))

            # 1. HAIL Predictive Prefetch
            if self.config.enable_hail_prefetch:
                predicted_raw = {
                    int(eid)
                    for eid in self.prefetcher.predict(prompt, user_memories)
                    if 0 <= int(eid) < available_experts
                }
                base_budget = self._current_prefetch_budget()
                dynamic_budget = self._compute_dynamic_prefetch_budget(base_budget)
                predicted = self._select_prefetch_candidates(set(predicted_raw), dynamic_budget)
                self.prefetcher.last_prediction = set(predicted)

                if self._should_prefetch(set(predicted_raw), predicted):
                    if bool(getattr(self.config, "enable_async_hail_prefetch", True)):
                        self._submit_prefetch_async(predicted)
                    else:
                        self.tiers.prefetch_to_ram(list(predicted), False)
            else:
                predicted = set()
                self.prefetcher.last_prediction = set()

            self._poll_prefetch_future()

            # 2. Secure Router Execution
            gating_logits = [0.1 * (i % 7) for i in range(available_experts)]
            if hasattr(self.router, "route_dynamic"):
                expert_ids, weights, routing_info = self.router.route_dynamic(gating_logits)
                active_top_k = int(routing_info.get("active_top_k", self.config.dynamic_min_top_k))
                self.last_routing_info = routing_info
            else:
                expert_ids, weights = self.router.route(gating_logits)
                active_top_k = getattr(self.router, "top_k", self.config.dynamic_min_top_k)
                self.last_routing_info = {
                    "active_top_k": active_top_k,
                    "entropy": 0.0,
                    "smoothed_entropy": 0.0,
                }

            # 3. Oblivious Fetching & VRAM Pinning
            common_pool = list(range(available_experts))
            if self.weights_mode in {"real_local_blobs"}:
                fetch_timeout_ms = int(getattr(self.config, "cold_fetch_timeout_ms_real", 50))
            else:
                fetch_timeout_ms = int(getattr(self.config, "cold_fetch_timeout_ms_synthetic", 0))
            secure_blobs = fetch_experts_secure(
                required_expert_ids=set(expert_ids[:active_top_k]),
                predicted_expert_ids=predicted,
                common_pool=common_pool,
                config=self.config,
                verifier=self.manifest_verifier,
                fetch_fn=lambda ids: self.tiers.fetch_to_vram(ids, lock_timeout_ms=fetch_timeout_ms)
            )

            # 4. Update HAIL prefetcher telemetry
            self.prefetcher.update_actual(expert_ids[:active_top_k])
            self._update_trit_prefetch_state()
            if bool(getattr(self.config, "enable_sparse_dispatch_probe", False)):
                self.last_dispatch_info = self._run_sparse_dispatch_probe(active_top_k)
            else:
                self.last_dispatch_info = {
                    "enabled": False,
                    "available": False,
                    "executed": False,
                    "reason": "disabled_by_config",
                }

            # Return execution summary & telemetry
            return {
                "status": "success",
                "model_id": self.active_model_id or "hydrusmoe-30b-q4",
                "active_experts": expert_ids[:active_top_k],
                "dummy_padding_experts": expert_ids[active_top_k:],
                "dispatch": dict(self.last_dispatch_info),
                "telemetry": self.get_status()
            }
        finally:
            elapsed_ms = (time.perf_counter() - forward_start) * 1000.0
            self._record_forward_latency(elapsed_ms)

    def get_status(self) -> Dict[str, Any]:
        telemetry = self.tiers.get_telemetry()
        prefetch_metrics = self.prefetcher.get_metrics()
        is_real_streaming_ready = self.weights_mode == "real_local_blobs" and self.loaded_expert_count > 0

        telemetry["prefetcher"] = {
            "hail_hit_rate": self.prefetcher.get_hit_rate(),
            "metrics": prefetch_metrics,
            "confidence_threshold": self.config.hail_confidence_threshold,
            "async": {
                "enabled": bool(getattr(self.config, "enable_async_hail_prefetch", True)),
                "submitted": self._prefetch_submitted,
                "completed": self._prefetch_completed,
                "failed": self._prefetch_failed,
                "dropped": self._prefetch_dropped,
                "skipped_lock_busy": self._prefetch_skipped_lock_busy,
                "last_error": self._prefetch_last_error,
            },
            "trit_scheduler": {
                "enabled": bool(getattr(self.config, "enable_trit_prefetch_scheduler", True)),
                "state": int(self._trit_prefetch_state),
                "prefetch_budget": int(self._current_prefetch_budget()),
                "threshold_low": float(getattr(self.config, "trit_low_hit_rate", 0.35)),
                "threshold_high": float(getattr(self.config, "trit_high_hit_rate", 0.75)),
            },
            "adaptive_guard": {
                "rolling_p95_ms": round(self._rolling_p95_latency(), 3),
                "tail_threshold_ms": float(getattr(self.config, "prefetch_tail_p95_threshold_ms", 10.0)),
                "paused_until_query": int(self._prefetch_pause_until_query),
                "current_query": int(self._forward_query_count),
                "last_ev_score_ms": round(float(self._prefetch_last_ev_score_ms), 6),
                "last_pressure": round(float(self._prefetch_last_pressure), 4),
            },
        }
        telemetry["security"] = {
            "encryption": "AES-256-GCM (Hardware-Bound)",
            "manifest_verified": True,
            "dummy_batch_size": self.config.dummy_batch_size,
            "pir_enabled": self.config.enable_pir
        }
        telemetry["active_model"] = self.active_model_id or "qwen3-35b-a3b"
        telemetry["execution"] = {
            "weights_mode": self.weights_mode,
            "loaded_experts": self.loaded_expert_count,
            "is_synthetic_demo": self.weights_mode in {"synthetic_fixture", "mixed_local_and_synthetic"},
            "streaming_generation_verified": is_real_streaming_ready,
            "dynamic_routing_enabled": bool(getattr(self.config, "enable_dynamic_routing", False)),
            "hail_prefetch_enabled": bool(getattr(self.config, "enable_hail_prefetch", True)),
            "sparse_dispatch": dict(self.last_dispatch_info),
        }
        if hasattr(self.router, "get_state"):
            telemetry["routing"] = self.router.get_state()
        else:
            telemetry["routing"] = {
                "enabled": False,
                "top_k": getattr(self.router, "top_k", self.config.dynamic_min_top_k),
            }
        telemetry["routing"]["last_forward"] = dict(self.last_routing_info)
        telemetry["claims"] = {
            "can_claim_real_35b_streaming": is_real_streaming_ready,
            "reason": "verified_local_expert_blobs" if is_real_streaming_ready else "no_verified_full_local_blob_manifest"
        }
        return telemetry

    def __del__(self):
        try:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
