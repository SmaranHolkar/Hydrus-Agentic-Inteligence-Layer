"""
Tensor-Train (TT) Decomposed Linear Projection Layer for GBPS Grounding.

Provides fast, low-rank tensor projection for grounded belief queries and claim validation.
Includes both PyTorch (GPU/GEMM) and pure-NumPy execution paths for zero-heavy-dependency environments.
"""

from typing import Dict, Any, Optional, Tuple
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    HAS_TORCH = False


if HAS_TORCH:
    class TTLinear(nn.Module):
        """
        PyTorch Tensor-Train linear projection.
        d1*d2*d3 must equal embedding dimension (e.g., 8*8*12 = 768).
        """
        def __init__(self, d1: int = 8, d2: int = 8, d3: int = 12, rank: int = 8):
            super().__init__()
            self.d1, self.d2, self.d3 = d1, d2, d3
            self.rank = rank

            self.G1 = nn.Parameter(torch.randn(d1, d1, rank) * 0.05)
            self.G2 = nn.Parameter(torch.randn(rank, d2, d2, rank) * 0.05)
            self.G3 = nn.Parameter(torch.randn(rank, d3, d3) * 0.05)
            self._cached_W: Optional[torch.Tensor] = None

        def set_rank(self, new_rank: int) -> None:
            if new_rank == self.rank and self._cached_W is not None:
                return
            current_r = self.G1.shape[-1]
            with torch.no_grad():
                if new_rank < current_r:
                    self.G1 = nn.Parameter(self.G1[:, :, :new_rank].clone())
                    self.G2 = nn.Parameter(self.G2[:new_rank, :, :, :new_rank].clone())
                    self.G3 = nn.Parameter(self.G3[:new_rank, :, :].clone())
                elif new_rank > current_r:
                    dev = self.G1.device
                    dt = self.G1.dtype
                    new_G1 = torch.randn(self.d1, self.d1, new_rank, device=dev, dtype=dt) * 0.05
                    new_G2 = torch.randn(new_rank, self.d2, self.d2, new_rank, device=dev, dtype=dt) * 0.05
                    new_G3 = torch.randn(new_rank, self.d3, self.d3, device=dev, dtype=dt) * 0.05
                    new_G1[:, :, :current_r] = self.G1.data
                    new_G2[:current_r, :, :, :current_r] = self.G2.data
                    new_G3[:current_r, :, :] = self.G3.data
                    self.G1, self.G2, self.G3 = nn.Parameter(new_G1), nn.Parameter(new_G2), nn.Parameter(new_G3)
            self._cached_W = None
            self.rank = new_rank

        def reconstruct(self) -> torch.Tensor:
            G12 = torch.einsum('iap,pjbq->ijabq', self.G1, self.G2)
            W = torch.einsum('ijabq,qkc->ijkabc', G12, self.G3)
            d = self.d1 * self.d2 * self.d3
            return W.reshape(d, d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                return x @ self.reconstruct()
            if self._cached_W is None or self._cached_W.device != x.device or self._cached_W.dtype != x.dtype:
                with torch.no_grad():
                    self._cached_W = self.reconstruct().detach()
            return x @ self._cached_W
else:
    class TTLinear:  # type: ignore
        """Fallback placeholder when PyTorch is not present."""
        def __init__(self, d1: int = 8, d2: int = 8, d3: int = 12, rank: int = 8):
            self.d1, self.d2, self.d3 = d1, d2, d3
            self.rank = rank


class NumpyTTLinear:
    """Pure-NumPy implementation of Tensor-Train linear projection for lightweight inference."""

    def __init__(self, d1: int = 8, d2: int = 8, d3: int = 12, rank: int = 8, seed: int = 42):
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.rank = rank
        rng = np.random.default_rng(seed)
        self.G1 = rng.normal(0.0, 0.05, size=(d1, d1, rank)).astype(np.float32)
        self.G2 = rng.normal(0.0, 0.05, size=(rank, d2, d2, rank)).astype(np.float32)
        self.G3 = rng.normal(0.0, 0.05, size=(rank, d3, d3)).astype(np.float32)
        self._cached_W: Optional[np.ndarray] = None

    def reconstruct(self) -> np.ndarray:
        if self._cached_W is not None:
            return self._cached_W
        G12 = np.einsum('iap,pjbq->ijabq', self.G1, self.G2)
        W = np.einsum('ijabq,qkc->ijkabc', G12, self.G3)
        d = self.d1 * self.d2 * self.d3
        self._cached_W = W.reshape(d, d)
        return self._cached_W

    def project(self, x: np.ndarray) -> np.ndarray:
        W = self.reconstruct()
        return np.dot(x, W)


def flop_estimate(d1: int = 8, d2: int = 8, d3: int = 12, rank: int = 8, batch: int = 1) -> Dict[str, Any]:
    """Calculate FLOP and parameter reduction of TTLinear vs standard dense matrix."""
    dim = d1 * d2 * d3
    dense_mults = batch * (dim ** 2)
    step1 = batch * d1 * d2 * d3 * d1 * rank
    step2 = batch * d1 * d2 * d3 * rank * d2 * rank
    step3 = batch * d1 * d2 * d3 * rank * d3
    tt_total = step1 + step2 + step3

    dense_params = dim ** 2
    tt_params = (d1 * d1 * rank) + (rank * d2 * d2 * rank) + (rank * d3 * d3)

    return {
        "dimension": dim,
        "dense_mults": dense_mults,
        "tt_mults": tt_total,
        "flop_reduction_x": round(dense_mults / max(1, tt_total), 2),
        "dense_params": dense_params,
        "tt_params": tt_params,
        "param_reduction_x": round(dense_params / max(1, tt_params), 2),
    }
