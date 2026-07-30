"""
Tensor-Train (TT) decomposed linear layer for HCL.

This is the classical-hardware analog of a quantum many-body idea: a dense
weight matrix (like HILHamiltonian.grounding_matrix) is factored into a chain
of small "cores," the same trick physicists use to classically simulate
quantum systems tractably (DMRG / tensor networks). No complex numbers,
no autograd-in-the-loop, no simulated superposition -- just structured
real-valued linear algebra.
"""

import torch
import torch.nn as nn
import math

class TTLinear(nn.Module):
    def __init__(self, d1: int, d2: int, d3: int, rank: int):
        """
        d1*d2*d3 must equal the embedding dim (e.g. 8*8*12 = 768).
        rank: the shared bond dimension r1 = r2 = rank.
        """
        super().__init__()
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.rank = rank

        self.G1 = nn.Parameter(torch.randn(d1, d1, rank) * 0.05)
        self.G2 = nn.Parameter(torch.randn(rank, d2, d2, rank) * 0.05)
        self.G3 = nn.Parameter(torch.randn(rank, d3, d3) * 0.05)
        
        self._cached_W = None

    def set_rank(self, new_rank: int):
        """
        Runtime rank change via SVD-style slicing/padding of existing cores.
        This allows HCL to trade fidelity for speed per-call without retraining.
        """
        if new_rank == self.rank and self._cached_W is not None:
            return

        current_r = self.G1.shape[-1]
        with torch.no_grad():
            if new_rank < current_r:
                # Truncate to a smaller rank (keep top singular components)
                self.G1 = nn.Parameter(self.G1[:, :, :new_rank].clone())
                self.G2 = nn.Parameter(self.G2[:new_rank, :, :, :new_rank].clone())
                self.G3 = nn.Parameter(self.G3[:new_rank, :, :].clone())
            elif new_rank > current_r:
                # Pad to a larger rank using small random initialized dimensions
                device = self.G1.device
                dtype = self.G1.dtype
                
                new_G1 = torch.randn(self.d1, self.d1, new_rank, device=device, dtype=dtype) * 0.05
                new_G2 = torch.randn(new_rank, self.d2, self.d2, new_rank, device=device, dtype=dtype) * 0.05
                new_G3 = torch.randn(new_rank, self.d3, self.d3, device=device, dtype=dtype) * 0.05
                
                new_G1[:, :, :current_r] = self.G1.data
                new_G2[:current_r, :, :, :current_r] = self.G2.data
                new_G3[:current_r, :, :] = self.G3.data
                
                self.G1 = nn.Parameter(new_G1)
                self.G2 = nn.Parameter(new_G2)
                self.G3 = nn.Parameter(new_G3)
                
        self._cached_W = None  # Clear cache
        self.rank = new_rank

    def reconstruct(self) -> torch.Tensor:
        """
        Reconstructs the dense weight matrix from the Tensor-Train cores.
        """
        # Contract G1 and G2 over rank dimension r1 (p)
        G12 = torch.einsum('iap,pjbq->ijabq', self.G1, self.G2)
        # Contract G12 and G3 over rank dimension r2 (q)
        W = torch.einsum('ijabq,qkc->ijkabc', G12, self.G3)
        
        d = self.d1 * self.d2 * self.d3
        return W.reshape(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass of TTLinear.
        
        If training, reconstructs the dense matrix on the fly for backprop.
        If evaluating, utilizes the cached dense matrix to run at full GEMM speed.
        """
        if self.training:
            W = self.reconstruct()
            return x @ W
        else:
            if self._cached_W is None or self._cached_W.device != x.device or self._cached_W.dtype != x.dtype:
                with torch.no_grad():
                    self._cached_W = self.reconstruct().detach()
            return x @ self._cached_W


def flop_estimate(d1: int, d2: int, d3: int, rank: int, batch: int = 1):
    """
    Multiply-add count for the sequential TT path vs an equivalent dense d x d matmul.
    Used for monitoring compression statistics.
    """
    dense = batch * (d1 * d2 * d3) ** 2
    step1 = batch * d1 * d2 * d3 * d1 * rank
    step2 = batch * d1 * d2 * d3 * rank * d2 * rank
    step3 = batch * d1 * d2 * d3 * rank * d3
    tt_total = step1 + step2 + step3
    
    dense_params = (d1 * d2 * d3) ** 2
    tt_params = (d1 * d1 * rank) + (rank * d2 * d2 * rank) + (rank * d3 * d3)
    
    return {
        "dense_mults": dense,
        "tt_mults": tt_total,
        "flop_reduction_x": dense / tt_total,
        "dense_params": dense_params,
        "tt_params": tt_params,
        "param_reduction_x": dense_params / tt_params,
    }
