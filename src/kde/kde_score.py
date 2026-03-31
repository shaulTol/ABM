"""
KDE-Based Score Estimation with Isotropic Gaussian Kernels

Efficient score computation ∇log f̂(x) from a kernel density estimate,
without routing through the full GMM infrastructure.

For isotropic Gaussian KDE:
    f̂(x) = (1/N) Σᵢ K_h(x − xᵢ)    where K_h(u) ∝ exp(−‖u‖²/(2h²))

The score (gradient of log-density) is:
    ∇log f̂(x) = Σᵢ wᵢ(x) · (xᵢ − x) / h²

where wᵢ(x) = K_h(x − xᵢ) / Σⱼ K_h(x − xⱼ)  are the responsibilities.

This is O(N·M·d) for M query points — no matrix inversions needed.
"""
import torch
import numpy as np
from typing import Union, Optional


class KDEScore:
    """
    Score function ∇log f̂(x) from isotropic Gaussian KDE.

    Stores kernel centers and bandwidth, provides efficient batched
    score evaluation. Designed to plug into GenericLangevinSimulator.

    Usage:
        kde = KDEScore(samples, bandwidth='scott')
        sim = GenericLangevinSimulator(score_fn=kde.score, samples=samples)
    """

    def __init__(
        self,
        centers: torch.Tensor,
        bandwidth: Union[float, str] = 'scott',
        device: Optional[torch.device] = None,
        batch_size: int = 2000,
    ):
        """
        Args:
            centers: (N, d) kernel centers (typically the data samples,
                     or a subsample thereof).
            bandwidth: Scalar bandwidth h, or 'scott'/'silverman' rule.
            device: Computation device.
            batch_size: Max query points per batch to limit memory.
                        For N centers and M queries, peak memory is
                        O(batch_size · N · d).
        """
        self.device = device or centers.device
        self.centers = centers.to(self.device)  # (N, d)
        self.N, self.d = self.centers.shape
        self.batch_size = batch_size

        # Compute bandwidth
        if isinstance(bandwidth, str):
            std = self.centers.std(dim=0).mean().item()
            if bandwidth == 'scott':
                self.h = std * (self.N ** (-1.0 / (self.d + 4)))
            elif bandwidth == 'silverman':
                factor = self.N * (self.d + 2) / 4.0
                self.h = std * (factor ** (-1.0 / (self.d + 4)))
            else:
                raise ValueError(f"Unknown rule: {bandwidth}")
        else:
            self.h = float(bandwidth)

        self.h2 = self.h ** 2  # h²

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇log f̂(x) for a batch of query points.

        ∇log f̂(x) = Σᵢ wᵢ(x) · (xᵢ − x) / h²
        where wᵢ(x) = softmax(−‖x − xᵢ‖²/(2h²))

        Args:
            x: (..., d) query points

        Returns:
            (..., d) score vectors
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d)  # (M, d)
        M = x_flat.shape[0]

        grad_out = torch.zeros_like(x_flat)  # (M, d)

        # Process in batches to limit memory: O(batch_size · N · d)
        for start in range(0, M, self.batch_size):
            end = min(start + self.batch_size, M)
            x_batch = x_flat[start:end]  # (B, d)
            B = x_batch.shape[0]

            # Differences: (B, N, d)
            diff = self.centers.unsqueeze(0) - x_batch.unsqueeze(1)  # xᵢ − x

            # Squared distances: (B, N)
            sq_dist = (diff ** 2).sum(dim=-1)

            # Log-responsibilities: −‖x − xᵢ‖²/(2h²)
            log_w = -sq_dist / (2.0 * self.h2)

            # Responsibilities via softmax (numerically stable)
            w = torch.softmax(log_w, dim=-1)  # (B, N)

            # Score: Σᵢ wᵢ · (xᵢ − x) / h²
            # w: (B, N, 1) · diff: (B, N, d) → sum over N → (B, d)
            grad_out[start:end] = (w.unsqueeze(-1) * diff).sum(dim=1) / self.h2

        return grad_out.reshape(orig_shape)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log f̂(x) (useful for diagnostics).

        Args:
            x: (..., d) query points

        Returns:
            (...,) log-density values
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.d)
        M = x_flat.shape[0]

        log_f = torch.zeros(M, device=self.device)

        for start in range(0, M, self.batch_size):
            end = min(start + self.batch_size, M)
            x_batch = x_flat[start:end]

            diff = x_batch.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, N, d)
            sq_dist = (diff ** 2).sum(dim=-1)  # (B, N)
            log_k = -sq_dist / (2.0 * self.h2)  # (B, N)

            # log f̂(x) = logsumexp(log_k) − log(N) − (d/2)log(2πh²)
            log_f[start:end] = (
                torch.logsumexp(log_k, dim=-1)
                - np.log(self.N)
                - 0.5 * self.d * np.log(2 * np.pi * self.h2)
            )

        return log_f.reshape(orig_shape)

    def __repr__(self) -> str:
        return (f"KDEScore(N={self.N}, d={self.d}, h={self.h:.4f}, "
                f"device={self.device})")
