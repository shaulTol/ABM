"""
Kernel Density Estimation Module

Converts raw samples into a Gaussian Mixture Model representation.

A KDE with Gaussian kernels is mathematically equivalent to a GMM where:
- Each sample is a component center (mean)
- All covariances are h²I (bandwidth determines variance)  
- All weights are uniform (1/N)

This allows reuse of existing GMM-based Langevin simulation infrastructure.
"""

import torch
import numpy as np
from typing import Union, Optional


class KernelDensityEstimate:
    """
    Kernel Density Estimate from samples.
    
    Converts samples to GMM parameters for use with Langevin simulation.
    
    Attributes:
        samples: (N, d) tensor of original samples
        bandwidth: scalar bandwidth value
        means: (N, d) tensor - same as samples
        covs: (N, d, d) tensor - h²I for each component
        weights: (N,) tensor - uniform 1/N
    """
    
    def __init__(
        self,
        samples: torch.Tensor,
        bandwidth: Union[float, str] = 'scott',
        device: Optional[torch.device] = None,
        weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize KDE from samples.
        
        Args:
            samples: (N, d) tensor of data samples
            bandwidth: Bandwidth parameter. Can be:
                - float: explicit bandwidth value
                - 'scott': Scott's rule (default)
                - 'silverman': Silverman's rule
            device: Computation device (default: same as samples)
            weights: Optional (N,) tensor of sample weights. If None, uses uniform weights (1/N).
                     Weights will be normalized to sum to 1.
        """
        # Store samples
        self.device = device or samples.device
        self.samples = samples.to(self.device)
        
        # Get dimensions
        self.n_samples = samples.shape[0]
        self.d = samples.shape[1] if samples.dim() > 1 else 1
        
        # Handle 1D case
        if samples.dim() == 1:
            self.samples = self.samples.unsqueeze(1)
        
        # Compute bandwidth
        self._bandwidth = self._compute_bandwidth(bandwidth)
        
        # Precompute GMM parameters
        self._means = self.samples  # (N, d)
        
        # Set weights (normalize if provided)
        if weights is not None:
            weights = weights.to(self.device)
            if weights.shape[0] != self.n_samples:
                raise ValueError(f"Weights shape {weights.shape[0]} must match number of samples {self.n_samples}")
            # Normalize weights to sum to 1
            self._weights = weights / weights.sum()
        else:
            # Uniform weights
            self._weights = torch.ones(self.n_samples, device=self.device) / self.n_samples
        
        self._covs = self._compute_covariances()
    
    def _compute_bandwidth(self, bandwidth: Union[float, str]) -> float:
        """
        Compute bandwidth using specified method.
        
        Scott's rule: h = N^(-1/(d+4)) * std
        Silverman's rule: h = (N*(d+2)/4)^(-1/(d+4)) * std
        
        Args:
            bandwidth: float value or rule name ('scott', 'silverman')
            
        Returns:
            Scalar bandwidth value
        """
        if isinstance(bandwidth, (int, float)):
            return float(bandwidth)
        
        # Compute standard deviation (average across dimensions)
        std = self.samples.std(dim=0).mean().item()
        
        if bandwidth == 'scott':
            # Scott's rule: h = N^(-1/(d+4)) * std
            h = (self.n_samples ** (-1.0 / (self.d + 4))) * std
        elif bandwidth == 'silverman':
            # Silverman's rule: h = (N*(d+2)/4)^(-1/(d+4)) * std
            factor = self.n_samples * (self.d + 2) / 4.0
            h = (factor ** (-1.0 / (self.d + 4))) * std
        else:
            raise ValueError(f"Unknown bandwidth rule: {bandwidth}. Use 'scott', 'silverman', or a float.")
        
        return h
    
    def _compute_covariances(self) -> torch.Tensor:
        """
        Compute covariance matrices for all components.
        
        For isotropic Gaussian kernels: Σ = h²I
        
        Returns:
            (N, d, d) tensor of covariance matrices
        """
        # All components have the same covariance: h²I
        cov = (self._bandwidth ** 2) * torch.eye(self.d, device=self.device)
        
        # Repeat for all N components
        covs = cov.unsqueeze(0).expand(self.n_samples, -1, -1).clone()
        
        return covs
    
    @property
    def means(self) -> torch.Tensor:
        """(N, d) tensor of component means (the original samples)."""
        return self._means
    
    @property
    def covs(self) -> torch.Tensor:
        """(N, d, d) tensor of component covariances (h²I for each)."""
        return self._covs
    
    @property
    def weights(self) -> torch.Tensor:
        """(N,) tensor of mixture weights (normalized to sum to 1)."""
        return self._weights
    
    @property
    def bandwidth(self) -> float:
        """Scalar bandwidth value."""
        return self._bandwidth
    
    def __repr__(self) -> str:
        return (f"KernelDensityEstimate(n_samples={self.n_samples}, d={self.d}, "
                f"bandwidth={self._bandwidth:.4f}, device={self.device})")

