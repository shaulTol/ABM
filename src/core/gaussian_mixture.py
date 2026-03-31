"""
Gaussian Mixture Module - N-dimensional mixture model with information theory

Key Features:
1. Arbitrary dimensional support
2. GPU-accelerated sampling and density computation
3. Information-theoretic measures (entropy, mutual information)
4. Numerical stability for covariance operations
"""

import torch
import torch.distributions as dist
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Import our GPU manager and device utils
try:
    from .gpu_manager import get_gpu_manager
    from .device_utils import safe_cholesky, safe_inverse, move_to_device, BatchProcessor
except ImportError:
    # For standalone testing
    import sys
    sys.path.append('..')
    from core.gpu_manager import get_gpu_manager
    from core.device_utils import safe_cholesky, safe_inverse, move_to_device, BatchProcessor


class GaussianMixture:
    """
    N-dimensional Gaussian Mixture Model with information theory methods
    
    Mathematical foundation:
    - Mixture: p(x) = Σᵢ πᵢ N(x; μᵢ, Σᵢ)
    - Entropy of labels: H(Z) = -Σᵢ πᵢ log πᵢ
    - Conditional entropy: H(Z|X) = -∫ p(x) Σᵢ p(i|x) log p(i|x) dx
    - Mutual information: I(Z;X) = H(Z) - H(Z|X)
    """
    
    def __init__(self, 
                 means: Union[List[np.ndarray], torch.Tensor],
                 covariances: Union[List[np.ndarray], torch.Tensor],
                 weights: Union[List[float], np.ndarray, torch.Tensor],
                 device: Optional[str] = None,
                 validate: bool = True,
                 regularization: float = 1e-6):
        """
        Initialize N-dimensional Gaussian mixture
        
        Args:
            means: List of mean vectors or tensor of shape (k, d)
            covariances: List of covariance matrices or tensor of shape (k, d, d)
            weights: Mixture weights (must sum to 1)
            device: Device placement ('cuda', 'cpu', or None for auto)
            validate: Whether to validate inputs
            regularization: Small value added to diagonal for stability
        """
        # Get GPU manager for device selection
        self.gpu_manager = get_gpu_manager()
        
        # Determine device
        if device is None:
            self.device = self.gpu_manager.device
        else:
            self.device = torch.device(device)
        
        # Convert inputs to tensors
        self.means = self._to_tensor(means, shape_check=2)
        self.covs = self._to_tensor(covariances, shape_check=3)
        self.weights = self._to_tensor(weights, shape_check=1)
        
        # Get dimensions
        self.k = self.means.shape[0]  # Number of components
        self.d = self.means.shape[1]  # Dimensionality
        
        # Validate inputs if requested
        if validate:
            self._validate_inputs()
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        # Regularization for numerical stability
        self.regularization = regularization
        if regularization > 0:
            eye = torch.eye(self.d, device=self.device, dtype=self.covs.dtype)
            self.covs = self.covs + regularization * eye.unsqueeze(0)
        
        # Precompute for efficiency
        self._precompute()
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.gpu_manager)
        
        # Cache for repeated computations
        self._cache = {}
        
    def _to_tensor(self, data, shape_check: int) -> torch.Tensor:
        """Convert input to tensor with proper shape"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(self.device)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float().to(self.device)
        elif isinstance(data, list):
            # Handle list of arrays
            if len(data) > 0 and isinstance(data[0], (np.ndarray, list)):
                tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            else:
                tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Ensure correct number of dimensions
        if shape_check > 0 and tensor.ndim != shape_check:
            if shape_check == 1 and tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            elif shape_check == 1 and tensor.ndim == 2 and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            else:
                raise ValueError(f"Expected {shape_check}D tensor, got {tensor.ndim}D")
        
        return tensor
    
    def _validate_inputs(self):
        """Validate input dimensions and properties"""
        # Check dimensions match
        if self.covs.shape[0] != self.k:
            raise ValueError(f"Number of covariances ({self.covs.shape[0]}) != number of means ({self.k})")
        if self.weights.shape[0] != self.k:
            raise ValueError(f"Number of weights ({self.weights.shape[0]}) != number of components ({self.k})")
        if self.covs.shape[1] != self.d or self.covs.shape[2] != self.d:
            raise ValueError(f"Covariance dimensions ({self.covs.shape[1]}x{self.covs.shape[2]}) != data dimension ({self.d})")
        
        # Check weights are positive
        if (self.weights < 0).any():
            raise ValueError("Weights must be non-negative")
        if self.weights.sum() == 0:
            raise ValueError("Weights sum to zero")
        
        # Check covariances are symmetric
        if not torch.allclose(self.covs, self.covs.transpose(-2, -1), atol=1e-6):
            warnings.warn("Covariance matrices are not symmetric, symmetrizing...")
            self.covs = (self.covs + self.covs.transpose(-2, -1)) / 2
    
    def _precompute(self):
        """Precompute expensive operations for efficiency"""
        # Compute Cholesky decomposition for sampling
        self.chol = torch.zeros_like(self.covs)
        for i in range(self.k):
            self.chol[i] = safe_cholesky(self.covs[i], self.regularization)
        
        # Compute log determinants for density calculation
        self.log_dets = torch.zeros(self.k, device=self.device)
        for i in range(self.k):
            # Use Cholesky for stable determinant
            self.log_dets[i] = 2 * torch.sum(torch.log(torch.diag(self.chol[i])))
        
        # Compute precision matrices using Cholesky (more stable!)
        # Since Σ = LL^T, we have Σ^(-1) = (L^T)^(-1) L^(-1)
        # torch.cholesky_inverse computes this efficiently
        self.precs = torch.zeros_like(self.covs)
        for i in range(self.k):
            try:
                # Most stable: use Cholesky factor directly
                self.precs[i] = torch.cholesky_inverse(self.chol[i])
            except:
                # Fallback if cholesky_inverse fails
                # Solve L L^T X = I for X = Σ^(-1)
                eye = torch.eye(self.d, device=self.device, dtype=self.chol[i].dtype)
                self.precs[i] = torch.cholesky_solve(eye, self.chol[i])
        
        # Precompute log weights
        self.log_weights = torch.log(self.weights + 1e-10)
        
        # Constants
        self.log_2pi = torch.log(torch.tensor(2 * np.pi, device=self.device))
    
    def sample(self, n: int, return_labels: bool = True, 
               batch_size: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample from the mixture distribution
        
        Args:
            n: Number of samples to generate
            return_labels: Whether to return component labels
            batch_size: Batch size for sampling (None = auto)
            
        Returns:
            samples: (n, d) tensor of samples
            labels: (n,) tensor of component labels (if return_labels=True)
        """
        if batch_size is None:
            batch_size = self.gpu_manager.estimate_batch_size(
                'projection', (n, self.d), torch.float32
            )
        
        # Sample component labels from categorical distribution
        labels = dist.Categorical(self.weights).sample((n,))
        
        # Initialize samples tensor
        samples = torch.zeros(n, self.d, device=self.device)
        
        # Sample from each component
        for i in range(self.k):
            mask = labels == i
            n_i = mask.sum().item()
            
            if n_i > 0:
                # Generate standard normal samples
                eps = torch.randn(n_i, self.d, device=self.device)
                
                # Transform using Cholesky factor: x = μ + L*ε
                samples[mask] = self.means[i] + eps @ self.chol[i].T
        
        if return_labels:
            return samples, labels
        else:
            return samples
    
    def log_prob(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute log probability of samples under the mixture
        
        Args:
            x: (n, d) tensor of samples
            batch_size: Batch size for computation (None = auto)
            
        Returns:
            (n,) tensor of log probabilities
        """
        n = x.shape[0]
        
        if batch_size is None:
            batch_size = self.gpu_manager.estimate_batch_size(
                'kde', x.shape, x.dtype
            )
        
        def compute_batch_log_prob(batch):
            batch_size_local = batch.shape[0]
            log_probs = torch.zeros(batch_size_local, self.k, device=self.device)
            
            for i in range(self.k):
                # Compute Mahalanobis distance
                diff = batch - self.means[i].unsqueeze(0)
                mahalanobis = torch.sum(diff @ self.precs[i] * diff, dim=1)
                
                # Compute log probability for component i
                log_probs[:, i] = (
                    -0.5 * (self.d * self.log_2pi + self.log_dets[i] + mahalanobis) 
                    + self.log_weights[i]
                )
            
            # Log-sum-exp for numerical stability
            return torch.logsumexp(log_probs, dim=1)
        
        # Process in batches
        return self.batch_processor.process_in_batches(
            x, compute_batch_log_prob, batch_size, 
            operation="log_prob", aggregate="concat"
        )
    
    def posterior_probabilities(self, x: torch.Tensor, 
                              batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute posterior probabilities p(z|x) for each component
        
        Args:
            x: (n, d) tensor of samples
            batch_size: Batch size for computation
            
        Returns:
            (n, k) tensor of posterior probabilities
        """
        n = x.shape[0]
        
        if batch_size is None:
            batch_size = self.gpu_manager.estimate_batch_size(
                'kde', x.shape, x.dtype
            )
        
        def compute_batch_posterior(batch):
            batch_size_local = batch.shape[0]
            log_probs = torch.zeros(batch_size_local, self.k, device=self.device)
            
            for i in range(self.k):
                diff = batch - self.means[i].unsqueeze(0)
                mahalanobis = torch.sum(diff @ self.precs[i] * diff, dim=1)
                
                log_probs[:, i] = (
                    -0.5 * (self.d * self.log_2pi + self.log_dets[i] + mahalanobis)
                    + self.log_weights[i]
                )
            
            # Normalize to get posteriors
            log_norm = torch.logsumexp(log_probs, dim=1, keepdim=True)
            return torch.exp(log_probs - log_norm)
        
        return self.batch_processor.process_in_batches(
            x, compute_batch_posterior, batch_size,
            operation="posterior", aggregate="concat"
        )
    
    def entropy_z(self) -> torch.Tensor:
        """
        Compute entropy of component labels H(Z)
        
        H(Z) = -Σᵢ πᵢ log πᵢ
        
        Returns:
            Scalar tensor with entropy value
        """
        # Use cached value if available
        if 'entropy_z' in self._cache:
            return self._cache['entropy_z']
        
        # Compute entropy
        eps = 1e-10
        entropy = -(self.weights * torch.log(self.weights + eps)).sum()
        
        # Cache result
        self._cache['entropy_z'] = entropy
        
        return entropy
    
    def entropy_z_given_x(self, n_samples: int = 5000,
                         batch_size: Optional[int] = None,
                         use_importance_sampling: bool = False) -> torch.Tensor:
        """
        Estimate conditional entropy H(Z|X) using Monte Carlo
        
        H(Z|X) = -E_X[Σᵢ p(i|x) log p(i|x)]
        
        Args:
            n_samples: Number of Monte Carlo samples
            batch_size: Batch size for computation
            use_importance_sampling: Use importance sampling for better estimate
            
        Returns:
            Scalar tensor with conditional entropy estimate
        """
        # Check cache
        cache_key = f'entropy_z_given_x_{n_samples}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Sample from the mixture
        samples, true_labels = self.sample(n_samples, return_labels=True)
        
        # Compute posterior probabilities
        posteriors = self.posterior_probabilities(samples, batch_size)
        
        # Compute entropy for each sample
        eps = 1e-10
        sample_entropies = -(posteriors * torch.log(posteriors + eps)).sum(dim=1)
        
        if use_importance_sampling:
            # Weight by true posterior for better estimate
            weights = posteriors[torch.arange(n_samples), true_labels]
            weighted_entropy = (sample_entropies * weights).mean() / weights.mean()
        else:
            # Simple Monte Carlo average
            weighted_entropy = sample_entropies.mean()
        
        # Cache result
        self._cache[cache_key] = weighted_entropy
        
        return weighted_entropy
    
    def mutual_information(self, n_samples: int = 5000,
                         batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute mutual information I(Z;X) = H(Z) - H(Z|X)
        
        This measures how much information the data X contains about 
        the component labels Z.
        
        Args:
            n_samples: Number of samples for Monte Carlo estimation
            batch_size: Batch size for computation
            
        Returns:
            Scalar tensor with mutual information
        """
        h_z = self.entropy_z()
        h_z_given_x = self.entropy_z_given_x(n_samples, batch_size)
        
        mi = h_z - h_z_given_x
        
        # MI should be non-negative
        if mi < 0:
            warnings.warn(f"Negative MI detected: {mi.item():.6f}. Setting to 0.")
            mi = torch.zeros_like(mi)
        
        return mi
    
    def kl_divergence(self, other: 'GaussianMixture', 
                     n_samples: int = 10000) -> torch.Tensor:
        """
        Estimate KL divergence KL(self || other) using Monte Carlo
        
        KL(p||q) = E_p[log p(x) - log q(x)]
        
        Args:
            other: Another GaussianMixture instance
            n_samples: Number of samples for estimation
            
        Returns:
            Scalar tensor with KL divergence estimate
        """
        # Sample from self
        samples = self.sample(n_samples, return_labels=False)
        
        # Compute log probabilities
        log_p = self.log_prob(samples)
        log_q = other.log_prob(samples)
        
        # Estimate KL divergence
        kl = (log_p - log_q).mean()
        
        # KL should be non-negative
        if kl < 0:
            warnings.warn(f"Negative KL detected: {kl.item():.6f}. Setting to 0.")
            kl = torch.zeros_like(kl)
        
        return kl
    
    def merge_components(self, threshold: float = 0.01) -> 'GaussianMixture':
        """
        Merge similar components based on KL divergence
        
        Args:
            threshold: Merge components if KL divergence below this
            
        Returns:
            New GaussianMixture with merged components
        """
        # Compute pairwise KL divergences between components
        merge_groups = []
        merged = set()
        
        for i in range(self.k):
            if i in merged:
                continue
            
            group = [i]
            for j in range(i + 1, self.k):
                if j in merged:
                    continue
                
                # Create single-component mixtures
                gm_i = GaussianMixture(
                    self.means[i:i+1], 
                    self.covs[i:i+1],
                    torch.ones(1),
                    device=self.device
                )
                gm_j = GaussianMixture(
                    self.means[j:j+1],
                    self.covs[j:j+1], 
                    torch.ones(1),
                    device=self.device
                )
                
                # Compute symmetric KL
                kl_ij = gm_i.kl_divergence(gm_j, n_samples=1000)
                kl_ji = gm_j.kl_divergence(gm_i, n_samples=1000)
                sym_kl = (kl_ij + kl_ji) / 2
                
                if sym_kl < threshold:
                    group.append(j)
                    merged.add(j)
            
            merge_groups.append(group)
        
        # Create merged components
        new_means = []
        new_covs = []
        new_weights = []
        
        for group in merge_groups:
            # Compute merged weight
            group_weight = sum(self.weights[i] for i in group)
            new_weights.append(group_weight)
            
            # Compute merged mean (weighted average)
            weighted_mean = sum(
                self.weights[i] * self.means[i] for i in group
            ) / group_weight
            new_means.append(weighted_mean)
            
            # Compute merged covariance (moment matching)
            weighted_cov = sum(
                self.weights[i] * (
                    self.covs[i] + 
                    torch.outer(self.means[i] - weighted_mean, 
                              self.means[i] - weighted_mean)
                ) for i in group
            ) / group_weight
            new_covs.append(weighted_cov)
        
        return GaussianMixture(
            torch.stack(new_means),
            torch.stack(new_covs),
            torch.tensor(new_weights, device=self.device),
            device=self.device
        )
    
    def to_dict(self) -> dict:
        """Export mixture parameters as dictionary"""
        return {
            'means': self.means.cpu().numpy(),
            'covariances': self.covs.cpu().numpy(),
            'weights': self.weights.cpu().numpy(),
            'k': self.k,
            'd': self.d,
            'device': str(self.device)
        }
    
    @classmethod
    def from_dict(cls, params: dict, device: Optional[str] = None) -> 'GaussianMixture':
        """Create GaussianMixture from dictionary"""
        return cls(
            params['means'],
            params['covariances'],
            params['weights'],
            device=device or params.get('device', None)
        )
    
    def __repr__(self) -> str:
        return (f"GaussianMixture(k={self.k}, d={self.d}, "
                f"device={self.device}, "
                f"weights={self.weights.cpu().numpy().round(3)})")


# Convenience functions for common mixture types

def create_bimodal_mixture(separation: float = 4.0,
                          variance: float = 1.0,
                          weight1: float = 0.5,
                          dimension: int = 2,
                          device: Optional[str] = None) -> GaussianMixture:
    """
    Create a bimodal Gaussian mixture
    
    Args:
        separation: Distance between component means
        variance: Variance for each component
        weight1: Weight of first component (weight2 = 1 - weight1)
        dimension: Number of dimensions
        device: Device placement
        
    Returns:
        GaussianMixture instance
    """
    # Create means
    mean1 = torch.zeros(dimension)
    mean1[0] = -separation / 2
    
    mean2 = torch.zeros(dimension)
    mean2[0] = separation / 2
    
    means = torch.stack([mean1, mean2])
    
    # Create covariances (spherical)
    cov = variance * torch.eye(dimension)
    covs = torch.stack([cov, cov])
    
    # Weights
    weights = torch.tensor([weight1, 1 - weight1])
    
    return GaussianMixture(means, covs, weights, device=device)


def create_ring_mixture(n_components: int = 8,
                       radius: float = 3.0,
                       variance: float = 0.5,
                       dimension: int = 2,
                       device: Optional[str] = None) -> GaussianMixture:
    """
    Create a ring-shaped mixture with components arranged in a circle
    
    Args:
        n_components: Number of components around the ring
        radius: Radius of the ring
        variance: Variance for each component
        dimension: Number of dimensions (uses first 2 for ring)
        device: Device placement
        
    Returns:
        GaussianMixture instance
    """
    means = []
    
    for i in range(n_components):
        angle = 2 * np.pi * i / n_components
        mean = torch.zeros(dimension)
        mean[0] = radius * np.cos(angle)
        if dimension > 1:
            mean[1] = radius * np.sin(angle)
        means.append(mean)
    
    means = torch.stack(means)
    
    # Create covariances
    cov = variance * torch.eye(dimension)
    covs = torch.stack([cov] * n_components)
    
    # Equal weights
    weights = torch.ones(n_components) / n_components
    
    return GaussianMixture(means, covs, weights, device=device)


def create_grid_mixture(grid_size: int = 3,
                       spacing: float = 2.0,
                       variance: float = 0.5,
                       dimension: int = 2,
                       device: Optional[str] = None) -> GaussianMixture:
    """
    Create a grid-arranged mixture
    
    Args:
        grid_size: Size of grid (grid_size x grid_size components)
        spacing: Distance between adjacent components
        variance: Variance for each component
        dimension: Number of dimensions (uses first 2 for grid)
        device: Device placement
        
    Returns:
        GaussianMixture instance
    """
    means = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            mean = torch.zeros(dimension)
            mean[0] = (i - grid_size/2 + 0.5) * spacing
            if dimension > 1:
                mean[1] = (j - grid_size/2 + 0.5) * spacing
            means.append(mean)
    
    means = torch.stack(means)
    
    # Create covariances
    cov = variance * torch.eye(dimension)
    n_components = grid_size * grid_size
    covs = torch.stack([cov] * n_components)
    
    # Equal weights
    weights = torch.ones(n_components) / n_components
    
    return GaussianMixture(means, covs, weights, device=device)

def create_multi_mode_mixture(
    n_components: int = 3,
    gap: float = 3.0,
    variance: float = 1.0,
    dimension: int = 2,
    arrangement: str = "line",  # "line", "circle", "grid"
    equal_weights: bool = True,
    device: Optional[str] = None
) -> GaussianMixture:
    """
    Create multi-component mixture with fixed gaps between components
    
    Args:
        n_components: Number of components
        gap: Distance between adjacent components
        variance: Variance for each component
        dimension: Number of dimensions
        arrangement: How to arrange components
        equal_weights: Use equal weights for all components
        device: Device placement
    """
    means = []
    
    if arrangement == "line":
        # Arrange components in a line along first axis
        for i in range(n_components):
            mean = torch.zeros(dimension)
            mean[0] = (i - (n_components-1)/2) * gap
            means.append(mean)
    
    elif arrangement == "circle" and dimension >= 2:
        # Arrange in circle (2D+)
        radius = gap * n_components / (2 * np.pi)
        for i in range(n_components):
            angle = 2 * np.pi * i / n_components
            mean = torch.zeros(dimension)
            mean[0] = radius * np.cos(angle)
            mean[1] = radius * np.sin(angle)
            means.append(mean)
    
    elif arrangement == "grid" and dimension >= 2:
        # Arrange in grid (2D+)
        grid_size = int(np.ceil(np.sqrt(n_components)))
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_components:
                    break
                mean = torch.zeros(dimension)
                mean[0] = (i - grid_size/2 + 0.5) * gap
                mean[1] = (j - grid_size/2 + 0.5) * gap
                means.append(mean)
                idx += 1
    else:
        raise ValueError(f"Invalid arrangement: {arrangement}")
    
    means = torch.stack(means)
    
    # Create covariances
    cov = variance * torch.eye(dimension)
    covs = torch.stack([cov] * n_components)
    
    # Weights
    if equal_weights:
        weights = torch.ones(n_components) / n_components
    else:
        # Could implement other weight schemes
        weights = torch.ones(n_components) / n_components
    
    return GaussianMixture(means, covs, weights, device=device)


def create_complex_mixture(
    n_components: int = 3,
    gaps: Optional[List[float]] = None,
    variances: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    dimension: int = 2,
    arrangement: str = "custom",
    device: Optional[str] = None
) -> GaussianMixture:
    """
    Create mixture with full control over all parameters
    
    Args:
        n_components: Number of components
        gaps: Custom gaps between components (length n_components-1)
        variances: Variance for each component (length n_components)
        weights: Weight for each component (must sum to 1)
        dimension: Number of dimensions
        arrangement: How to arrange components
        device: Device placement
    """
    # Default values
    if gaps is None:
        gaps = [3.0] * (n_components - 1)
    if variances is None:
        variances = [1.0] * n_components
    if weights is None:
        weights = [1.0/n_components] * n_components
    
    # Validate inputs
    if len(gaps) != n_components - 1:
        raise ValueError(f"Expected {n_components-1} gaps, got {len(gaps)}")
    if len(variances) != n_components:
        raise ValueError(f"Expected {n_components} variances, got {len(variances)}")
    if len(weights) != n_components:
        raise ValueError(f"Expected {n_components} weights, got {len(weights)}")
    
    # Create means based on gaps
    means = []
    current_pos = 0.0
    
    for i in range(n_components):
        mean = torch.zeros(dimension)
        if i == 0:
            mean[0] = -(sum(gaps) / 2)  # Center the mixture
        else:
            current_pos += gaps[i-1]
            mean[0] = -(sum(gaps) / 2) + current_pos
        means.append(mean)
    
    means = torch.stack(means)
    
    # Create covariances
    covs = []
    for var in variances:
        cov = var * torch.eye(dimension)
        covs.append(cov)
    covs = torch.stack(covs)
    
    # Convert weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()  # Normalize
    
    return GaussianMixture(means, covs, weights, device=device)
