"""
Sliced-Like Autocorrelation from Full d-Dimensional Trajectories

This module computes a "sliced-like" SSA score by:
1. Simulating full d-dimensional Langevin trajectories
2. Projecting them to 1D along random unit directions
3. Computing 1D autocorrelation (with per-trajectory normalization) for each projection
4. Averaging SSA scores across projections

This provides a sanity check: it should behave like the sliced method but uses
the full d-dimensional simulation instead of projecting the GMM first.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Dict
import warnings

# Import simulation and metrics
try:
    from ..simulation.langevin import simulate_d_trajectories
    from ..metrics.autocorrelation import compute_autocorrelation, compute_ssa_score
    from .projection_strategies import uniform_random_projections
except ImportError:
    from src.simulation.langevin import simulate_d_trajectories
    from src.metrics.autocorrelation import compute_autocorrelation, compute_ssa_score
    from src.slicing.projection_strategies import uniform_random_projections


def sliced_like_autocorrelation(
    gmm_means: torch.Tensor,
    gmm_covs: torch.Tensor,
    gmm_weights: torch.Tensor,
    n_trajectories: int = 100,
    T: int = 1000,
    max_lag: int = 500,
    dt: float = 0.01,
    n_projections: int = 50,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    return_stats: bool = False
) -> Tuple[float, Optional[Dict]]:
    """
    Compute sliced-like SSA score from full d-dimensional trajectories
    
    Process:
    1. Simulate d-dimensional Langevin trajectories X_{n,t} ∈ ℝ^d
    2. Draw m random unit directions θ_1, ..., θ_m
    3. For each θ_j:
       - Project trajectories: x_{n,t}^{(j)} = θ_j^T X_{n,t}
       - Compute 1D ACF with per-trajectory normalization
       - Compute SSA_j = Σ_{h=1}^L ρ_j(h)²
    4. Return mean SSA across projections
    
    Args:
        gmm_means: (k, d) GMM component means
        gmm_covs: (k, d, d) GMM component covariance matrices
        gmm_weights: (k,) GMM mixture weights
        n_trajectories: Number of trajectories to simulate
        T: Length of each trajectory (number of steps)
        max_lag: Maximum lag L for autocorrelation computation
        dt: Time step size for Langevin simulation
        n_projections: Number of random projection directions m
        device: Computation device
        seed: Random seed for reproducibility
        return_stats: If True, also return std and quantiles across projections
        
    Returns:
        mean_ssa: Mean SSA score across all projections
        stats: Optional dict with 'std', 'min', 'max', 'q25', 'q50', 'q75' if return_stats=True
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    d = gmm_means.shape[1]
    
    # Step 1: Simulate d-dimensional Langevin trajectories
    trajectories = simulate_d_trajectories(
        means=gmm_means,
        covs=gmm_covs,
        weights=gmm_weights,
        n_trajectories=n_trajectories,
        T=T,
        dt=dt,
        device=device,
        seed=seed
    )  # Shape: (n_trajectories, T+1, d)
    
    # Step 2: Draw m random unit directions
    projections = uniform_random_projections(
        d=d,
        n_projections=n_projections,
        device=device,
        seed=seed if seed is not None else None
    )  # Shape: (n_projections, d)
    
    # Step 3: For each projection, compute 1D ACF and SSA
    ssa_scores = []
    
    for j in range(n_projections):
        theta_j = projections[j]  # (d,)
        
        # Project trajectories to 1D: x_{n,t}^{(j)} = θ_j^T X_{n,t}
        # trajectories: (N, T+1, d), theta_j: (d,)
        # Result: (N, T+1)
        projected_trajectories = torch.matmul(trajectories, theta_j)  # (N, T+1)
        
        # Compute 1D autocorrelation with per-trajectory normalization
        rho, mu, sigma = compute_autocorrelation(
            trajectories=projected_trajectories,
            max_lag=max_lag,
            demean=True
        )
        
        # Compute SSA_j = Σ_{h=1}^L ρ_j(h)²
        ssa_j = (rho[1:] ** 2).sum().item()  # Exclude lag 0
        ssa_scores.append(ssa_j)
    
    # Step 4: Compute mean and optionally statistics
    ssa_array = np.array(ssa_scores)
    mean_ssa = ssa_array.mean()
    
    if return_stats:
        stats = {
            'mean': mean_ssa,
            'std': ssa_array.std(),
            'min': ssa_array.min(),
            'max': ssa_array.max(),
            'q25': np.percentile(ssa_array, 25),
            'q50': np.percentile(ssa_array, 50),
            'q75': np.percentile(ssa_array, 75)
        }
        return mean_ssa, stats
    else:
        return mean_ssa, None


def test_sliced_like_autocorrelation():
    """Test the sliced-like autocorrelation function"""
    print("Testing Sliced-Like Autocorrelation...")
    
    device = torch.device('cpu')
    
    # Create a simple 2D bimodal mixture
    d = 2
    k = 2
    separation = 4.0
    variance = 1.0
    
    means = torch.tensor([
        [-separation/2, 0.0],
        [separation/2, 0.0]
    ], device=device)
    
    covs = torch.stack([
        variance * torch.eye(d, device=device),
        variance * torch.eye(d, device=device)
    ])
    
    weights = torch.tensor([0.5, 0.5], device=device)
    
    # Compute sliced-like SSA
    mean_ssa, stats = sliced_like_autocorrelation(
        gmm_means=means,
        gmm_covs=covs,
        gmm_weights=weights,
        n_trajectories=100,
        T=500,
        max_lag=100,
        n_projections=20,
        device=device,
        seed=42,
        return_stats=True
    )
    
    print(f"Mean SSA: {mean_ssa:.4f}")
    print(f"Stats: {stats}")
    print("✓ Sliced-like autocorrelation test passed!")


if __name__ == "__main__":
    test_sliced_like_autocorrelation()

