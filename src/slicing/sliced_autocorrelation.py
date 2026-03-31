"""
Sliced Autocorrelation Analysis

Computes autocorrelation scores on random 1D projections of high-dimensional data.
"""
import torch
import numpy as np
from typing import Optional, Tuple
import warnings

# Import simulation and metrics
try:
    from ..simulation.langevin import simulate_trajectories, simulate_d_trajectories
    from ..metrics.autocorrelation import compute_ssa_score, compute_vector_ssa_score
    from .projection_strategies import uniform_random_projections, variance_biased_projections
except ImportError:
    from src.simulation.langevin import simulate_trajectories, simulate_d_trajectories
    from src.metrics.autocorrelation import compute_ssa_score, compute_vector_ssa_score
    from src.slicing.projection_strategies import uniform_random_projections, variance_biased_projections


def project_gmm_to_1d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    projection: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project a high-dimensional GMM to 1D along a direction
    
    For Gaussian N(μ, Σ), projection Y = θᵀX gives:
        Y ~ N(θᵀμ, θᵀΣθ)
    
    Args:
        means: (k, d) component means
        covariances: (k, d, d) component covariances
        weights: (k,) mixture weights
        projection: (d,) unit projection vector
        
    Returns:
        means_1d: (k,) projected means
        variances_1d: (k,) projected variances
        weights_1d: (k,) unchanged weights
    """
    # Project means: θᵀμ
    means_1d = torch.matmul(means, projection)  # (k,)
    
    # Project variances: θᵀΣθ
    # For each component: θᵀ Σ θ
    variances_1d = torch.zeros(means.shape[0], device=means.device)
    for i in range(means.shape[0]):
        variances_1d[i] = torch.matmul(
            projection,
            torch.matmul(covariances[i], projection)
        )
    
    # Ensure positive variances
    variances_1d = variances_1d.clamp_min(1e-6)
    
    return means_1d, variances_1d, weights


def sliced_autocorrelation(
    samples: torch.Tensor,
    gmm_means: Optional[torch.Tensor] = None,
    gmm_covs: Optional[torch.Tensor] = None,
    gmm_weights: Optional[torch.Tensor] = None,
    n_projections_init: int = 8,
    max_iterations: int = 30,
    epsilon: float = 1e-3,
    projection_type: str = 'uniform',
    n_trajectories: int = 100,
    T: int = 1000,
    max_lag: int = 100,
    dt: float = 0.01,
    projection_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> float:
    """
    Compute Sliced Autocorrelation Score with Adaptive Convergence
    
    Workflow:
    1. Start with n_projections_init random projections
    2. For each batch of projections:
        a. Project GMM to 1D: f₁D = GMM₁D(θ)
        b. Simulate Langevin trajectories on f₁D
        c. Compute SSA score Sᵢ
    3. Check convergence: if |SSA_new - SSA_old| / SSA_old < epsilon, stop
    4. Otherwise, add more projections up to max_iterations
    5. Return average: S̄ = (1/n) Σ Sᵢ
    
    Args:
        samples: (N, d) tensor of data samples (for covariance estimation)
        gmm_means: (k, d) GMM means (if known)
        gmm_covs: (k, d, d) GMM covariances (if known)
        gmm_weights: (k,) GMM weights (if known)
        n_projections_init: Initial number of projections per iteration
        max_iterations: Maximum number of iterations (total projections = n_projections_init * max_iterations)
        epsilon: Convergence threshold (relative change in SSA)
        projection_type: 'uniform' or 'variance_biased'
        n_trajectories: Number of Langevin trajectories per projection
        T: Length of each trajectory
        max_lag: Maximum lag for autocorrelation
        dt: Langevin time step
        projection_seed: Random seed for projections
        device: Computation device
        verbose: Print progress
        
    Returns:
        Average SSA score across projections
    """
    device = device or torch.device('cpu')
    d = samples.shape[1]
    
    # Set seed if provided
    if projection_seed is not None:
        torch.manual_seed(projection_seed)
    
    # If GMM parameters not provided, must be supplied
    if gmm_means is None:
        raise ValueError("GMM parameters (means, covs, weights) must be provided")
    
    # Adaptive convergence loop
    ssa_scores = []
    ssa_running_avg = []
    total_projections = 0
    
    for iteration in range(max_iterations):
        # Generate projections for this iteration
        n_proj_this_iter = n_projections_init
        
        if projection_type == 'uniform':
            projections = uniform_random_projections(d, n_proj_this_iter, device)
        elif projection_type == 'variance_biased':
            projections = variance_biased_projections(
                samples, n_proj_this_iter, device, temperature=1.0, pca_mix=0.5
            )
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
        
        # Compute SSA for each projection in this iteration
        for i in range(n_proj_this_iter):
            total_projections += 1
            
            # Project GMM to 1D
            means_1d, vars_1d, weights_1d = project_gmm_to_1d(
                gmm_means, gmm_covs, gmm_weights, projections[i]
            )
            
            # Simulate trajectories
            try:
                trajectories = simulate_trajectories(
                    means_1d, vars_1d, weights_1d,
                    n_trajectories=n_trajectories,
                    T=T,
                    dt=dt,
                    device=device
                )
                
                # Compute SSA score
                S, _ = compute_ssa_score(trajectories, max_lag)
                ssa_scores.append(S)
                
            except Exception as e:
                warnings.warn(f"Projection {total_projections} failed: {e}")
                continue
        
        if len(ssa_scores) == 0:
            raise RuntimeError("All projections failed")
        
        # Compute running average
        current_avg = np.mean(ssa_scores)
        ssa_running_avg.append(current_avg)
        
        if verbose:
            print(f"  Iteration {iteration+1}: {total_projections} projections, SSA={current_avg:.4f}")
        
        # Check convergence (need at least 2 iterations)
        if len(ssa_running_avg) >= 2:
            prev_avg = ssa_running_avg[-2]
            relative_change = abs(current_avg - prev_avg) / abs(prev_avg)
            
            if relative_change < epsilon:
                if verbose:
                    print(f"  Converged! Change={relative_change:.6f} < {epsilon}")
                break
    
    # Final average
    mean_ssa = np.mean(ssa_scores)
    
    if verbose:
        print(f"  Final SSA: {mean_ssa:.4f} (±{np.std(ssa_scores):.4f}) using {total_projections} projections")
    
    return mean_ssa


def full_autocorrelation(
    samples: torch.Tensor,
    gmm_means: Optional[torch.Tensor] = None,
    gmm_covs: Optional[torch.Tensor] = None,
    gmm_weights: Optional[torch.Tensor] = None,
    n_trajectories: int = 100,
    T: int = 1000,
    max_lag: int = 100,
    dt: float = 0.01,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    normalization: str = 'whitened'
) -> float:
    """
    Compute Full (Non-Sliced) Autocorrelation Score on d-dimensional trajectories
    
    Workflow:
    1. Simulate d-dimensional Langevin trajectories directly from full GMM
    2. Compute vector autocorrelation (whitened or trace-based)
    3. Return SSA score
    
    Args:
        samples: (N, d) tensor of data samples (for API compatibility, not used)
        gmm_means: (k, d) GMM means (required)
        gmm_covs: (k, d, d) GMM covariances (required)
        gmm_weights: (k,) GMM weights (required)
        n_trajectories: Number of Langevin trajectories
        T: Length of each trajectory
        max_lag: Maximum lag for autocorrelation
        dt: Langevin time step
        device: Computation device
        verbose: Print progress
        normalization: 'whitened' or 'trace' (default: 'whitened')
        
    Returns:
        SSA score (scalar)
    """
    device = device or torch.device('cpu')
    
    # If GMM parameters not provided, must be supplied
    if gmm_means is None:
        raise ValueError("GMM parameters (means, covs, weights) must be provided")
    
    if verbose:
        print(f"  Simulating {n_trajectories} d-dimensional trajectories (T={T})...")
    
    # Simulate d-dimensional trajectories directly from full GMM
    try:
        trajectories = simulate_d_trajectories(
            gmm_means,
            gmm_covs,
            gmm_weights,
            n_trajectories=n_trajectories,
            T=T,
            dt=dt,
            device=device
        )
        
        if verbose:
            print(f"  Trajectories shape: {trajectories.shape}")
        
        # Compute vector autocorrelation and SSA score
        S, _ = compute_vector_ssa_score(
            trajectories, max_lag,
            gmm_means=gmm_means,
            gmm_covs=gmm_covs,
            gmm_weights=gmm_weights,
            normalization=normalization
        )
        
        if verbose:
            print(f"  Full Autocorrelation (SSA): {S:.4f}")
        
        return S
        
    except Exception as e:
        warnings.warn(f"Full autocorrelation computation failed: {e}")
        raise


def test_sliced_autocorrelation():
    """Test sliced autocorrelation on a simple 2D GMM"""
    print("Testing Sliced Autocorrelation...")
    
    # Create a simple 2D bimodal GMM
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    covs = torch.stack([
        torch.eye(2) * 0.5,
        torch.eye(2) * 0.5
    ])
    weights = torch.tensor([0.5, 0.5])
    
    # Generate samples (not used in current implementation but required for API)
    samples = torch.randn(1000, 2)
    
    # Compute sliced autocorrelation
    ssa = sliced_autocorrelation(
        samples,
        gmm_means=means,
        gmm_covs=covs,
        gmm_weights=weights,
        n_projections_init=5,
        max_iterations=10,
        epsilon=1e-3,
        n_trajectories=50,
        T=500,
        max_lag=50,
        dt=0.01,
        projection_seed=42,
        verbose=True
    )
    
    print(f"Sliced SSA Score: {ssa:.4f}")
    assert ssa > 0, "SSA should be positive"
    
    print("✓ Sliced autocorrelation tests passed!")


if __name__ == "__main__":
    test_sliced_autocorrelation()

