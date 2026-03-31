"""
Autocorrelation Metrics for Time Series Analysis

Computes the autocorrelation function and Sum of Squared Autocorrelations (SSA) score
from trajectory data.
"""
import torch
import numpy as np
from typing import Tuple, Optional


def compute_stationary_moments(
    gmm_means: torch.Tensor,
    gmm_covs: torch.Tensor,
    gmm_weights: torch.Tensor,
    regularization: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute theoretical mean and covariance of stationary distribution f (GMM)
    
    For GMM: f(x) = Σ_i π_i N(x; μ_i, Σ_i)
    
    Args:
        gmm_means: (k, d) component means
        gmm_covs: (k, d, d) component covariance matrices
        gmm_weights: (k,) mixture weights
        regularization: Small value for numerical stability
        
    Returns:
        mu_f: (d,) mean vector
        cov_f: (d, d) covariance matrix (with regularization added)
    """
    device = gmm_means.device
    d = gmm_means.shape[1]
    
    # Mean: μ_f = Σ_i π_i μ_i
    mu_f = (gmm_weights.unsqueeze(-1) * gmm_means).sum(dim=0)  # (d,)
    
    # Covariance: Σ_f = Σ_i π_i [Σ_i + (μ_i - μ_f)(μ_i - μ_f)^T]
    cov_f = torch.zeros(d, d, device=device)
    for i in range(len(gmm_weights)):
        diff = gmm_means[i] - mu_f  # (d,)
        outer = torch.outer(diff, diff)  # (d, d)
        cov_f += gmm_weights[i] * (gmm_covs[i] + outer)
    
    # Add regularization for numerical stability
    cov_f = cov_f + regularization * torch.eye(d, device=device)
    
    return mu_f, cov_f


def compute_autocorrelation(
    trajectories: torch.Tensor,
    max_lag: int,
    demean: bool = True
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute empirical autocorrelation function from trajectories
    Uses per-trajectory mean and variance normalization.
    
    Args:
        trajectories: (N, T+1) tensor of position trajectories
        max_lag: Maximum lag L to compute
        demean: Whether to subtract the mean (standard practice)
        
    Returns:
        rho: (max_lag+1,) tensor of autocorrelations [ρ(0), ρ(1), ..., ρ(L)]
        mu: Empirical mean (averaged across trajectories)
        sigma: Empirical standard deviation (averaged across trajectories)
    """
    N, T_plus_1 = trajectories.shape
    T = T_plus_1 - 1
    eps = 1e-12
    
    # Compute per-trajectory mean: mu_n = (N, 1)
    mu_n = trajectories.mean(dim=1, keepdim=True)  # (N, 1)
    
    # Center per trajectory
    if demean:
        centered = trajectories - mu_n  # (N, T+1)
    else:
        centered = trajectories
    
    # Compute per-trajectory variance: var_n = (N,)
    var_n = centered.var(dim=1, unbiased=False)  # (N,)
    
    # Compute autocorrelations
    rho = torch.zeros(max_lag + 1, device=trajectories.device)
    
    for lag in range(max_lag + 1):
        if lag > T:
            # Not enough data for this lag
            rho[lag] = 0.0
            continue
        
        # Extract pairs (X_t, X_{t+lag})
        x_t = centered[:, :T + 1 - lag]  # (N, T+1-lag)
        x_h = centered[:, lag:T + 1]  # (N, T+1-lag)
        
        # Compute per-trajectory covariance: cov_n = (N,)
        cov_n = (x_t * x_h).mean(dim=1)  # (N,)
        
        # Normalize per trajectory and average: rho[lag] = mean(cov_n / (var_n + eps))
        rho[lag] = (cov_n / (var_n + eps)).mean()
    
    # Return global mean and std for backward compatibility
    mu = mu_n.mean().item()
    sigma = torch.sqrt(var_n.mean()).item()
    
    return rho, mu, sigma


def compute_vector_autocorrelation(
    trajectories: torch.Tensor,
    max_lag: int,
    gmm_means: torch.Tensor,
    gmm_covs: torch.Tensor,
    gmm_weights: torch.Tensor,
    demean: bool = True,
    normalization: str = 'whitened'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute autocorrelation function from d-dimensional trajectories
    
    Two normalization methods available:
    
    **'whitened' (default)**: Uses whitening with standard autocorrelation
    1. Center by global mean (stationary mean μ_f): Y_{n,t} = X_{n,t} - μ_f
    2. Whiten: Z_{n,t} = Σ_f^{-1/2} Y_{n,t}
    3. Standard autocorrelation: 
       - C(h) = (1/(N(T+1-h))) Σ_{n,t} z_{n,t} · z_{n,t+h} (global autocovariance at lag h)
       - C(0) = (1/(N(T+1))) Σ_{n,t} ||z_{n,t}||^2 (global variance, autocovariance at lag 0)
       - ρ(h) = C(h) / C(0) (standard normalization)
    
    **'trace'**: Uses trace-based normalization
    1. Center by global mean (stationary mean μ_f): Y_{n,t} = X_{n,t} - μ_f
    2. Compute autocovariance matrices: C(τ) = (1/(N(T+1-τ))) Σ_{n,t} Y_{n,t} Y_{n,t+τ}^T
    3. Trace-based autocorrelation:
       - tr_C(0) = tr(C(0)) = E[||Y_t||^2]
       - tr_C(τ) = tr(C(τ)) = E[Y_t · Y_{t+τ}]
       - ρ(τ) = tr_C(τ) / tr_C(0)
    
    Args:
        trajectories: (N, T+1, d) tensor of d-dimensional position trajectories
        max_lag: Maximum lag H to compute
        gmm_means: (k, d) GMM component means (required)
        gmm_covs: (k, d, d) GMM component covariance matrices (required)
        gmm_weights: (k,) GMM weights (required)
        demean: Whether to subtract the mean (standard practice)
        normalization: 'whitened' or 'trace' (default: 'whitened')
        
    Returns:
        rho: (max_lag+1,) tensor of autocorrelations [ρ(0), ρ(1), ..., ρ(H)]
        mu_f: (d,) mean vector of stationary distribution
        cov_f: (d, d) covariance matrix of stationary distribution
    """
    N, T_plus_1, d = trajectories.shape
    T = T_plus_1 - 1
    eps = 1e-12
    
    # Validate normalization method
    if normalization not in ['whitened', 'trace']:
        raise ValueError(f"normalization must be 'whitened' or 'trace', got '{normalization}'")
    
    # Validate GMM parameters
    if gmm_means is None or gmm_covs is None or gmm_weights is None:
        raise ValueError("gmm_means, gmm_covs, and gmm_weights are required")
    
    # Compute stationary moments from GMM (analytically, not from trajectories)
    mu_f, cov_f = compute_stationary_moments(gmm_means, gmm_covs, gmm_weights)
    
    # Ensure mu_f matches trajectories dtype and device to avoid dtype/device mismatch
    mu_f = mu_f.to(dtype=trajectories.dtype, device=trajectories.device)
    cov_f = cov_f.to(dtype=trajectories.dtype, device=trajectories.device)
    
    # Center by global mean (stationary mean μ_f)
    if demean:
        centered = trajectories - mu_f.unsqueeze(0).unsqueeze(0)  # (N, T+1, d)
    else:
        centered = trajectories
    
    # Compute autocorrelations
    rho = torch.zeros(max_lag + 1, device=trajectories.device)
    
    if normalization == 'whitened':
        # WHITENED METHOD: Use whitening with standard autocorrelation
        # Compute Σ_f^{-1/2} via eigendecomposition (once, outside loop)
        # Σ_f = Q Λ Q^T, so Σ_f^{-1/2} = Q Λ^{-1/2} Q^T
        eigenvals, eigenvecs = torch.linalg.eigh(cov_f)  # eigenvals: (d,), eigenvecs: (d, d)
        eigenvals_inv_sqrt = 1.0 / torch.sqrt(eigenvals.clamp_min(1e-12))  # (d,)
        cov_f_inv_sqrt = torch.matmul(
            eigenvecs * eigenvals_inv_sqrt.unsqueeze(0),
            eigenvecs.t()
        )  # (d, d)
        
        # Ensure dtype matches trajectories to avoid dtype mismatch in matrix multiplication
        cov_f_inv_sqrt = cov_f_inv_sqrt.to(dtype=trajectories.dtype)
        
        # Pre-whiten all trajectories: Z = Σ_f^{-1/2} Y (once, outside loop)
        # This avoids recomputing whitening for each lag
        # centered: (N, T+1, d), cov_f_inv_sqrt: (d, d)
        centered_flat = centered.reshape(-1, d)  # (N*(T+1), d)
        whitened_flat = torch.matmul(centered_flat, cov_f_inv_sqrt.t())  # (N*(T+1), d)
        whitened = whitened_flat.reshape(N, T_plus_1, d)  # (N, T+1, d)
        
        # Compute global variance (autocovariance at lag 0): C(0) = (1/(N(T+1))) Σ_{n,t} ||z_{n,t}||^2
        whitened_flat_all = whitened.reshape(-1, d)  # (N*(T+1), d) # alternative
        C_0 = (whitened_flat_all.pow(2).sum(dim=-1)).mean()  # Scalar: global variance  # ANOTHER OPTION COULD BE TO DO THAT BY SAMPLING FROM f RATHER THAN THE TRAJECTORIES. 

        # Alternative
        '''
        n_samples_c0 = 10000  # Number of samples for better approximation
        labels = torch.multinomial(gmm_weights, n_samples_c0, replacement=True)
        samples_f = torch.zeros(n_samples_c0, d, device=trajectories.device, dtype=trajectories.dtype)
        for i in range(len(gmm_weights)):
            mask = (labels == i)
            n_i = mask.sum().item()
            if n_i > 0:
                # Sample from N(μ_i, Σ_i) using Cholesky
                L_i = torch.linalg.cholesky(gmm_covs[i] + 1e-6 * torch.eye(d, device=trajectories.device, dtype=trajectories.dtype))
                eps = torch.randn(n_i, d, device=trajectories.device, dtype=trajectories.dtype)
                samples_f[mask] = gmm_means[i] + eps @ L_i.t()

        # Center and whiten: Z = Σ_f^{-1/2}(X - μ_f)
        samples_f_centered = samples_f - mu_f.unsqueeze(0)  # (n_samples_c0, d)
        samples_f_whitened = torch.matmul(samples_f_centered, cov_f_inv_sqrt.t())  # (n_samples_c0, d)

        # C(0) = E[||Z||^2] from samples of f
        C_0 = (samples_f_whitened.pow(2).sum(dim=-1)).mean()  # Scalar: global variance from f
        '''
            
        
        for lag in range(max_lag + 1):
            if lag > T:
                # Not enough data for this lag
                rho[lag] = 0.0
                continue
            
            # Extract pairs from pre-whitened trajectories
            z_t = whitened[:, :T + 1 - lag, :]  # (N, T+1-lag, d)
            z_h = whitened[:, lag:T + 1, :]  # (N, T+1-lag, d)
            
            # Compute global autocovariance: C(h) = (1/(N(T+1-h))) Σ_{n,t} z_{n,t} · z_{n,t+h}
            # Flatten to compute dot products
            z_t_flat = z_t.reshape(-1, d)  # (N*(T+1-lag), d)
            z_h_flat = z_h.reshape(-1, d)  # (N*(T+1-lag), d)
            
            # Dot product: z_{n,t} · z_{n,t+h} for all pairs
            dot_products = (z_t_flat * z_h_flat).sum(dim=-1)  # (N*(T+1-lag),)
            
            # Global autocovariance: mean over all pairs
            C_h = dot_products.mean()  # Scalar
            
            # Standard autocorrelation: ρ(h) = C(h) / C(0)
            rho[lag] = C_h / (C_0 + eps)
    
    elif normalization == 'trace':
        # TRACE METHOD: Use trace-based normalization
        # Compute autocovariance matrix at lag 0: C(0) = (1/(N(T+1))) Σ_{n,t} Y_{n,t} Y_{n,t}^T
        centered_flat_all = centered.reshape(-1, d)  # (N*(T+1), d)
        n_total = float(N * (T + 1))  # Ensure it's a Python float for division
        C_0_matrix = torch.matmul(centered_flat_all.t(), centered_flat_all) / n_total  # (d, d)
        tr_C_0 = torch.trace(C_0_matrix)  # Scalar: tr(C(0))
        
        for lag in range(max_lag + 1):
            if lag > T:
                # Not enough data for this lag
                rho[lag] = 0.0
                continue
            
            # Extract pairs from centered trajectories
            y_t = centered[:, :T + 1 - lag, :]  # (N, T+1-lag, d)
            y_h = centered[:, lag:T + 1, :]  # (N, T+1-lag, d)
            
            # Compute autocovariance matrix: C(τ) = (1/(N(T+1-τ))) Σ_{n,t} Y_{n,t} Y_{n,t+τ}^T
            # Flatten to compute outer products
            y_t_flat = y_t.reshape(-1, d)  # (N*(T+1-lag), d)
            y_h_flat = y_h.reshape(-1, d)  # (N*(T+1-lag), d)
            
            # Matrix multiplication: Y_t^T @ Y_h gives (d, d) autocovariance matrix
            n_total_lag = float(N * (T + 1 - lag))  # Ensure it's a Python float for division
            C_tau_matrix = torch.matmul(y_t_flat.t(), y_h_flat) / n_total_lag  # (d, d)
            
            # Trace-based autocorrelation: ρ(τ) = tr(C(τ)) / tr(C(0))
            tr_C_tau = torch.trace(C_tau_matrix)  # Scalar: tr(C(τ))
            rho[lag] = tr_C_tau / (tr_C_0 + eps)
    
    return rho, mu_f, cov_f


def compute_ssa_score(
    trajectories: torch.Tensor,
    max_lag: int,
    exclude_zero_lag: bool = True
) -> Tuple[float, torch.Tensor]:
    """
    Compute Sum of Squared Autocorrelations (SSA) score, normalized by number of lags.

    S = (1/L) * Σ_{ℓ=1}^{L} ρ(ℓ)²

    This is the mean squared autocorrelation, making the score comparable
    across different max_lag settings.

    Args:
        trajectories: (N, T+1) tensor of position trajectories
        max_lag: Maximum lag L
        exclude_zero_lag: Whether to exclude ρ(0) from the sum (standard)

    Returns:
        S: SSA score (scalar), normalized by number of lags
        rho: (max_lag+1,) autocorrelation function
    """
    # Compute autocorrelation function
    rho, mu, sigma = compute_autocorrelation(trajectories, max_lag, demean=True)
    
    # Compute SSA score (normalized by number of lags)
    if exclude_zero_lag:
        n_lags = max_lag  # lags 1 to L
        S = (rho[1:] ** 2).sum().item() / n_lags
    else:
        n_lags = max_lag + 1  # lags 0 to L
        S = (rho ** 2).sum().item() / n_lags
    
    return S, rho


def compute_vector_ssa_score(
    trajectories: torch.Tensor,
    max_lag: int,
    gmm_means: torch.Tensor,
    gmm_covs: torch.Tensor,
    gmm_weights: torch.Tensor,
    exclude_zero_lag: bool = True,
    normalization: str = 'whitened'
) -> Tuple[float, torch.Tensor]:
    """
    Compute Sum of Squared Autocorrelations (SSA) score for vector trajectories,
    normalized by number of lags.

    S = (1/L) * Σ_{ℓ=1}^{L} ρ(ℓ)²

    This is the mean squared autocorrelation, making the score comparable
    across different max_lag settings.

    Args:
        trajectories: (N, T+1, d) tensor of d-dimensional position trajectories
        max_lag: Maximum lag L
        gmm_means: (k, d) GMM component means (required)
        gmm_covs: (k, d, d) GMM component covariance matrices (required)
        gmm_weights: (k,) GMM weights (required)
        exclude_zero_lag: Whether to exclude ρ(0) from the sum (standard)
        normalization: 'whitened' or 'trace' (default: 'whitened')

    Returns:
        S: SSA score (scalar), normalized by number of lags
        rho: (max_lag+1,) autocorrelation function
    """
    # Compute autocorrelation function
    rho, mu_f, cov_f = compute_vector_autocorrelation(
        trajectories, max_lag, gmm_means, gmm_covs, gmm_weights, 
        demean=True, normalization=normalization
    )
    
    # Compute SSA score (normalized by number of lags)
    if exclude_zero_lag:
        n_lags = max_lag  # lags 1 to L
        S = (rho[1:] ** 2).sum().item() / n_lags
    else:
        n_lags = max_lag + 1  # lags 0 to L
        S = (rho ** 2).sum().item() / n_lags
    
    return S, rho


def analyze_trajectories(
    trajectories: torch.Tensor,
    max_lag: int = 100,
    verbose: bool = False
) -> dict:
    """
    Comprehensive analysis of trajectory autocorrelation
    
    Args:
        trajectories: (N, T+1) tensor
        max_lag: Maximum lag to analyze
        verbose: Print results
        
    Returns:
        Dictionary with analysis results
    """
    N, T_plus_1 = trajectories.shape
    T = T_plus_1 - 1
    
    # Compute autocorrelation
    rho, mu, sigma = compute_autocorrelation(trajectories, max_lag)
    
    # Compute SSA score
    S, _ = compute_ssa_score(trajectories, max_lag)
    
    # Estimate correlation length (integral of ACF)
    # L_corr ≈ Σ ρ(ℓ) for ℓ > 0
    L_corr = rho[1:].sum().item()
    
    # Find decorrelation time (first zero crossing or 1/e point)
    decorr_time = None
    threshold = 1.0 / np.e
    for lag in range(1, len(rho)):
        if abs(rho[lag].item()) < threshold:
            decorr_time = lag
            break
    
    results = {
        'N_trajectories': N,
        'T_length': T,
        'mean': mu,
        'std': sigma,
        'ssa_score': S,
        'correlation_length': L_corr,
        'decorr_time': decorr_time,
        'rho': rho.cpu().numpy(),
        'max_lag': max_lag
    }
    
    if verbose:
        print(f"Trajectory Analysis:")
        print(f"  N = {N}, T = {T}")
        print(f"  Mean: {mu:.4f}, Std: {sigma:.4f}")
        print(f"  SSA Score: {S:.4f}")
        print(f"  Correlation Length: {L_corr:.2f}")
        print(f"  Decorrelation Time: {decorr_time if decorr_time else '>'+str(max_lag)}")
    
    return results


def test_autocorrelation():
    """Test autocorrelation computation"""
    print("Testing Autocorrelation Metrics...")
    
    # Create synthetic trajectories with known autocorrelation
    # AR(1) process: X_t = α X_{t-1} + ε_t
    # Theoretical: ρ(ℓ) = α^ℓ
    alpha = 0.8
    N = 100
    T = 500
    
    torch.manual_seed(42)
    trajectories = torch.zeros(N, T + 1)
    
    for n in range(N):
        x = 0.0
        for t in range(T + 1):
            trajectories[n, t] = x
            x = alpha * x + torch.randn(1).item()
    
    # Compute autocorrelation
    max_lag = 20
    rho, mu, sigma = compute_autocorrelation(trajectories, max_lag)
    
    print(f"Empirical ρ(1) = {rho[1]:.3f}, Theoretical = {alpha:.3f}")
    assert abs(rho[1] - alpha) < 0.1, "ρ(1) should be close to α"
    
    # Compute SSA score
    S, _ = compute_ssa_score(trajectories, max_lag)
    
    # Theoretical SSA for AR(1): Σ α^{2ℓ} = α²/(1-α²)
    theoretical_S = sum(alpha**(2*lag) for lag in range(1, max_lag + 1))
    print(f"Empirical S = {S:.3f}, Theoretical ≈ {theoretical_S:.3f}")
    
    # Should be roughly close
    assert abs(S - theoretical_S) < theoretical_S * 0.3, "SSA should be reasonably close"
    
    print("✓ Autocorrelation tests passed!")


if __name__ == "__main__":
    test_autocorrelation()

