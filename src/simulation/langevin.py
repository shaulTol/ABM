"""
Langevin Dynamics Simulator for d-dimensional Gaussian Mixtures

Implements Variance-Scaled Overdamped Langevin Diffusion:
    dX_t = (σ_f²/2) * ∇log f(X_t) dt + σ_f * dW_t
where:
    - f is the d-dimensional Gaussian Mixture (stationary distribution)
    - σ_f² = tr(Cov_f) / d is the scalar diffusion coefficient
    - σ_f = √(σ_f²)
    - dW_t is d-dimensional standard Brownian motion

Variance Scaling Convention:
    The diffusion coefficient is the scalar σ_f² = tr(Σ_f)/d, which averages
    the variance across all dimensions. This makes the construction scale- and
    rotation-invariant and avoids pathological behavior when Σ_f is anisotropic
    (e.g., between-mode variance inflating noise along the separation direction).

Discretization (Euler-Maruyama):
    X_{t+1} = X_t + (σ_f²/2) * ∇log f(X_t) * Δt + σ_f * √Δt * Z_t
    where Z_t ~ N(0, I_d) and σ_f² = tr(Cov_f)/d.
"""
import torch
import numpy as np
from typing import Optional


class LangevinSimulatorD:
    """
    d-Dimensional Overdamped Langevin Dynamics Simulator for Gaussian Mixtures
    """
    
    def __init__(self,
                 means: torch.Tensor,
                 covs: torch.Tensor,
                 weights: torch.Tensor,
                 dt: float = 0.01,
                 device: Optional[torch.device] = None,
                 regularization: float = 1e-6):
        """
        Initialize simulator with d-dimensional Gaussian Mixture parameters
        
        Uses variance-scaled diffusion convention:
            dX_t = (σ_f²/2) * ∇log f(X_t) dt + σ_f * dW_t
        where σ_f² = tr(Cov_f) / d is the average variance of f across dimensions.

        Args:
            means: (k, d) tensor of component means
            covs: (k, d, d) tensor of component covariance matrices
            weights: (k,) tensor of mixture weights (normalized)
            dt: Time step size
            device: Computation device
            regularization: Small value added to covariance matrices for numerical stability
        """
        self.device = device or torch.device('cpu')
        
        # Convert to tensors on device
        self.means = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        self.covs = torch.as_tensor(covs, dtype=torch.float32, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        # Number of components and dimension
        self.k = len(self.weights)
        self.d = self.means.shape[1]
        
        # Add regularization to covariance matrices for numerical stability
        self.covs_reg = self.covs + regularization * torch.eye(self.d, device=self.device).unsqueeze(0)
        
        # Precompute inverse covariance matrices (precision matrices)
        # Use Cholesky decomposition for numerical stability
        try:
            L = torch.linalg.cholesky(self.covs_reg)  # (k, d, d) lower triangular
            self.precisions = torch.cholesky_solve(
                torch.eye(self.d, device=self.device).unsqueeze(0).expand(self.k, -1, -1),
                L
            )  # (k, d, d)
        except RuntimeError:
            # Fallback to direct inverse if Cholesky fails
            self.precisions = torch.linalg.inv(self.covs_reg)
        
        # Time step
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)
        
        # Compute covariance matrix of the stationary distribution f
        # For mixture: Cov_f(X) = Σᵢ πᵢ [Σᵢ + (μᵢ - μ̄)(μᵢ - μ̄)ᵀ]
        # where μ̄ = Σᵢ πᵢ μᵢ
        mu_bar = (self.weights.unsqueeze(1) * self.means).sum(dim=0)  # (d,)
        
        # Compute full covariance matrix of mixture
        self.cov_f = torch.zeros(self.d, self.d, device=self.device)
        for i in range(self.k):
            diff = self.means[i] - mu_bar  # (d,)
            outer = torch.outer(diff, diff)  # (d, d)
            self.cov_f += self.weights[i] * (self.covs[i] + outer)
        
        # Scalar diffusion coefficient: σ_f² = tr(Cov_f) / d
        self.var_f = torch.trace(self.cov_f).item() / self.d  # Scalar
        self.sigma_f = np.sqrt(self.var_f)  # Scalar σ_f

        # Precompute log constants for vectorized pdf/grad_log_pdf
        self.log_dets = torch.logdet(self.covs_reg)  # (k,)
        self.log_norm = -0.5 * (self.d * np.log(2 * np.pi) + self.log_dets)  # (k,)
        self.log_weights = torch.log(self.weights.clamp_min(1e-30))  # (k,)
    
    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the mixture PDF at points x

        Args:
            x: (..., d) tensor of evaluation points

        Returns:
            (..., ) tensor of PDF values
        """
        x_shape = x.shape[:-1]  # (...)
        x_flat = x.reshape(-1, self.d)  # (n_points, d)

        # All components at once — no Python loop
        # diff: (k, n_points, d)
        diff = x_flat.unsqueeze(0) - self.means.unsqueeze(1)

        # Quadratic form via batched matmul: (k, n_points, d)
        prec_diff = torch.bmm(diff, self.precisions)
        quad = (diff * prec_diff).sum(dim=-1)  # (k, n_points)

        # Log component densities
        log_component_pdf = self.log_weights[:, None] + self.log_norm[:, None] - 0.5 * quad  # (k, n_points)

        # Sum in log-space via logsumexp, then exponentiate
        log_pdf = torch.logsumexp(log_component_pdf, dim=0)  # (n_points,)
        pdf_total = torch.exp(log_pdf)

        return pdf_total.reshape(x_shape)
    
    def grad_log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of log PDF: ∇log f(x)

        For d-dimensional GMM:
            ∇log f(x) = Σᵢ wᵢ(x) · Σᵢ⁻¹(μᵢ - x)
        where wᵢ(x) = πᵢ Nᵢ(x) / f(x) are the responsibility weights.

        Args:
            x: (..., d) tensor of evaluation points

        Returns:
            (..., d) tensor of gradient vectors
        """
        x_shape = x.shape[:-1]  # (...)
        x_flat = x.reshape(-1, self.d)  # (n_points, d)

        # All components at once — no Python loop
        # diff: (k, n_points, d)
        diff = x_flat.unsqueeze(0) - self.means.unsqueeze(1)

        # Quadratic form via batched matmul: prec_diff = diff @ Σ⁻¹
        prec_diff = torch.bmm(diff, self.precisions)  # (k, n_points, d)
        quad = (diff * prec_diff).sum(dim=-1)  # (k, n_points)

        # Log component densities
        log_component_pdf = self.log_weights[:, None] + self.log_norm[:, None] - 0.5 * quad  # (k, n_points)

        # Responsibilities via softmax (numerically stable)
        responsibilities = torch.softmax(log_component_pdf, dim=0)  # (k, n_points)

        # Gradient: Σ_i resp_i * Σ_i⁻¹(μ_i - x) = -Σ_i resp_i * prec_diff_i
        grad = -(responsibilities.unsqueeze(-1) * prec_diff).sum(dim=0)  # (n_points, d)

        return grad.reshape(*x_shape, self.d)
    
    def sample_initial(self, n: int) -> torch.Tensor:
        """
        Sample initial positions from the stationary distribution
        
        Args:
            n: Number of samples
            
        Returns:
            (n, d) tensor of initial positions
        """
        # Sample component labels
        labels = torch.multinomial(self.weights, n, replacement=True)
        
        # Sample from corresponding multivariate Gaussians
        x0 = torch.zeros(n, self.d, device=self.device)
        for i in range(self.k):
            mask = (labels == i)
            n_i = mask.sum().item()
            if n_i > 0:
                # Sample from N(μᵢ, Σᵢ) using Cholesky decomposition
                L = torch.linalg.cholesky(self.covs_reg[i])
                z = torch.randn(n_i, self.d, device=self.device)
                x0[mask] = torch.matmul(z, L.t()) + self.means[i].unsqueeze(0)
        
        return x0
    
    def step(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform one Langevin step with scalar variance-scaled diffusion

        Implements the SDE discretization:
            X_{t+Δt} = X_t + (σ_f²/2) * ∇log f(X_t) * Δt + σ_f * √Δt * Z
        where Z ~ N(0, I_d) and σ_f² = tr(Cov_f)/d.

        NOTE: Currently the process is defined for a specific f (not translation-invariant).
        For any f, if we translate f by a constant, we get a different process.
        In the future, we might want to center the process (e.g., by subtracting μ_f in the
        drift term) so that translating f results in the same process (translation-invariance).

        Currently, we de facto center the process post-hoc when computing autocorrelation:
        we compute autocorrelation of Y = X - μ_f (the centered trajectories) rather than X.
        This centering happens in `compute_vector_autocorrelation` in `src/metrics/autocorrelation.py`
        (see line ~178: `centered = trajectories - mu_f`).

        Args:
            x: Current positions (..., d)
            z: Optional pre-sampled noise (n_points, d). If None, samples fresh noise.
               Providing pre-sampled noise enables differentiability via reparameterization.

        Returns:
            Next positions (..., d)
        """
        # Reshape for matrix operations
        x_shape = x.shape[:-1]  # (...)
        x_flat = x.reshape(-1, self.d)  # (n_points, d)
        n_points = x_flat.shape[0]

        # Drift term: (σ_f²/2) * ∇log f(x) * dt
        grad = self.grad_log_pdf(x)  # (..., d)
        grad_flat = grad.reshape(-1, self.d)  # (n_points, d)

        # Scalar diffusion: σ_f² * grad (no matrix multiply needed)
        drift_flat = 0.5 * self.var_f * self.dt * grad_flat  # (n_points, d)
        drift = drift_flat.reshape(*x_shape, self.d)

        # Diffusion term: σ_f * √Δt * Z, where Z ~ N(0, I_d)
        if z is None:
            z = torch.randn(n_points, self.d, device=self.device)  # (n_points, d)
        noise_flat = self.sigma_f * self.sqrt_dt * z  # (n_points, d)
        noise = noise_flat.reshape(*x_shape, self.d)

        return x + drift + noise
    
    def simulate(self, x0: torch.Tensor, T: int, 
                 presampled_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate trajectories from initial positions
        
        Args:
            x0: (n, d) tensor of initial positions
            T: Number of time steps
            presampled_noise: Optional (T, n, d) tensor of pre-sampled standard normal noise.
                              If provided, enables differentiability via reparameterization trick.
                              If None, samples fresh noise at each step (not differentiable).
            
        Returns:
            (n, T+1, d) tensor of trajectories (including t=0)
        """
        n = x0.shape[0]
        trajectory = torch.zeros(n, T + 1, self.d, device=self.device)
        trajectory[:, 0, :] = x0
        
        x = x0
        for t in range(T):
            z = presampled_noise[t] if presampled_noise is not None else None
            x = self.step(x, z=z)
            trajectory[:, t + 1, :] = x
        
        return trajectory
    
    def simulate_differentiable(self, x0: torch.Tensor, T: int, 
                                 seed: Optional[int] = None) -> torch.Tensor:
        """
        Simulate trajectories with differentiability support.
        
        Pre-samples all noise before simulation, enabling gradient flow through
        the entire trajectory via the reparameterization trick.
        
        Args:
            x0: (n, d) tensor of initial positions
            T: Number of time steps
            seed: Optional random seed for noise generation
            
        Returns:
            (n, T+1, d) tensor of trajectories (including t=0)
        """
        n = x0.shape[0]
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Pre-sample all noise for reparameterization trick
        presampled_noise = torch.randn(T, n, self.d, device=self.device)
        
        return self.simulate(x0, T, presampled_noise=presampled_noise)


def simulate_d_trajectories(
    means: torch.Tensor,
    covs: torch.Tensor,
    weights: torch.Tensor,
    n_trajectories: int = 100,
    T: int = 1000,
    dt: float = 0.01,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Convenience function to simulate multiple d-dimensional trajectories
    
    Args:
        means: (k, d) GMM component means
        covs: (k, d, d) GMM component covariances
        weights: (k,) GMM mixture weights
        n_trajectories: Number of trajectories
        T: Length of each trajectory
        dt: Time step
        device: Computation device
        seed: Random seed for reproducibility
        
    Returns:
        (n_trajectories, T+1, d) tensor of position trajectories
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create simulator
    sim = LangevinSimulatorD(means, covs, weights, dt, device)
    
    # Sample initial conditions
    x0 = sim.sample_initial(n_trajectories)
    
    # Simulate
    trajectories = sim.simulate(x0, T)
    
    return trajectories
