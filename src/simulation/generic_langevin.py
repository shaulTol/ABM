"""
Generic Langevin Dynamics Simulator

Accepts an arbitrary score function ∇log f(x) instead of requiring GMM parameters.
This enables DA direction extraction and SSA computation from any density estimate
(KDE, learned score networks, etc.) — not just analytical GMMs.

The canonical diffusion is:
    dX_t = (σ_f²/2) · ∇log f(X_t) dt + σ_f · dW_t

where σ_f² = tr(Cov_f)/d is the scalar diffusion coefficient.
"""
import torch
import numpy as np
from typing import Callable, Optional, Union


class GenericLangevinSimulator:
    """
    Overdamped Langevin simulator driven by an arbitrary score function.

    Unlike LangevinSimulatorD which requires GMM parameters, this class accepts:
    - A callable score function: x → ∇log f(x)
    - Static samples from f (for initial conditions and moment estimation)
    - Optionally, pre-computed stationary moments (mu_f, cov_f, var_f)

    If moments are not provided, they are estimated empirically from the samples.
    """

    def __init__(
        self,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        samples: torch.Tensor,
        dt: float = 0.01,
        device: Optional[torch.device] = None,
        mu_f: Optional[torch.Tensor] = None,
        cov_f: Optional[torch.Tensor] = None,
        var_f: Optional[float] = None,
    ):
        """
        Args:
            score_fn: Callable mapping (N, d) → (N, d), returning ∇log f(x).
                      Must handle batched input.
            samples: (M, d) static samples from the stationary distribution f.
                     Used for initial conditions (resampled) and moment estimation.
            dt: Euler-Maruyama time step.
            device: Computation device.
            mu_f: (d,) pre-computed stationary mean. If None, estimated from samples.
            cov_f: (d, d) pre-computed stationary covariance. If None, estimated from samples.
            var_f: Pre-computed scalar diffusion coefficient σ_f² = tr(Cov_f)/d.
                   If None, computed from cov_f.
        """
        self.device = device or torch.device('cpu')
        self.score_fn = score_fn
        self.samples = samples.to(self.device)
        self.d = samples.shape[1]
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)

        # Stationary moments — empirical if not provided
        if mu_f is not None:
            self.mu_f = mu_f.to(self.device)
        else:
            self.mu_f = self.samples.mean(dim=0)

        if var_f is not None:
            # Skip cov_f computation entirely — critical for high-D (e.g. 196k pixels)
            self.var_f = var_f
            self.cov_f = cov_f.to(self.device) if cov_f is not None else None
        elif cov_f is not None:
            self.cov_f = cov_f.to(self.device)
            self.var_f = torch.trace(self.cov_f).item() / self.d
        else:
            centered = self.samples - self.mu_f.unsqueeze(0)
            self.cov_f = (centered.t() @ centered) / (self.samples.shape[0] - 1)
            self.var_f = torch.trace(self.cov_f).item() / self.d

        self.sigma_f = np.sqrt(self.var_f)

    def grad_log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ∇log f(x) via the provided score function."""
        return self.score_fn(x)

    def sample_initial(self, n: int) -> torch.Tensor:
        """
        Sample initial conditions by resampling (with replacement) from the
        stored samples. This is the nonparametric analogue of sampling from f.
        """
        idx = torch.randint(0, self.samples.shape[0], (n,), device=self.device)
        return self.samples[idx].clone()

    def step(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One Euler-Maruyama step:
            X_{t+dt} = X_t + (σ_f²/2) · ∇log f(X_t) · dt + σ_f · √dt · Z

        Args:
            x: (..., d) current positions
            z: Optional (N, d) pre-sampled standard normal noise.
        """
        grad = self.score_fn(x)  # (..., d)
        drift = 0.5 * self.var_f * self.dt * grad
        if z is None:
            z = torch.randn_like(x)
        noise = self.sigma_f * self.sqrt_dt * z
        return x + drift + noise

    def simulate(self, x0: torch.Tensor, T: int,
                 presampled_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate trajectories.

        Args:
            x0: (n, d) initial positions
            T: Number of time steps
            presampled_noise: Optional (T, n, d) noise tensor

        Returns:
            (n, T+1, d) trajectories
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
