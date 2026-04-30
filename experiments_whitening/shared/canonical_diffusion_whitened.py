"""
Thin wrapper: run canonical Langevin diffusion in whitened coordinates.

Takes a WhiteningTransform + original score function + samples,
creates a GenericLangevinSimulator in whitened coordinates with var_f ≈ 1.

Two modes:
  1. GMM (analytic): Pass GMM parameters → whitens score analytically.
  2. Generic (empirical): Pass score_fn + samples → whitens both.
"""
import sys, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import torch
from src.simulation.generic_langevin import GenericLangevinSimulator
from experiments_whitening.shared.whitening import WhiteningTransform


def make_whitened_simulator(
    wt: WhiteningTransform,
    score_fn,
    samples: np.ndarray,
    dt: float = 0.01,
    device=None,
):
    """
    Create a GenericLangevinSimulator in whitened coordinates.

    Args:
        wt: WhiteningTransform (already fitted).
        score_fn: Original score function in original coordinates.
                  Callable: (N, d) torch.Tensor -> (N, d) torch.Tensor.
        samples: (M, d) original-space samples for initial conditions.
        dt: Time step for Euler-Maruyama.
        device: Torch device.

    Returns:
        GenericLangevinSimulator operating in whitened space.
    """
    device = device or torch.device('cpu')

    # Whiten the samples for initial conditions
    samples_tilde = wt.whiten(samples)
    samples_tilde_t = torch.tensor(samples_tilde, dtype=torch.float32, device=device)

    # Transform the score function to whitened coordinates
    score_tilde = wt.transform_score_torch(score_fn)

    # Empirical var_f in whitened space (should be ≈ 1)
    var_f_tilde = float(np.var(samples_tilde, axis=0).mean())

    # Mean in whitened space (should be ≈ 0)
    mu_tilde = torch.tensor(samples_tilde.mean(axis=0), dtype=torch.float32, device=device)

    sim = GenericLangevinSimulator(
        score_fn=score_tilde,
        samples=samples_tilde_t,
        dt=dt,
        device=device,
        mu_f=mu_tilde,
        var_f=var_f_tilde,
    )
    return sim


def make_whitened_gmm_simulator(
    wt: WhiteningTransform,
    gmm_means: torch.Tensor,
    gmm_covs: torch.Tensor,
    gmm_weights: torch.Tensor,
    dt: float = 0.01,
    device=None,
    n_samples: int = 2000,
    seed: int = 0,
):
    """
    Create a whitened Langevin simulator from GMM parameters.

    Uses the GMM's analytic score (via LangevinSimulatorD.grad_log_pdf)
    as the original score function, then whitens.

    Args:
        wt: WhiteningTransform (fitted from analytic Sigma_f).
        gmm_means, gmm_covs, gmm_weights: GMM parameters.
        dt: Time step.
        device: Torch device.
        n_samples: Number of samples for initial conditions.
        seed: Random seed for sampling.

    Returns:
        GenericLangevinSimulator in whitened coordinates.
    """
    from src.simulation.langevin import LangevinSimulatorD

    device = device or torch.device('cpu')

    # Build original-space GMM simulator for its score function
    sim_orig = LangevinSimulatorD(gmm_means, gmm_covs, gmm_weights, dt=dt, device=device)

    # Sample initial conditions from the GMM
    torch.manual_seed(seed)
    x0 = sim_orig.sample_initial(n_samples)
    samples_np = x0.cpu().numpy()

    return make_whitened_simulator(
        wt=wt,
        score_fn=sim_orig.grad_log_pdf,
        samples=samples_np,
        dt=dt,
        device=device,
    )
