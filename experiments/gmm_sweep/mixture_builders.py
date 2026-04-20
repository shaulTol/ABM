"""Helper functions to build GaussianMixture objects for sweep experiments."""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.gaussian_mixture import GaussianMixture


def build_bimodal_nd(dim, separation_per_dim, variance_per_dim):
    """Build a bimodal mixture with per-dimension separation and variance.

    Args:
        dim: Dimensionality.
        separation_per_dim: List/array of length dim, separation in each dimension.
        variance_per_dim: List/array of length dim, variance in each dimension.
    """
    means = torch.zeros(2, dim)
    means[1] = torch.tensor(separation_per_dim, dtype=torch.float32)
    covs = torch.zeros(2, dim, dim)
    for k in range(2):
        covs[k] = torch.diag(torch.tensor(variance_per_dim, dtype=torch.float32))
    weights = torch.tensor([0.5, 0.5])
    return GaussianMixture(means, covs, weights, device='cpu')


def build_ring_mixture(n_modes, radius=3.0, variance=0.5, dim=2):
    """K modes equally spaced on a circle (first 2 dims), isotropic variance."""
    angles = torch.linspace(0, 2 * np.pi, n_modes + 1)[:n_modes]
    means = torch.zeros(n_modes, dim)
    means[:, 0] = radius * torch.cos(angles)
    means[:, 1] = radius * torch.sin(angles)
    covs = (variance * torch.eye(dim)).unsqueeze(0).expand(n_modes, -1, -1).clone()
    weights = torch.ones(n_modes) / n_modes
    return GaussianMixture(means, covs, weights, device='cpu')


def build_random_gmm(dim, seed):
    """Build a random GMM with reproducible parameters.

    Args:
        dim: Dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        (GaussianMixture, metadata_dict) where metadata has k, separation, variance, weights.
    """
    rng = np.random.RandomState(seed)
    k = 2
    separation = rng.uniform(1.0, 8.0)
    variance = rng.uniform(0.5, 3.0)
    w1 = rng.uniform(0.5, 0.95)
    weights_np = np.array([w1, 1.0 - w1])

    means = torch.zeros(2, dim)
    means[1, 0] = separation
    covs = (variance * torch.eye(dim)).unsqueeze(0).expand(2, -1, -1).clone()
    gm = GaussianMixture(means, covs, torch.tensor(weights_np, dtype=torch.float32), device='cpu')

    metadata = {
        'k': int(k),
        'separation': float(separation),
        'variance': float(variance),
        'weights': weights_np.tolist(),
    }
    return gm, metadata


def build_line_fixed_distance(n_modes, distance=4.0, variance=0.5, dim=2):
    """K modes on a line with fixed inter-mode distance.

    Modes at positions 0, d, 2d, ..., (K-1)*d along dimension 0.
    """
    n_modes = int(n_modes)
    means = torch.zeros(n_modes, dim)
    for i in range(n_modes):
        means[i, 0] = i * distance
    covs = (variance * torch.eye(dim)).unsqueeze(0).expand(n_modes, -1, -1).clone()
    weights = torch.ones(n_modes) / n_modes
    return GaussianMixture(means, covs, weights, device='cpu')


def build_ring_fixed_distance(n_modes, distance=4.0, variance=0.5, dim=2):
    """K modes on a circle with FIXED inter-mode distance (radius grows with k).

    Adjacent chord distance = 2R sin(π/k) = distance, so R = distance / (2 sin(π/k)).
    For k=2 the modes are on opposite ends (distance apart).
    """
    n_modes = int(n_modes)
    if n_modes < 2:
        # Single mode at origin
        means = torch.zeros(1, dim)
        covs = (variance * torch.eye(dim)).unsqueeze(0).clone()
        weights = torch.ones(1)
        return GaussianMixture(means, covs, weights, device='cpu')
    radius = distance / (2.0 * np.sin(np.pi / n_modes))
    angles = torch.linspace(0, 2 * np.pi, n_modes + 1)[:n_modes]
    means = torch.zeros(n_modes, dim)
    means[:, 0] = radius * torch.cos(angles)
    means[:, 1] = radius * torch.sin(angles)
    covs = (variance * torch.eye(dim)).unsqueeze(0).expand(n_modes, -1, -1).clone()
    weights = torch.ones(n_modes) / n_modes
    return GaussianMixture(means, covs, weights, device='cpu')
