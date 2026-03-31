"""SSA and MI computation wrapper."""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.gaussian_mixture import GaussianMixture
from src.slicing.sliced_autocorrelation import full_autocorrelation


def _move_gm_to_device(gm, device):
    """Move GaussianMixture tensor attributes to the given device."""
    target = torch.device(device)
    if gm.means.device == target:
        return gm
    for attr in ('means', 'covs', 'weights', 'chol', 'log_dets', 'precisions'):
        try:
            tensor = getattr(gm, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(gm, attr, tensor.to(device))
        except AttributeError:
            pass
    gm.device = target
    gm._cache = {}
    return gm


def compute_ssa_and_mi(gm: GaussianMixture, n_samples_ssa=2000, n_samples_mi=5000,
                       n_trajectories=10000, T=1000, max_lag=500, dt=0.01,
                       device='cpu') -> dict:
    """Compute SSA and MI for a GaussianMixture.

    Args:
        device: 'cpu' or 'cuda'. GaussianMixture is moved to this device for computation.

    Returns:
        dict with keys 'ssa' (float) and 'mi' (float).
    """
    gm = _move_gm_to_device(gm, device)

    with torch.no_grad():
        # MI — fall back to CPU if GPU fails
        try:
            mi = gm.mutual_information(n_samples=n_samples_mi).item()
        except RuntimeError:
            gm_cpu = _move_gm_to_device(gm, 'cpu')
            mi = gm_cpu.mutual_information(n_samples=n_samples_mi).item()
            gm = _move_gm_to_device(gm_cpu, device)

        samples = gm.sample(n_samples_ssa, return_labels=False)

        ssa = full_autocorrelation(
            samples,
            gmm_means=gm.means,
            gmm_covs=gm.covs,
            gmm_weights=gm.weights,
            n_trajectories=n_trajectories,
            T=T,
            max_lag=max_lag,
            dt=dt,
            device=torch.device(device),
            normalization='trace',
        )

    return {'ssa': float(ssa), 'mi': float(mi)}
