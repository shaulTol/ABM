"""All 10 sweep configurations for SSA vs MI comparison."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mixture_builders import build_bimodal_nd, build_ring_mixture, build_ring_fixed_distance, build_line_fixed_distance


# --- 2D sweeps ---

def _build_sep_2d(sep):
    return build_bimodal_nd(dim=2, separation_per_dim=[sep, 0.0], variance_per_dim=[1.0, 1.0])

def _build_var_2d(var):
    return build_bimodal_nd(dim=2, separation_per_dim=[4.0, 0.0], variance_per_dim=[var, var])

def _build_nmodes_2d(k):
    return build_ring_mixture(n_modes=int(k), radius=3.0, variance=0.5, dim=2)

def _build_nmodes_fixdist_2d(k):
    return build_line_fixed_distance(n_modes=int(k), distance=4.0, variance=0.5, dim=2)

def _build_nmodes_fixdist_10d(k):
    return build_ring_fixed_distance(n_modes=int(k), distance=4.0, variance=0.5, dim=10)

def _build_weight_10d(w1):
    import torch
    means = torch.zeros(2, 10)
    means[1] = torch.tensor([4.0] * 10, dtype=torch.float32)
    covs = torch.eye(10).unsqueeze(0).expand(2, -1, -1).clone()
    weights = torch.tensor([w1, 1.0 - w1])
    from src.core.gaussian_mixture import GaussianMixture
    return GaussianMixture(means, covs, weights, device='cpu')

def _build_weight_2d(w1):
    import torch
    means = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    covs = torch.eye(2).unsqueeze(0).expand(2, -1, -1).clone()
    weights = torch.tensor([w1, 1.0 - w1])
    from src.core.gaussian_mixture import GaussianMixture
    return GaussianMixture(means, covs, weights, device='cpu')

# --- 10D sweeps ---

def _build_sep_10d_all(sep):
    return build_bimodal_nd(dim=10, separation_per_dim=[sep] * 10, variance_per_dim=[1.0] * 10)

def _build_sep_10d_dim0(sep):
    seps = [0.0] * 10
    seps[0] = sep
    return build_bimodal_nd(dim=10, separation_per_dim=seps, variance_per_dim=[1.0] * 10)

def _build_sep_10d_half(sep):
    seps = [sep] * 5 + [0.0] * 5
    return build_bimodal_nd(dim=10, separation_per_dim=seps, variance_per_dim=[1.0] * 10)

def _build_var_10d_all(var):
    return build_bimodal_nd(dim=10, separation_per_dim=[4.0] * 10, variance_per_dim=[var] * 10)

def _build_var_10d_dim0(var):
    variances = [1.0] * 10
    variances[0] = var
    return build_bimodal_nd(dim=10, separation_per_dim=[4.0] * 10, variance_per_dim=variances)

def _build_var_10d_half(var):
    variances = [var] * 5 + [1.0] * 5
    return build_bimodal_nd(dim=10, separation_per_dim=[4.0] * 10, variance_per_dim=variances)


SWEEP_CONFIGS = [
    {
        'name': 'sep_2d',
        'description': '2D Bimodal: Varying Separation',
        'dim': 2,
        'param_name': 'Separation',
        'param_values': np.linspace(0.5, 8.0, 30),
        'build_mixture': _build_sep_2d,
    },
    {
        'name': 'var_2d',
        'description': '2D Bimodal: Varying Variance',
        'dim': 2,
        'param_name': 'Variance',
        'param_values': np.linspace(0.1, 5.0, 30),
        'build_mixture': _build_var_2d,
    },
    {
        'name': 'nmodes_2d',
        'description': '2D Ring: Varying Number of Modes',
        'dim': 2,
        'param_name': 'Number of Modes',
        'param_values': np.array([1, 2, 3, 4, 5, 6, 8]),
        'build_mixture': _build_nmodes_2d,
    },
    {
        'name': 'weight_2d',
        'description': '2D Bimodal: Varying Weight Asymmetry',
        'dim': 2,
        'param_name': 'Weight of Mode 1',
        'param_values': np.linspace(0.5, 0.95, 20),
        'build_mixture': _build_weight_2d,
    },
    {
        'name': 'nmodes_fixdist_2d',
        'description': '2D Line: Varying Modes (Fixed Distance=4.0)',
        'dim': 2,
        'param_name': 'Number of Modes',
        'param_values': np.array([1, 2, 3, 4, 5]),
        'build_mixture': _build_nmodes_fixdist_2d,
    },
    {
        'name': 'nmodes_fixdist_10d',
        'description': '10D Ring: Varying Modes (Fixed Distance=4.0)',
        'dim': 10,
        'param_name': 'Number of Modes',
        'param_values': np.array([1, 2, 3, 4, 5, 6, 8]),
        'build_mixture': _build_nmodes_fixdist_10d,
    },
    {
        'name': 'weight_10d',
        'description': '10D Bimodal: Varying Weight Asymmetry',
        'dim': 10,
        'param_name': 'Weight of Mode 1',
        'param_values': np.linspace(0.5, 0.95, 20),
        'build_mixture': _build_weight_10d,
    },
    {
        'name': 'sep_10d_all',
        'description': '10D Bimodal: Separation in All Dims',
        'dim': 10,
        'param_name': 'Separation (per dim)',
        'param_values': np.linspace(0.5, 8.0, 30),
        'build_mixture': _build_sep_10d_all,
    },
    {
        'name': 'sep_10d_dim0',
        'description': '10D Bimodal: Separation in Dim 0 Only',
        'dim': 10,
        'param_name': 'Separation (dim 0)',
        'param_values': np.linspace(0.5, 8.0, 30),
        'build_mixture': _build_sep_10d_dim0,
    },
    {
        'name': 'sep_10d_half',
        'description': '10D Bimodal: Separation in Dims 0-4',
        'dim': 10,
        'param_name': 'Separation (dims 0-4)',
        'param_values': np.linspace(0.5, 8.0, 30),
        'build_mixture': _build_sep_10d_half,
    },
    {
        'name': 'var_10d_all',
        'description': '10D Bimodal: Varying Variance (All Dims)',
        'dim': 10,
        'param_name': 'Variance',
        'param_values': np.linspace(0.1, 5.0, 30),
        'build_mixture': _build_var_10d_all,
    },
    {
        'name': 'var_10d_dim0',
        'description': '10D Bimodal: Varying Variance (Dim 0 Only)',
        'dim': 10,
        'param_name': 'Variance (dim 0)',
        'param_values': np.linspace(0.1, 5.0, 30),
        'build_mixture': _build_var_10d_dim0,
    },
    {
        'name': 'var_10d_half',
        'description': '10D Bimodal: Varying Variance (Dims 0-4)',
        'dim': 10,
        'param_name': 'Variance (dims 0-4)',
        'param_values': np.linspace(0.1, 5.0, 30),
        'build_mixture': _build_var_10d_half,
    },
]
