"""
Pipeline Module - Algorithm 1 end to end

save_lags   : simulate the canonical diffusion, save C_sym(τ) to an NPZ
compute_ssa : SSA (Ŝ) with the 1/N stopping rule, trajectory bootstrap
compute_da  : DA lag selection τ*(m), m̂, directions, matched Monte Carlo null
"""
from .save_lags import save_lags, compute_symmetrized_autocovariance
from .compute_ssa import compute_ssa, bootstrap_ssa, ssa_from_rho_sq
from .compute_da import (
    compute_da,
    da_significance,
    select_tau_star,
    n_metastable,
    dip_pvalues,
)

__all__ = [
    'save_lags',
    'compute_symmetrized_autocovariance',
    'compute_ssa',
    'bootstrap_ssa',
    'ssa_from_rho_sq',
    'compute_da',
    'da_significance',
    'select_tau_star',
    'n_metastable',
    'dip_pvalues',
]
