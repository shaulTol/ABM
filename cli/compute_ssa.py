#!/usr/bin/env python3
"""
Shared pipeline: given saved C(τ), compute SSA (Ŝ) + optional bootstrap CI.

Reads an NPZ file produced by cli/save_lags.py and computes the
Sum of Squared Autocorrelations with free-convolution noise floor truncation.

Usage (as library):
    from cli.compute_ssa import compute_ssa, bootstrap_ssa

Usage (as CLI):
    python -m cli.compute_ssa results/alanine_C_tau_full_h_half_10k.npz
"""
import os, sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.autocorrelation import aggregate_ssa
from src.noise_floor.free_conv_utils import edges


def compute_ssa(C_tau_path, floor='freeconv', floor_value=None):
    """Compute SSA (Ŝ) from saved autocovariance data.

    Args:
        C_tau_path: path to NPZ from save_lags.py.
        floor: 'freeconv' (default), '1overN', or 'fixed'.
        floor_value: explicit floor value when floor='fixed'.

    Returns:
        dict with keys: s_hat, rho_sq, phys_times, tau_star, n_lags_used.
    """
    data = np.load(C_tau_path)
    eigenvalues = data['eigenvalues']   # (n_lags, d)
    phys_times = data['phys_times']     # (n_lags,)
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    N = int(data['N'])
    dt = float(data['dt'])
    sigma = np.sqrt(sigma_f_sq)

    # Compute ρ²(τ) = (λ₁(τ) / (d · σ²_f))² for each lag
    rho_sq = (eigenvalues[:, 0] / (d * sigma_f_sq)) ** 2

    # Determine noise floor
    if floor == 'freeconv':
        # Free-conv upper edge at each lag
        floors = np.array([edges(tau, sigma, gamma=gamma)[1] for tau in phys_times])
        rho_sq_floor = (floors / (d * sigma_f_sq)) ** 2
        # Truncate: use lags where rho_sq > floor
        above = rho_sq > rho_sq_floor
        if above.any():
            last_above = np.where(above)[0][-1]
            n_used = last_above + 1
        else:
            n_used = 1
    elif floor == '1overN':
        rho_sq_floor_val = 1.0 / N
        above = rho_sq > rho_sq_floor_val
        n_used = np.where(above)[0][-1] + 1 if above.any() else 1
    elif floor == 'fixed':
        above = rho_sq > floor_value
        n_used = np.where(above)[0][-1] + 1 if above.any() else 1
    else:
        raise ValueError(f"Unknown floor type: {floor}")

    # Skip lag 0 (ρ(0)=1 always), integrate lags 1..n_used
    rho_sq_used = rho_sq[1:n_used]
    s_hat = aggregate_ssa(rho_sq_used, dt=dt)

    tau_star = phys_times[n_used - 1] if n_used > 0 else 0

    return {
        's_hat': float(s_hat),
        'rho_sq': rho_sq,
        'phys_times': phys_times,
        'tau_star': float(tau_star),
        'n_lags_used': int(n_used),
    }


def bootstrap_ssa(C_tau_path, B=2000, floor='freeconv', seed=42):
    """Bootstrap confidence interval for SSA.

    Resamples N trajectories with replacement B times, recomputes Ŝ each time.
    Requires that the NPZ contains per-trajectory data (Z_all key).

    If Z_all is not available, falls back to a simpler lag-resampling bootstrap
    using the stored eigenvalues.

    Args:
        C_tau_path: path to NPZ.
        B: number of bootstrap replicates.
        floor: noise floor type.
        seed: RNG seed.

    Returns:
        dict with keys: s_hat, se, ci_lo, ci_hi, s_hat_boots.
    """
    rng = np.random.RandomState(seed)
    base = compute_ssa(C_tau_path, floor=floor)

    data = np.load(C_tau_path)
    dt = float(data['dt'])

    if 'Z_all' in data:
        # Full trajectory bootstrap
        Z_all = data['Z_all']   # (N, T+1, d)
        N = Z_all.shape[0]
        lags = (data['phys_times'] / dt).astype(int)
        from cli.save_lags import compute_symmetrized_autocovariance

        boots = np.zeros(B)
        for b in range(B):
            idx = rng.choice(N, N, replace=True)
            Z_b = Z_all[idx]
            C_b = compute_symmetrized_autocovariance(Z_b, lags)
            d_dim = Z_b.shape[2]
            sigma_f_sq = float(data['sigma_f_sq'])
            eigs_b = np.array([np.linalg.eigvalsh(C_b[i])[::-1] for i in range(len(lags))])
            rho_sq_b = (eigs_b[:, 0] / (d_dim * sigma_f_sq)) ** 2
            rho_sq_used = rho_sq_b[1:base['n_lags_used']]
            boots[b] = aggregate_ssa(rho_sq_used, dt=dt)
    else:
        # Fallback: lag-block bootstrap (less accurate)
        rho_sq = base['rho_sq']
        n_used = base['n_lags_used']
        boots = np.zeros(B)
        for b in range(B):
            idx = rng.choice(n_used - 1, n_used - 1, replace=True) + 1
            boots[b] = aggregate_ssa(rho_sq[idx], dt=dt)

    se = float(np.std(boots))
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    return {
        's_hat': base['s_hat'],
        'se': se,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        's_hat_boots': boots,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m cli.compute_ssa <C_tau.npz> [--bootstrap B]')
        sys.exit(1)
    path = sys.argv[1]
    result = compute_ssa(path)
    print(f"Ŝ = {result['s_hat']:.4f}  (τ* = {result['tau_star']:.1f}, "
          f"{result['n_lags_used']} lags used)")

    if '--bootstrap' in sys.argv:
        idx = sys.argv.index('--bootstrap')
        B = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 2000
        boot = bootstrap_ssa(path, B=B)
        print(f"Bootstrap (B={B}): Ŝ = {boot['s_hat']:.4f} ± {boot['se']:.4f} "
              f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")
