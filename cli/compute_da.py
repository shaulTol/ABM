#!/usr/bin/env python3
"""
Shared pipeline: given saved C(τ), compute DA directions + significance.

Reads an NPZ file produced by cli/save_lags.py and computes the
Diffusion-Autocorrelation (DA) directions — eigenvectors of the symmetrized
autocovariance C_sym(τ*) at the selected lag.

Usage (as library):
    from cli.compute_da import compute_da, da_significance, select_tau_star

Usage (as CLI):
    python -m cli.compute_da results/alanine_C_tau_full_h_half_10k.npz
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.noise_floor.free_conv_utils import edges, edge_tauinf


def select_tau_star(eigenvalues, phys_times, sigma_f_sq, gamma, k=2):
    """Auto-select lag τ* where the k-th eigenvalue crosses the free-conv floor.

    The DA directions are meaningful as long as the top-k eigenvalues stay
    above the free-convolution upper edge λ₊(τ). τ* is the largest lag
    where λ_k(τ) > λ₊(τ).

    Args:
        eigenvalues: (n_lags, d) eigenvalues sorted descending.
        phys_times: (n_lags,) physical times.
        sigma_f_sq: scalar diffusion coefficient.
        gamma: aspect ratio d/N.
        k: number of DA directions to protect (default 2).

    Returns:
        tau_star_idx: index into phys_times.
        tau_star: physical time.
    """
    sigma = np.sqrt(sigma_f_sq)
    n_lags = len(phys_times)

    # Find last lag where λ_k > λ₊(τ)
    tau_star_idx = 0
    for i in range(n_lags):
        _, lp = edges(phys_times[i], sigma, gamma=gamma)
        if eigenvalues[i, k - 1] > lp:
            tau_star_idx = i

    return tau_star_idx, float(phys_times[tau_star_idx])


def compute_da(C_tau_path, tau_idx=None, n_directions=6):
    """Compute DA directions from saved autocovariance.

    Args:
        C_tau_path: path to NPZ from save_lags.py.
        tau_idx: lag index to use. If None, auto-selects via free-conv criterion.
        n_directions: number of eigenvectors to return.

    Returns:
        dict with keys: eigenvalues, eigenvectors, tau_star, tau_star_idx,
                         gap_ratio, sigma_f_sq, gamma, d.
    """
    data = np.load(C_tau_path)
    C_tau_all = data['C_tau_all']       # (n_lags, d, d)
    eigenvalues = data['eigenvalues']   # (n_lags, d)
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])

    # Select lag
    if tau_idx is None:
        tau_idx, tau_star = select_tau_star(eigenvalues, phys_times,
                                            sigma_f_sq, gamma, k=2)
    else:
        tau_star = float(phys_times[tau_idx])

    # Eigendecompose C_sym(τ*)
    C_star = C_tau_all[tau_idx]
    evals, evecs = np.linalg.eigh(C_star)
    # Sort descending
    order = np.argsort(evals)[::-1].copy()
    evals = evals[order]
    evecs = evecs[:, order]

    # Truncate to n_directions
    evals = evals[:n_directions]
    evecs = evecs[:, :n_directions]

    # Gap ratio
    sigma = np.sqrt(sigma_f_sq)
    _, lp = edges(tau_star, sigma, gamma=gamma)
    gap_ratio = evals[0] / lp if lp > 0 else float('inf')

    return {
        'eigenvalues': evals,
        'eigenvectors': evecs,
        'tau_star': tau_star,
        'tau_star_idx': int(tau_idx),
        'gap_ratio': float(gap_ratio),
        'noise_floor': float(lp),
        'sigma_f_sq': sigma_f_sq,
        'gamma': gamma,
        'd': d,
    }


def da_significance(C_tau_path, B=500, tau_idx=None, seed=42):
    """Monte Carlo null test for DA spike significance.

    Generates B null replicates by permuting the time axis of each trajectory,
    destroying temporal correlation while preserving marginal distribution.
    Computes top eigenvalue of C_sym(τ*) under null and compares with observed.

    Args:
        C_tau_path: path to NPZ.
        B: number of null replicates.
        tau_idx: lag index (auto-selected if None).
        seed: RNG seed.

    Returns:
        dict with keys: observed_evals, null_max_evals, p_values, tau_star.
    """
    data = np.load(C_tau_path)
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])

    # Get observed DA
    result = compute_da(C_tau_path, tau_idx=tau_idx)
    observed_evals = result['eigenvalues']
    tau_idx_used = result['tau_star_idx']

    if 'Z_all' not in data:
        raise ValueError("NPZ must contain 'Z_all' (per-trajectory data) "
                         "for significance testing")

    Z_all = data['Z_all']   # (N, T+1, d)
    N, T_plus_1, d = Z_all.shape
    lags = (data['phys_times'] / float(data['dt'])).astype(int)
    lag = lags[tau_idx_used]

    rng = np.random.RandomState(seed)
    null_max_evals = np.zeros(B)

    for b in range(B):
        # Permute time axis independently per trajectory
        Z_perm = Z_all.copy()
        for n in range(N):
            perm = rng.permutation(T_plus_1)
            Z_perm[n] = Z_perm[n, perm]

        # Compute C_sym(τ*) under null
        Z_t = Z_perm[:, :T_plus_1 - lag, :]
        Z_th = Z_perm[:, lag:, :]
        Z_t_flat = Z_t.reshape(-1, d)
        Z_th_flat = Z_th.reshape(-1, d)
        C = Z_t_flat.T @ Z_th_flat / Z_t_flat.shape[0]
        C_sym = (C + C.T) / 2
        null_max_evals[b] = np.linalg.eigvalsh(C_sym)[-1]

    # p-values for each observed eigenvalue
    p_values = np.array([
        (null_max_evals >= ev).mean() for ev in observed_evals
    ])

    return {
        'observed_evals': observed_evals,
        'null_max_evals': null_max_evals,
        'p_values': p_values,
        'tau_star': result['tau_star'],
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m cli.compute_da <C_tau.npz> [--significance B]')
        sys.exit(1)
    path = sys.argv[1]
    result = compute_da(path)
    print(f"DA at τ* = {result['tau_star']:.1f}")
    print(f"  λ₁ = {result['eigenvalues'][0]:.4e}  "
          f"(gap = {result['gap_ratio']:.1f}× floor)")
    if len(result['eigenvalues']) > 1:
        print(f"  λ₂ = {result['eigenvalues'][1]:.4e}")
    print(f"  Floor λ₊ = {result['noise_floor']:.4e}")

    if '--significance' in sys.argv:
        idx = sys.argv.index('--significance')
        B = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 500
        sig = da_significance(path, B=B)
        print(f"\nMC null (B={B}):")
        for i, (ev, p) in enumerate(zip(sig['observed_evals'], sig['p_values'])):
            print(f"  DA{i+1}: λ={ev:.4e}, p={p:.4f}")
