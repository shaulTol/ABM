"""
Shared utilities for noise-floor-based DA lag selection.

Two floor types:
1. Permutation floor (operative): shuffle trajectory indices to destroy temporal
   correlation, compute top eigenvalue of Ĉ_perm(τ). Repeat n_perm times, take
   95th percentile. This is the ground-truth threshold.
2. Analytic floor (sanity check): exact for isotropic Gaussian null.
   λ_floor(τ) = exp(-τ/2)·σ²·(1+√γ)² + σ²·√γ·√(1-exp(-τ))
   where σ² = σ_f², γ = d/N.
"""
import numpy as np


def analytic_floor(tau, sigma_f_sq, d, N):
    """
    Analytic noise floor for C(τ) eigenvalues (exact for isotropic Gaussian).

    Args:
        tau: lag (scalar or array), in units of the canonical SDE time
        sigma_f_sq: scalar diffusion coefficient = (1/d)·tr(Cov_f)
        d: dimension
        N: number of trajectories

    Returns:
        λ_floor(τ): noise floor eigenvalue(s), same shape as tau
    """
    tau = np.asarray(tau, dtype=float)
    gamma = d / N
    sqrt_gamma = np.sqrt(gamma)
    rho = np.exp(-tau / 2)
    # Term 1: scaled MP upper edge of the correlated part
    term1 = rho * sigma_f_sq * (1 + sqrt_gamma) ** 2
    # Term 2: noise from the independent part
    term2 = sigma_f_sq * sqrt_gamma * np.sqrt(1 - np.exp(-tau))
    return term1 + term2


def permutation_floor(x0, x_tau, n_perm=20, quantile=0.95, symmetrize=True):
    """
    Permutation-based noise floor for C(τ).

    Destroys temporal correlation by shuffling trajectory indices, computes
    top eigenvalue of Ĉ_perm(τ). Repeats n_perm times, returns the quantile.

    Args:
        x0: (N, d) array, centered initial positions
        x_tau: (N, d) array, centered positions at lag τ
        n_perm: number of permutation replicates
        quantile: quantile of top eigenvalue distribution (default 0.95)
        symmetrize: whether to symmetrize C_perm (default True)

    Returns:
        floor: scalar, the quantile of the top eigenvalue across permutations
    """
    N, d = x0.shape
    top_eigs = np.zeros(n_perm)
    rng = np.random.default_rng()

    for i in range(n_perm):
        perm = rng.permutation(N)
        x_tau_perm = x_tau[perm]
        C_perm = x_tau_perm.T @ x0 / N
        if symmetrize:
            C_perm = 0.5 * (C_perm + C_perm.T)
        eigs = np.linalg.eigvalsh(C_perm)
        top_eigs[i] = eigs[-1]  # largest eigenvalue

    return np.quantile(top_eigs, quantile)


def find_tau_star(lags, eigenvalues_at_lags, floor_at_lags, m=1):
    """
    Find τ* = largest lag where the top m eigenvalues ALL exceed the floor.

    Args:
        lags: (n_lags,) array of lag values
        eigenvalues_at_lags: (n_lags, d) array, eigenvalues sorted descending
        floor_at_lags: (n_lags,) array, noise floor at each lag
        m: number of DA directions required

    Returns:
        tau_star_idx: index into lags (or -1 if never above floor)
        tau_star: the lag value
    """
    lags = np.asarray(lags)
    eigenvalues_at_lags = np.asarray(eigenvalues_at_lags)
    floor_at_lags = np.asarray(floor_at_lags)

    # At each lag, check if λ_1, ..., λ_m all exceed the floor
    tau_star_idx = -1
    for i in range(len(lags)):
        all_above = True
        for j in range(m):
            if eigenvalues_at_lags[i, j] <= floor_at_lags[i]:
                all_above = False
                break
        if all_above:
            tau_star_idx = i

    if tau_star_idx >= 0:
        return tau_star_idx, lags[tau_star_idx]
    else:
        return -1, None


def find_first_crossing(lags, eigenvalues_at_lags, floor_at_lags, m=1):
    """
    Find τ* = last lag BEFORE the first time λ_m drops below the floor.

    Scans forward from small to large τ. At the first lag where λ_m < floor,
    returns the previous checkpoint. This avoids late stochastic bounces.

    Args:
        lags: (n_lags,) array of lag values
        eigenvalues_at_lags: (n_lags, d) array, eigenvalues sorted descending
        floor_at_lags: (n_lags,) array, noise floor at each lag
        m: number of DA directions required (checks λ_m, the m-th eigenvalue)

    Returns:
        tau_star_idx: index into lags (or len(lags)-1 if never crosses)
        tau_star: the lag value
    """
    lags = np.asarray(lags)
    eigenvalues_at_lags = np.asarray(eigenvalues_at_lags)
    floor_at_lags = np.asarray(floor_at_lags)

    for i in range(len(lags)):
        # Check if the m-th eigenvalue (index m-1) drops below floor
        if eigenvalues_at_lags[i, m - 1] < floor_at_lags[i]:
            if i == 0:
                return 0, lags[0]
            return i - 1, lags[i - 1]

    # Never crossed — entire grid is above floor
    return len(lags) - 1, lags[-1]
