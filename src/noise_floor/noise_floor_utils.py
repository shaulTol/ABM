"""
Shared utilities for noise-floor-based DA lag selection.

Two analytic bounds on the noise floor for C(τ) eigenvalues under the
isotropic Gaussian null (dX = -½X dt + dW, stationary at N(0, σ²I)):

Conservative (upper bound on noise, from Bai-Yin submultiplicativity):
  λ_floor_upper(τ) = e^{-τ/2}·σ²·(1+√γ)² + σ²·(1+√γ)²·√(1-e^{-τ})

Aggressive (lower bound on noise, from Frobenius norm):
  λ_floor_lower(τ) = e^{-τ/2}·σ²·(1+√γ)² + σ²·√(d/N)·√(1-e^{-τ})

where σ² = σ_f², γ = d/N. Both terms share the MP bulk edge (1+√γ)²
for the signal term; they differ only in the cross-product term:
  upper: (1+√γ)²   (Bai-Yin, not tight but rigorous)
  lower: √(d/N)    (Frobenius lower bound)
"""
import numpy as np


def analytic_floor_upper(tau, sigma_f_sq, d, N):
    """
    Conservative (upper) noise floor — upper bound on noise eigenvalue.

    λ_floor(τ) = e^{-τ/2}·σ²·(1+√γ)² + σ²·(1+√γ)²·√(1-e^{-τ})
    """
    tau = np.asarray(tau, dtype=float)
    gamma = d / N
    sqrt_gamma = np.sqrt(gamma)
    mp_edge = (1 + sqrt_gamma) ** 2
    rho = np.exp(-tau / 2)
    term1 = rho * sigma_f_sq * mp_edge
    term2 = sigma_f_sq * mp_edge * np.sqrt(1 - np.exp(-tau))
    return term1 + term2


def analytic_floor_lower(tau, sigma_f_sq, d, N):
    """
    Aggressive (lower) noise floor — lower bound on noise eigenvalue.

    λ_floor(τ) = e^{-τ/2}·σ²·(1+√γ)² + σ²·√(d/N)·√(1-e^{-τ})
    """
    tau = np.asarray(tau, dtype=float)
    gamma = d / N
    sqrt_gamma = np.sqrt(gamma)
    mp_edge = (1 + sqrt_gamma) ** 2
    rho = np.exp(-tau / 2)
    term1 = rho * sigma_f_sq * mp_edge
    term2 = sigma_f_sq * sqrt_gamma * np.sqrt(1 - np.exp(-tau))
    return term1 + term2


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
        if eigenvalues_at_lags[i, m - 1] < floor_at_lags[i]:
            if i == 0:
                return 0, lags[0]
            return i - 1, lags[i - 1]

    # Never crossed — entire grid is above floor
    return len(lags) - 1, lags[-1]
