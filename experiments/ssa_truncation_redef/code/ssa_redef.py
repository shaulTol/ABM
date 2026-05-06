#!/usr/bin/env python3
"""
Core module for SSA truncation redefinition.

Two definitions:
  S_old = (1/L) * Σ_{ℓ=1}^{L} ρ(ℓ)²          (mean over all lags)
  S_hat = Δt  * Σ_{ℓ=1}^{L*} ρ(ℓ)²            (Riemann sum, truncated at noise floor)

where L* = τ*/Δt, and τ* is from the noise-floor criterion (m=1).
"""
import sys
import os
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiments.noise_floor.noise_floor_utils import analytic_floor_upper, find_first_crossing


def compute_s_old(rho_sq, exclude_zero=True):
    """
    S_old = (1/L) * Σ_{ℓ=1}^{L} ρ(ℓ)².

    Args:
        rho_sq: array of ρ(ℓ)² values. If length L+1, index 0 is lag 0.
                If length L, assumed to already exclude lag 0.
        exclude_zero: if True, skip index 0 (lag 0).

    Returns:
        S_old (float)
    """
    rho_sq = np.asarray(rho_sq)
    if exclude_zero:
        vals = rho_sq[1:]  # skip lag 0
    else:
        vals = rho_sq
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def compute_s_hat(rho_sq, lags, dt, tau_star, exclude_zero=True):
    """
    S_hat = Δt * Σ_{ℓ : lag_ℓ <= τ* and lag_ℓ > 0} ρ(ℓ)².

    Riemann sum of ρ² up to the noise-floor cutoff τ*.

    Args:
        rho_sq: array of ρ(ℓ)² values, one per lag.
        lags: array of lag values (physical time, i.e. lag_index * dt or snapshot lags).
        dt: time step Δt (for uniform spacing) or lag spacing.
        tau_star: cutoff τ* from noise floor.
        exclude_zero: if True, skip lags <= 0.

    Returns:
        S_hat (float)
    """
    rho_sq = np.asarray(rho_sq)
    lags = np.asarray(lags, dtype=float)
    assert len(rho_sq) == len(lags), f"rho_sq ({len(rho_sq)}) and lags ({len(lags)}) must match"

    total = 0.0
    for i in range(len(lags)):
        if exclude_zero and lags[i] <= 0:
            continue
        if lags[i] > tau_star:
            break
        total += rho_sq[i] * dt
    return float(total)


def find_tau_star_from_eigenvalues(lags, eigenvalues_at_lags, var_f, d, N, m=1):
    """
    Find τ* via the noise-floor criterion (conservative upper bound).

    Uses analytic_floor_upper with m=1: the first lag where λ_1 drops
    below the MP noise floor.

    Args:
        lags: (n_lags,) array of physical-time lag values.
        eigenvalues_at_lags: (n_lags, d_eff) eigenvalues sorted descending.
        var_f: σ_f² = tr(Σ_f)/d.
        d: embedding dimension.
        N: number of trajectories.
        m: eigenvalue index to check (1 = leading eigenvalue).

    Returns:
        tau_star_idx, tau_star
    """
    lags = np.asarray(lags, dtype=float)
    floor = analytic_floor_upper(lags, var_f, d, N)
    return find_first_crossing(lags, eigenvalues_at_lags, floor, m=m)


def find_tau_star_from_rho_sq(rho_sq, lags, dt, var_f, d, N):
    """
    Find τ* for the scalar ρ² case (no eigenvalue decomposition available).

    Strategy: ρ²(τ) should exceed the noise level. For a single scalar
    autocorrelation, the noise floor at lag τ is approximately:
      floor_rho_sq(τ) ≈ 1/N  (from the CLT: Var[ρ̂] ≈ 1/N for N trajectories)

    We find the first lag where ρ²(τ) < 1/N.

    Args:
        rho_sq: array of ρ(ℓ)².
        lags: array of lag values.
        dt: time step.
        var_f: σ_f².
        d: dimension.
        N: number of trajectories.

    Returns:
        tau_star_idx, tau_star
    """
    rho_sq = np.asarray(rho_sq)
    lags = np.asarray(lags, dtype=float)
    floor = 1.0 / N  # CLT-based noise floor for ρ²

    for i in range(len(lags)):
        if lags[i] <= 0:
            continue
        if rho_sq[i] < floor:
            if i == 0:
                return 0, lags[0]
            return i - 1, lags[i - 1]

    return len(lags) - 1, lags[-1]


def gaussian_baseline(d, var_f, dt, T, max_lag=None):
    """
    Analytic S_hat for isotropic Gaussian (OU process).

    For the OU process dX = -(1/2)X dt + σ_f dW, stationary at N(0, σ_f²I):
      ρ(τ) = exp(-τ/2)
      ρ²(τ) = exp(-τ)

    S_old = (1/L) Σ_{ℓ=1}^L exp(-2*ℓ*dt/2) = (1/L) Σ exp(-ℓ*dt)
    S_hat = dt * Σ_{ℓ=1}^{L*} exp(-ℓ*dt)  (truncated at L* from floor)

    For the Gaussian case, S_hat converges to ∫_0^∞ e^{-t} dt = 1 as T→∞.
    But with finite T and noise floor, τ* < ∞.

    Args:
        d: dimension.
        var_f: σ_f².
        dt: time step.
        T: total simulation time steps.
        max_lag: maximum lag (defaults to T).

    Returns:
        dict with s_old_analytic, s_hat_analytic, rho_sq_analytic, lags.
    """
    if max_lag is None:
        max_lag = T
    lags = np.arange(0, max_lag + 1) * dt  # physical times
    # ρ(τ) = exp(-τ/2), so ρ²(τ) = exp(-τ)
    rho_sq = np.exp(-lags)

    # S_old: mean of ρ² for lags 1..L
    s_old = rho_sq[1:].mean()

    # S_hat: Riemann sum (all lags, since Gaussian has no noise floor issue)
    s_hat = dt * rho_sq[1:].sum()

    # Also compute the infinite-sum limit
    # Σ_{ℓ=1}^∞ dt * exp(-ℓ*dt) = dt * exp(-dt)/(1 - exp(-dt))
    s_hat_inf = dt * np.exp(-dt) / (1 - np.exp(-dt))

    return {
        's_old_analytic': float(s_old),
        's_hat_analytic': float(s_hat),
        's_hat_infinite': float(s_hat_inf),
        'rho_sq_analytic': rho_sq.tolist(),
        'lags': lags.tolist(),
    }


def compute_both_ssa(rho_sq, lags, dt, tau_star):
    """
    Compute both S_old and S_hat from the same ρ² array.

    Args:
        rho_sq: array of ρ(ℓ)² (including lag 0).
        lags: array of lag values (physical time).
        dt: time step.
        tau_star: noise-floor cutoff.

    Returns:
        dict with s_old, s_hat, tau_star, L_star (number of lags in sum).
    """
    rho_sq = np.asarray(rho_sq)
    lags = np.asarray(lags, dtype=float)

    s_old = compute_s_old(rho_sq, exclude_zero=True)
    s_hat = compute_s_hat(rho_sq, lags, dt, tau_star, exclude_zero=True)

    # Count how many lags are in the S_hat sum
    L_star = sum(1 for i in range(len(lags)) if lags[i] > 0 and lags[i] <= tau_star)

    return {
        's_old': float(s_old),
        's_hat': float(s_hat),
        'tau_star': float(tau_star),
        'L_star': int(L_star),
    }
