"""Silverman modality test for 1D projections.

Implements:
  1. Critical bandwidth h_crit: smallest bandwidth at which KDE has < n_modes modes.
  2. Bootstrap p-value: fraction of B resamples (smoothed at h_crit) with >= n_modes modes.

Reference: Silverman (1981), "Using kernel density estimates to investigate multimodality."
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmax


def count_modes(data, bandwidth, grid_size=2048):
    """Count modes (local maxima) of KDE with given bandwidth.

    Args:
        data: (N,) array of 1D observations.
        bandwidth: scalar bandwidth for Gaussian KDE.
        grid_size: number of evaluation points.

    Returns:
        Number of modes (int).
    """
    x_grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, grid_size)
    # Use scipy KDE with fixed bandwidth
    kde = gaussian_kde(data, bw_method=bandwidth / data.std(ddof=1))
    density = kde(x_grid)
    peaks = argrelmax(density, order=1)[0]
    return max(len(peaks), 1)  # at least 1 mode


def silverman_critical_bandwidth(data, n_modes=2, tol=1e-4, max_iter=100):
    """Find h_crit: smallest bandwidth where KDE has fewer than n_modes modes.

    Binary search over bandwidth. At h < h_crit, KDE has >= n_modes modes.
    At h >= h_crit, KDE has < n_modes modes.

    Args:
        data: (N,) array.
        n_modes: target number of modes to test for.
        tol: relative tolerance for binary search convergence.
        max_iter: maximum iterations.

    Returns:
        h_crit (float).
    """
    data = np.asarray(data, dtype=float)
    sd = data.std(ddof=1)
    if sd == 0:
        return 1.0

    # Search range: very small h (many modes) to very large h (1 mode)
    h_lo = sd * 0.001
    h_hi = sd * 5.0

    # Ensure h_lo gives >= n_modes
    while count_modes(data, h_lo) < n_modes and h_lo > sd * 1e-6:
        h_lo *= 0.5

    # Ensure h_hi gives < n_modes
    while count_modes(data, h_hi) >= n_modes and h_hi < sd * 50:
        h_hi *= 2.0

    # If even smallest h doesn't give n_modes, data is unimodal
    if count_modes(data, h_lo) < n_modes:
        return h_lo

    # Binary search
    for _ in range(max_iter):
        h_mid = (h_lo + h_hi) / 2
        if (h_hi - h_lo) / h_mid < tol:
            break
        n = count_modes(data, h_mid)
        if n >= n_modes:
            h_lo = h_mid
        else:
            h_hi = h_mid

    return h_hi  # smallest h with < n_modes modes


def silverman_bootstrap(data, h_crit, B=1000, n_modes=2, seed=42):
    """Bootstrap p-value for Silverman's modality test.

    Tests H0: the density has fewer than n_modes modes.
    Small p-value → reject H0 → evidence for >= n_modes modes.

    Procedure (following Hall & York 2001, Ameijeiras-Alonso et al. 2019):
      For each bootstrap replicate b = 1..B:
        1. Resample N points with replacement from data.
        2. Add Gaussian noise with std = h_crit to each point.
        3. Rescale to preserve original variance.
        4. Find the critical bandwidth h_crit_b for this resample.
      p = fraction of resamples where h_crit_b > h_crit.

    A small p means: the observed h_crit is unusually large (modes are
    well-separated), rejecting the null of fewer modes.

    Args:
        data: (N,) array.
        h_crit: critical bandwidth from silverman_critical_bandwidth().
        B: number of bootstrap replicates.
        n_modes: target number of modes.
        seed: random seed.

    Returns:
        p_value (float): proportion of replicates with h_crit_b > h_crit.
    """
    data = np.asarray(data, dtype=float)
    N = len(data)
    rng = np.random.default_rng(seed)

    # Rescaling factor: var(smoothed resample) should match var(data)
    var_data = data.var(ddof=1)
    # After adding noise: var = var_data + h_crit^2
    # Rescale by factor c so that c^2 * (var_data + h_crit^2) = var_data
    c = np.sqrt(var_data / (var_data + h_crit**2))

    count_exceed = 0
    for _ in range(B):
        # Resample with replacement
        idx = rng.integers(0, N, size=N)
        x_star = data[idx]
        # Add smoothing noise (generates from kernel density at h_crit)
        x_star = x_star + h_crit * rng.standard_normal(N)
        # Rescale to preserve variance
        x_star = data.mean() + c * (x_star - data.mean())
        # Find critical bandwidth for this resample
        h_crit_b = silverman_critical_bandwidth(x_star, n_modes=n_modes)
        if h_crit_b > h_crit:
            count_exceed += 1

    return count_exceed / B


def silverman_test(data, n_modes=2, B=1000, seed=42):
    """Full Silverman modality test.

    Tests H0: density has fewer than n_modes modes.
    Small p → reject H0 → significant evidence for >= n_modes modes.

    Args:
        data: (N,) array.
        n_modes: test for at least this many modes.
        B: bootstrap replicates.
        seed: random seed.

    Returns:
        dict with keys: h_crit, p_value, n_modes_observed (at h_crit - epsilon).
              p_value < alpha means REJECT unimodality (significant multimodality).
    """
    data = np.asarray(data, dtype=float)
    h_crit = silverman_critical_bandwidth(data, n_modes=n_modes)
    p_value = silverman_bootstrap(data, h_crit, B=B, n_modes=n_modes, seed=seed)
    # Modes at slightly below h_crit (to confirm the data itself has >= n_modes)
    n_obs = count_modes(data, h_crit * 0.99)
    return {
        "h_crit": float(h_crit),
        "p_value": float(p_value),
        "n_modes_observed": int(n_obs),
        "n_modes_tested": n_modes,
        "B": B,
    }
