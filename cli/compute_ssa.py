#!/usr/bin/env python3
"""
Shared pipeline: given saved C(τ), compute SSA (Ŝ) + optional bootstrap SE.

Reads an NPZ file produced by cli/save_lags.py and computes the paper's
estimator (eq:ssa_estimator, eq:tmax_stopping):
    ρ̂(τ) = tr Ĉ(τ) / tr Ĉ(0),
    Ŝ = Δt Σ_{ℓ=1}^{L*} ρ̂(ℓΔt)²,
with L* the first lag at which ρ̂² ≤ 1/N (that lag included; the full lag grid
if ρ̂² never falls to the floor, in which case Ŝ is a finite-horizon lower bound).

Usage (as library):
    from cli.compute_ssa import compute_ssa, bootstrap_ssa

Usage (as CLI):
    python -m cli.compute_ssa results/alanine_C_tau_full_h_half_10k.npz [--bootstrap B]
"""
import os, sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.autocorrelation import aggregate_ssa


def ssa_from_rho_sq(rho_sq, lags, dt, N):
    """Ŝ from ρ̂² at positive integer lags, truncated at the first 1/N crossing.

    Args:
        rho_sq: (L,) ρ̂(τ)² at lags `lags` (lag 0 excluded).
        lags: (L,) increasing positive integer lags (in steps).
        dt: time step.
        N: number of trajectories (floor 1/N).

    Returns:
        (s_hat, n_used, converged): n_used lags integrated; converged is False if
        ρ̂² never reached 1/N on the grid.
    """
    rho_sq = np.asarray(rho_sq, dtype=float)
    lags = np.asarray(lags)
    below = np.where(rho_sq <= 1.0 / N)[0]
    n_used = int(below[0]) + 1 if len(below) else len(rho_sq)
    if np.array_equal(lags[:n_used], np.arange(1, n_used + 1)):
        s_hat = aggregate_ssa(rho_sq[:n_used], dt=dt)          # Δt Σ ρ̂², every step
    else:
        s_hat = aggregate_ssa(rho_sq[:n_used], lags=lags[:n_used] * dt)  # sparse grid: trapezoid
    return float(s_hat), n_used, bool(len(below))


def _rho_sq_from_traces(traces, lags):
    if lags[0] != 0:
        raise ValueError("lag 0 must be the first saved lag (it gives tr Ĉ(0))")
    return (traces[1:] / traces[0]) ** 2


def compute_ssa(C_tau_path):
    """Compute SSA (Ŝ) from saved autocovariance data.

    Args:
        C_tau_path: path to NPZ from save_lags.py (lag 0 must be saved).

    Returns:
        dict with keys: s_hat, rho_sq, phys_times, tau_star, n_lags_used, converged.
        rho_sq and phys_times exclude lag 0.
    """
    data = np.load(C_tau_path)
    eigenvalues = data['eigenvalues']   # (n_lags, d)
    phys_times = data['phys_times']     # (n_lags,)
    N = int(data['N'])
    dt = float(data['dt'])
    lags = np.rint(phys_times / dt).astype(int)

    rho_sq = _rho_sq_from_traces(eigenvalues.sum(axis=1), lags)   # tr C_sym = tr C
    s_hat, n_used, converged = ssa_from_rho_sq(rho_sq, lags[1:], dt, N)

    return {
        's_hat': s_hat,
        'rho_sq': rho_sq,
        'phys_times': phys_times[1:],
        'tau_star': float(phys_times[n_used]),
        'n_lags_used': n_used,
        'converged': converged,
    }


def bootstrap_ssa(C_tau_path, B=2000, seed=42):
    """Trajectory bootstrap SE for SSA.

    Resamples the N trajectories with replacement and recomputes Ŝ end-to-end
    (pooled mean, trajectory-averaged Ĉ(τ), 1/N stopping rule, sum) for each of
    the B replicates. Requires per-trajectory data (Z_all; see
    save_lags(..., save_trajectories=True)).

    Returns:
        dict with keys: s_hat, se, ci_lo, ci_hi, s_hat_boots.
    """
    from cli.save_lags import compute_symmetrized_autocovariance

    data = np.load(C_tau_path)
    if 'Z_all' not in data:
        raise ValueError("NPZ must contain 'Z_all' (save_lags(..., save_trajectories=True))")
    base = compute_ssa(C_tau_path)
    Z_all = data['Z_all']   # (N, T+1, d)
    N = Z_all.shape[0]
    dt = float(data['dt'])
    lags = np.rint(data['phys_times'] / dt).astype(int)

    rng = np.random.RandomState(seed)
    boots = np.zeros(B)
    for b in range(B):
        Z_b = Z_all[rng.choice(N, N, replace=True)]
        Z_b = Z_b - Z_b.mean(axis=(0, 1))
        C_b = compute_symmetrized_autocovariance(Z_b, lags)
        rho_sq_b = _rho_sq_from_traces(np.trace(C_b, axis1=1, axis2=2), lags)
        boots[b] = ssa_from_rho_sq(rho_sq_b, lags[1:], dt, N)[0]

    return {
        's_hat': base['s_hat'],
        'se': float(np.std(boots, ddof=1)),
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
        's_hat_boots': boots,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m cli.compute_ssa <C_tau.npz> [--bootstrap B]')
        sys.exit(1)
    path = sys.argv[1]
    result = compute_ssa(path)
    print(f"Ŝ = {result['s_hat']:.4f}  (τ* = {result['tau_star']:.3f}, "
          f"{result['n_lags_used']} lags used"
          f"{'' if result['converged'] else '; 1/N floor not reached, lower bound'})")

    if '--bootstrap' in sys.argv:
        idx = sys.argv.index('--bootstrap')
        B = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 2000
        boot = bootstrap_ssa(path, B=B)
        print(f"Bootstrap (B={B}): Ŝ = {boot['s_hat']:.4f} ± {boot['se']:.4f} "
              f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")
