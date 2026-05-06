#!/usr/bin/env python3
"""
Gaussian Baseline Sanity Check.

For isotropic Gaussian N(0, σ²I), the OU process gives:
  ρ(τ) = exp(-τ/2), so ρ²(τ) = exp(-τ).

Both S_old and S_hat should be small (no multimodality).
S_hat should converge to the analytic Riemann integral.

Runs both analytic computation and a numerical simulation.
"""
import sys
import os
import json
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)

from ssa_redef import gaussian_baseline, compute_both_ssa, find_tau_star_from_rho_sq
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation, compute_stationary_moments


def run_analytic():
    """Compute analytic S_old and S_hat for Gaussian."""
    print("=== Analytic Gaussian Baseline ===")

    configs = [
        {'d': 2, 'var_f': 1.0, 'dt': 0.01, 'T': 1000, 'max_lag': 500, 'label': '2D, dt=0.01'},
        {'d': 10, 'var_f': 1.0, 'dt': 0.01, 'T': 1000, 'max_lag': 500, 'label': '10D, dt=0.01'},
        {'d': 768, 'var_f': 0.27, 'dt': 0.0005, 'T': 2000, 'max_lag': 2000, 'label': 'CLIP-like'},
    ]

    results = {}
    for cfg in configs:
        res = gaussian_baseline(cfg['d'], cfg['var_f'], cfg['dt'], cfg['T'], cfg['max_lag'])
        print(f"  {cfg['label']}: S_old={res['s_old_analytic']:.6f}, "
              f"S_hat={res['s_hat_analytic']:.6f}, S_hat_∞={res['s_hat_infinite']:.6f}")
        results[cfg['label']] = {**res, **cfg}

    return results


def run_numerical(d=2, var_f=1.0, dt=0.01, T=1000, max_lag=500, N=10000, seed=42):
    """Run numerical simulation and compare."""
    print(f"\n=== Numerical Gaussian Baseline (d={d}, N={N}) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Single-component GMM = Gaussian
    means = torch.zeros(1, d)
    covs = (var_f * torch.eye(d)).unsqueeze(0)
    weights = torch.tensor([1.0])

    torch.manual_seed(seed)
    trajectories = simulate_d_trajectories(
        means.to(device), covs.to(device), weights.to(device),
        n_trajectories=N, T=T, dt=dt, device=device,
    )

    rho, mu_f, cov_f = compute_vector_autocorrelation(
        trajectories, max_lag,
        gmm_means=means.to(device), gmm_covs=covs.to(device),
        gmm_weights=weights.to(device),
        demean=True, normalization='trace',
    )
    rho_np = rho.cpu().numpy()
    rho_sq = rho_np ** 2
    lags = np.arange(max_lag + 1) * dt

    # Find τ* and compute both
    tau_star_idx, tau_star = find_tau_star_from_rho_sq(
        rho_sq, lags, dt, var_f, d, N
    )
    both = compute_both_ssa(rho_sq, lags, dt, tau_star)

    # Analytic comparison
    analytic = gaussian_baseline(d, var_f, dt, T, max_lag)

    print(f"  Numerical: S_old={both['s_old']:.6f}, S_hat={both['s_hat']:.6f}, τ*={tau_star:.2f}")
    print(f"  Analytic:  S_old={analytic['s_old_analytic']:.6f}, S_hat={analytic['s_hat_analytic']:.6f}")
    print(f"  Difference: ΔS_old={abs(both['s_old'] - analytic['s_old_analytic']):.6f}, "
          f"ΔS_hat={abs(both['s_hat'] - analytic['s_hat_analytic']):.6f}")

    return {
        'numerical': both,
        'analytic': {
            's_old': analytic['s_old_analytic'],
            's_hat': analytic['s_hat_analytic'],
            's_hat_inf': analytic['s_hat_infinite'],
        },
        'tau_star': tau_star,
        'd': d,
        'var_f': var_f,
        'dt': dt,
        'T': T,
        'N': N,
    }


def main():
    results_dir = os.path.join(BASEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {}
    results['analytic'] = run_analytic()
    results['numerical_2d'] = run_numerical(d=2)
    results['numerical_10d'] = run_numerical(d=10)

    out_path = os.path.join(results_dir, 'gaussian_baseline.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Sanity checks
    for key in ['numerical_2d', 'numerical_10d']:
        r = results[key]
        s_old = r['numerical']['s_old']
        s_hat = r['numerical']['s_hat']
        assert s_old < 0.1, f"Gaussian S_old should be small, got {s_old}"
        print(f"\n  ✓ {key}: S_old={s_old:.6f} < 0.1 (unimodal → low SSA)")


if __name__ == '__main__':
    main()
