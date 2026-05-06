#!/usr/bin/env python3
"""
Gaussian baseline with empirical ρ² profile saved.

Runs long enough (T=2000, dt=0.01, max_lag=2000 → physical time 20s)
so that ρ² decays past the noise floor (~1/N at τ≈ln(N)≈9.2 for N=10000).

Saves:
  - Per-lag ρ² profile (empirical + analytic)
  - Cumulative Ŝ(τ) = ∫₀^τ ρ²(t)dt converging to 1
  - Noise floor crossing point τ*
  - Figure showing convergence
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

from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa
from ssa_redef import find_tau_star_from_rho_sq

# Parameters — long enough to reach noise floor
D = 10
VAR_F = 1.0
DT = 0.01
T = 2000
MAX_LAG = 2000  # physical time = 20s, ln(10000)≈9.2
N = 10000
SEED = 42


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, d={D}, N={N}, T={T}, dt={DT}, max_lag={MAX_LAG}")
    print(f"Physical time: {T*DT:.1f}s, noise floor τ* ≈ ln(N) = {np.log(N):.1f}s")

    # Single Gaussian = 1-component GMM
    means = torch.zeros(1, D)
    covs = (VAR_F * torch.eye(D)).unsqueeze(0)
    weights = torch.tensor([1.0])

    # Simulate
    torch.manual_seed(SEED)
    trajectories = simulate_d_trajectories(
        means.to(device), covs.to(device), weights.to(device),
        n_trajectories=N, T=T, dt=DT, device=device,
    )
    print(f"Trajectories: {trajectories.shape}")

    # Compute ρ
    rho, mu_f, cov_f = compute_vector_autocorrelation(
        trajectories, MAX_LAG,
        gmm_means=means.to(device), gmm_covs=covs.to(device),
        gmm_weights=weights.to(device),
        demean=True, normalization='trace',
    )
    rho_np = rho.cpu().numpy()
    rho_sq_emp = rho_np ** 2
    lags = np.arange(MAX_LAG + 1) * DT

    # Analytic
    rho_sq_ana = np.exp(-lags)  # ρ²(τ) = exp(-τ) for OU process

    # Find τ*
    tau_star_idx, tau_star = find_tau_star_from_rho_sq(
        rho_sq_emp, lags, DT, VAR_F, D, N
    )
    print(f"τ* = {tau_star:.2f}s (index {tau_star_idx}), ln(N) = {np.log(N):.2f}")

    # Cumulative Ŝ(τ) = ∫₀^τ ρ²(t)dt
    cumsum_emp = np.cumsum(rho_sq_emp[1:]) * DT
    cumsum_ana = np.cumsum(rho_sq_ana[1:]) * DT
    lags_cumsum = lags[1:]

    # Final values
    mask_trunc = lags[1:] <= tau_star
    s_hat_emp = aggregate_ssa(rho_sq_emp[1:][mask_trunc], dt=DT)
    s_hat_full = aggregate_ssa(rho_sq_emp[1:], dt=DT)
    s_old = aggregate_ssa(rho_sq_emp[1:], normalize='mean')

    print(f"Ŝ (truncated at τ*): {s_hat_emp:.6f}")
    print(f"Ŝ (full):            {s_hat_full:.6f}")
    print(f"S_old (mean):        {s_old:.6f}")
    print(f"Analytic ∫₀^∞ exp(-τ)dτ = 1.0")

    # Save results
    results_dir = os.path.join(BASEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'd': D, 'var_f': VAR_F, 'dt': DT, 'T': T, 'max_lag': MAX_LAG, 'N': N,
        'tau_star': float(tau_star),
        's_hat_trunc': float(s_hat_emp),
        's_hat_full': float(s_hat_full),
        's_old': float(s_old),
        'rho_sq_empirical': rho_sq_emp.tolist(),
        'rho_sq_analytic': rho_sq_ana.tolist(),
        'cumsum_empirical': cumsum_emp.tolist(),
        'cumsum_analytic': cumsum_ana.tolist(),
        'lags': lags.tolist(),
    }

    out_path = os.path.join(results_dir, 'gaussian_profile.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
