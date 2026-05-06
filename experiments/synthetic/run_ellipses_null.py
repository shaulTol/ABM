#!/usr/bin/env python3
"""
MC Null Test + Dip Test for Parallel Ellipses DA Directions.

Compares observed DA eigenvalues against a pipeline-matched Gaussian null:
- Observed: bimodal GMM (means ± (0,3), cov diag(25,1))
- Null: single Gaussian with cov = mixture covariance Σ_hat

Also runs Hartigan's dip test on projections onto DA1 and PC1.

LOCAL run, ~10 min with B=1000.
"""
import sys
import os
import time
import argparse
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.metrics.da_directions import extract_da_directions
from src.metrics.autocorrelation import compute_stationary_moments
from src.simulation.langevin import LangevinSimulatorD

import diptest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1000, help='Number of null replicates')
    parser.add_argument('--lag', type=int, default=800)
    parser.add_argument('--n_traj', type=int, default=10000)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, B={args.B}, lag={args.lag}, N={args.n_traj}, dt={args.dt}")

    # --- Observed GMM ---
    means = torch.tensor([[0.0, -3.0], [0.0, 3.0]])
    covs = torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                          [[25.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([0.5, 0.5])

    # Compute mixture covariance (for null + PCA)
    mu_f, cov_f = compute_stationary_moments(means, covs, weights)
    print(f"Mixture cov:\n{cov_f.numpy()}")
    # cov_f should be [[25, 0], [0, 10]]

    # PCA from cov_f
    pca_evals, pca_evecs = torch.linalg.eigh(cov_f)
    pc1 = pca_evecs[:, -1].numpy()  # largest eigenvalue
    print(f"PC1: {pc1}")

    # --- Observed DA ---
    print("\nComputing observed DA directions...")
    t0 = time.time()
    obs_evals, obs_evecs, obs_C_tau = extract_da_directions(
        means, covs, weights,
        lag=args.lag, n_trajectories=args.n_traj, dt=args.dt,
        device=device, seed=args.seed,
    )
    obs_evals_np = obs_evals.cpu().numpy()
    obs_evecs_np = obs_evecs.cpu().numpy()
    da1 = obs_evecs_np[:, 0]
    print(f"  Observed eigenvalues: {obs_evals_np}")
    print(f"  DA1: {da1}  (time: {time.time()-t0:.1f}s)")

    # --- Null model: single Gaussian with cov = Σ_hat ---
    null_means = torch.tensor([[0.0, 0.0]])
    null_covs = cov_f.unsqueeze(0)  # (1, 2, 2)
    null_weights = torch.tensor([1.0])

    print(f"\nRunning {args.B} null replicates...")
    null_evals_all = np.zeros((args.B, 2))  # store all eigenvalues per replicate
    t0 = time.time()
    for b in range(args.B):
        evals_b, _, _ = extract_da_directions(
            null_means, null_covs, null_weights,
            lag=args.lag, n_trajectories=args.n_traj, dt=args.dt,
            device=device, seed=1000 + b,
        )
        null_evals_all[b] = evals_b.cpu().numpy()
        if (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {b+1}/{args.B} done ({elapsed:.0f}s, "
                  f"{elapsed/(b+1):.1f}s/rep)")

    elapsed = time.time() - t0
    print(f"Null replicates done in {elapsed:.0f}s ({elapsed/args.B:.2f}s/rep)")

    # --- p-values: direction i compared to i-th order statistic of null ---
    d = 2
    p_values = np.zeros(d)
    for i in range(d):
        obs_i = obs_evals_np[i]
        null_i = null_evals_all[:, i]  # i-th largest eigenvalue across replicates
        p_values[i] = (1 + np.sum(null_i >= obs_i)) / (1 + args.B)

    print(f"\nResults:")
    print(f"  Observed λ₁ = {obs_evals_np[0]:.6f}, null median = {np.median(null_evals_all[:,0]):.6f}, p = {p_values[0]:.4f}")
    print(f"  Observed λ₂ = {obs_evals_np[1]:.6f}, null median = {np.median(null_evals_all[:,1]):.6f}, p = {p_values[1]:.4f}")

    # --- Dip tests ---
    # Sample from stationary distribution for projections
    torch.manual_seed(args.seed)
    sim = LangevinSimulatorD(means, covs, weights, dt=args.dt, device=device)
    samples = sim.sample_initial(args.n_traj).cpu().numpy()

    proj_da1 = samples @ da1
    proj_pc1 = samples @ pc1

    dip_da1, pval_da1 = diptest.diptest(proj_da1)
    dip_pc1, pval_pc1 = diptest.diptest(proj_pc1)

    print(f"\nDip tests (N={args.n_traj}):")
    print(f"  DA1: dip={dip_da1:.4f}, p={pval_da1:.2e}")
    print(f"  PC1 (control): dip={dip_pc1:.4f}, p={pval_pc1:.2e}")

    # --- Save ---
    out_path = os.path.join(args.output_dir, 'ellipses_null_results.npz')
    np.savez(out_path,
             # Observed
             obs_eigenvalues=obs_evals_np,
             obs_da1=da1,
             obs_pc1=pc1,
             obs_C_tau=obs_C_tau.cpu().numpy(),
             # Null eigenvalues (B, d)
             null_eigenvalues=null_evals_all,
             # p-values
             p_values=p_values,
             # Dip test
             dip_da1=dip_da1, pval_da1=pval_da1,
             dip_pc1=dip_pc1, pval_pc1=pval_pc1,
             # Projections
             proj_da1=proj_da1, proj_pc1=proj_pc1,
             # Params
             B=args.B, lag=args.lag, n_traj=args.n_traj, dt=args.dt,
             seed=args.seed)
    print(f"\nSaved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"PARALLEL ELLIPSES MC NULL + DIP TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  DA1 angle from y-axis: {np.degrees(np.arccos(np.abs(da1[1]))):.1f}°")
    print(f"  PC1 angle from x-axis: {np.degrees(np.arccos(np.abs(pc1[0]))):.1f}°")
    print(f"  λ₁ MC p-value: {p_values[0]:.4f} ({'PASS' if p_values[0] < 0.05 else 'FAIL'} at α=0.05)")
    print(f"  DA1 dip p-value: {pval_da1:.2e} ({'PASS' if pval_da1 < 0.05 else 'FAIL'} at α=0.05)")
    print(f"  PC1 dip p-value: {pval_pc1:.2e} (control)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
