#!/usr/bin/env python3
"""
Exp 4: Dispersion vs Fragmentation — whitened variant.

Replicates figures/make_fig_dispersion_vs_fragmentation.py but with whitening.
Two distributions:
  - Unimodal blob: Sigma_f = 25 * I_2
  - Parallel ellipses: Sigma_f = diag(26.25, 10)

After whitening, both distributions have Cov(X_tilde) ≈ I.
We then run Langevin with var_f ≈ 1 and compare SSA + DA.
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import torch
from src.simulation.langevin import LangevinSimulatorD
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa
from src.metrics.da_directions import extract_da_directions
from experiments_whitening.shared.whitening import WhiteningTransform
from experiments_whitening.shared.canonical_diffusion_whitened import make_whitened_gmm_simulator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def compute_ssa_truncated(traj, mu_f_t, cov_f_t, gmm_means, gmm_covs, gmm_weights, dt, max_lag):
    """Compute Ŝ with 1/N noise floor truncation."""
    rho, _, _ = compute_vector_autocorrelation(
        traj, max_lag, gmm_means, gmm_covs, gmm_weights,
        demean=True, normalization='trace')
    rho_sq = (rho[1:] ** 2).cpu().numpy()
    N = traj.shape[0]
    nf = 1.0 / N
    tau_star_idx = len(rho_sq)
    for ell in range(len(rho_sq)):
        if rho_sq[ell] < nf:
            tau_star_idx = ell
            break
    S = aggregate_ssa(rho_sq[:tau_star_idx], dt=dt, normalize='riemann')
    return S, rho_sq


def run_whitened_ssa(means, covs, weights, Sigma_f, mu_f, label, dt=0.01, T=5000, max_lag=4000, n_traj=2000):
    """Run whitened Langevin and compute SSA."""
    print(f"\n--- {label} (whitened) ---")

    # Build whitening transform from analytic covariance
    wt = WhiteningTransform(analytic_cov=Sigma_f, mu=mu_f)
    print(f"  Whitening: cond_before={wt.cond_before:.1f}, cond_after={wt.cond_after:.1f}, "
          f"sigma_tilde_sq={wt.sigma_tilde_sq:.4f}")

    # Build whitened simulator
    sim_w = make_whitened_gmm_simulator(
        wt, means, covs, weights, dt=dt, device=device, n_samples=n_traj, seed=seed)

    # Verify whitened covariance ≈ I
    samples_tilde = sim_w.samples.cpu().numpy()
    cov_tilde = np.cov(samples_tilde, rowvar=False)
    cov_err = np.linalg.norm(cov_tilde - np.eye(cov_tilde.shape[0]), 'fro') / cov_tilde.shape[0]
    print(f"  ||Cov(X_tilde) - I||_F / d = {cov_err:.4f}")
    print(f"  var_f_tilde = {sim_w.var_f:.4f}")

    # Simulate whitened trajectories
    torch.manual_seed(seed + 100)
    x0 = sim_w.sample_initial(n_traj)
    traj = sim_w.simulate(x0, T)  # (n_traj, T+1, d)

    # For SSA computation, we need "GMM params" in whitened space.
    # Use empirical moments from whitened samples as stand-in.
    d = means.shape[1]
    mu_tilde = torch.zeros(d, device=device)
    cov_tilde_t = torch.eye(d, device=device)
    # Use a single-component "GMM" with identity covariance in whitened space
    gmm_means_w = mu_tilde.unsqueeze(0)
    gmm_covs_w = cov_tilde_t.unsqueeze(0)
    gmm_weights_w = torch.ones(1, device=device)

    rho, _, _ = compute_vector_autocorrelation(
        traj, max_lag, gmm_means_w, gmm_covs_w, gmm_weights_w,
        demean=True, normalization='trace')
    rho_sq = (rho[1:] ** 2).cpu().numpy()
    N = n_traj
    nf = 1.0 / N
    tau_star_idx = len(rho_sq)
    for ell in range(len(rho_sq)):
        if rho_sq[ell] < nf:
            tau_star_idx = ell
            break
    ssa = aggregate_ssa(rho_sq[:tau_star_idx], dt=dt, normalize='riemann')
    tau_star = (tau_star_idx + 1) * dt if tau_star_idx < len(rho_sq) else None

    print(f"  SSA_whitened = {ssa:.4f}, tau* = {tau_star}")

    return {
        'label': label,
        'ssa_whitened': float(ssa),
        'tau_star': tau_star,
        'sigma_tilde_sq': wt.sigma_tilde_sq,
        'cov_err': float(cov_err),
        'var_f_tilde': float(sim_w.var_f),
        'diagnostics': wt.diagnostics(),
    }


def run_original_ssa(means, covs, weights, label, dt=0.01, T=5000, max_lag=4000, n_traj=2000):
    """Run original (unwhitened) Langevin for comparison."""
    print(f"\n--- {label} (original) ---")

    from src.simulation.langevin import simulate_d_trajectories
    torch.manual_seed(seed + 100)
    traj = simulate_d_trajectories(
        means, covs, weights, n_trajectories=n_traj, T=T, dt=dt,
        device=device, seed=seed + 100)

    rho, mu_f, cov_f = compute_vector_autocorrelation(
        traj, max_lag, means, covs, weights,
        demean=True, normalization='trace')
    rho_sq = (rho[1:] ** 2).cpu().numpy()
    N = n_traj
    nf = 1.0 / N
    tau_star_idx = len(rho_sq)
    for ell in range(len(rho_sq)):
        if rho_sq[ell] < nf:
            tau_star_idx = ell
            break
    ssa = aggregate_ssa(rho_sq[:tau_star_idx], dt=dt, normalize='riemann')
    tau_star = (tau_star_idx + 1) * dt if tau_star_idx < len(rho_sq) else None

    print(f"  SSA_original = {ssa:.4f}, tau* = {tau_star}")
    return {'label': label, 'ssa_original': float(ssa), 'tau_star': tau_star}


def main():
    dt = 0.01
    T = 5000
    max_lag = 4000
    n_traj = 2000

    # ── Parallel ellipses (GMM) ──
    means_gmm = torch.tensor([[0.0, -3.0], [0.0, 3.0]])
    covs_gmm = torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                              [[25.0, 0.0], [0.0, 1.0]]])
    weights_gmm = torch.tensor([0.5, 0.5])

    # Analytic Sigma_f for GMM
    mu_bar = (weights_gmm.unsqueeze(1) * means_gmm).sum(0).numpy()
    Sigma_f_gmm = np.zeros((2, 2))
    for i in range(2):
        diff = means_gmm[i].numpy() - mu_bar
        Sigma_f_gmm += weights_gmm[i].item() * (covs_gmm[i].numpy() + np.outer(diff, diff))
    print(f"GMM Sigma_f = diag({Sigma_f_gmm[0,0]:.2f}, {Sigma_f_gmm[1,1]:.2f})")

    # ── Unimodal blob ──
    means_blob = torch.tensor([[0.0, 0.0]])
    covs_blob = torch.tensor([[[25.0, 0.0], [0.0, 25.0]]])
    weights_blob = torch.tensor([1.0])
    Sigma_f_blob = np.array([[25.0, 0.0], [0.0, 25.0]])
    mu_blob = np.array([0.0, 0.0])

    # Run whitened
    res_gmm_w = run_whitened_ssa(means_gmm, covs_gmm, weights_gmm,
                                  Sigma_f_gmm, mu_bar, "Parallel Ellipses",
                                  dt=dt, T=T, max_lag=max_lag, n_traj=n_traj)
    res_blob_w = run_whitened_ssa(means_blob, covs_blob, weights_blob,
                                  Sigma_f_blob, mu_blob, "Unimodal Blob",
                                  dt=dt, T=T, max_lag=max_lag, n_traj=n_traj)

    # Run original
    res_gmm_o = run_original_ssa(means_gmm, covs_gmm, weights_gmm,
                                  "Parallel Ellipses",
                                  dt=dt, T=T, max_lag=max_lag, n_traj=n_traj)
    res_blob_o = run_original_ssa(means_blob, covs_blob, weights_blob,
                                   "Unimodal Blob",
                                   dt=dt, T=T, max_lag=max_lag, n_traj=n_traj)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON: Original vs Whitened SSA")
    print("=" * 60)
    print(f"Parallel Ellipses:  SSA_orig = {res_gmm_o['ssa_original']:.4f}  "
          f"SSA_whit = {res_gmm_w['ssa_whitened']:.4f}")
    print(f"Unimodal Blob:      SSA_orig = {res_blob_o['ssa_original']:.4f}  "
          f"SSA_whit = {res_blob_w['ssa_whitened']:.4f}")
    print(f"GMM > Blob (orig):  {res_gmm_o['ssa_original'] > res_blob_o['ssa_original']}")
    print(f"GMM > Blob (whit):  {res_gmm_w['ssa_whitened'] > res_blob_w['ssa_whitened']}")

    # Save
    results = {
        'gmm_whitened': res_gmm_w,
        'blob_whitened': res_blob_w,
        'gmm_original': res_gmm_o,
        'blob_original': res_blob_o,
        'dt': dt, 'T': T, 'max_lag': max_lag, 'n_traj': n_traj,
    }
    out_json = os.path.join(OUT_DIR, 'dispersion_whitened.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == '__main__':
    main()
