#!/usr/bin/env python3
"""
Permutation Null Tests for DA Directions.

Tests whether observed C(τ*) eigenvalues are consistent with independent
(X_0, X_τ*) pairs. Permuting the pairing breaks lag-τ* coupling; the
permuted eigenvalues should cluster near zero if there is real coupling.

Pure numpy — no torch required, no simulation needed, runs in seconds.

Applies to all four directions:
  1. Parallel ellipses DA1 (lag=800)
  2. Alanine DA1 (τ*=10,000ps)
  3. Alanine DA2 (τ*=10,000ps)
  4. SDXL elderly DA1 (τ*=2000)

For ellipses: re-simulates (X_0, X_τ) via numpy GMM Langevin.
For alanine: uses raw MD trajectory pairs (same marginal as KDE-Langevin).
For SDXL: reconstructs z_τ from z_0 and C(τ*) via pseudoinverse.
"""
import sys
import os
import time
import argparse
import numpy as np
from scipy import stats as sp_stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


# ================================================================
# Permutation test engine
# ================================================================

def permutation_null_eigenvalues(X0_c, Xtau_c, B=10000, n_eigvals=5, seed=42):
    """
    Compute null eigenvalue distribution by permuting the (X_0, X_τ) pairing.

    Args:
        X0_c: (N, d) centered X_0 samples
        Xtau_c: (N, d) centered X_τ samples (same N)
        B: number of permutation replicates
        n_eigvals: number of top eigenvalues to store
        seed: random seed

    Returns:
        null_evals: (B, n_eigvals) top eigenvalues per permutation
    """
    N, d = X0_c.shape
    rng = np.random.RandomState(seed)
    m = min(n_eigvals, min(N, d))
    null_evals = np.zeros((B, m))

    for b in range(B):
        perm = rng.permutation(N)
        C_perm = Xtau_c[perm].T @ X0_c / N  # (d, d)
        C_perm_sym = 0.5 * (C_perm + C_perm.T)

        if d <= N:
            evals = np.linalg.eigvalsh(C_perm_sym)[::-1]  # descending
        else:
            # Dual trick for d > N: eigenvalues of (d,d) = eigenvalues of (N,N)
            # C_sym = 0.5(Xtau_perm^T X0 + X0^T Xtau_perm) / N
            # Use the N×N formulation
            M = (Xtau_c[perm].T @ X0_c + X0_c.T @ Xtau_c[perm]) / (2 * N)
            evals = np.linalg.eigvalsh(M)[::-1]

        null_evals[b] = evals[:m].copy()

    return null_evals


def compute_pvalue(obs, null_array):
    """One-sided MC p-value: P(null >= obs)."""
    B = len(null_array)
    return (1 + np.sum(null_array >= obs)) / (1 + B)


# ================================================================
# 1. Parallel Ellipses
# ================================================================

def gmm_score_2d(x, means, covs_inv, weights, log_norms):
    """Score ∇log f(x) for 2D GMM. x: (N, 2), returns (N, 2)."""
    N = x.shape[0]
    K = len(weights)
    # Compute responsibilities and weighted gradients
    log_w = np.zeros((N, K))
    grads = np.zeros((N, K, 2))
    for k in range(K):
        diff = x - means[k]  # (N, 2)
        # log N(x; mu_k, Sigma_k) = -0.5 diff^T Sigma_inv diff + log_norm
        maha = np.sum(diff @ covs_inv[k] * diff, axis=1)  # (N,)
        log_w[:, k] = -0.5 * maha + log_norms[k] + np.log(weights[k])
        grads[:, k] = -diff @ covs_inv[k]  # (N, 2)

    # Log-sum-exp for stability
    max_log_w = log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w - max_log_w)
    w /= w.sum(axis=1, keepdims=True)  # (N, K) responsibilities

    # Weighted sum of gradients
    score = np.sum(w[:, :, None] * grads, axis=1)  # (N, 2)
    return score


def run_ellipses_permutation(B_perm=10000, lag=800, N=10000, dt=0.01, seed=42):
    """Re-simulate ellipses GMM and run permutation test."""
    print("=== Parallel Ellipses Permutation Test ===")
    means = np.array([[0.0, -3.0], [0.0, 3.0]])
    covs = np.array([[[25.0, 0.0], [0.0, 1.0]],
                      [[25.0, 0.0], [0.0, 1.0]]])
    weights = np.array([0.5, 0.5])

    # Precompute
    covs_inv = np.array([np.linalg.inv(c) for c in covs])
    log_norms = np.array([-0.5 * np.log(np.linalg.det(2 * np.pi * c)) for c in covs])

    # Mixture covariance for var_f
    mu_f = (weights[:, None] * means).sum(0)
    cov_f = np.zeros((2, 2))
    for k in range(2):
        diff = means[k] - mu_f
        cov_f += weights[k] * (covs[k] + np.outer(diff, diff))
    var_f = np.trace(cov_f) / 2.0
    sigma_f = np.sqrt(var_f)

    # Sample X_0 from mixture
    rng = np.random.RandomState(seed)
    comp = rng.choice(2, size=N, p=weights)
    X0 = np.zeros((N, 2))
    for k in range(2):
        mask = comp == k
        nk = mask.sum()
        X0[mask] = rng.multivariate_normal(means[k], covs[k], size=nk)

    # Langevin simulation: lag steps
    x = X0.copy()
    sqrt_dt = np.sqrt(dt)
    t0 = time.time()
    for t in range(lag):
        score = gmm_score_2d(x, means, covs_inv, weights, log_norms)
        x = x + 0.5 * var_f * dt * score + sigma_f * sqrt_dt * rng.randn(N, 2)
    X_tau = x
    print(f"  Simulated {N} trajectories × {lag} steps in {time.time()-t0:.1f}s")

    # Observed C(τ)
    X0_c = X0 - mu_f
    Xtau_c = X_tau - mu_f
    C_obs = Xtau_c.T @ X0_c / N
    C_obs_sym = 0.5 * (C_obs + C_obs.T)
    obs_evals = np.linalg.eigvalsh(C_obs_sym)[::-1].copy()
    print(f"  Observed eigenvalues: {obs_evals}")

    # PCA
    C0 = X0_c.T @ X0_c / N
    pca_evals, pca_evecs = np.linalg.eigh(C0)
    pc1 = pca_evecs[:, -1]
    da1 = np.linalg.eigh(C_obs_sym)[1][:, -1]
    # Sign convention: DA1 ≈ y-axis
    if abs(da1[1]) < abs(da1[0]):
        da1 = np.linalg.eigh(C_obs_sym)[1][:, 0]

    # Permutation null
    t0 = time.time()
    null_evals = permutation_null_eigenvalues(X0_c, Xtau_c, B=B_perm, n_eigvals=2, seed=seed+1)
    print(f"  Permutation null ({B_perm} reps) in {time.time()-t0:.1f}s")

    p_da1 = compute_pvalue(obs_evals[0], null_evals[:, 0])
    print(f"  λ₁^obs={obs_evals[0]:.4f}, null median={np.median(null_evals[:,0]):.4f}, p={p_da1:.4e}")

    return {
        'obs_evals': obs_evals,
        'null_evals': null_evals,
        'p_da1': p_da1,
        'X0_c': X0_c, 'Xtau_c': Xtau_c,
        'da1': da1, 'pc1': pc1,
    }


# ================================================================
# 2. Alanine
# ================================================================

def run_alanine_permutation(B_perm=10000, N_pairs=2000, lag_frames=10000, seed=42):
    """Permutation test on alanine MD trajectory pairs."""
    print("\n=== Alanine Permutation Test ===")

    # Load observed eigenvalues from saved C(τ) at τ=10,000ps
    ctau_path = os.path.join(project_root,
        'experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz')
    ctau = np.load(ctau_path)
    C_tau_all = ctau['C_tau']  # (38, 45, 45)
    C0 = np.array(ctau['C0'], dtype=np.float64)
    phys_times = ctau['phys_times']
    gt_phi = ctau['gt_phi']
    gt_psi = ctau['gt_psi']

    # Use last lag (τ=10,000ps)
    tau_idx = len(phys_times) - 1
    tau_ps = phys_times[tau_idx]
    C_tau_obs = C_tau_all[tau_idx]
    C_tau_obs_sym = 0.5 * (C_tau_obs + C_tau_obs.T)
    obs_evals = np.linalg.eigvalsh(C_tau_obs_sym)[::-1].copy()
    obs_evecs = np.linalg.eigh(C_tau_obs_sym)[1][:, ::-1].copy()
    print(f"  τ={tau_ps:.0f}ps, obs λ₁={obs_evals[0]:.6f}, λ₂={obs_evals[1]:.6f}")

    # PCA from C(0)
    C0_sym = 0.5 * (C0 + C0.T)
    pca_evals, pca_evecs = np.linalg.eigh(C0_sym)
    pc1 = pca_evecs[:, -1]

    da1 = obs_evecs[:, 0]
    da2 = obs_evecs[:, 1]

    # Load MD data for permutation pairs
    print("  Loading MD trajectory...")
    # Try local cache first, then mdshare
    local_filename = os.path.join(project_root,
        'experiments/da_directions/alanine_dipeptide/data',
        'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    if not os.path.exists(local_filename):
        import mdshare
        local_filename = mdshare.fetch(
            'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
            working_directory='data'
        )
    with np.load(local_filename) as fh:
        data = np.concatenate([fh[f'arr_{i}'] for i in range(3)])  # (750000, 45)

    N_total = data.shape[0]
    d = data.shape[1]
    print(f"  MD data: {data.shape}, lag_frames={lag_frames}")

    # Extract (X_0, X_tau) pairs from MD trajectory
    max_pairs = N_total - lag_frames
    stride = max(1, max_pairs // N_pairs)
    indices = np.arange(0, max_pairs, stride)[:N_pairs]
    X0 = data[indices]              # (N_pairs, 45)
    Xtau = data[indices + lag_frames]  # (N_pairs, 45)
    N_actual = X0.shape[0]
    print(f"  Extracted {N_actual} MD pairs at stride={stride}")

    mu = X0.mean(axis=0)
    X0_c = X0 - mu
    Xtau_c = Xtau - mu

    # Permutation null
    t0 = time.time()
    null_evals = permutation_null_eigenvalues(X0_c, Xtau_c, B=B_perm, n_eigvals=5, seed=seed+2)
    print(f"  Permutation null ({B_perm} reps) in {time.time()-t0:.1f}s")

    p_da1 = compute_pvalue(obs_evals[0], null_evals[:, 0])
    p_da2 = compute_pvalue(obs_evals[1], null_evals[:, 1])
    print(f"  DA1: λ₁^obs={obs_evals[0]:.6f}, null median={np.median(null_evals[:,0]):.6f}, p={p_da1:.4e}")
    print(f"  DA2: λ₂^obs={obs_evals[1]:.6f}, null median={np.median(null_evals[:,1]):.6f}, p={p_da2:.4e}")

    # Correlations with CVs for sign convention
    proj_da1 = data @ da1
    r_da1_psi = np.corrcoef(proj_da1, gt_psi)[0, 1]
    proj_da2 = data @ da2
    r_da2_phi = np.corrcoef(proj_da2, gt_phi)[0, 1]

    return {
        'obs_evals': obs_evals[:5],
        'null_evals': null_evals,
        'p_da1': p_da1,
        'p_da2': p_da2,
        'da1': da1, 'da2': da2, 'pc1': pc1,
        'N_pairs': N_actual, 'tau_ps': tau_ps,
        'r_da1_psi': r_da1_psi, 'r_da2_phi': r_da2_phi,
    }


# ================================================================
# 3. SDXL Elderly
# ================================================================

def run_sdxl_permutation(B_perm=10000, seed=42, z_tau_clip_path=None):
    """Permutation test on SDXL elderly DA1.

    Requires z_tau CLIP embeddings (200, 768) saved as a .npz file.
    If not provided, looks for a default path. If unavailable, the SDXL
    permutation test is skipped (pseudoinverse reconstruction is too lossy
    because C(τ*) is stored in float32 and d > N).

    Args:
        z_tau_clip_path: path to npz with key 'z_tau' shape (200, 768)
    """
    print("\n=== SDXL Elderly Permutation Test ===")

    # Load observed C(τ*) and eigenvalues
    da_path = os.path.join(project_root,
        'experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz')
    da = np.load(da_path)
    C_tau_star = da['C_tau_star'].astype(np.float64)  # (768, 768)
    mu_clip = da['mu_clip']  # (768,)
    obs_evals_full = da['eigenvalues_full_at_tau_star']  # (768,) descending
    tau_star = int(da['tau_star_freeconv'])
    da1_eigvec = da['da1_eigvec']

    print(f"  τ*={tau_star}, obs λ₁={obs_evals_full[0]:.6f}, λ₂={obs_evals_full[1]:.6f}")

    # Load z_0 CLIP embeddings
    z0_path = os.path.join(project_root,
        'experiments/sdxl/results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_clip.npz')
    z0 = np.load(z0_path)['z0'].astype(np.float64)  # (200, 768)
    N, d = z0.shape
    z0_c = z0 - mu_clip.astype(np.float64)  # (200, 768)

    # Load z_tau CLIP embeddings
    if z_tau_clip_path is None:
        z_tau_clip_path = os.path.join(project_root,
            'experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_z_tau.npz')
    if not os.path.exists(z_tau_clip_path):
        print(f"  z_tau CLIP embeddings not found at {z_tau_clip_path}")
        print("  Pseudoinverse reconstruction is too lossy (float32 C, d>N).")
        print("  SKIPPING SDXL permutation test. Need to save z_tau from cluster.")
        return None

    z_tau = np.load(z_tau_clip_path)['z_tau'].astype(np.float64)  # (200, 768)
    z_tau_c = z_tau - mu_clip.astype(np.float64)
    print(f"  z0: {z0.shape}, z_tau: {z_tau.shape}")

    # Verify: C from loaded data should match stored C(τ*)
    C_check = z_tau_c.T @ z0_c / N
    C_check_sym = 0.5 * (C_check + C_check.T)
    check_evals = np.linalg.eigvalsh(C_check_sym)[::-1]
    check_err = np.max(np.abs(check_evals[:5] - obs_evals_full[:5]))
    print(f"  Cross-check error (top-5 evals): {check_err:.2e}")

    # Permutation null
    t0 = time.time()
    null_evals = permutation_null_eigenvalues(z0_c, z_tau_c, B=B_perm, n_eigvals=10, seed=seed+3)
    print(f"  Permutation null ({B_perm} reps) in {time.time()-t0:.1f}s")

    p_da1 = compute_pvalue(obs_evals_full[0], null_evals[:, 0])
    print(f"  DA1: λ₁^obs={obs_evals_full[0]:.6f}, null median={np.median(null_evals[:,0]):.6f}, p={p_da1:.4e}")

    return {
        'obs_evals': obs_evals_full[:10],
        'null_evals': null_evals,
        'p_da1': p_da1,
        'tau_star': tau_star,
        'da1': da1_eigvec,
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--skip_ellipses', action='store_true')
    parser.add_argument('--skip_alanine', action='store_true')
    parser.add_argument('--skip_sdxl', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # 1. Ellipses
    if not args.skip_ellipses:
        ell = run_ellipses_permutation(B_perm=args.B)
        np.savez(os.path.join(args.output_dir, 'ellipses_perm_null.npz'),
                 obs_eigenvalues=ell['obs_evals'],
                 null_eigenvalues=ell['null_evals'],
                 p_da1=ell['p_da1'],
                 da1=ell['da1'], pc1=ell['pc1'],
                 B=args.B)
        results['ellipses'] = ell

    # 2. Alanine
    if not args.skip_alanine:
        ala = run_alanine_permutation(B_perm=args.B)
        np.savez(os.path.join(args.output_dir, 'alanine_perm_null.npz'),
                 obs_eigenvalues=ala['obs_evals'],
                 null_eigenvalues=ala['null_evals'],
                 p_da1=ala['p_da1'], p_da2=ala['p_da2'],
                 da1=ala['da1'], da2=ala['da2'], pc1=ala['pc1'],
                 N_pairs=ala['N_pairs'], tau_ps=ala['tau_ps'],
                 r_da1_psi=ala['r_da1_psi'], r_da2_phi=ala['r_da2_phi'],
                 B=args.B)
        results['alanine'] = ala

    # 3. SDXL
    if not args.skip_sdxl:
        sdxl = run_sdxl_permutation(B_perm=args.B)
        np.savez(os.path.join(args.output_dir, 'sdxl_perm_null.npz'),
                 obs_eigenvalues=sdxl['obs_evals'],
                 null_eigenvalues=sdxl['null_evals'],
                 p_da1=sdxl['p_da1'],
                 tau_star=sdxl['tau_star'],
                 da1=sdxl['da1'],
                 recon_err=sdxl['recon_err'],
                 B=args.B)
        results['sdxl'] = sdxl

    # Summary table
    print(f"\n{'='*80}")
    print(f"PERMUTATION NULL TEST SUMMARY (B={args.B})")
    print(f"{'='*80}")
    header = f"{'Experiment':<20} {'Direction':<8} {'λ^obs':>10} {'Null med':>10} {'p-value':>10} {'α':>6} {'Pass':>6}"
    print(header)
    print('-' * 80)

    if 'ellipses' in results:
        e = results['ellipses']
        p = 'Yes' if e['p_da1'] < 0.05 else 'No'
        print(f"{'Parallel ellipses':<20} {'DA1':<8} {e['obs_evals'][0]:>10.4f} "
              f"{np.median(e['null_evals'][:,0]):>10.4f} {e['p_da1']:>10.4e} {'0.050':>6} {p:>6}")

    if 'alanine' in results:
        a = results['alanine']
        for i, (name, p_val) in enumerate([('DA1', a['p_da1']), ('DA2', a['p_da2'])]):
            p = 'Yes' if p_val < 0.025 else 'No'
            print(f"{'Alanine':<20} {name:<8} {a['obs_evals'][i]:>10.6f} "
                  f"{np.median(a['null_evals'][:,i]):>10.6f} {p_val:>10.4e} {'0.025':>6} {p:>6}")

    if 'sdxl' in results:
        s = results['sdxl']
        p = 'Yes' if s['p_da1'] < 0.05 else 'No'
        print(f"{'SDXL elderly':<20} {'DA1':<8} {s['obs_evals'][0]:>10.6f} "
              f"{np.median(s['null_evals'][:,0]):>10.6f} {s['p_da1']:>10.4e} {'0.050':>6} {p:>6}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
