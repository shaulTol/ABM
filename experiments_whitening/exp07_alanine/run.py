#!/usr/bin/env python3
"""
Exp 7: Alanine Dipeptide — whitened variant.

Applies whitening to the 45D pairwise-distance space before running
KDE-DA. Empirical Sigma_hat from 2000 KDE centers.

Pipeline:
  1. Load alanine data (750k frames, 45D)
  2. Subsample 10k → 2k KDE centers
  3. Compute empirical Sigma_hat from 2k centers → WhiteningTransform
  4. Whiten centers → refit KDE in whitened space (new Scott bandwidth)
  5. Transform score to whitened coordinates
  6. Build GenericLangevinSimulator with var_f ≈ 1
  7. Extract DA directions at canonical lag τ = 10,000ps (100k steps at dt=0.1)
  8. Compare with unwhitened DA directions
"""
import sys, os, json, time
import numpy as np
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import torch
from src.metrics.da_directions import extract_da_directions
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore
from experiments_whitening.shared.whitening import WhiteningTransform

EXPDIR = os.path.dirname(os.path.abspath(__file__))
ALANINE_DIR = os.path.join(ROOT, 'experiments', 'da_directions')
RESDIR_ORIG = os.path.join(ALANINE_DIR, 'results')
OUT_DIR = os.path.join(EXPDIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + (phi, psi)."""
    data_dir = os.path.join(ALANINE_DIR, 'alanine_dipeptide', 'data')
    dist_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')

    if not os.path.exists(dist_file) or not os.path.exists(dihed_file):
        import mdshare
        os.makedirs(data_dir, exist_ok=True)
        dist_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
            working_directory=data_dir)
        dihed_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
            working_directory=data_dir)

    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    all_diheds = np.concatenate([dihed_data[f'arr_{i}'] for i in range(3)])

    return dict(
        all_dists=all_dists,
        ground_truth={'phi': all_diheds[:, 0], 'psi': all_diheds[:, 1]},
    )


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # ── 1. Load data ──
    print('[1/7] Loading alanine data...')
    d = load_alanine()
    all_dists = d['all_dists']
    N_total, dim = all_dists.shape
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean
    print(f'  {N_total:,} frames, {dim}D')

    # ── 2. Subsample ──
    print('\n[2/7] Subsampling...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    center_idx = np.random.choice(10000, 2000, replace=False)
    X_centers = X_sub[center_idx]
    print(f'  10k subsample → 2000 KDE centers')

    # ── 3. Whitening transform from 2k centers ──
    print('\n[3/7] Computing whitening transform...')
    wt = WhiteningTransform(X_centers)
    diag = wt.diagnostics()
    print(f'  cond_before={diag["cond_before"]:.1f}, cond_after={diag["cond_after"]:.1f}')
    print(f'  sigma_tilde_sq={diag["sigma_tilde_sq"]:.4f}')

    # Verify whitened covariance
    X_centers_tilde = wt.whiten(X_centers)
    cov_tilde = np.cov(X_centers_tilde, rowvar=False)
    cov_err = np.linalg.norm(cov_tilde - np.eye(dim), 'fro') / dim
    print(f'  ||Cov(X_tilde) - I||_F / d = {cov_err:.4f}')

    # Save whitening matrix
    wt.save(os.path.join(OUT_DIR, 'W_alanine.npz'))

    # ── 4. Build KDE in whitened space ──
    print('\n[4/7] Building KDE in whitened space...')
    X_sub_tilde = wt.whiten(X_sub)
    X_centers_tilde_t = torch.tensor(X_centers_tilde, dtype=torch.float32, device=device)
    X_sub_tilde_t = torch.tensor(X_sub_tilde, dtype=torch.float32, device=device)

    kde_tilde = KDEScore(X_centers_tilde_t, bandwidth='scott', device=device, batch_size=1000)
    print(f'  KDE bandwidth (whitened): h={kde_tilde.h:.4f}')

    # ── 5. dt stability test ──
    print('\n[5/7] dt stability test (whitened)...')
    best_dt = None
    for test_dt in [0.1, 0.01, 0.001]:
        sim_test = GenericLangevinSimulator(
            score_fn=kde_tilde.score, samples=X_sub_tilde_t,
            dt=test_dt, device=device)
        torch.manual_seed(0)
        x_test = sim_test.sample_initial(100)
        stable = True
        for step in range(1000):
            x_test = sim_test.step(x_test)
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f'  dt={test_dt}: UNSTABLE (NaN/Inf at step {step})')
                stable = False
                break
        if stable:
            final_norms = x_test.norm(dim=1)
            init_norms = sim_test.samples.norm(dim=1).mean()
            ratio = (final_norms.max() / init_norms).item()
            print(f'  dt={test_dt}: STABLE (var_f={sim_test.var_f:.4f})')
        if stable and best_dt is None:
            best_dt = test_dt

    if best_dt is None:
        best_dt = 0.001
    print(f'  -> Using dt={best_dt}')

    # ── 6. DA extraction in whitened space ──
    print('\n[6/7] DA extraction (whitened)...')
    kde_sim = GenericLangevinSimulator(
        score_fn=kde_tilde.score, samples=X_sub_tilde_t,
        dt=best_dt, device=device)
    print(f'  var_f (whitened) = {kde_sim.var_f:.4f}')

    n_traj = 2000
    # Canonical lag: tau = 10,000 ps → steps = 10000 / dt
    tau_phys = 10000.0  # ps
    da_lag = max(1, int(tau_phys / best_dt))
    print(f'  DA lag = {da_lag:,} steps (tau = {tau_phys:.0f} ps)')

    t0 = time.time()
    da_evals, da_evecs, C_tau = extract_da_directions(
        simulator=kde_sim, lag=da_lag, n_trajectories=n_traj,
        dt=best_dt, device=device, seed=42)
    elapsed = time.time() - t0
    print(f'  [{elapsed:.1f}s]')

    print(f'  Top-5 eigenvalues: {da_evals[:5].numpy()}')
    gap12 = da_evals[0].item() / da_evals[1].item() if da_evals[1].item() > 0 else float('inf')
    print(f'  lambda1/lambda2 = {gap12:.1f}x')

    # Project all data to whitened DA directions
    all_tilde = wt.whiten(all_dists)
    all_tilde_centered = all_tilde - all_tilde.mean(axis=0)
    da_proj = all_tilde_centered @ da_evecs.numpy()

    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(da_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  Whitened-DA{i+1}: {", ".join(gt_strs)}')

    # ── 7. Compare with original (unwhitened) DA ──
    print('\n[7/7] Comparison with original DA...')
    # Load original results if available
    orig_file = os.path.join(RESDIR_ORIG, 'alanine_da_results_kde.npz')
    if os.path.exists(orig_file):
        orig = np.load(orig_file, allow_pickle=True)
        orig_da1 = orig['kde_da1_evecs'][:, 0] if 'kde_da1_evecs' in orig else None
        if orig_da1 is not None:
            # Back-project whitened DA1 to original space for comparison
            # DA direction in whitened space v_tilde → original space: W_inv @ v_tilde
            da1_tilde = da_evecs[:, 0].numpy()
            da1_orig_space = wt.W_inv @ da1_tilde
            da1_orig_space /= np.linalg.norm(da1_orig_space)

            cos_sim = abs(np.dot(da1_orig_space, orig_da1) /
                          (np.linalg.norm(da1_orig_space) * np.linalg.norm(orig_da1)))
            print(f'  |cos(DA1_whitened, DA1_original)| = {cos_sim:.3f}')
    else:
        print(f'  Original results not found at {orig_file}, skipping comparison')

    # Save results
    results = {
        'tau_phys_ps': tau_phys,
        'dt': best_dt,
        'da_lag_steps': da_lag,
        'n_traj': n_traj,
        'gap12': gap12,
        'eigenvalues_top5': da_evals[:5].numpy().tolist(),
        'whitening_diagnostics': diag,
        'cov_err': float(cov_err),
        'total_time_s': time.time() - t_start,
    }
    # Add DA correlations with ground truth
    for i in range(3):
        for k, v in d['ground_truth'].items():
            r, p = pearsonr(da_proj[:, i], v)
            results[f'da{i+1}_r_{k}'] = float(r)
            results[f'da{i+1}_abs_r_{k}'] = float(abs(r))

    out_json = os.path.join(OUT_DIR, 'alanine_whitened_da.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_json}')

    np.savez(
        os.path.join(OUT_DIR, 'alanine_whitened_da.npz'),
        eigenvalues=da_evals.numpy(),
        eigenvectors=da_evecs.numpy(),
        C_tau=C_tau.numpy(),
        da_proj=da_proj[:, :5],
    )
    print(f'Total time: {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
