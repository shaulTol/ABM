#!/usr/bin/env python3
"""
KDE-DA for alanine dipeptide — standalone script.

Loads existing GMM results (TICA, PCA, GMM-DA, ground truth, optimal lags),
builds a KDE scorer with Scott's bandwidth, validates dt stability, then
extracts DA directions at both DA1 and DA2 optimal physical times.

The key insight: KDE with Scott's rule in 45D produces a much smoother score
than the 4-component GMM (h ~ 0.83*std), so larger dt is stable, reducing
step count from 10M+ to ~100k.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time
from scipy.stats import pearsonr

from src.metrics.da_directions import extract_da_directions
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')


def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + (phi, psi) ground truth."""
    data_dir = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')
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

    # ── 1. Load existing GMM results ──
    gmm_file = os.path.join(RESDIR, 'alanine_da_results_gmm_only.npz')
    print(f'[1/6] Loading GMM results from {gmm_file}...')
    gmm = np.load(gmm_file, allow_pickle=True)

    # Extract GMM-DA optimal physical times
    dt_gmm = float(gmm['dt'])
    checkpoints = gmm['conv_checkpoints']

    # DA1: peak gap12
    opt_idx1 = int(gmm['conv_optimal_idx'])
    opt_phys1 = checkpoints[opt_idx1] * dt_gmm
    print(f'  DA1 optimal phys_t = {opt_phys1:.1f} (from GMM gap12)')

    # DA2: peak gap23
    opt_idx2 = int(gmm['conv_optimal_idx_da2'])
    opt_phys2 = checkpoints[opt_idx2] * dt_gmm
    print(f'  DA2 optimal phys_t = {opt_phys2:.1f} (from GMM gap23)')

    # GMM-DA directions for comparison
    gmm_da1_evec = gmm['da_gmm_evecs'][:, 0]  # at DA1 lag
    gmm_da2_evec_at_da2 = gmm['conv_optimal_eigenvectors_da2'][:, 1]  # DA2 at DA2 lag

    # ── 2. Load raw data + build KDE ──
    print(f'\n[2/6] Loading data and building KDE scorer...')
    d = load_alanine()
    all_dists = d['all_dists']
    N_total, dim = all_dists.shape
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean
    print(f'  {N_total:,} frames, {dim}D')

    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

    center_idx = np.random.choice(10000, 2000, replace=False)
    kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)
    print(f'  KDE: h={kde.h:.4f}, 2000 centers')

    # ── 3. dt stability test ──
    print(f'\n[3/6] dt stability test...')
    best_dt = None
    for test_dt in [0.1, 0.01, 0.001]:
        kde_sim_test = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_sub_t, dt=test_dt, device=device)

        torch.manual_seed(0)
        x_test = kde_sim_test.sample_initial(100)
        init_norms = x_test.norm(dim=1)
        stable = True
        for step in range(1000):
            x_test = kde_sim_test.step(x_test)
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f'  dt={test_dt}: UNSTABLE (NaN/Inf at step {step})')
                stable = False
                break
        if stable:
            final_norms = x_test.norm(dim=1)
            ratio = (final_norms / init_norms).max().item()
            if ratio > 10:
                print(f'  dt={test_dt}: UNSTABLE (norm ratio {ratio:.1f}x > 10)')
                stable = False
            else:
                sigma_f_sq = kde_sim_test.var_f
                noise_drift = (sigma_f_sq * test_dt)**0.5 / (sigma_f_sq * test_dt / 2)
                print(f'  dt={test_dt}: STABLE (max norm ratio {ratio:.1f}x, '
                      f'sigma_f^2={sigma_f_sq:.4f}, noise/drift={noise_drift:.1f}x)')

        if stable and best_dt is None:
            best_dt = test_dt

    if best_dt is None:
        print('  WARNING: No stable dt found! Falling back to dt=0.001')
        best_dt = 0.001

    print(f'  -> Using dt={best_dt}')

    # ── Build simulator with chosen dt ──
    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=best_dt, device=device)
    print(f'  sigma_f^2 (empirical) = {kde_sim.var_f:.4f}')

    n_traj = 2000

    # ── 4. DA at DA2 optimal lag (phys_t from GMM) ──
    da2_lag = max(1, int(opt_phys2 / best_dt))
    print(f'\n[4/6] KDE-DA at DA2 lag={da2_lag:,} steps (phys_t={opt_phys2:.1f})...')
    t0 = time.time()
    da2_evals, da2_evecs, _ = extract_da_directions(
        simulator=kde_sim, lag=da2_lag, n_trajectories=n_traj,
        dt=best_dt, device=device, seed=42)
    da2_proj = all_centered @ da2_evecs.numpy()
    print(f'  [{time.time()-t0:.1f}s]')

    print(f'  Top-3 eigenvalues: {da2_evals[:3].numpy()}')
    gap23_kde = da2_evals[1].item() / da2_evals[2].item() if da2_evals[2].item() > 0 else float('inf')
    print(f'  lambda2/lambda3 = {gap23_kde:.1f}x')
    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(da2_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  KDE-DA{i+1} (at DA2 lag): {", ".join(gt_strs)}')

    # Cross-correlation with GMM DA2
    kde_da2_vec = da2_evecs[:, 1].numpy()
    cos_gmm_kde_da2 = abs(np.dot(kde_da2_vec, gmm_da2_evec_at_da2) /
                          (np.linalg.norm(kde_da2_vec) * np.linalg.norm(gmm_da2_evec_at_da2)))
    r_cross_da2 = abs(pearsonr(all_centered @ kde_da2_vec,
                                all_centered @ gmm_da2_evec_at_da2)[0])
    print(f'  DA2 GMM<->KDE: |cos|={cos_gmm_kde_da2:.3f}, |r(proj)|={r_cross_da2:.3f}')

    # ── 5. DA at DA1 optimal lag (phys_t from GMM) ──
    da1_lag = max(1, int(opt_phys1 / best_dt))
    print(f'\n[5/6] KDE-DA at DA1 lag={da1_lag:,} steps (phys_t={opt_phys1:.1f})...')
    t0 = time.time()
    da1_evals, da1_evecs, _ = extract_da_directions(
        simulator=kde_sim, lag=da1_lag, n_trajectories=n_traj,
        dt=best_dt, device=device, seed=42)
    da1_proj = all_centered @ da1_evecs.numpy()
    elapsed_da1 = time.time() - t0
    print(f'  [{elapsed_da1:.1f}s]')

    print(f'  Top-3 eigenvalues: {da1_evals[:3].numpy()}')
    gap12_kde = da1_evals[0].item() / da1_evals[1].item() if da1_evals[1].item() > 0 else float('inf')
    print(f'  lambda1/lambda2 = {gap12_kde:.1f}x')
    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(da1_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  KDE-DA{i+1} (at DA1 lag): {", ".join(gt_strs)}')

    # Cross-correlation with GMM DA1
    kde_da1_vec = da1_evecs[:, 0].numpy()
    cos_gmm_kde_da1 = abs(np.dot(kde_da1_vec, gmm_da1_evec) /
                          (np.linalg.norm(kde_da1_vec) * np.linalg.norm(gmm_da1_evec)))
    r_cross_da1 = abs(pearsonr(all_centered @ kde_da1_vec,
                                all_centered @ gmm_da1_evec)[0])
    print(f'  DA1 GMM<->KDE: |cos|={cos_gmm_kde_da1:.3f}, |r(proj)|={r_cross_da1:.3f}')

    # ── 6. Summary + save ──
    print(f'\n[6/6] Summary')
    print(f'  {"Method":<12} {"DA1 |r(psi)|":>14} {"DA2 |r(phi)|":>14}')
    print(f'  {"-"*42}')

    gmm_da1_r_psi = abs(pearsonr(all_centered @ gmm_da1_evec, d['ground_truth']['psi'])[0])
    gmm_da2_r_phi = abs(pearsonr(all_centered @ gmm_da2_evec_at_da2, d['ground_truth']['phi'])[0])
    kde_da1_r_psi = abs(pearsonr(da1_proj[:, 0], d['ground_truth']['psi'])[0])
    kde_da2_r_phi = abs(pearsonr(da2_proj[:, 1], d['ground_truth']['phi'])[0])

    print(f'  {"GMM-DA":<12} {gmm_da1_r_psi:>14.3f} {gmm_da2_r_phi:>14.3f}')
    print(f'  {"KDE-DA":<12} {kde_da1_r_psi:>14.3f} {kde_da2_r_phi:>14.3f}')
    print(f'  DA1 GMM<->KDE |cos| = {cos_gmm_kde_da1:.3f}')
    print(f'  DA2 GMM<->KDE |cos| = {cos_gmm_kde_da2:.3f}')

    # Save
    out_path = os.path.join(RESDIR, 'alanine_da_results_kde.npz')
    np.savez_compressed(out_path,
        # KDE config
        kde_dt=best_dt,
        kde_h=kde.h,
        kde_n_centers=2000,
        n_traj=n_traj,
        # DA1 (at DA1 lag)
        da1_lag=da1_lag,
        da1_phys_t=opt_phys1,
        da1_evals=da1_evals[:5].numpy(),
        da1_evecs=da1_evecs[:, :5].numpy(),
        da1_gap12=gap12_kde,
        # DA2 (at DA2 lag)
        da2_lag=da2_lag,
        da2_phys_t=opt_phys2,
        da2_evals=da2_evals[:5].numpy(),
        da2_evecs=da2_evecs[:, :5].numpy(),
        da2_gap23=gap23_kde,
        # Cross-validation
        cos_gmm_kde_da1=cos_gmm_kde_da1,
        cos_gmm_kde_da2=cos_gmm_kde_da2,
        r_cross_da1=r_cross_da1,
        r_cross_da2=r_cross_da2,
        # Ground truth correlations
        kde_da1_r_psi=kde_da1_r_psi,
        kde_da1_r_phi=abs(pearsonr(da1_proj[:, 0], d['ground_truth']['phi'])[0]),
        kde_da2_r_psi=abs(pearsonr(da2_proj[:, 1], d['ground_truth']['psi'])[0]),
        kde_da2_r_phi=kde_da2_r_phi,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\n  Saved: {out_path} ({size_mb:.1f} MB)')

    elapsed_total = time.time() - t_start
    print(f'\nTotal time: {elapsed_total/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
