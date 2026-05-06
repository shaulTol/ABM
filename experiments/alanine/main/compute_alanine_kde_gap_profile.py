#!/usr/bin/env python3
"""
Compute KDE eigenvalue gap convergence profile for alanine dipeptide.

Runs da_convergence_profile() with KDE score at the same physical time
grid as the GMM run, producing gap12 and gap23 profiles. This validates
that the KDE-selected τ* matches the GMM-selected τ*.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch
import time

from src.metrics.da_directions import da_convergence_profile
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # Load GMM results for checkpoint grid and PCA1 reference
    gmm = np.load(os.path.join(RESDIR, 'alanine_da_results_gmm_only.npz'), allow_pickle=True)
    gmm_dt = float(gmm['dt'])
    gmm_cps = gmm['conv_checkpoints']
    phys_times = gmm_cps * gmm_dt  # physical times from GMM run

    pca1 = torch.tensor(gmm['pca_evecs'][:, 0], dtype=torch.float32, device=device)

    # Load raw data
    print('[1/3] Loading data and building KDE...')
    dist_file = os.path.join(EXPDIR, 'alanine_dipeptide', 'data',
                             'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dist_data = np.load(dist_file)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    N_total, dim = all_dists.shape
    print(f'  {N_total:,} frames, {dim}D')

    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

    center_idx = np.random.choice(10000, 2000, replace=False)
    kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)
    print(f'  KDE: h={kde.h:.4f}, 2000 centers')

    # Build KDE simulator at dt=0.1
    kde_dt = 0.1
    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=kde_dt, device=device)
    print(f'  sigma_f^2={kde_sim.var_f:.4f}, dt={kde_dt}')

    # Convert physical times to KDE step counts
    kde_cps = sorted(set(max(1, int(pt / kde_dt)) for pt in phys_times))
    print(f'\n[2/3] Running KDE convergence profile ({len(kde_cps)} checkpoints, '
          f'max={kde_cps[-1]:,} steps, n_traj=2000)...')
    print(f'  Physical time range: {kde_cps[0]*kde_dt:.1f} to {kde_cps[-1]*kde_dt:.1f}')

    t0 = time.time()
    profile = da_convergence_profile(
        simulator=kde_sim,
        checkpoints=kde_cps,
        n_trajectories=2000,
        dt=kde_dt,
        seed=42,
        reference_direction=pca1,
        device=device,
    )
    elapsed = time.time() - t0
    print(f'  [{elapsed:.1f}s = {elapsed/60:.1f} min]')

    # Report results
    cps = profile['checkpoints']
    kde_phys = cps * kde_dt
    gap12 = profile['gap_profile']
    gap23 = profile['gap23_profile']

    opt1 = profile['optimal_idx']
    opt2 = profile['optimal_idx_da2']
    opt_phys1 = kde_phys[opt1]
    opt_phys2 = kde_phys[opt2]

    print(f'\n[3/3] Results:')
    print(f'  DA1 peak gap12 = {gap12[opt1]:.1f}x at phys_t = {opt_phys1:.1f} (GMM: 1010.2)')
    print(f'  DA2 peak gap23 = {gap23[opt2]:.1f}x at phys_t = {opt_phys2:.1f} (GMM: 92.5)')

    print(f'\n  Gap12 profile:')
    for i, (pt, g12, g23) in enumerate(zip(kde_phys, gap12, gap23)):
        marker1 = ' <-- DA1 peak' if i == opt1 else ''
        marker2 = ' <-- DA2 peak' if i == opt2 else ''
        print(f'    phys_t={pt:8.1f}: gap12={g12:7.1f}x  gap23={g23:7.1f}x{marker1}{marker2}')

    # Ground truth correlations at optimal lags
    dihed_file = os.path.join(EXPDIR, 'alanine_dipeptide', 'data',
                              'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    dihed_data = np.load(dihed_file)
    all_diheds = np.concatenate([dihed_data[f'arr_{i}'] for i in range(3)])
    phi_gt, psi_gt = all_diheds[:, 0], all_diheds[:, 1]
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean

    from scipy.stats import pearsonr

    # DA1 at peak gap12
    da1_evec = profile['optimal_eigenvectors'][:, 0]
    da1_proj = all_centered @ da1_evec
    print(f'\n  KDE-DA1 at tau*={opt_phys1:.1f}:')
    print(f'    |r(psi)| = {abs(pearsonr(da1_proj, psi_gt)[0]):.4f}')
    print(f'    |r(phi)| = {abs(pearsonr(da1_proj, phi_gt)[0]):.4f}')

    # DA2 at peak gap23
    da2_evec = profile['optimal_eigenvectors_da2'][:, 1]
    da2_proj = all_centered @ da2_evec
    print(f'  KDE-DA2 at tau*={opt_phys2:.1f}:')
    print(f'    |r(phi)| = {abs(pearsonr(da2_proj, phi_gt)[0]):.4f}')
    print(f'    |r(psi)| = {abs(pearsonr(da2_proj, psi_gt)[0]):.4f}')

    # Save
    out_path = os.path.join(RESDIR, 'alanine_kde_gap_profile.npz')
    np.savez_compressed(out_path,
        checkpoints=cps,
        phys_times=kde_phys,
        gap12=gap12,
        gap23=gap23,
        optimal_idx_da1=opt1,
        optimal_idx_da2=opt2,
        optimal_phys_t_da1=opt_phys1,
        optimal_phys_t_da2=opt_phys2,
        all_eigenvalues=profile['all_eigenvalues'],
        angles_deg=profile['angles_deg'],
        leading_eigenvectors=profile['leading_eigenvectors'],
        optimal_eigenvalues_da1=profile['optimal_eigenvalues'],
        optimal_eigenvectors_da1=profile['optimal_eigenvectors'],
        optimal_eigenvalues_da2=profile['optimal_eigenvalues_da2'],
        optimal_eigenvectors_da2=profile['optimal_eigenvectors_da2'],
        kde_dt=kde_dt,
        kde_h=kde.h,
    )
    print(f'\n  Saved: {out_path}')

    print(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
