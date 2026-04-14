#!/usr/bin/env python3
"""
Alanine Dipeptide: noise-floor-based DA lag selection.

d=45, N=2000, KDE score (h=0.018, 2000 centers), dt=0.1.
Re-runs canonical Langevin diffusion at the same 28-checkpoint geometric grid,
computes C(τ) eigenvalues + permutation floor at each checkpoint.
Finds τ* for m=2 using permutation floor criterion.
Compares to old gap-max results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import time
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures.style_config import setup_style, SSA_COLOR, MI_COLOR, TEXT_WIDTH

setup_style()

from src.simulation.generic_langevin import GenericLangevinSimulator
# Import KDEScore by loading the file directly (src/kde/__init__.py has a broken import)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('kde_score',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'kde', 'kde_score.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
KDEScore = _mod.KDEScore
from experiments.noise_floor.noise_floor_utils import (
    analytic_floor, permutation_floor, find_tau_star
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures')
DA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'da_directions')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_global = time.time()
    print(f'Device: {device}')

    # --- Load existing results for reference ---
    gmm_path = os.path.join(DA_DIR, 'results', 'alanine_da_results_gmm_only.npz')
    gmm = np.load(gmm_path, allow_pickle=True)
    sigma_f_sq = float(gmm['sigma_f_sq'])
    gt_phi = gmm['gt_phi']  # (750000,)
    gt_psi = gmm['gt_psi']  # (750000,)
    pca_evecs = gmm['pca_evecs']  # (45, 5)

    # Old results
    old_da1_tau_ps = 717.9
    old_da1_r_psi = 0.676
    old_da2_tau_ps = 33.2
    old_da2_r_phi = 0.754

    # Load KDE gap profile for checkpoint grid reference
    kde_profile = np.load(os.path.join(DA_DIR, 'results', 'alanine_kde_gap_profile.npz'),
                          allow_pickle=True)
    old_eigenvalues = kde_profile['all_eigenvalues']  # (28, 45)
    old_checkpoints = kde_profile['checkpoints']  # (28,) in steps
    kde_dt = float(kde_profile['kde_dt'])  # 0.1
    old_phys_times = kde_profile['phys_times']  # (28,)

    print(f'Old results: DA1 τ*={old_da1_tau_ps}ps |r(ψ)|={old_da1_r_psi}')
    print(f'             DA2 τ*={old_da2_tau_ps}ps |r(φ)|={old_da2_r_phi}')
    print(f'sigma_f^2 = {sigma_f_sq:.6f}, d=45, KDE dt={kde_dt}')

    # --- Build KDE score ---
    print('\n[1/4] Loading data and building KDE...')
    dist_file = os.path.join(DA_DIR, 'alanine_dipeptide', 'data',
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

    # Build simulator
    N_traj = 2000
    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=kde_dt, device=device)
    var_f = kde_sim.var_f
    print(f'  var_f (from simulator) = {var_f:.6f}')

    # --- Checkpoint grid (same as old experiment) ---
    checkpoints = old_checkpoints.tolist()  # step counts
    phys_times = np.array(checkpoints) * kde_dt

    # --- Run Langevin + compute eigenvalues + permutation floor ---
    print(f'\n[2/4] Running Langevin ({len(checkpoints)} checkpoints, '
          f'max={checkpoints[-1]:,} steps, N={N_traj})...')
    t0 = time.time()

    torch.manual_seed(42)
    x0 = kde_sim.sample_initial(N_traj)  # (N, 45)
    mu_f = kde_sim.mu_f.to(device=device)
    x0_centered = (x0 - mu_f.unsqueeze(0)).cpu().numpy()

    x = x0.clone()
    current_step = 0

    all_eigenvalues = []
    all_eigenvectors = []  # top 3 at each checkpoint
    perm_floors = []
    analytic_floors_list = []

    for cp_idx, cp in enumerate(checkpoints):
        steps = cp - current_step
        for _ in range(steps):
            x = kde_sim.step(x)
        current_step = cp

        x_centered = (x - mu_f.unsqueeze(0)).cpu().numpy()

        # C(τ) eigendecomposition
        C_tau = x_centered.T @ x0_centered / N_traj
        C_tau_sym = 0.5 * (C_tau + C_tau.T)
        evals, evecs = np.linalg.eigh(C_tau_sym)
        # Sort descending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx].copy()
        evecs = evecs[:, idx].copy()

        all_eigenvalues.append(evals)
        all_eigenvectors.append(evecs[:, :3].copy())  # top 3

        # Permutation floor
        pf = permutation_floor(x0_centered, x_centered, n_perm=20, quantile=0.95)
        perm_floors.append(pf)

        # Analytic floor
        af = analytic_floor(cp * kde_dt, sigma_f_sq, dim, N_traj)
        analytic_floors_list.append(af)

        if (cp_idx + 1) % 7 == 0 or cp_idx == len(checkpoints) - 1:
            elapsed = time.time() - t0
            print(f'  checkpoint {cp_idx+1}/{len(checkpoints)}: '
                  f'τ={cp} (t={cp*kde_dt:.1f}ps), λ₁={evals[0]:.5f}, '
                  f'perm_floor={pf:.5f} [{elapsed:.0f}s]')

    all_eigenvalues = np.array(all_eigenvalues)  # (28, 45)
    perm_floors = np.array(perm_floors)
    analytic_floors_arr = np.array(analytic_floors_list)
    checkpoints = np.array(checkpoints)

    elapsed_total = time.time() - t0
    print(f'  Langevin done [{elapsed_total:.0f}s = {elapsed_total/60:.1f}min]')

    # --- Find τ* with permutation floor ---
    print('\n[3/4] Finding τ* ...')

    # m=2: both λ_1 and λ_2 must exceed perm floor
    tau_star_idx_m2, tau_star_m2 = find_tau_star(
        checkpoints, all_eigenvalues, perm_floors, m=2
    )
    # m=1: only λ_1
    tau_star_idx_m1, tau_star_m1 = find_tau_star(
        checkpoints, all_eigenvalues, perm_floors, m=1
    )

    print(f'  τ* (m=1): step={tau_star_m1}, phys_t={tau_star_m1*kde_dt:.1f}ps')
    print(f'    λ_1 = {all_eigenvalues[tau_star_idx_m1, 0]:.5f}, '
          f'floor = {perm_floors[tau_star_idx_m1]:.5f}')
    print(f'  τ* (m=2): step={tau_star_m2}, phys_t={tau_star_m2*kde_dt:.1f}ps')
    print(f'    λ_1 = {all_eigenvalues[tau_star_idx_m2, 0]:.5f}, '
          f'λ_2 = {all_eigenvalues[tau_star_idx_m2, 1]:.5f}, '
          f'floor = {perm_floors[tau_star_idx_m2]:.5f}')

    # Also find τ* using analytic floor for comparison
    tau_star_idx_af2, tau_star_af2 = find_tau_star(
        checkpoints, all_eigenvalues, analytic_floors_arr, m=2
    )
    tau_star_idx_af1, tau_star_af1 = find_tau_star(
        checkpoints, all_eigenvalues, analytic_floors_arr, m=1
    )
    print(f'\n  Analytic floor τ* (m=1): step={tau_star_af1}, phys_t={tau_star_af1*kde_dt:.1f}ps')
    print(f'    λ_1 = {all_eigenvalues[tau_star_idx_af1, 0]:.5f}, '
          f'floor = {analytic_floors_arr[tau_star_idx_af1]:.5f}')
    print(f'  Analytic floor τ* (m=2): step={tau_star_af2}, phys_t={tau_star_af2*kde_dt:.1f}ps')
    print(f'    λ_1 = {all_eigenvalues[tau_star_idx_af2, 0]:.5f}, '
          f'λ_2 = {all_eigenvalues[tau_star_idx_af2, 1]:.5f}, '
          f'floor = {analytic_floors_arr[tau_star_idx_af2]:.5f}')

    from scipy.stats import pearsonr
    all_dists_centered = all_dists - all_dists.mean(axis=0)
    old_da1 = kde_profile['optimal_eigenvectors_da1'][:, 0]
    old_da2 = kde_profile['optimal_eigenvectors_da2'][:, 1]  # 2nd eigvec at DA2 lag

    def compute_da_correlations(evecs_at_lag, label):
        """Compute and print DA correlations at a given lag."""
        da1 = evecs_at_lag[:, 0]
        da2 = evecs_at_lag[:, 1]
        proj1 = all_dists_centered @ da1
        proj2 = all_dists_centered @ da2
        r1_psi = abs(pearsonr(proj1, gt_psi)[0])
        r1_phi = abs(pearsonr(proj1, gt_phi)[0])
        r2_psi = abs(pearsonr(proj2, gt_psi)[0])
        r2_phi = abs(pearsonr(proj2, gt_phi)[0])
        cos1 = abs(np.dot(da1, old_da1))
        cos2 = abs(np.dot(da2, old_da2))
        ang1 = np.degrees(np.arccos(np.clip(cos1, 0, 1)))
        ang2 = np.degrees(np.arccos(np.clip(cos2, 0, 1)))
        print(f'\n  {label}:')
        print(f'    DA1: |r(ψ)|={r1_psi:.3f}, |r(φ)|={r1_phi:.3f}')
        print(f'    DA2: |r(ψ)|={r2_psi:.3f}, |r(φ)|={r2_phi:.3f}')
        print(f'    Angle(DA1, old DA1) = {ang1:.1f}°')
        print(f'    Angle(DA2, old DA2) = {ang2:.1f}°')
        return {
            'da1_r_psi': float(r1_psi), 'da1_r_phi': float(r1_phi),
            'da2_r_psi': float(r2_psi), 'da2_r_phi': float(r2_phi),
            'angle_da1_deg': float(ang1), 'angle_da2_deg': float(ang2),
            'cos_da1': float(cos1), 'cos_da2': float(cos2),
        }

    # Perm floor τ*(m=2)
    perm_corr = compute_da_correlations(
        all_eigenvectors[tau_star_idx_m2],
        f'Perm floor τ*(m=2) = {tau_star_m2*kde_dt:.1f}ps')

    # Analytic floor τ*(m=2)
    af_corr = compute_da_correlations(
        all_eigenvectors[tau_star_idx_af2],
        f'Analytic floor τ*(m=2) = {tau_star_af2*kde_dt:.1f}ps')

    # Gap-max (old) — already have correlations, just print for comparison
    print(f'\n  Old gap-max (DA1 τ*=717.9ps, DA2 τ*=33.2ps):')
    print(f'    DA1: |r(ψ)|={old_da1_r_psi:.3f}')
    print(f'    DA2: |r(φ)|={old_da2_r_phi:.3f}')

    # --- Save JSON ---
    print('\n[4/4] Saving results...')
    results = {
        'experiment': 'alanine_noise_floor',
        'parameters': {
            'd': dim, 'N': N_traj, 'dt': kde_dt,
            'sigma_f_sq': sigma_f_sq,
            'gamma': dim / N_traj,
            'kde_h': float(kde.h),
        },
        'checkpoints': checkpoints.tolist(),
        'phys_times': phys_times.tolist(),
        'eigenvalues_top5': all_eigenvalues[:, :5].tolist(),
        'perm_floor': perm_floors.tolist(),
        'analytic_floor': analytic_floors_arr.tolist(),
        'tau_star_m1': {
            'step': int(tau_star_m1),
            'phys_t_ps': float(tau_star_m1 * kde_dt),
            'lambda_1': float(all_eigenvalues[tau_star_idx_m1, 0]),
            'floor': float(perm_floors[tau_star_idx_m1]),
        },
        'tau_star_m2': {
            'step': int(tau_star_m2),
            'phys_t_ps': float(tau_star_m2 * kde_dt),
            'lambda_1': float(all_eigenvalues[tau_star_idx_m2, 0]),
            'lambda_2': float(all_eigenvalues[tau_star_idx_m2, 1]),
            'floor': float(perm_floors[tau_star_idx_m2]),
        },
        'tau_star_analytic_m1': {
            'step': int(tau_star_af1),
            'phys_t_ps': float(tau_star_af1 * kde_dt),
            'lambda_1': float(all_eigenvalues[tau_star_idx_af1, 0]),
            'floor': float(analytic_floors_arr[tau_star_idx_af1]),
        },
        'tau_star_analytic_m2': {
            'step': int(tau_star_af2),
            'phys_t_ps': float(tau_star_af2 * kde_dt),
            'lambda_1': float(all_eigenvalues[tau_star_idx_af2, 0]),
            'lambda_2': float(all_eigenvalues[tau_star_idx_af2, 1]),
            'floor': float(analytic_floors_arr[tau_star_idx_af2]),
        },
        'perm_floor_correlations': perm_corr,
        'analytic_floor_correlations': af_corr,
        'old_correlations': {
            'da1_tau_ps': old_da1_tau_ps,
            'da1_r_psi': old_da1_r_psi,
            'da2_tau_ps': old_da2_tau_ps,
            'da2_r_phi': old_da2_r_phi,
        },
    }

    json_path = os.path.join(OUTDIR, 'noise_floor_alanine.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {json_path}')

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.6))

    ax.plot(phys_times, all_eigenvalues[:, 0], 'o-', color=SSA_COLOR,
            markersize=4, lw=1.4, label=r'$\lambda_1(\tau)$')
    ax.plot(phys_times, all_eigenvalues[:, 1], 's-', color='#2ca02c',
            markersize=3.5, lw=1.2, label=r'$\lambda_2(\tau)$')
    ax.plot(phys_times, all_eigenvalues[:, 2], '^-', color='#ff7f0e',
            markersize=3, lw=1.0, alpha=0.7, label=r'$\lambda_3(\tau)$')
    ax.plot(phys_times, perm_floors, 'x-', color=MI_COLOR,
            markersize=5, lw=1.2, label='Permutation floor (95th pctile)')
    ax.plot(phys_times, analytic_floors_arr, '--', color='#999999',
            lw=1.0, label='Analytic floor (Gaussian null)')

    # Mark τ*(m=2) — permutation floor
    t_star_ps = tau_star_m2 * kde_dt
    ax.axvline(t_star_ps, color='black', ls=':', lw=0.8, alpha=0.7)
    ax.annotate(rf'$\tau^*_{{perm}}$ = {t_star_ps:.0f} ps',
                xy=(t_star_ps, all_eigenvalues[tau_star_idx_m2, 0]),
                xytext=(t_star_ps * 1.5, all_eigenvalues[tau_star_idx_m2, 0] * 1.05),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.6))

    # Mark τ*(m=2) — analytic floor
    t_star_af_ps = tau_star_af2 * kde_dt
    ax.axvline(t_star_af_ps, color='#999999', ls=':', lw=0.8, alpha=0.7)
    ax.annotate(rf'$\tau^*_{{analytic}}$ = {t_star_af_ps:.0f} ps',
                xy=(t_star_af_ps, all_eigenvalues[tau_star_idx_af2, 1]),
                xytext=(t_star_af_ps * 1.8, all_eigenvalues[tau_star_idx_af2, 1] * 0.5),
                fontsize=8, color='#666666',
                arrowprops=dict(arrowstyle='->', lw=0.6, color='#666666'))

    ax.set_xlabel(r'Lag $\tau$ (ps)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Alanine Dipeptide: noise floor lag selection (d=45, N=2000)', fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig_path = os.path.join(FIGDIR, 'fig_alanine_noise_floor.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved {fig_path}')

    total = time.time() - t_global
    print(f'\nTotal: {total:.0f}s = {total/60:.1f}min')


if __name__ == '__main__':
    main()
