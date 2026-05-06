#!/usr/bin/env python3
"""
Save full autocovariance matrices C_sym(tau) for alanine dipeptide at h = h_Scott/2.

Identical to save_alanine_C_tau.py except KDE bandwidth = Scott/2 = 0.0091.
This tests whether halving the bandwidth eliminates the structured negative
eigenvalues without degrading DA quality.

Output: results/alanine_C_tau_full_h_half_50k.npz (same format, extended to 50000ps)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time
from scipy.stats import pearsonr
from scipy.linalg import eigh as scipy_eigh

from src.metrics.da_directions import extract_da_directions
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore

import importlib.util as _ilu
_fc_spec = _ilu.spec_from_file_location('free_conv_utils',
    os.path.join(os.path.dirname(__file__), '..', 'noise_floor', 'free_conv_utils.py'))
_fc_mod = _ilu.module_from_spec(_fc_spec)
_fc_spec.loader.exec_module(_fc_mod)
free_conv_edges = _fc_mod.edges

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)
DA_DIR = os.path.join(os.path.dirname(__file__), '..', 'da_directions')


def load_alanine():
    data_dir = os.path.join(DA_DIR, 'alanine_dipeptide', 'data')
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
    return all_dists, all_diheds[:, 0], all_diheds[:, 1]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_start = time.time()

    print('[1/4] Loading alanine dipeptide data...')
    all_dists, gt_phi, gt_psi = load_alanine()
    N_total, dim = all_dists.shape
    md_mean = all_dists.mean(axis=0)
    centered = all_dists - md_mean
    print(f'  {N_total:,} frames, {dim}D, device={device}')

    # 2. Build KDE scorer — same centers as canonical, but h = scott/2
    print('\n[2/4] Building KDE scorer (h = Scott/2)...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

    center_idx = np.random.choice(10000, 2000, replace=False)
    centers = X_sub_t[center_idx]

    # Get Scott bandwidth, then halve
    kde_scott = KDEScore(centers, bandwidth='scott', device=device, batch_size=1000)
    h_scott = kde_scott.h
    h_half = h_scott / 2.0
    print(f'  h_Scott = {h_scott:.4f}, using h = {h_half:.4f}')

    kde = KDEScore(centers, bandwidth=h_half, device=device, batch_size=1000)

    # dt stability test
    print('  dt stability test...')
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
                print(f'    dt={test_dt}: UNSTABLE (NaN/Inf at step {step})')
                stable = False
                break
        if stable:
            final_norms = x_test.norm(dim=1)
            ratio = (final_norms / init_norms).max().item()
            if ratio > 10:
                print(f'    dt={test_dt}: UNSTABLE (norm ratio {ratio:.1f}x)')
                stable = False
            else:
                print(f'    dt={test_dt}: STABLE (norm ratio {ratio:.1f}x, '
                      f'sigma_f^2={kde_sim_test.var_f:.4f})')
        if stable and best_dt is None:
            best_dt = test_dt

    if best_dt is None:
        best_dt = 0.001
    print(f'  -> Using dt={best_dt}')

    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=best_dt, device=device)
    sigma_f_sq = kde_sim.var_f
    print(f'  sigma_f^2 = {sigma_f_sq:.6f}')

    n_traj = 2000

    # 3. Convergence grid (same as canonical)
    phys_times = np.geomspace(0.1, 50000.0, 50)
    kde_lags = np.maximum(1, np.round(phys_times / best_dt).astype(int))
    unique_mask = np.concatenate([[True], np.diff(kde_lags) > 0])
    kde_lags = kde_lags[unique_mask]
    phys_times = phys_times[unique_mask]
    n_checkpoints = len(kde_lags)

    print(f'\n[3/4] Computing C_tau at {n_checkpoints} lags...')

    C_tau_all = np.zeros((n_checkpoints, dim, dim), dtype=np.float64)
    eigenvalues_all = np.zeros((n_checkpoints, dim))

    sigma_f = np.sqrt(sigma_f_sq)
    gamma = dim / n_traj
    consecutive_below = 0
    n_computed = 0

    for i, (lag, pt) in enumerate(zip(kde_lags, phys_times)):
        t0 = time.time()
        evals, evecs, C_tau_sym = extract_da_directions(
            simulator=kde_sim, lag=int(lag), n_trajectories=n_traj,
            dt=best_dt, device=device, seed=42)
        elapsed = time.time() - t0

        evals_np = evals.cpu().numpy()
        evecs_np = evecs.cpu().numpy()
        C_tau_all[i] = C_tau_sym.cpu().numpy()
        eigenvalues_all[i] = evals_np

        # Free-conv noise floor
        _, lam_plus = free_conv_edges(pt, sigma_f, gamma=gamma)
        lam2 = evals_np[1] if len(evals_np) > 1 else 0.0
        below_floor = lam2 < lam_plus

        gap12 = evals_np[0] / lam2 if lam2 > 0 else float('inf')

        # DA correlations
        da1_proj = centered @ evecs_np[:, 0]
        da2_proj = centered @ evecs_np[:, 1]
        r1_psi = abs(pearsonr(da1_proj, gt_psi)[0])
        r2_phi = abs(pearsonr(da2_proj, gt_phi)[0])
        lam_min = evals_np[-1]

        print(f'  [{i+1:2d}/{n_checkpoints}] lag={lag:>6d} ({pt:>8.1f}ps)  '
              f'gap12={gap12:.2f}x  DA1|r(ψ)|={r1_psi:.3f}  DA2|r(φ)|={r2_phi:.3f}  '
              f'λ_min={lam_min:.2e}  '
              f'λ₂={lam2:.6f} floor={lam_plus:.6f} {"BELOW" if below_floor else "above"}  '
              f'[{elapsed:.1f}s]')

        n_computed = i + 1

        if below_floor:
            consecutive_below += 1
            if consecutive_below >= 2:
                print(f'\n  *** Stopping: λ₂ below free-conv floor ***')
                break
        else:
            consecutive_below = 0

    # Trim
    kde_lags = kde_lags[:n_computed]
    phys_times = phys_times[:n_computed]
    C_tau_all = C_tau_all[:n_computed]
    eigenvalues_all = eigenvalues_all[:n_computed]

    # C(0)
    print('\n  Computing C(0)...')
    torch.manual_seed(42)
    x0 = kde_sim.sample_initial(n_traj)
    mu_f = kde_sim.mu_f
    x0_c = x0 - mu_f.unsqueeze(0)
    C0 = (x0_c.t() @ x0_c / n_traj).cpu().numpy()

    # 4. Save
    print(f'\n[4/4] Saving ({n_computed} lags)...')
    out_path = os.path.join(RESDIR, 'alanine_C_tau_full_h_half_50k.npz')
    np.savez_compressed(out_path,
        lags=kde_lags,
        phys_times=phys_times,
        C_tau=C_tau_all,
        C0=C0,
        eigenvalues=eigenvalues_all,
        N=np.array(n_traj),
        d=np.array(dim),
        sigma_f_sq=np.array(sigma_f_sq),
        dt=np.array(best_dt),
        kde_h=np.array(h_half),
        kde_h_scott=np.array(h_scott),
        kde_n_centers=np.array(2000),
        gamma=np.array(gamma),
        gt_phi=gt_phi,
        gt_psi=gt_psi,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path} ({size_mb:.1f} MB)')

    # Quick comparison at tau* if we got that far
    idx_star = np.argmin(np.abs(phys_times - 362.6))
    if phys_times[idx_star] > 300:
        print(f'\n  === Quick diagnostic at tau*={phys_times[idx_star]:.1f}ps ===')
        evals_star, evecs_star = scipy_eigh(C_tau_all[idx_star])
        v_min = evecs_star[:, 0]
        proj_min = centered @ v_min
        r_min_phi = abs(pearsonr(proj_min, gt_phi)[0])
        r_min_psi = abs(pearsonr(proj_min, gt_psi)[0])
        print(f'    λ_min = {evals_star[0]:.4e}, |r(φ)| = {r_min_phi:.4f}, |r(ψ)| = {r_min_psi:.4f}')
        print(f'    λ₁ = {evals_star[-1]:.4e}')
        v_da1 = evecs_star[:, -1]
        r_da1_psi = abs(pearsonr(centered @ v_da1, gt_psi)[0])
        r_da1_phi = abs(pearsonr(centered @ v_da1, gt_phi)[0])
        print(f'    DA1: |r(ψ)| = {r_da1_psi:.4f}, |r(φ)| = {r_da1_phi:.4f}')
        v_da2 = evecs_star[:, -2]
        r_da2_psi = abs(pearsonr(centered @ v_da2, gt_psi)[0])
        r_da2_phi = abs(pearsonr(centered @ v_da2, gt_phi)[0])
        print(f'    DA2: |r(ψ)| = {r_da2_psi:.4f}, |r(φ)| = {r_da2_phi:.4f}')
        print(f'    λ₁/λ₂ gap = {evals_star[-1]/evals_star[-2]:.2f}x')

    elapsed_total = time.time() - t_start
    print(f'\nTotal time: {elapsed_total/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
