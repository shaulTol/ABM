#!/usr/bin/env python3
"""
Save full autocovariance matrices C_sym(tau) for alanine dipeptide.

Matches compute_alanine_kde_convergence.py exactly:
- Same KDE scorer (Scott bandwidth, 2000 centers, seed=42)
- Same lag grid (geomspace 0.1..2000ps, 30 points)
- Same stopping criterion (lambda_2 < free-conv upper edge, 2 consecutive)
- Same n_traj=2000, dt=best_dt

Output: results/alanine_C_tau_full.npz
  lags:       (L,) int     — lag step indices
  phys_times: (L,) float   — physical times (lags * dt)
  C_tau:      (L, d, d)    — symmetrized autocovariance at each lag
  C0:         (d, d)       — sample covariance C(0)
  eigenvalues:(L, d)       — eigenvalues at each lag (for consistency check)
  N, d, sigma_f_sq, dt: scalars
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time

from src.metrics.da_directions import extract_da_directions
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore

# Import free-conv utils
import importlib.util as _ilu
_fc_spec = _ilu.spec_from_file_location('free_conv_utils',
    os.path.join(os.path.dirname(__file__), '..', 'noise_floor', 'free_conv_utils.py'))
_fc_mod = _ilu.module_from_spec(_fc_spec)
_fc_spec.loader.exec_module(_fc_mod)
free_conv_edges = _fc_mod.edges

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)

# Reuse data loader from da_directions experiment
DA_DIR = os.path.join(os.path.dirname(__file__), '..', 'da_directions')


def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + dihedrals."""
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
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    return all_dists


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # 1. Load data
    print('[1/4] Loading alanine dipeptide data...')
    all_dists = load_alanine()
    N_total, dim = all_dists.shape
    print(f'  {N_total:,} frames, {dim}D')

    # 2. Build KDE scorer (identical to compute_alanine_kde_convergence.py)
    print('\n[2/4] Building KDE scorer...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

    center_idx = np.random.choice(10000, 2000, replace=False)
    kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)
    print(f'  KDE: h={kde.h:.4f}, 2000 centers')

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

    # 3. Convergence grid
    phys_times = np.geomspace(0.1, 2000.0, 30)
    kde_lags = np.maximum(1, np.round(phys_times / best_dt).astype(int))
    unique_mask = np.concatenate([[True], np.diff(kde_lags) > 0])
    kde_lags = kde_lags[unique_mask]
    phys_times = phys_times[unique_mask]
    n_checkpoints = len(kde_lags)

    print(f'\n[3/4] Computing C_tau at {n_checkpoints} lags...')
    print(f'  Physical times: {phys_times[0]:.1f}ps to {phys_times[-1]:.1f}ps')

    # Pre-allocate
    C_tau_all = np.zeros((n_checkpoints, dim, dim), dtype=np.float64)
    eigenvalues_all = np.zeros((n_checkpoints, dim))

    # Free-conv stopping criterion
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

        C_tau_all[i] = C_tau_sym.numpy()
        eigenvalues_all[i] = evals.numpy()

        # Free-conv noise floor
        _, lam_plus = free_conv_edges(pt, sigma_f, gamma=gamma)
        lam2 = evals[1].item() if len(evals) > 1 else 0.0
        below_floor = lam2 < lam_plus

        gap12 = evals[0].item() / lam2 if lam2 > 0 else float('inf')
        print(f'  [{i+1:2d}/{n_checkpoints}] lag={lag:>6d} ({pt:>8.1f}ps)  '
              f'gap12={gap12:.2f}x  '
              f'lambda_2={lam2:.6f} floor={lam_plus:.6f} '
              f'{"BELOW" if below_floor else "above"}  [{elapsed:.1f}s]')

        n_computed = i + 1

        if below_floor:
            consecutive_below += 1
            if consecutive_below >= 2:
                print(f'\n  *** Stopping: lambda_2 below free-conv floor for 2 consecutive lags ***')
                break
        else:
            consecutive_below = 0

    # Trim to computed
    kde_lags = kde_lags[:n_computed]
    phys_times = phys_times[:n_computed]
    C_tau_all = C_tau_all[:n_computed]
    eigenvalues_all = eigenvalues_all[:n_computed]

    # Compute C(0) = sample covariance from initial samples
    print('\n  Computing C(0)...')
    torch.manual_seed(42)
    x0 = kde_sim.sample_initial(n_traj)
    mu_f = kde_sim.mu_f
    x0_c = x0 - mu_f.unsqueeze(0)
    C0 = (x0_c.t() @ x0_c / n_traj).numpy()

    # 4. Save
    print(f'\n[4/4] Saving ({n_computed} lags, {dim}x{dim} matrices)...')
    out_path = os.path.join(RESDIR, 'alanine_C_tau_full.npz')
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
        kde_h=np.array(kde.h),
        kde_n_centers=np.array(2000),
        gamma=np.array(gamma),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path} ({size_mb:.1f} MB)')

    # Verification
    print('\n  Verification:')
    print(f'    C_tau shape: {C_tau_all.shape}')
    print(f'    C0 shape: {C0.shape}')
    print(f'    Symmetry check (max |C-C^T|): {max(np.max(np.abs(C_tau_all[i] - C_tau_all[i].T)) for i in range(n_computed)):.2e}')
    print(f'    tr(C0) = {np.trace(C0):.6f}, sigma_f^2 * d = {sigma_f_sq * dim:.6f}')

    elapsed_total = time.time() - t_start
    print(f'\nTotal time: {elapsed_total/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
