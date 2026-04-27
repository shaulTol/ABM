#!/usr/bin/env python3
"""
KDE-DA convergence grid for alanine dipeptide.

Runs KDE-DA at log-spaced convergence checkpoints (0.1ps to 2000ps),
saving the full eigenvector grid so noise-floor τ* selection can be applied
purely from KDE (no GMM dependency).

Uses free-convolution noise floor (exact analytic upper edge) as a stopping
criterion: stops when λ₁ drops below the upper edge for 2 consecutive lags.

Also saves TICA projections and ground truth dihedrals for figure generation.

Output: results/alanine_da_results_kde_convergence.npz
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
# Import free-conv utils via direct path (sys.path has project root)
import importlib.util as _ilu
_fc_spec = _ilu.spec_from_file_location('free_conv_utils',
    os.path.join(os.path.dirname(__file__), '..', 'noise_floor', 'free_conv_utils.py'))
_fc_mod = _ilu.module_from_spec(_fc_spec)
_fc_spec.loader.exec_module(_fc_mod)
free_conv_edges = _fc_mod.edges

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)


def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + dihedrals."""
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
    dists_list = [dist_data[f'arr_{i}'] for i in range(3)]
    diheds_list = [dihed_data[f'arr_{i}'] for i in range(3)]
    all_dists = np.concatenate(dists_list)
    all_diheds = np.concatenate(diheds_list)
    return dists_list, all_dists, all_diheds[:, 0], all_diheds[:, 1]


def compute_tica(dists_list, all_dists, lag=10, n_components=5):
    """Compute TICA at given lag."""
    dim = all_dists.shape[1]
    data_mean = all_dists.mean(axis=0)

    C0 = np.zeros((dim, dim))
    Ctau = np.zeros((dim, dim))
    n_c0 = 0
    n_pairs = 0

    for traj in dists_list:
        c = traj - data_mean
        T = len(traj)
        C0 += c.T @ c
        n_c0 += T
        Ctau += c[:T - lag].T @ c[lag:]
        n_pairs += T - lag

    C0 /= n_c0
    Ctau /= n_pairs
    Ctau_sym = (Ctau + Ctau.T) / 2

    ev0, U0 = scipy_eigh(C0)
    keep = ev0 > ev0.max() * 1e-10
    W = U0[:, keep] @ np.diag(1.0 / np.sqrt(ev0[keep]))
    Cw = W.T @ Ctau_sym @ W
    Cw = (Cw + Cw.T) / 2
    tica_evals, tica_evecs_w = scipy_eigh(Cw)
    tica_evecs = W @ tica_evecs_w

    idx = np.argsort(tica_evals)[::-1]
    tica_evals = tica_evals[idx]
    tica_evecs = tica_evecs[:, idx]

    all_centered = all_dists - data_mean
    proj = all_centered @ tica_evecs[:, :n_components]
    return tica_evals[:n_components], tica_evecs[:, :n_components], proj


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # ── 1. Load data ──
    print('[1/5] Loading alanine dipeptide data...')
    dists_list, all_dists, gt_phi, gt_psi = load_alanine()
    N_total, dim = all_dists.shape
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean
    print(f'  {N_total:,} frames, {dim}D')

    # ── 2. Build KDE scorer ──
    print('\n[2/5] Building KDE scorer...')
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

    # ── 3. Convergence grid (same physical times as GMM) ──
    # Physical times from GMM: log-spaced 0.1ps to 2000ps, 30 points
    phys_times = np.geomspace(0.1, 2000.0, 30)
    # Convert to KDE lag steps
    kde_lags = np.maximum(1, np.round(phys_times / best_dt).astype(int))
    # Remove duplicates while keeping physical times aligned
    unique_mask = np.concatenate([[True], np.diff(kde_lags) > 0])
    kde_lags = kde_lags[unique_mask]
    phys_times = phys_times[unique_mask]
    n_checkpoints = len(kde_lags)

    print(f'\n[3/5] Running KDE-DA at {n_checkpoints} convergence checkpoints...')
    print(f'  Physical times: {phys_times[0]:.1f}ps to {phys_times[-1]:.1f}ps')
    print(f'  KDE lags: {kde_lags[0]} to {kde_lags[-1]} steps')

    k_save = 5  # save top-5 eigenvectors
    conv_eigenvalues = np.zeros((n_checkpoints, dim))
    conv_top_k_eigenvectors = np.zeros((n_checkpoints, dim, k_save))
    conv_gap_profile = np.zeros(n_checkpoints)
    conv_gap23_profile = np.zeros(n_checkpoints)

    # Free-convolution stopping criterion
    sigma_f = np.sqrt(sigma_f_sq)
    gamma = dim / n_traj
    consecutive_below = 0
    n_computed = 0

    for i, (lag, pt) in enumerate(zip(kde_lags, phys_times)):
        t0 = time.time()
        evals, evecs, _ = extract_da_directions(
            simulator=kde_sim, lag=int(lag), n_trajectories=n_traj,
            dt=best_dt, device=device, seed=42)
        elapsed = time.time() - t0

        evals_np = evals.numpy()
        evecs_np = evecs.numpy()
        conv_eigenvalues[i] = evals_np
        conv_top_k_eigenvectors[i] = evecs_np[:, :k_save]
        conv_gap_profile[i] = evals_np[0] / evals_np[1] if evals_np[1] > 0 else float('inf')
        conv_gap23_profile[i] = evals_np[1] / evals_np[2] if evals_np[2] > 0 else float('inf')

        # Free-conv noise floor upper edge
        tau_fc = pt  # physical time = free-conv tau
        _, lam_plus = free_conv_edges(tau_fc, sigma_f, gamma=gamma)

        # Quick correlation check for DA1 and DA2
        da1_proj = all_centered @ evecs_np[:, 0]
        da2_proj = all_centered @ evecs_np[:, 1]
        r1_psi = abs(pearsonr(da1_proj, gt_psi)[0])
        r2_phi = abs(pearsonr(da2_proj, gt_phi)[0])

        # Check λ₂ (m=2): DA2 hits noise before DA1
        lam2 = evals_np[1] if len(evals_np) > 1 else 0.0
        below_floor = lam2 < lam_plus
        print(f'  [{i+1:2d}/{n_checkpoints}] lag={lag:>6d} ({pt:>8.1f}ps)  '
              f'gap12={conv_gap_profile[i]:.2f}x  gap23={conv_gap23_profile[i]:.2f}x  '
              f'DA1|r(ψ)|={r1_psi:.3f}  DA2|r(φ)|={r2_phi:.3f}  '
              f'λ₂={lam2:.6f} floor={lam_plus:.6f} {"BELOW" if below_floor else "above"}  '
              f'[{elapsed:.1f}s]')

        n_computed = i + 1

        # Stopping criterion: λ₁ below floor for 2 consecutive checkpoints
        if below_floor:
            consecutive_below += 1
            if consecutive_below >= 2:
                print(f'\n  *** Stopping: λ₂ below free-conv floor for 2 consecutive lags ***')
                break
        else:
            consecutive_below = 0

    # Trim arrays to only computed checkpoints
    kde_lags = kde_lags[:n_computed]
    phys_times = phys_times[:n_computed]
    conv_eigenvalues = conv_eigenvalues[:n_computed]
    conv_top_k_eigenvectors = conv_top_k_eigenvectors[:n_computed]
    conv_gap_profile = conv_gap_profile[:n_computed]
    conv_gap23_profile = conv_gap23_profile[:n_computed]

    print(f'\n  Computed {n_computed}/{n_checkpoints} checkpoints '
          f'(up to {phys_times[-1]:.1f}ps)')

    # ── 4. TICA (lag=10) ──
    print('\n[4/5] Computing TICA (lag=10)...')
    tica_evals, tica_evecs, tica_proj = compute_tica(dists_list, all_dists, lag=10)
    r_tica1_psi = abs(pearsonr(tica_proj[:, 0], gt_psi)[0])
    r_tica2_phi = abs(pearsonr(tica_proj[:, 1], gt_phi)[0])
    print(f'  TICA1 |r(ψ)|={r_tica1_psi:.4f}, TICA2 |r(φ)|={r_tica2_phi:.4f}')

    # ── 5. Save ──
    print('\n[5/5] Saving...')
    out_path = os.path.join(RESDIR, 'alanine_da_results_kde_convergence.npz')
    np.savez_compressed(out_path,
        # KDE config
        dt=best_dt,
        kde_h=kde.h,
        kde_n_centers=2000,
        n_traj=n_traj,
        sigma_f_sq=sigma_f_sq,
        # Convergence grid
        conv_checkpoints=kde_lags,
        conv_phys_times=phys_times,
        conv_eigenvalues=conv_eigenvalues,
        conv_top_k_eigenvectors=conv_top_k_eigenvectors,
        conv_gap_profile=conv_gap_profile,
        conv_gap23_profile=conv_gap23_profile,
        # TICA
        tica_evals=tica_evals,
        tica_evecs=tica_evecs,
        tica_proj=tica_proj,
        # Ground truth
        gt_phi=gt_phi,
        gt_psi=gt_psi,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path} ({size_mb:.1f} MB)')

    elapsed_total = time.time() - t_start
    print(f'\nTotal time: {elapsed_total/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
