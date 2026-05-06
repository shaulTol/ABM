#!/usr/bin/env python3
"""
Diagnostic: antisymmetric part of raw C(tau*) + negative-spike eigenvector analysis.

Q1: Recompute raw (unsymmetrized) C(tau*), measure antisymmetric/symmetric ratio.
Q2: Project negative-spike eigenvectors onto phi/psi dihedrals.
"""
import sys, os
import json
import numpy as np
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore
from scipy.stats import pearsonr

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
DA_DIR = os.path.join(os.path.dirname(__file__), '..', 'da_directions')


def load_alanine():
    """Load alanine dipeptide: 45D distances + dihedrals."""
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
    device = torch.device('cpu')
    t_start = time.time()

    # Load saved C_tau data
    saved = np.load(os.path.join(RESDIR, 'alanine_C_tau_full.npz'))
    tau_star = float(saved['phys_times'][23])
    lag_star = int(saved['lags'][23])
    sigma_f_sq = float(saved['sigma_f_sq'])
    dt = float(saved['dt'])
    gamma = float(saved['gamma'])
    C_tau_sym_saved = saved['C_tau'][23]
    d = int(saved['d'])
    N_traj = int(saved['N'])

    print(f'tau* = {tau_star:.1f}ps, lag = {lag_star} steps, dt = {dt}')

    # Load data
    print('\n[1/4] Loading alanine data...')
    all_dists, gt_phi, gt_psi = load_alanine()
    N_total = all_dists.shape[0]
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean

    # Build KDE scorer (same as save_alanine_C_tau.py)
    print('\n[2/4] Building KDE scorer...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)
    center_idx = np.random.choice(10000, 2000, replace=False)
    kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)

    # dt stability (same procedure to get best_dt)
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
                stable = False
                break
        if stable:
            final_norms = x_test.norm(dim=1)
            ratio = (final_norms / init_norms).max().item()
            if ratio > 10:
                stable = False
        if stable and best_dt is None:
            best_dt = test_dt
    if best_dt is None:
        best_dt = 0.001
    print(f'  dt = {best_dt}')

    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=best_dt, device=device)
    mu_f = kde_sim.mu_f

    # ================================================================
    # Q1: Recompute raw (unsymmetrized) C(tau*)
    # ================================================================
    print(f'\n[3/4] Recomputing raw C(tau*) at lag={lag_star}...')
    torch.manual_seed(42)  # same seed as original
    n_traj = N_traj

    x0 = kde_sim.sample_initial(n_traj)
    x = x0.clone()
    for t in range(lag_star):
        x = kde_sim.step(x)
    x_tau = x

    x0_c = x0 - mu_f.unsqueeze(0)
    x_tau_c = x_tau - mu_f.unsqueeze(0)

    # Raw (unsymmetrized) cross-covariance
    C_raw = (x_tau_c.t() @ x0_c / n_traj).numpy()
    C_sym = 0.5 * (C_raw + C_raw.T)
    C_anti = 0.5 * (C_raw - C_raw.T)

    F_sym = np.linalg.norm(C_sym, 'fro')
    F_anti = np.linalg.norm(C_anti, 'fro')
    F_raw = np.linalg.norm(C_raw, 'fro')

    print(f'\n  === Q1: Antisymmetric analysis ===')
    print(f'  ||C_raw||_F    = {F_raw:.6e}')
    print(f'  ||C_sym||_F    = {F_sym:.6e}')
    print(f'  ||C_anti||_F   = {F_anti:.6e}')
    print(f'  ||C_anti|| / ||C_sym|| = {F_anti/F_sym:.4f} ({F_anti/F_sym*100:.2f}%)')

    # Eigenvalues of antisymmetric part (purely imaginary for skew-symmetric)
    anti_eigvals = np.linalg.eigvals(C_anti)
    # Sort by magnitude of imaginary part
    anti_sorted = sorted(anti_eigvals, key=lambda z: -abs(z.imag))
    print(f'\n  Top eigenvalues of C_anti (should be ±i pairs):')
    for i in range(min(6, len(anti_sorted))):
        v = anti_sorted[i]
        print(f'    [{i}] {v.real:+.4e} {v.imag:+.4e}i   |imag| = {abs(v.imag):.4e}')

    # Verify consistency with saved C_tau_sym
    cos_sim = np.sum(C_sym * C_tau_sym_saved) / (np.linalg.norm(C_sym, 'fro') * np.linalg.norm(C_tau_sym_saved, 'fro'))
    print(f'\n  Cosine similarity C_sym(recomputed) vs C_sym(saved): {cos_sim:.6f}')

    # ================================================================
    # Q2: Eigenvector projections onto dihedrals
    # ================================================================
    print(f'\n[4/4] Eigenvector projections...')

    # Eigendecompose the saved symmetrized C_tau
    eigvals, eigvecs = np.linalg.eigh(C_tau_sym_saved)
    # Sort descending by eigenvalue
    idx = np.argsort(eigvals)[::-1].copy()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Positive spikes: top 2 (DA1, DA2)
    # Negative spikes: bottom 3 (most negative eigenvalues)
    pos_indices = [0, 1]  # DA1, DA2
    neg_indices = [d-1, d-2, d-3]  # most negative eigenvalues

    labels = {
        0: 'DA1 (pos spike 1)',
        1: 'DA2 (pos spike 2)',
        d-1: 'Neg spike 1 (most negative)',
        d-2: 'Neg spike 2',
        d-3: 'Neg spike 3',
    }

    print(f'\n  === Q2: Eigenvector correlations with dihedrals ===')
    print(f'  {"Direction":<30s}  {"eigenvalue":>12s}  {"|r(phi)|":>8s}  {"|r(psi)|":>8s}')
    print(f'  {"-"*30}  {"-"*12}  {"-"*8}  {"-"*8}')

    results = {}
    for i in pos_indices + neg_indices:
        v = eigvecs[:, i]
        proj = all_centered @ v

        r_phi = abs(pearsonr(proj, gt_phi)[0])
        r_psi = abs(pearsonr(proj, gt_psi)[0])

        label = labels[i]
        print(f'  {label:<30s}  {eigvals[i]:>12.4e}  {r_phi:>8.4f}  {r_psi:>8.4f}')

        results[label] = {
            'eigenvalue': float(eigvals[i]),
            'r_phi': float(r_phi),
            'r_psi': float(r_psi),
            'eigvec_idx': i,
        }

    # Cosine similarities between pos and neg spike directions
    print(f'\n  === Cosine similarities between spike directions ===')
    all_indices = pos_indices + neg_indices
    for i in range(len(all_indices)):
        for j in range(i+1, len(all_indices)):
            ii, jj = all_indices[i], all_indices[j]
            cos = abs(np.dot(eigvecs[:, ii], eigvecs[:, jj]))
            print(f'  |cos({labels[ii]}, {labels[jj]})| = {cos:.4f}')

    # Save JSON
    json_data = {
        'tau_ps': tau_star,
        'lag_steps': lag_star,
        'Q1_antisymmetric': {
            'F_raw': float(F_raw),
            'F_sym': float(F_sym),
            'F_anti': float(F_anti),
            'ratio_anti_over_sym': float(F_anti / F_sym),
            'ratio_percent': float(F_anti / F_sym * 100),
            'top_anti_eigenvalues_imag': [float(abs(anti_sorted[i].imag)) for i in range(min(6, len(anti_sorted)))],
            'cosine_recomputed_vs_saved': float(cos_sim),
        },
        'Q2_eigenvector_correlations': results,
    }

    json_path = os.path.join(RESDIR, 'diagnostic_taustar_negspikes.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\nSaved: {json_path}')

    elapsed = time.time() - t_start
    print(f'Total time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
