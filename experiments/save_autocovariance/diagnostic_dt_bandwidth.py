#!/usr/bin/env python3
"""
Diagnostic checks 2 & 3: discretization error and KDE bandwidth sensitivity.

Check 2: Re-run at dt=0.05 (half current), lag=7252 (same physical tau*=362.6ps).
Check 3: Re-run at h=0.0091 (half) and h=0.0366 (double), dt=0.1, lag=3626.

For each, report lambda_min, |r(phi)|, |r(psi)| of the most-negative eigenvector,
plus DA1/DA2 for sanity.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import time
from scipy.stats import pearsonr
from scipy.linalg import eigh as scipy_eigh

from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore


def load_alanine():
    data_dir = os.path.join(os.path.dirname(__file__),
                            '..', 'alanine', 'main', 'alanine_dipeptide', 'data')
    dist_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    all_diheds = np.concatenate([dihed_data[f'arr_{i}'] for i in range(3)])
    return all_dists, all_diheds[:, 0], all_diheds[:, 1]


def run_single_lag(kde_sim, n_traj, lag, seed=42):
    """Run DA at a single lag. Returns (evals_asc, evecs_asc)."""
    torch.manual_seed(seed)
    x0 = kde_sim.sample_initial(n_traj)
    mu_f = kde_sim.mu_f
    x = x0.clone()
    for t in range(lag):
        x = kde_sim.step(x)
    x0_c = x0 - mu_f.unsqueeze(0)
    x_tau_c = x - mu_f.unsqueeze(0)
    C_tau = (x_tau_c.t() @ x0_c) / n_traj
    C_sym = 0.5 * (C_tau + C_tau.t())
    evals, evecs = scipy_eigh(C_sym.cpu().numpy())
    return evals, evecs


def report(label, evals, evecs, centered, gt_phi, gt_psi):
    v_min = evecs[:, 0]
    proj = centered @ v_min
    r_phi = abs(pearsonr(proj, gt_phi)[0])
    r_psi = abs(pearsonr(proj, gt_psi)[0])
    v_da1 = evecs[:, -1]
    r_da1_psi = abs(pearsonr(centered @ v_da1, gt_psi)[0])
    v_da2 = evecs[:, -2]
    r_da2_phi = abs(pearsonr(centered @ v_da2, gt_phi)[0])
    n_neg = int(np.sum(evals < 0))
    print(f'  {label}:')
    print(f'    lam_min={evals[0]:.4e}  |r(phi)|={r_phi:.4f}  |r(psi)|={r_psi:.4f}  #neg={n_neg}')
    print(f'    DA1: lam1={evals[-1]:.4e}  |r(psi)|={r_da1_psi:.4f}')
    print(f'    DA2: lam2={evals[-2]:.4e}  |r(phi)|={r_da2_phi:.4f}')
    return {
        'lam_min': evals[0], 'r_min_phi': r_phi, 'r_min_psi': r_psi,
        'lam1': evals[-1], 'r_da1_psi': r_da1_psi,
        'lam2': evals[-2], 'r_da2_phi': r_da2_phi,
        'n_neg': n_neg,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    t_start = time.time()

    # Load data
    print('Loading alanine dipeptide data...')
    all_dists, gt_phi, gt_psi = load_alanine()
    N_total, dim = all_dists.shape
    md_mean = all_dists.mean(axis=0)
    centered = all_dists - md_mean

    # Build KDE samples (same seed as canonical)
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)
    center_idx = np.random.choice(10000, 2000, replace=False)
    centers = X_sub_t[center_idx]

    n_traj = 2000
    tau_star_phys = 362.6  # ps

    results = {}

    # ── Check 2: Discretization error ──
    print('\n' + '='*70)
    print('CHECK 2: Discretization error (dt=0.05 vs dt=0.1)')
    print('='*70)

    for dt_test in [0.1, 0.05, 0.025]:
        lag = int(round(tau_star_phys / dt_test))
        actual_phys = lag * dt_test
        label = f'dt={dt_test}, lag={lag} (tau={actual_phys:.1f}ps)'
        print(f'\n  Running {label}...')

        kde = KDEScore(centers, bandwidth='scott', device=device, batch_size=1000)
        sim = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_sub_t, dt=dt_test, device=device)

        t0 = time.time()
        evals, evecs = run_single_lag(sim, n_traj, lag, seed=42)
        elapsed = time.time() - t0
        r = report(label, evals, evecs, centered, gt_phi, gt_psi)
        r['dt'] = dt_test
        r['lag'] = lag
        r['elapsed'] = elapsed
        results[f'dt_{dt_test}'] = r
        print(f'    Time: {elapsed:.1f}s')

    # ── Check 3: KDE bandwidth sensitivity ──
    print('\n' + '='*70)
    print('CHECK 3: KDE bandwidth sensitivity')
    print('='*70)

    dt = 0.1
    lag = int(round(tau_star_phys / dt))

    # Get Scott bandwidth for reference
    kde_scott = KDEScore(centers, bandwidth='scott', device=device, batch_size=1000)
    h_scott = kde_scott.h
    print(f'  Scott bandwidth: h={h_scott:.4f}')

    for h_mult, h_label in [(0.5, 'h/2'), (1.0, 'h_scott'), (2.0, '2h')]:
        h_val = h_scott * h_mult
        label = f'{h_label} (h={h_val:.4f})'
        print(f'\n  Running {label}...')

        kde = KDEScore(centers, bandwidth=h_val, device=device, batch_size=1000)
        sim = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_sub_t, dt=dt, device=device)

        t0 = time.time()
        evals, evecs = run_single_lag(sim, n_traj, lag, seed=42)
        elapsed = time.time() - t0
        r = report(label, evals, evecs, centered, gt_phi, gt_psi)
        r['h'] = h_val
        r['elapsed'] = elapsed
        results[f'h_{h_label}'] = r
        print(f'    Time: {elapsed:.1f}s')

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'diagnostic_dt_bandwidth.npz')
    # Flatten results dict for npz
    for key, val in results.items():
        for k, v in val.items():
            results[f'{key}_{k}'] = v
    # Just save as text summary too
    print(f'\nTotal time: {time.time() - t_start:.0f}s')
    print('DONE')


if __name__ == '__main__':
    main()
