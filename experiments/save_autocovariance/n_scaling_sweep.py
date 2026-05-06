#!/usr/bin/env python3
"""
N-scaling sweep: does |λ_min| of C_sym(τ*) scale as 1/√N?

Re-simulates at τ*=3626 steps (362.6ps) for N ∈ {500, 1000, 2000, 4000, 8000}.
Reports λ_min, its eigenvector's |r(φ)|, |r(ψ)|, plus DA1/DA2 for sanity.

Uses the original pipeline (no burn-in) to match the canonical run.
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
    """Load alanine dipeptide: 45D heavy-atom distances + dihedrals."""
    data_dir = os.path.join(os.path.dirname(__file__),
                            '..', 'da_directions', 'alanine_dipeptide', 'data')
    dist_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')

    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    all_diheds = np.concatenate([dihed_data[f'arr_{i}'] for i in range(3)])
    return all_dists, all_diheds[:, 0], all_diheds[:, 1]


def run_single_lag(kde_sim, n_traj, lag, seed=42):
    """
    Run extract_da_directions-equivalent at a single lag.
    Returns (C_sym, evals_asc, evecs_asc) with evals in ascending order.
    """
    torch.manual_seed(seed)
    x0 = kde_sim.sample_initial(n_traj)
    mu_f = kde_sim.mu_f

    # Forward simulate τ steps
    x = x0.clone()
    for t in range(lag):
        x = kde_sim.step(x)
    x_tau = x

    # Center by simulator's mu_f (same as original pipeline)
    x0_c = x0 - mu_f.unsqueeze(0)
    x_tau_c = x_tau - mu_f.unsqueeze(0)

    # C(τ) and symmetrize
    C_tau = (x_tau_c.t() @ x0_c) / n_traj
    C_sym = 0.5 * (C_tau + C_tau.t())

    # Eigendecompose (CPU for scipy)
    C_np = C_sym.cpu().numpy()
    evals, evecs = scipy_eigh(C_np)  # ascending
    return C_np, evals, evecs


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # Load data
    print('Loading alanine dipeptide data...')
    all_dists, gt_phi, gt_psi = load_alanine()
    N_total, dim = all_dists.shape
    md_mean = all_dists.mean(axis=0)
    centered = all_dists - md_mean

    # Build KDE scorer (identical to canonical pipeline)
    print('Building KDE scorer...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, 10000, replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)
    center_idx = np.random.choice(10000, 2000, replace=False)
    kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)
    print(f'  KDE: h={kde.h:.4f}')

    dt = 0.1  # same as canonical
    kde_sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_sub_t, dt=dt, device=device)

    lag = 3626  # τ* = 362.6ps
    N_values = [500, 1000, 2000, 4000, 8000]

    print(f'\nN-scaling sweep at τ*={lag*dt:.1f}ps (lag={lag})')
    print(f'{"="*80}')
    print(f'{"N":>6} {"λ_min":>12} {"v_min|r(φ)|":>12} {"v_min|r(ψ)|":>12} '
          f'{"λ₁":>12} {"DA1|r(ψ)|":>12} {"λ₂":>12} {"DA2|r(φ)|":>12} {"time":>8}')
    print(f'{"-"*80}')

    results = []
    for N in N_values:
        t0 = time.time()
        C_np, evals, evecs = run_single_lag(kde_sim, N, lag, seed=42)
        elapsed = time.time() - t0

        # Most-negative eigenvector (index 0, ascending)
        v_min = evecs[:, 0]
        proj_min = centered @ v_min
        r_min_phi = abs(pearsonr(proj_min, gt_phi)[0])
        r_min_psi = abs(pearsonr(proj_min, gt_psi)[0])

        # DA1 (index -1, largest)
        v_da1 = evecs[:, -1]
        proj_da1 = centered @ v_da1
        r_da1_psi = abs(pearsonr(proj_da1, gt_psi)[0])

        # DA2 (index -2)
        v_da2 = evecs[:, -2]
        proj_da2 = centered @ v_da2
        r_da2_phi = abs(pearsonr(proj_da2, gt_phi)[0])

        print(f'{N:>6} {evals[0]:>12.4e} {r_min_phi:>12.4f} {r_min_psi:>12.4f} '
              f'{evals[-1]:>12.4e} {r_da1_psi:>12.4f} {evals[-2]:>12.4e} {r_da2_phi:>12.4f} '
              f'{elapsed:>7.1f}s')

        results.append({
            'N': N, 'lam_min': evals[0],
            'r_min_phi': r_min_phi, 'r_min_psi': r_min_psi,
            'lam1': evals[-1], 'r_da1_psi': r_da1_psi,
            'lam2': evals[-2], 'r_da2_phi': r_da2_phi,
            'n_negative': int(np.sum(evals < 0)),
        })

    # Scaling analysis
    print(f'\n{"="*80}')
    print('Scaling analysis:')
    lam_mins = [r['lam_min'] for r in results]
    Ns = [r['N'] for r in results]
    for i in range(1, len(results)):
        N_ratio = Ns[i] / Ns[0]
        lam_ratio = abs(lam_mins[i]) / abs(lam_mins[0])
        expected_ratio_sqrt = 1.0 / np.sqrt(N_ratio)  # 1/√N scaling
        print(f'  N={Ns[i]:>5} vs N={Ns[0]}: |λ_min| ratio={lam_ratio:.3f}, '
              f'expected if 1/√N: {expected_ratio_sqrt:.3f}, '
              f'expected if 1/N: {1.0/N_ratio:.3f}')

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'n_scaling_sweep.npz')
    np.savez(out_path,
             N_values=np.array(Ns),
             lam_min=np.array(lam_mins),
             r_min_phi=np.array([r['r_min_phi'] for r in results]),
             r_min_psi=np.array([r['r_min_psi'] for r in results]),
             lam1=np.array([r['lam1'] for r in results]),
             lam2=np.array([r['lam2'] for r in results]),
             n_negative=np.array([r['n_negative'] for r in results]),
             lag=lag, dt=dt)
    print(f'\nSaved: {out_path}')

    elapsed_total = time.time() - t_start
    print(f'Total time: {elapsed_total:.0f}s')


if __name__ == '__main__':
    main()
