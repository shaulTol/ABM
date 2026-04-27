#!/usr/bin/env python3
"""
Generate TICA implied timescale and direction sensitivity figure.
Saves to figures/fig_tica_its.png.
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(project_root, 'figures')


def load_alanine():
    thesis_data = os.path.expanduser(
        '~/github/thesis/modality_score/autocorrelation-mixture-analysis/'
        'experiments/da_directions/alanine_dipeptide/data')
    local_data = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')

    for dd in [thesis_data, local_data]:
        dist_f = os.path.join(dd, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
        dihed_f = os.path.join(dd, 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
        if os.path.exists(dist_f) and os.path.exists(dihed_f):
            break
    else:
        raise FileNotFoundError("Alanine data not found")

    dist_data = np.load(dist_f)
    dihed_data = np.load(dihed_f)
    dists_list = [dist_data[f'arr_{i}'] for i in range(3)]
    diheds_list = [dihed_data[f'arr_{i}'] for i in range(3)]
    all_dists = np.concatenate(dists_list)
    all_diheds = np.concatenate(diheds_list)
    return dists_list, all_dists, all_diheds[:, 0], all_diheds[:, 1]


def compute_tica_at_lag(dists_list, all_dists, lag, n_components=5):
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
    print('Loading alanine dipeptide data...')
    dists_list, all_dists, phi, psi = load_alanine()
    N, dim = all_dists.shape
    print(f'  {N:,} frames, {dim}D')

    lags = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    n_ts = 4

    all_timescales = np.full((len(lags), n_ts), np.nan)
    corr_tica1_psi = np.full(len(lags), np.nan)
    corr_tica1_phi = np.full(len(lags), np.nan)
    corr_tica2_psi = np.full(len(lags), np.nan)
    corr_tica2_phi = np.full(len(lags), np.nan)

    print(f'Computing TICA at {len(lags)} lags...')
    for j, lag in enumerate(lags):
        evals, evecs, proj = compute_tica_at_lag(dists_list, all_dists, lag, n_components=n_ts)

        for i in range(n_ts):
            lam = evals[i]
            if 0 < lam < 1:
                all_timescales[j, i] = -lag / np.log(lam)

        corr_tica1_psi[j] = abs(pearsonr(proj[:, 0], psi)[0])
        corr_tica1_phi[j] = abs(pearsonr(proj[:, 0], phi)[0])
        corr_tica2_psi[j] = abs(pearsonr(proj[:, 1], psi)[0])
        corr_tica2_phi[j] = abs(pearsonr(proj[:, 1], phi)[0])

        ts_str = ', '.join(f'{t:.1f}' if np.isfinite(t) else 'inf'
                           for t in all_timescales[j])
        print(f'  lag={lag:>3}: ts=[{ts_str}]  '
              f'|r(TICA1,ψ)|={corr_tica1_psi[j]:.3f}  '
              f'|r(TICA2,φ)|={corr_tica2_phi[j]:.3f}')

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Implied Timescales
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(n_ts):
        ax1.plot(all_timescales[:, i], lags, 'o-', color=colors[i],
                 markersize=6, label=f'$t_{i+1}$')
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='lag=10 (paper)')
    ax1.set_ylabel('TICA lag (frames = ps)')
    ax1.set_xlabel('Implied timescale (ps)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Direction Sensitivity
    ax2.plot(corr_tica1_psi, lags, 'o-', color='blue', markersize=6,
             label=r'TICA1 $|r(\psi)|$')
    ax2.plot(corr_tica1_phi, lags, 'o--', color='blue', markersize=5, alpha=0.6,
             label=r'TICA1 $|r(\phi)|$')
    ax2.plot(corr_tica2_phi, lags, 'o-', color='red', markersize=6,
             label=r'TICA2 $|r(\phi)|$')
    ax2.plot(corr_tica2_psi, lags, 'o--', color='red', markersize=5, alpha=0.6,
             label=r'TICA2 $|r(\psi)|$')
    ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('TICA lag (frames = ps)')
    ax2.set_xlabel('|Pearson r|')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    fig_path = os.path.join(FIGDIR, 'fig_tica_its.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {fig_path}')


if __name__ == '__main__':
    main()
