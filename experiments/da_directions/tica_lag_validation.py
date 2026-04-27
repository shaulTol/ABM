#!/usr/bin/env python3
"""
TICA lag validation for alanine dipeptide (45D pairwise distances).

1. Implied timescale (ITS) convergence: compute TICA at multiple lags,
   plot top 3-4 implied timescales vs lag. They should plateau.
2. Direction sensitivity: report |r(ψ)| and |r(φ)| for TICA1/TICA2
   at each lag. Correlations should be stable.

Saves:
  - figures/fig_alanine_tica_its.png
  - results/tica_lag_validation.txt
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
FIGDIR = os.path.join(EXPDIR, 'figures')
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)


def load_alanine():
    """Load alanine dipeptide 45D distances + dihedrals."""
    # Try thesis repo (has the data)
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
        import mdshare
        os.makedirs(local_data, exist_ok=True)
        dist_f = mdshare.fetch(
            'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
            working_directory=local_data)
        dihed_f = mdshare.fetch(
            'alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
            working_directory=local_data)

    dist_data = np.load(dist_f)
    dihed_data = np.load(dihed_f)
    dists_list = [dist_data[f'arr_{i}'] for i in range(3)]
    diheds_list = [dihed_data[f'arr_{i}'] for i in range(3)]

    all_dists = np.concatenate(dists_list)
    all_diheds = np.concatenate(diheds_list)
    return dists_list, all_dists, all_diheds[:, 0], all_diheds[:, 1]


def compute_tica_at_lag(dists_list, all_dists, lag, n_components=5):
    """Compute TICA at a given lag. Returns eigenvalues, eigenvectors, projections."""
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

    # Whitened generalized eigenproblem
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
    print(f'  {N:,} frames, {dim}D, 3 trajectories (each 250ns, 1ps timestep)')

    lags = [1, 2, 5, 10, 20, 50, 100, 200]
    n_ts = 4  # number of implied timescales to track

    # Storage
    all_timescales = np.full((len(lags), n_ts), np.nan)
    all_evals = np.full((len(lags), n_ts), np.nan)
    corr_tica1_psi = np.full(len(lags), np.nan)
    corr_tica1_phi = np.full(len(lags), np.nan)
    corr_tica2_psi = np.full(len(lags), np.nan)
    corr_tica2_phi = np.full(len(lags), np.nan)

    print(f'\nComputing TICA at {len(lags)} lags...')
    for j, lag in enumerate(lags):
        evals, evecs, proj = compute_tica_at_lag(dists_list, all_dists, lag, n_components=n_ts)
        all_evals[j] = evals

        for i in range(n_ts):
            lam = evals[i]
            if 0 < lam < 1:
                all_timescales[j, i] = -lag / np.log(lam)

        r1_psi = abs(pearsonr(proj[:, 0], psi)[0])
        r1_phi = abs(pearsonr(proj[:, 0], phi)[0])
        r2_psi = abs(pearsonr(proj[:, 1], psi)[0])
        r2_phi = abs(pearsonr(proj[:, 1], phi)[0])
        corr_tica1_psi[j] = r1_psi
        corr_tica1_phi[j] = r1_phi
        corr_tica2_psi[j] = r2_psi
        corr_tica2_phi[j] = r2_phi

        ts_str = ', '.join(f'{t:.1f}' if np.isfinite(t) else 'inf'
                           for t in all_timescales[j])
        print(f'  lag={lag:>3}: λ=[{evals[0]:.4f}, {evals[1]:.4f}, {evals[2]:.4f}, {evals[3]:.4f}]  '
              f'ts=[{ts_str}] ps  '
              f'|r(TICA1,ψ)|={r1_psi:.3f}  |r(TICA2,φ)|={r2_phi:.3f}')

    # ── Figure: 2-panel ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Implied timescales vs lag
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(n_ts):
        ax1.plot(lags, all_timescales[:, i], 'o-', color=colors[i],
                 markersize=6, label=f'$t_{i+1}$')
    ax1.set_xlabel('TICA lag (frames = ps)')
    ax1.set_ylabel('Implied timescale (ps)')
    ax1.set_title('Implied Timescale Convergence')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='lag=10 (paper)')
    # Re-add legend with the vline
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=9)

    # Panel 2: Ground truth correlations vs lag
    ax2.plot(lags, corr_tica1_psi, 'bo-', markersize=6, label=r'TICA1 $|r(\psi)|$')
    ax2.plot(lags, corr_tica1_phi, 'bs--', markersize=5, alpha=0.6, label=r'TICA1 $|r(\phi)|$')
    ax2.plot(lags, corr_tica2_phi, 'ro-', markersize=6, label=r'TICA2 $|r(\phi)|$')
    ax2.plot(lags, corr_tica2_psi, 'rs--', markersize=5, alpha=0.6, label=r'TICA2 $|r(\psi)|$')
    ax2.set_xlabel('TICA lag (frames = ps)')
    ax2.set_ylabel('|Pearson r|')
    ax2.set_title('TICA Direction Sensitivity to Lag')
    ax2.set_xscale('log')
    ax2.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig_path = os.path.join(FIGDIR, 'fig_alanine_tica_its.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure saved: {fig_path}')

    # ── Summary table ──
    lines = []
    lines.append('TICA Lag Validation — Alanine Dipeptide (45D pairwise distances)')
    lines.append('Dataset: MDShare alanine-dipeptide-3x250ns, 1ps timestep, 750000 frames')
    lines.append('=' * 90)
    lines.append('')
    lines.append('1. Implied Timescales (ps)')
    lines.append('-' * 90)
    header = f'{"lag":>5}' + ''.join(f'{"t"+str(i+1):>12}' for i in range(n_ts))
    lines.append(header)
    lines.append('-' * 90)
    for j, lag in enumerate(lags):
        row = f'{lag:>5}'
        for i in range(n_ts):
            ts = all_timescales[j, i]
            row += f'{ts:>12.1f}' if np.isfinite(ts) else f'{"inf":>12}'
        lines.append(row)

    lines.append('')
    lines.append('2. TICA Direction Correlations with Ground Truth Dihedrals')
    lines.append('-' * 90)
    lines.append(f'{"lag":>5} {"TICA1 |r(psi)|":>16} {"TICA1 |r(phi)|":>16} '
                 f'{"TICA2 |r(psi)|":>16} {"TICA2 |r(phi)|":>16}')
    lines.append('-' * 90)
    for j, lag in enumerate(lags):
        lines.append(f'{lag:>5} {corr_tica1_psi[j]:>16.4f} {corr_tica1_phi[j]:>16.4f} '
                     f'{corr_tica2_psi[j]:>16.4f} {corr_tica2_phi[j]:>16.4f}')

    lines.append('')
    lines.append('3. Assessment')
    lines.append('-' * 90)

    # Check plateau: t1 at lag=10 vs lag=20
    t1_10 = all_timescales[lags.index(10), 0]
    t1_20 = all_timescales[lags.index(20), 0]
    t1_50 = all_timescales[lags.index(50), 0]
    rel_change_10_20 = abs(t1_20 - t1_10) / t1_10 * 100 if np.isfinite(t1_10) and np.isfinite(t1_20) else float('inf')
    rel_change_10_50 = abs(t1_50 - t1_10) / t1_10 * 100 if np.isfinite(t1_10) and np.isfinite(t1_50) else float('inf')

    lines.append(f't1 relative change lag=10→20: {rel_change_10_20:.1f}%')
    lines.append(f't1 relative change lag=10→50: {rel_change_10_50:.1f}%')

    # Direction stability
    r_psi_range = corr_tica1_psi[lags.index(5):lags.index(50)+1]
    r_phi_range = corr_tica2_phi[lags.index(5):lags.index(50)+1]
    lines.append(f'TICA1 |r(psi)| range [lag=5..50]: {r_psi_range.min():.4f} — {r_psi_range.max():.4f}')
    lines.append(f'TICA2 |r(phi)| range [lag=5..50]: {r_phi_range.min():.4f} — {r_phi_range.max():.4f}')

    plateau_ok = rel_change_10_20 < 10
    direction_ok = (r_psi_range.max() - r_psi_range.min()) < 0.05
    lines.append(f'ITS plateau by lag=10: {"YES" if plateau_ok else "NO"} ({rel_change_10_20:.1f}% change to lag=20)')
    lines.append(f'Direction stable lag=5..50: {"YES" if direction_ok else "NO"} (spread={r_psi_range.max()-r_psi_range.min():.4f})')

    text = '\n'.join(lines)
    print(f'\n{text}')

    txt_path = os.path.join(RESDIR, 'tica_lag_validation.txt')
    with open(txt_path, 'w') as f:
        f.write(text + '\n')
    print(f'\nSummary saved: {txt_path}')


if __name__ == '__main__':
    main()
