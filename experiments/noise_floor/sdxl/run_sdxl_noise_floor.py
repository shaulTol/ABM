#!/usr/bin/env python3
"""
SDXL Square: noise-floor-based DA lag selection (from cached data).

Uses cached eigenvalue profiles from clip_da_autoresearch_square_T2500.npz.
d=768 (CLIP ViT-L/14), N=200, dt=0.001, T=2500.

Computes both conservative and aggressive analytic floors.
sigma_f_sq from CLIP embeddings.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures.style_config import setup_style, SSA_COLOR, MI_COLOR, TEXT_WIDTH

setup_style()

from experiments.noise_floor.noise_floor_utils import (
    analytic_floor_upper, analytic_floor_lower, find_first_crossing
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures')
SDXL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sdxl')


def main():
    np.random.seed(42)

    # --- Load cached data ---
    sq = np.load(os.path.join(SDXL_DIR, 'results_autoresearch',
                              'clip_da_autoresearch_square_T2500.npz'),
                 allow_pickle=True)
    lags = sq['snapshot_lags']  # (20,)
    mu_clip = sq['mu_clip']    # (768,)

    # CLIP embeddings for sigma_f_sq
    proj = np.load(os.path.join(os.path.dirname(__file__), '..', '..', 'autoresearch',
                                'results_face_boxed', 'phase3_rank2',
                                'projections_vitl14.npz'))
    clip_emb = proj['embeddings']  # (200, 768)

    d = clip_emb.shape[1]   # 768
    N = clip_emb.shape[0]   # 200
    dt = 0.001

    # sigma_f_sq from CLIP embeddings
    centered = clip_emb - mu_clip
    cov_trace = np.sum(centered ** 2) / N
    sigma_f_sq = cov_trace / d
    gamma = d / N

    print(f'd={d}, N={N}, dt={dt}')
    print(f'sigma_f_sq = {sigma_f_sq:.6f}, gamma = {gamma:.3f}')
    print(f'Lags: {lags}')

    # Collect eigenvalues at each lag (top 10 saved)
    all_eigenvalues = np.zeros((len(lags), 10))
    for i, lag in enumerate(lags):
        all_eigenvalues[i] = sq[f'eigenvalues_lag{lag}']

    print(f'\nEigenvalue range: λ_1 [{all_eigenvalues[:,0].min():.5f}, {all_eigenvalues[:,0].max():.5f}]')

    # --- Analytic floors ---
    phys_times = lags * dt
    floor_upper = analytic_floor_upper(phys_times, sigma_f_sq, d, N)
    floor_lower = analytic_floor_lower(phys_times, sigma_f_sq, d, N)

    print(f'Floor upper range: [{floor_upper.min():.5f}, {floor_upper.max():.5f}]')
    print(f'Floor lower range: [{floor_lower.min():.5f}, {floor_lower.max():.5f}]')

    # --- Find τ* ---
    tau_idx_cons_m1, tau_cons_m1 = find_first_crossing(lags, all_eigenvalues, floor_upper, m=1)
    tau_idx_aggr_m1, tau_aggr_m1 = find_first_crossing(lags, all_eigenvalues, floor_lower, m=1)

    print(f'\nConservative τ*(m=1): lag={tau_cons_m1}')
    if tau_cons_m1 is not None:
        print(f'  λ₁={all_eigenvalues[tau_idx_cons_m1,0]:.5f}, floor={floor_upper[tau_idx_cons_m1]:.5f}')
    print(f'Aggressive τ*(m=1): lag={tau_aggr_m1}')
    if tau_aggr_m1 is not None:
        print(f'  λ₁={all_eigenvalues[tau_idx_aggr_m1,0]:.5f}, floor={floor_lower[tau_idx_aggr_m1]:.5f}')

    # Check if λ₁ ever exceeds either floor
    above_upper = all_eigenvalues[:, 0] > floor_upper
    above_lower = all_eigenvalues[:, 0] > floor_lower
    print(f'\nλ₁ > upper floor at any lag: {above_upper.any()} ({above_upper.sum()}/{len(lags)} lags)')
    print(f'λ₁ > lower floor at any lag: {above_lower.any()} ({above_lower.sum()}/{len(lags)} lags)')

    # --- Save JSON ---
    results = {
        'experiment': 'sdxl_square_noise_floor',
        'parameters': {
            'd': d, 'N': N, 'dt': dt,
            'sigma_f_sq': float(sigma_f_sq),
            'gamma': float(gamma),
        },
        'lags': lags.tolist(),
        'eigenvalues_top10': all_eigenvalues.tolist(),
        'floor_upper': floor_upper.tolist(),
        'floor_lower': floor_lower.tolist(),
        'conservative': {
            'tau_star_m1_lag': int(tau_cons_m1) if tau_cons_m1 is not None else None,
        },
        'aggressive': {
            'tau_star_m1_lag': int(tau_aggr_m1) if tau_aggr_m1 is not None else None,
        },
        'old_gap_max_tau': 1189,
        'old_gender_r': 0.905,
    }

    json_path = os.path.join(OUTDIR, 'noise_floor_sdxl.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {json_path}')

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.6))

    ax.plot(lags, all_eigenvalues[:, 0], 'o-', color=SSA_COLOR,
            markersize=4, lw=1.4, label=r'$\lambda_1(\tau)$')
    ax.plot(lags, all_eigenvalues[:, 1], 's-', color='#2ca02c',
            markersize=3.5, lw=1.2, label=r'$\lambda_2(\tau)$')
    ax.plot(lags, all_eigenvalues[:, 2], '^-', color='#ff7f0e',
            markersize=3, lw=1.0, alpha=0.7, label=r'$\lambda_3(\tau)$')
    ax.plot(lags, floor_upper, ':', color=MI_COLOR,
            lw=1.2, label='Conservative floor (upper)')
    ax.plot(lags, floor_lower, '--', color=MI_COLOR,
            lw=1.2, label='Aggressive floor (lower)')

    ax.set_xlabel(r'Lag $\tau$ (Langevin steps)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('SDXL Square: noise floor lag selection (d=768, N=200)', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig_path = os.path.join(FIGDIR, 'fig_sdxl_noise_floor.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved {fig_path}')


if __name__ == '__main__':
    main()
