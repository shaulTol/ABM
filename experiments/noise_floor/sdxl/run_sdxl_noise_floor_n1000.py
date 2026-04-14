#!/usr/bin/env python3
"""
SDXL Square N=1000: noise-floor analysis (from cached eigenvalues).

Uses eigenvalue profiles from clip_da_autoresearch_square_N1000_T1500.npz.
d=768, N=1000, dt=0.001, T=1500.

sigma_f_sq estimated from N=200 CLIP embeddings (population property,
sample-size independent to first order).

Computes both conservative and aggressive analytic floors.
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

    # --- Load eigenvalue data ---
    results_dir = os.path.join(SDXL_DIR, 'results_autoresearch')
    da_npz = np.load(os.path.join(results_dir,
                     'clip_da_autoresearch_square_N1000_T1500.npz'),
                     allow_pickle=True)
    lags = da_npz['snapshot_lags']  # (20,)
    dt = 0.001

    all_eigenvalues = np.zeros((len(lags), 10))
    for i, lag in enumerate(lags):
        all_eigenvalues[i] = da_npz[f'eigenvalues_lag{lag}']

    # sigma_f_sq = (1/d) * tr(Cov_f) in CLIP space, computed from N=200 embeddings.
    # This is a population property of the distribution, independent of N.
    d = 768
    N = 1000  # N for the floor computation
    sigma_f_sq = 0.000340  # from N=200 CLIP embeddings (see run_sdxl_noise_floor.py)
    gamma = d / N

    print(f'd={d}, N={N}, dt={dt}')
    print(f'sigma_f_sq = {sigma_f_sq:.6f} (from N=200 CLIP embeddings)')
    print(f'gamma = {gamma:.3f}')
    print(f'Lags: {lags}')
    print(f'λ_1 range: [{all_eigenvalues[:,0].min():.5f}, {all_eigenvalues[:,0].max():.5f}]')

    # --- Analytic floors ---
    phys_times = lags * dt
    floor_upper = analytic_floor_upper(phys_times, sigma_f_sq, d, N)
    floor_lower = analytic_floor_lower(phys_times, sigma_f_sq, d, N)

    print(f'\nFloor upper range: [{floor_upper.min():.5f}, {floor_upper.max():.5f}]')
    print(f'Floor lower range: [{floor_lower.min():.5f}, {floor_lower.max():.5f}]')

    # --- Find τ* ---
    tau_idx_cons_m1, tau_cons_m1 = find_first_crossing(lags, all_eigenvalues, floor_upper, m=1)
    tau_idx_aggr_m1, tau_aggr_m1 = find_first_crossing(lags, all_eigenvalues, floor_lower, m=1)
    tau_idx_cons_m2, tau_cons_m2 = find_first_crossing(lags, all_eigenvalues, floor_upper, m=2)
    tau_idx_aggr_m2, tau_aggr_m2 = find_first_crossing(lags, all_eigenvalues, floor_lower, m=2)

    print(f'\nConservative (upper bound):')
    print(f'  τ*(m=1): lag={tau_cons_m1}')
    print(f'    λ₁={all_eigenvalues[tau_idx_cons_m1,0]:.5f}, floor={floor_upper[tau_idx_cons_m1]:.5f}')
    print(f'  τ*(m=2): lag={tau_cons_m2}')
    print(f'    λ₂={all_eigenvalues[tau_idx_cons_m2,1]:.5f}, floor={floor_upper[tau_idx_cons_m2]:.5f}')

    print(f'Aggressive (lower bound):')
    print(f'  τ*(m=1): lag={tau_aggr_m1}')
    print(f'    λ₁={all_eigenvalues[tau_idx_aggr_m1,0]:.5f}, floor={floor_lower[tau_idx_aggr_m1]:.5f}')
    print(f'  τ*(m=2): lag={tau_aggr_m2}')
    print(f'    λ₂={all_eigenvalues[tau_idx_aggr_m2,1]:.5f}, floor={floor_lower[tau_idx_aggr_m2]:.5f}')

    # Compare at each lag
    print(f'\nPer-lag comparison:')
    for i, lag in enumerate(lags):
        status_u = '✓' if all_eigenvalues[i, 0] > floor_upper[i] else '✗'
        status_l = '✓' if all_eigenvalues[i, 0] > floor_lower[i] else '✗'
        print(f'  lag={lag:4d}: λ₁={all_eigenvalues[i,0]:.5f} '
              f'upper={floor_upper[i]:.5f} {status_u}  '
              f'lower={floor_lower[i]:.5f} {status_l}')

    # Gap-max comparison
    print(f'\nOld gap-max (N=200): τ*=1189, DA1~gender ρ=0.905')
    print(f'New gap-max (N=1000): τ*=1265, DA1~gender ρ=0.843')

    # --- Save JSON ---
    results = {
        'experiment': 'sdxl_square_noise_floor_N1000',
        'parameters': {
            'd': d, 'N': N, 'dt': dt, 'T': 1500,
            'sigma_f_sq': float(sigma_f_sq),
            'gamma': float(gamma),
        },
        'lags': lags.tolist(),
        'eigenvalues_top5': all_eigenvalues[:, :5].tolist(),
        'floor_upper': floor_upper.tolist(),
        'floor_lower': floor_lower.tolist(),
        'conservative': {
            'tau_star_m1': int(tau_cons_m1),
            'tau_star_m2': int(tau_cons_m2),
        },
        'aggressive': {
            'tau_star_m1': int(tau_aggr_m1),
            'tau_star_m2': int(tau_aggr_m2),
        },
    }

    json_path = os.path.join(OUTDIR, 'noise_floor_sdxl_N1000.json')
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

    # Mark τ* if found
    if tau_cons_m1 is not None:
        ax.axvline(tau_cons_m1, color='black', ls=':', lw=0.8, alpha=0.7)
        ax.annotate(rf'$\tau^*_{{cons}}$ = {tau_cons_m1}',
                    xy=(tau_cons_m1, all_eigenvalues[tau_idx_cons_m1, 0]),
                    xytext=(tau_cons_m1 + 100, all_eigenvalues[tau_idx_cons_m1, 0] * 1.1),
                    fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.6))

    ax.set_xlabel(r'Lag $\tau$ (Langevin steps)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('SDXL Square: noise floor (d=768, N=1000)', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig_path = os.path.join(FIGDIR, 'fig_sdxl_noise_floor_N1000.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved {fig_path}')


if __name__ == '__main__':
    main()
