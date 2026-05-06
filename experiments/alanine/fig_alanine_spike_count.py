#!/usr/bin/env python3
"""
Fig A6: Spike count vs lag τ for alanine dipeptide.

At each lag τ, counts how many eigenvalues of C_sym(τ) exceed the
free-convolution upper edge λ₊(τ). Shows that the number of significant
DA directions decreases with τ (modes peel off in order of rate).

Reads: results/alanine_C_tau_full_h_half_10k.npz
Output: fig_alanine_spike_count_vs_tau.png
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.noise_floor.free_conv_utils import edges, edge_tauinf
from src.plotting.style_config import setup_style

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(EXPDIR, 'results', 'alanine_C_tau_full_h_half_10k.npz')


def main():
    setup_style()

    data = np.load(DATAPATH)
    eigenvalues = data['eigenvalues']   # (n_lags, d), sorted descending
    phys_times = data['phys_times']     # (n_lags,)
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])

    sigma = np.sqrt(sigma_f_sq)

    # Count spikes above free-conv floor at each lag
    spike_counts = np.zeros(len(phys_times), dtype=int)
    for i, tau in enumerate(phys_times):
        _, lp = edges(tau, sigma, gamma=gamma)
        spike_counts[i] = np.sum(eigenvalues[i] > lp)

    # Also compute tau-infinity floor
    _, lp_inf = edge_tauinf(sigma, gamma)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(phys_times, spike_counts, 'ko-', markersize=4, lw=1.5)

    ax.set_xscale('log')
    ax.set_xlabel(r'Lag $\tau$ (ps)', fontsize=10)
    ax.set_ylabel('Number of spikes above $\\lambda_+(\\tau)$', fontsize=10)
    ax.set_ylim(-0.5, max(spike_counts) + 1)
    ax.set_yticks(range(0, max(spike_counts) + 2))

    # Annotate key transitions
    for count_val in [1, 2]:
        # Find where count drops to this value
        where = np.where(spike_counts == count_val)[0]
        if len(where) > 0:
            first = where[0]
            ax.annotate(f'm={count_val}',
                       xy=(phys_times[first], count_val),
                       xytext=(phys_times[first] * 2, count_val + 0.8),
                       fontsize=8, ha='left',
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    fig.tight_layout()
    out_path = os.path.join(EXPDIR, 'fig_alanine_spike_count_vs_tau.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')

    # Summary
    for i, tau in enumerate(phys_times):
        if spike_counts[i] != (spike_counts[i-1] if i > 0 else -1):
            print(f'  τ={tau:>9.1f}ps  spikes={spike_counts[i]}')


if __name__ == '__main__':
    main()
