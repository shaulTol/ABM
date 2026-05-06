#!/usr/bin/env python3
"""
Alanine eigenvalue rank plot: 3 rows x 2 cols, 6 lags.

Per panel: eigenvalues lambda_i (sorted descending) vs rank i on log y-axis.
Shows DA1 and DA2 standing above the free-conv noise floor.

Output: figures/fig_alanine_eigenvalue_rank.png
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'noise_floor'))
from free_conv_utils import edges as fc_edges, edge_tauinf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _spike_fig_style import (
    HIST_EMPIRICAL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF,
    CURVE_LW, VLINE_LW, VLINE_STYLE,
)
from style_config import setup_style
setup_style()

FIGDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(FIGDIR, '..', 'experiments', 'save_autocovariance',
                        'results', 'alanine_C_tau_full_h_half_10k.npz')

SELECTED_INDICES = [28, 29, 32, 34, 36, 37]


def main():
    data = np.load(DATAPATH)
    eigenvalues = data['eigenvalues']   # (38, 45), sorted descending
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])

    sigma = np.sqrt(sigma_f_sq)
    _, lp_inf = edge_tauinf(sigma, gamma)

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.35, left=0.12, right=0.95,
                        top=0.92, bottom=0.06)

    ranks = np.arange(1, d + 1)

    for panel_i, idx in enumerate(SELECTED_INDICES):
        row, col = divmod(panel_i, 2)
        ax = axes[row, col]

        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx].copy()  # 45 values, descending

        _, lp_tau = fc_edges(tau_ps, sigma, gamma=gamma)

        # Absolute values for log scale (some eigenvalues may be negative)
        abs_eigs = np.abs(eigs)

        # Plot all eigenvalues as stems
        # Bulk eigenvalues (i >= 3) in light blue
        ax.stem(ranks[2:], abs_eigs[2:], linefmt=HIST_EMPIRICAL,
                markerfmt='o', basefmt='none', label='Bulk')
        # Style the bulk markers
        markerline, stemlines, _ = ax.stem(ranks[2:], abs_eigs[2:],
                                            linefmt='-', markerfmt='o', basefmt='none')
        plt.setp(stemlines, color=HIST_EMPIRICAL, linewidth=1.0)
        plt.setp(markerline, color=HIST_EMPIRICAL, markersize=3)

        # DA1 and DA2 in red
        markerline1, stemlines1, _ = ax.stem(ranks[:2], abs_eigs[:2],
                                              linefmt='-', markerfmt='o', basefmt='none')
        plt.setp(stemlines1, color=FREECONV_CURVE, linewidth=1.5)
        plt.setp(markerline1, color=FREECONV_CURVE, markersize=5)

        # Labels for DA1 and DA2
        ax.annotate('DA1', xy=(1, abs_eigs[0]), xytext=(4, abs_eigs[0] * 1.3),
                    fontsize=6.5, color=FREECONV_CURVE, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=FREECONV_CURVE, lw=0.5))
        ax.annotate('DA2', xy=(2, abs_eigs[1]), xytext=(5, abs_eigs[1] * 1.3),
                    fontsize=6.5, color=FREECONV_CURVE, fontstyle='italic',
                    arrowprops=dict(arrowstyle='-', color=FREECONV_CURVE, lw=0.5))

        # Floor lines
        ax.axhline(lp_tau, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=3)
        ax.axhline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
                   zorder=3, alpha=0.8)

        # Log y-axis
        ax.set_yscale('log')
        ax.set_xlim(0.5, d + 0.5)

        # Panel title
        tau_str = f'{tau_ps:.1f}' if tau_ps < 1000 else f'{tau_ps:.0f}'
        ax.set_title(fr'$\tau = {tau_str}$ ps', fontsize=9)

        if row == 2:
            ax.set_xlabel('Rank $i$', fontsize=9)
        if col == 0:
            ax.set_ylabel(r'$|\lambda_i|$', fontsize=9)
        ax.tick_params(labelsize=7)

    # Shared legend at top
    legend_handles = [
        Line2D([], [], color=FREECONV_CURVE, marker='o', markersize=5, lw=1.5,
               label='DA spikes'),
        Line2D([], [], color=HIST_EMPIRICAL, marker='o', markersize=3, lw=1.0,
               label='Bulk eigenvalues'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=r'$\lambda_+(\tau)$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               alpha=0.8, label=r'$\lambda_+(\infty)$'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=7, frameon=False, bbox_to_anchor=(0.52, 0.99))

    out_path = os.path.join(FIGDIR, 'fig_alanine_eigenvalue_rank.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')

    # Summary
    for idx in SELECTED_INDICES:
        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx]
        _, lp = fc_edges(tau_ps, sigma, gamma=gamma)
        print(f'  idx={idx:2d}  tau={tau_ps:>9.1f}ps  lam1={eigs[0]:.4e}  '
              f'lam2={eigs[1]:.4e}  lp={lp:.4e}  ratio1={eigs[0]/lp:.1f}x  '
              f'ratio2={eigs[1]/lp:.1f}x')


if __name__ == '__main__':
    main()
