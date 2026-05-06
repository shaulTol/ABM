#!/usr/bin/env python3
"""
Alanine peeled bulk histogram: 3 rows x 2 cols, 6 lags.

Per panel: histogram of 43 non-spike eigenvalues with peeled free-conv density
overlay (tight fit) vs un-peeled naive density (loose contrast).

Output: figures/fig_alanine_bulk_peeled.png
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density, edge_tauinf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _spike_fig_style import (
    HIST_EMPIRICAL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF,
    HIST_ALPHA_EMP, CURVE_LW, VLINE_LW, VLINE_STYLE,
)
from style_config import setup_style
setup_style()

FIGDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(FIGDIR, '..', 'experiments', 'save_autocovariance',
                        'results', 'alanine_C_tau_full_h_half_10k.npz')

SELECTED_INDICES = [28, 29, 32, 34, 36, 37]

NAIVE_COLOR = '#8E44AD'   # purple for un-peeled


def main():
    data = np.load(DATAPATH)
    eigenvalues = data['eigenvalues']   # (38, 45), sorted descending
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    N = int(data['N'])
    C0 = data['C0']

    sigma_naive = np.sqrt(sigma_f_sq)

    # C(0) eigenvalues for peeling (PSD, sorted descending)
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()
    tr_C0 = eigs_C0.sum()

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7.5))
    plt.subplots_adjust(hspace=0.50, wspace=0.35, left=0.12, right=0.95,
                        top=0.91, bottom=0.06)

    for panel_i, idx in enumerate(SELECTED_INDICES):
        row, col = divmod(panel_i, 2)
        ax = axes[row, col]

        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx].copy()  # 45 values, descending

        lam1 = eigs[0]
        lam2 = eigs[1]
        bulk_eigs = eigs[2:]  # 43 non-spike eigenvalues

        # Peeled bulk variance from C(0): remove top 2 C(0) eigenvalues
        n_spikes = 2
        sigma_bulk_sq = (tr_C0 - eigs_C0[:n_spikes].sum()) / d
        sigma_bulk = np.sqrt(max(sigma_bulk_sq, 1e-20))

        # Peeled free-conv edges and density
        lm_peel, lp_peel = fc_edges(tau_ps, sigma_bulk, gamma=gamma)
        _, lp_peel_inf = edge_tauinf(sigma_bulk, gamma)

        pad_peel = 0.25 * (lp_peel - lm_peel)
        lam_grid_peel = np.linspace(lm_peel - pad_peel, lp_peel + pad_peel, 600)
        rho_grid_peel = fc_density(lam_grid_peel, tau_ps, sigma_bulk, gamma=gamma)

        # Naive (un-peeled) free-conv edges and density
        lm_naive, lp_naive = fc_edges(tau_ps, sigma_naive, gamma=gamma)
        pad_naive = 0.25 * (lp_naive - lm_naive)
        lam_grid_naive = np.linspace(lm_naive - pad_naive, lp_naive + pad_naive, 600)
        rho_grid_naive = fc_density(lam_grid_naive, tau_ps, sigma_naive, gamma=gamma)

        # Uniform bins in data space
        eig_range = bulk_eigs.max() - bulk_eigs.min()
        bins = np.linspace(bulk_eigs.min() - 0.05 * eig_range,
                           bulk_eigs.max() + 0.05 * eig_range, 22)

        # Histogram of 43 bulk eigenvalues
        ax.hist(bulk_eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
                alpha=HIST_ALPHA_EMP, edgecolor='white', linewidth=0.3, zorder=2)

        # Peeled free-conv density (red, tight fit)
        ax.plot(lam_grid_peel, rho_grid_peel, color=FREECONV_CURVE,
                lw=CURVE_LW, zorder=4)

        # Naive free-conv density (purple, loose contrast)
        ax.plot(lam_grid_naive, rho_grid_naive, color=NAIVE_COLOR,
                lw=CURVE_LW - 0.3, ls='--', zorder=3, alpha=0.7)

        # Floor lines
        # Peeled lambda_+(tau) - dark green dotted
        ax.axvline(lp_peel, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
        # Naive lambda_+(tau) - purple dotted
        ax.axvline(lp_naive, color=NAIVE_COLOR, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
                   zorder=4, alpha=0.6)
        # Peeled lambda_+(inf) - orange dotted
        ax.axvline(lp_peel_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
                   zorder=5, alpha=0.8)

        # DA1 and DA2 as red stems
        ax.set_xscale('symlog', linthresh=1e-5)

        # Get y-range for stem heights (use axis transform)
        trans = ax.get_xaxis_transform()
        ax.plot([lam1, lam1], [0, 0.85], color=FREECONV_CURVE, lw=1.5,
                transform=trans, clip_on=False, zorder=6)
        ax.plot(lam1, 0.85, marker='o', color=FREECONV_CURVE, markersize=4,
                transform=trans, clip_on=False, zorder=7)
        ax.text(lam1, 0.90, 'DA1', fontsize=6, color=FREECONV_CURVE,
                ha='center', va='bottom', fontweight='bold',
                transform=trans)

        ax.plot([lam2, lam2], [0, 0.70], color=FREECONV_CURVE, lw=1.2,
                transform=trans, clip_on=False, zorder=6)
        ax.plot(lam2, 0.70, marker='o', color=FREECONV_CURVE, markersize=3.5,
                transform=trans, clip_on=False, zorder=7)
        ax.text(lam2, 0.75, 'DA2', fontsize=6, color=FREECONV_CURVE,
                ha='center', va='bottom', fontstyle='italic',
                transform=trans)

        # Panel title
        tau_str = f'{tau_ps:.1f}' if tau_ps < 1000 else f'{tau_ps:.0f}'
        ax.set_title(fr'$\tau = {tau_str}$ ps', fontsize=9)

        if row == 2:
            ax.set_xlabel(r'$\lambda$', fontsize=9)
        if col == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)

    # Shared legend at top
    legend_handles = [
        Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP, edgecolor='white',
              label='Bulk eigenvalues ($i \\geq 3$)'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label=r'Peeled density ($\sigma^2_{\mathrm{bulk}}$)'),
        Line2D([], [], color=NAIVE_COLOR, lw=CURVE_LW - 0.3, ls='--', alpha=0.7,
               label=r'Naive density ($\sigma^2_f$)'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=r'$\lambda_+^{\mathrm{peel}}(\tau)$'),
        Line2D([], [], color=NAIVE_COLOR, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               alpha=0.6, label=r'$\lambda_+^{\mathrm{naive}}(\tau)$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               alpha=0.8, label=r'$\lambda_+^{\mathrm{peel}}(\infty)$'),
        Line2D([], [], color=FREECONV_CURVE, marker='o', ls='-', markersize=4,
               lw=1.5, label='DA spikes'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=6, frameon=False, bbox_to_anchor=(0.52, 1.0),
               columnspacing=1.0)

    out_path = os.path.join(FIGDIR, 'fig_alanine_bulk_peeled.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')

    # Summary
    s_bulk_sq = (tr_C0 - eigs_C0[:2].sum()) / d
    s_bulk = np.sqrt(max(s_bulk_sq, 1e-20))
    print(f'\n  sigma_naive^2={sigma_f_sq:.4e}  sigma_bulk^2={s_bulk_sq:.4e}')
    for idx in SELECTED_INDICES:
        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx]
        _, lp_p = fc_edges(tau_ps, s_bulk, gamma=gamma)
        _, lp_n = fc_edges(tau_ps, sigma_naive, gamma=gamma)
        print(f'  idx={idx:2d}  tau={tau_ps:>9.1f}ps  '
              f'lam1={eigs[0]:.4e}  lam2={eigs[1]:.4e}  '
              f'lp_peel={lp_p:.4e}  lp_naive={lp_n:.4e}')


if __name__ == '__main__':
    main()
