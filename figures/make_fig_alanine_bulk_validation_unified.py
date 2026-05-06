#!/usr/bin/env python3
"""
Unified alanine bulk-validation figure: 3 rows x 2 cols, 6 lags.

Replaces the old 4-panel row figure. Uses h/2 canonical data (h_Scott/2).
Layout: 3 rows x 2 cols. Lags at indices [28, 29, 32, 34, 36, 37].

Output: figures/fig_alanine_bulk_validation.png
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density, edge_tauinf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def main():
    data = np.load(DATAPATH)
    eigenvalues = data['eigenvalues']   # (38, 45), sorted descending
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    N = int(data['N'])

    sigma = np.sqrt(sigma_f_sq)

    # lambda_+(inf)
    _, lp_inf = edge_tauinf(sigma, gamma)

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.35, left=0.10, right=0.95,
                        top=0.92, bottom=0.06)

    for panel_i, idx in enumerate(SELECTED_INDICES):
        row, col = divmod(panel_i, 2)
        ax = axes[row, col]

        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx].copy()  # 45 values, descending

        # Free-conv edges at this tau (no peeling, naive sigma)
        lm_tau, lp_tau = fc_edges(tau_ps, sigma, gamma=gamma)

        # Density curve over support
        pad = 0.25 * (lp_tau - lm_tau)
        lam_grid = np.linspace(lm_tau - pad, lp_tau + pad, 600)
        rho_grid = fc_density(lam_grid, tau_ps, sigma, gamma=gamma)

        # Symlog-spaced bins: more bins near 0 where symlog stretches the axis
        linthresh = 1e-5
        def _symlog(x):
            return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)
        def _inv_symlog(y):
            return np.sign(y) * linthresh * (10**np.abs(y) - 1)

        sl_lo = _symlog(eigs.min() - 0.02 * (eigs.max() - eigs.min()))
        sl_hi = _symlog(eigs.max() + 0.02 * (eigs.max() - eigs.min()))
        bins = _inv_symlog(np.linspace(sl_lo, sl_hi, 40))

        # Histogram of all 45 eigenvalues
        ax.hist(eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
                alpha=HIST_ALPHA_EMP, edgecolor='white', linewidth=0.3, zorder=2)

        # Free-conv density
        ax.plot(lam_grid, rho_grid, color=FREECONV_CURVE, lw=CURVE_LW, zorder=4)

        # lambda_+(tau) floor (dark green dotted)
        ax.axvline(lp_tau, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)

        # lambda_+(inf) floor (orange dotted) — overlaps with tau line at this gamma
        ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
                   zorder=4, alpha=0.8)

        # Symlog x-axis and y-axis
        ax.set_xscale('symlog', linthresh=1e-5)
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(bottom=0)

        # Mark DA1 and DA2 positions (use axis transform for consistent placement)
        lam1 = eigs[0]
        lam2 = eigs[1]

        # DA1, DA2 markers at top of plot
        trans = ax.get_xaxis_transform()
        ax.plot(lam1, 0.92, marker='v', color=FREECONV_CURVE, markersize=4,
                zorder=6, transform=trans, clip_on=False)
        ax.text(lam1, 0.96, 'DA1', fontsize=6, color=FREECONV_CURVE,
                ha='center', va='bottom', fontweight='bold', transform=trans)

        if lam2 > lp_tau * 1.5:  # only label DA2 if clearly above floor
            ax.plot(lam2, 0.82, marker='v', color=FREECONV_CURVE, markersize=3,
                    zorder=6, transform=trans, clip_on=False)
            ax.text(lam2, 0.86, 'DA2', fontsize=6, color=FREECONV_CURVE,
                    ha='center', va='bottom', fontstyle='italic', transform=trans)

        # Panel label
        tau_str = f'{tau_ps:.1f}' if tau_ps < 1000 else f'{tau_ps:.0f}'
        ax.set_title(fr'$\tau = {tau_str}$ ps', fontsize=9)

        if row == 2:
            ax.set_xlabel(r'$\lambda$', fontsize=9)
        if col == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=7)

    # Shared legend at top
    legend_handles = [
        Line2D([], [], color=HIST_EMPIRICAL, lw=6, alpha=HIST_ALPHA_EMP,
               label='Eigenvalues'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label='Free-conv. density'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=r'$\lambda_+(\tau)$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               label=r'$\lambda_+(\infty)$'),
        Line2D([], [], color=FREECONV_CURVE, marker='v', ls='none', markersize=4,
               label='DA spike'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.52, 0.99))

    out_path = os.path.join(FIGDIR, 'fig_alanine_bulk_validation.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')

    # Print summary
    for idx in SELECTED_INDICES:
        tau_ps = phys_times[idx]
        eigs = eigenvalues[idx]
        lm, lp = fc_edges(tau_ps, sigma, gamma=gamma)
        print(f'  idx={idx:2d}  tau={tau_ps:>9.1f}ps  lam1={eigs[0]:.4e}  '
              f'lam2={eigs[1]:.4e}  lp(tau)={lp:.4e}  lp(inf)={lp_inf:.4e}')


if __name__ == '__main__':
    main()
