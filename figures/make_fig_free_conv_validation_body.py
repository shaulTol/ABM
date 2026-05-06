#!/usr/bin/env python3
"""
Compact free-convolution validation figure for the paper body.

Layout: 1 row x 4 cols (also saved as 4 individual panels).
  gamma = 0.25
  Cols: tau in {0, 0.5, 2, 10}

Output: figures/fig_free_conv_validation_body.{pdf,png}
        figures/fig_free_conv_body_tau{0,0p5,2,10}.{pdf,png}
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

GAMMA = 0.25
TAUS = [0.0, 0.5, 2.0, 10.0]
TAU_NAMES = {0.0: 'tau0', 0.5: 'tau0p5', 2.0: 'tau2', 10.0: 'tau10'}
N_FIXED = 500
SIGMA = 1.0
N_SEEDS = 1
DT = 0.01


def simulate_ou_csym(d, N, tau, dt, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((N, d))
    if tau == 0.0:
        x_tau = x0.copy()
    else:
        n_steps = max(int(round(tau / dt)), 1)
        x = x0.copy()
        for _ in range(n_steps):
            x = x - 0.5 * x * dt + np.sqrt(dt) * rng.standard_normal((N, d))
        x_tau = x
    C_raw = (x_tau.T @ x0) / N
    C_sym = 0.5 * (C_raw + C_raw.T)
    return np.linalg.eigvalsh(C_sym)


def plot_one_panel(ax, tau, all_eigs, lp_inf, gamma, show_ylabel=True,
                   show_legend=False):
    """Plot a single panel with histogram + density + floor lines."""
    lm, lp = fc_edges(tau, SIGMA, gamma=gamma)
    pad = 0.15 * (lp - lm)
    lam_grid = np.linspace(lm - pad, lp + pad, 600)
    rho_grid = fc_density(lam_grid, tau, SIGMA, gamma=gamma)

    # Histogram
    eig_min, eig_max = all_eigs.min(), all_eigs.max()
    margin = 0.03 * (eig_max - eig_min)
    bins = np.linspace(eig_min - margin, eig_max + margin, 35)
    ax.hist(all_eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
            alpha=HIST_ALPHA_EMP, edgecolor='none', zorder=2)
    ax.plot(lam_grid, rho_grid, color=FREECONV_CURVE,
            lw=CURVE_LW, zorder=4)

    # Floor lines — both on all panels
    ax.axvline(lp, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
    ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE,
               lw=VLINE_LW - 0.2, zorder=5, alpha=0.8)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(-1.2, 2.7)

    # Zero line
    ax.axvline(0, color='black', ls='-', lw=0.3, alpha=0.2)

    # Title
    tau_str = '0' if tau == 0 else f'{tau:g}'
    ax.set_title(fr'$\tau = {tau_str}$', fontsize=9)

    # Axes labels
    ax.set_xlabel(r'$\lambda$', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Density', fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.set_xticks([-1, 0, 1, 2])
    ax.tick_params(labelsize=6.5)

    # Legend
    if show_legend:
        legend_handles = [
            Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP,
                  label='Empirical'),
            Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
                   label='Analytic'),
            Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
                   label=r'$\lambda_+(\tau)$'),
            Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
                   alpha=0.8, label=r'$\lambda_+(\infty)$'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', ncol=1,
                  fontsize=6, frameon=True, fancybox=True,
                  edgecolor='#cccccc',
                  handlelength=1.2, handletextpad=0.4)


def main():
    gamma = GAMMA
    d = int(round(gamma * N_FIXED))
    print(f'gamma={gamma}, d={d}')

    # Pre-compute eigenvalues
    all_data = {}
    for ci, tau in enumerate(TAUS):
        eigs = simulate_ou_csym(d, N_FIXED, tau, DT, seed=42 + ci * 10000)
        all_data[ci] = eigs
        print(f'  tau={tau}: {len(eigs)} eigs, '
              f'range [{eigs.min():.4f}, {eigs.max():.4f}]')

    _, lp_inf = edge_tauinf(SIGMA, gamma)

    # --- Combined 1x4 + legend strip figure ---
    # 5 columns: 4 panels + 1 narrow legend strip
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.3),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.32],
                                          'wspace': 0.08})
    plt.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.25)

    for ci, tau in enumerate(TAUS):
        plot_one_panel(axes[ci], tau, all_data[ci], lp_inf, gamma,
                       show_ylabel=(ci == 0),
                       show_legend=False)

    # Legend in the 5th "panel" (invisible axes)
    ax_leg = axes[4]
    ax_leg.axis('off')
    legend_handles = [
        Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP,
              label='Empirical'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label='Analytic'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=r'$\lambda_+(\tau)$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               alpha=0.8, label=r'$\lambda_+(\infty)$'),
    ]
    ax_leg.legend(handles=legend_handles, loc='center', ncol=1,
                  fontsize=6.5, frameon=True, fancybox=True,
                  edgecolor='#cccccc',
                  handlelength=1.2, handletextpad=0.4)

    base = os.path.join(FIGDIR, 'fig_free_conv_validation_body')
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    print(f'\nSaved: {base}.pdf')
    print(f'Saved: {base}.png')

    # --- 4 individual panels ---
    for ci, tau in enumerate(TAUS):
        fig_s, ax_s = plt.subplots(figsize=(2.5, 2.0))
        is_first = (ci == 0)
        plt.subplots_adjust(left=(0.18 if is_first else 0.08),
                            right=0.95, top=0.85, bottom=0.18)

        plot_one_panel(ax_s, tau, all_data[ci], lp_inf, gamma,
                       show_ylabel=is_first,
                       show_legend=is_first)

        name = TAU_NAMES[tau]
        path = os.path.join(FIGDIR, f'fig_free_conv_body_{name}')
        fig_s.savefig(path + '.pdf')
        fig_s.savefig(path + '.png', dpi=300)
        plt.close(fig_s)
        print(f'Saved: {path}.pdf')

    print('Done.')


if __name__ == '__main__':
    main()
