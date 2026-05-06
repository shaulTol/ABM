#!/usr/bin/env python3
"""
Unified free-convolution validation figure.

Two output modes:
  1. Single 6x4 grid PNG: fig_free_conv_validation.png
  2. Multi-page PDF with one 2x2 page per gamma: fig_free_conv_validation_pages.pdf

Rows/pages: gamma in {0.25, 0.5, 0.75, 1, 2, 5}. Fixed N=500, d=gamma*N.
Cols: tau in {0, 0.5, 2, 10}.

Output: figures/fig_free_conv_validation.png
        figures/fig_free_conv_validation_pages.pdf
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.noise_floor.free_conv_utils import edges as fc_edges, density as fc_density, edge_tauinf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.plotting.spike_fig_style import (
    HIST_EMPIRICAL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF,
    HIST_ALPHA_EMP, CURVE_LW, VLINE_LW, VLINE_STYLE,
)
from src.plotting.style_config import setup_style
setup_style()

FIGDIR = os.path.dirname(os.path.abspath(__file__))

GAMMAS = [0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
TAUS = [0.0, 0.5, 2.0, 10.0]
N_FIXED = 500
SIGMA = 1.0
N_SEEDS = 1
DT = 0.01

ZERO_TOL = 1e-8
ATOM_COLOR = '#7B2D8E'

# Gamma name map for filenames
GAMMA_NAMES = {0.25: 'g0p25', 0.5: 'g0p5', 0.75: 'g0p75',
               1.0: 'g1p0', 2.0: 'g2p0', 5.0: 'g5p0'}


def simulate_ou_csym(d, N, tau, dt, seed):
    """Simulate OU Langevin and return eigenvalues of C_sym(tau)."""
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


def compute_shared_axes(all_data, gi, gamma):
    """Compute shared x-axis and y-axis limits across all taus for a given gamma."""
    # Gather all eigenvalue ranges across taus
    x_min, x_max = np.inf, -np.inf
    for ci in range(len(TAUS)):
        eigs = all_data[(gi, ci)]
        # For atom cases, look at non-zero eigenvalues only for x-range
        tau = TAUS[ci]
        if tau == 0.0:
            atom_mass = max(0.0, 1.0 - 1.0 / gamma)
        else:
            atom_mass = max(0.0, 1.0 - 2.0 / gamma)
        if atom_mass > 0:
            nonzero = eigs[np.abs(eigs) > ZERO_TOL]
            if len(nonzero) > 0:
                x_min = min(x_min, nonzero.min())
                x_max = max(x_max, nonzero.max())
        else:
            x_min = min(x_min, eigs.min())
            x_max = max(x_max, eigs.max())

    # Also include free-conv edges at tau=0 (widest support)
    lm0, lp0 = fc_edges(0.0, SIGMA, gamma=gamma)
    x_min = min(x_min, lm0)
    x_max = max(x_max, lp0)

    x_pad = 0.08 * (x_max - x_min)
    return x_min - x_pad, x_max + x_pad


def plot_panel(ax, gamma, tau, all_eigs, lp_inf, show_title=True,
               show_xlabel=True, show_ylabel=True, xlim=None, ylim=None):
    """Plot a single panel: histogram + density + floor lines + annotations."""
    d = int(round(gamma * N_FIXED))

    # Atom mass depends on tau
    if tau == 0.0:
        atom_mass = max(0.0, 1.0 - 1.0 / gamma)
    else:
        atom_mass = max(0.0, 1.0 - 2.0 / gamma)
    bulk_mass = 1.0 - atom_mass

    # Free-conv edges and density
    lm, lp = fc_edges(tau, SIGMA, gamma=gamma)
    pad = 0.15 * (lp - lm)
    lam_grid = np.linspace(lm - pad, lp + pad, 600)
    rho_grid = fc_density(lam_grid, tau, SIGMA, gamma=gamma)

    # --- Histogram ---
    _atom_frac = 0.0
    if atom_mass > 0:
        # Separate zeros from non-zero bulk
        nonzero_mask = np.abs(all_eigs) > ZERO_TOL
        bulk_eigs = all_eigs[nonzero_mask]
        _atom_frac = np.sum(~nonzero_mask) / len(all_eigs)

        eig_min, eig_max = bulk_eigs.min(), bulk_eigs.max()
        margin = 0.03 * (eig_max - eig_min)
        bins = np.linspace(eig_min - margin, eig_max + margin, 35)

        ax.hist(bulk_eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
                alpha=HIST_ALPHA_EMP, edgecolor='none', zorder=2)
        ax.plot(lam_grid, rho_grid / bulk_mass, color=FREECONV_CURVE,
                lw=CURVE_LW, zorder=4)
    else:
        eig_min, eig_max = all_eigs.min(), all_eigs.max()
        margin = 0.03 * (eig_max - eig_min)
        bins = np.linspace(eig_min - margin, eig_max + margin, 35)
        ax.hist(all_eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
                alpha=HIST_ALPHA_EMP, edgecolor='none', zorder=2)
        ax.plot(lam_grid, rho_grid, color=FREECONV_CURVE,
                lw=CURVE_LW, zorder=4)

    # lambda_+(tau) vertical (dark green dotted)
    ax.axvline(lp, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)

    # lambda_+(inf) vertical (orange dotted) -- always shown
    ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE,
               lw=VLINE_LW - 0.2, zorder=5, alpha=0.8)

    # gamma=2, tau>0: log y so the density spike near 0 doesn't crush the bulk
    if gamma == 2.0 and tau > 0:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3)
    elif ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    if xlim is not None:
        ax.set_xlim(xlim)

    # Zero line
    ax.axvline(0, color='black', ls='-', lw=0.3, alpha=0.2)

    # Purple atom box at lambda=0 (drawn after y-scale is set)
    if _atom_frac > 0:
        box_text = (f'Atom at $\\lambda=0$\n'
                    f'Predicted $w_0 = {atom_mass:.2f}$\n'
                    f'Sampled  $w_0 = {_atom_frac:.2f}$')
        ax.annotate(box_text,
                    xy=(0, 0), xycoords='data',
                    xytext=(0.72, 0.92), textcoords='axes fraction',
                    fontsize=5, color=ATOM_COLOR, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5',
                              edgecolor=ATOM_COLOR, lw=1.0, alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color=ATOM_COLOR, lw=1.2,
                                    connectionstyle='arc3,rad=0.3'),
                    zorder=8)

    # Labels
    if show_title:
        tau_str = '0' if tau == 0 else f'{tau:g}'
        ax.set_title(fr'$\tau = {tau_str}$', fontsize=9)
    if show_xlabel:
        ax.set_xlabel(r'$\lambda$', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Density', fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6.5)


def make_legend_handles():
    return [
        Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP,
              label='Empirical'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label='Free-conv.'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=r'$\lambda_+(\tau)$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2,
               alpha=0.8, label=r'$\lambda_+(\infty)$'),
    ]


def main():
    # Pre-compute all eigenvalues
    all_data = {}  # (gi, ci) -> eigs array
    for gi, gamma in enumerate(GAMMAS):
        d = int(round(gamma * N_FIXED))
        print(f'gamma={gamma}, d={d}, N={N_FIXED}')
        for ci, tau in enumerate(TAUS):
            eigs_list = []
            for s in range(N_SEEDS):
                eigs = simulate_ou_csym(d, N_FIXED, tau, DT,
                                        seed=42 + s + gi * 100 + ci * 10000)
                eigs_list.append(eigs)
            pooled = np.concatenate(eigs_list)
            all_data[(gi, ci)] = pooled
            print(f'  tau={tau}: {len(pooled)} eigs, '
                  f'range [{pooled.min():.4f}, {pooled.max():.4f}]')

    # --- Output 1: Single 6x4 grid PNG ---
    n_rows = len(GAMMAS)
    n_cols = len(TAUS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 10.5))
    plt.subplots_adjust(hspace=0.30, wspace=0.20, left=0.07, right=0.97,
                        top=0.94, bottom=0.04)

    for gi, gamma in enumerate(GAMMAS):
        _, lp_inf = edge_tauinf(SIGMA, gamma)
        xlim = compute_shared_axes(all_data, gi, gamma)
        for ci, tau in enumerate(TAUS):
            ax = axes[gi, ci]
            plot_panel(ax, gamma, tau, all_data[(gi, ci)], lp_inf,
                       show_title=(gi == 0),
                       show_xlabel=(gi == n_rows - 1),
                       show_ylabel=(ci == 0),
                       xlim=xlim)
            # Row label on left
            if ci == 0:
                ax.set_ylabel(fr'$\gamma = {gamma:g}$', fontsize=8)

    # Legend in top-left of grid
    axes[0, 0].legend(handles=make_legend_handles(), loc='upper left', ncol=1,
                      fontsize=6.5, frameon=True, fancybox=True, edgecolor='#cccccc',
                      handlelength=1.2, handletextpad=0.4)

    out_png = os.path.join(FIGDIR, 'fig_free_conv_validation.png')
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f'\nSaved: {out_png}')

    # --- Output 2: Multi-page PDF + per-gamma PNGs ---
    out_pdf = os.path.join(FIGDIR, 'fig_free_conv_validation_pages.pdf')
    with PdfPages(out_pdf) as pdf:
        for gi, gamma in enumerate(GAMMAS):
            d = int(round(gamma * N_FIXED))
            _, lp_inf = edge_tauinf(SIGMA, gamma)
            xlim = compute_shared_axes(all_data, gi, gamma)

            for output_type in ('pdf_page', 'png'):
                fig_page, axes_page = plt.subplots(2, 2, figsize=(7, 5))
                plt.subplots_adjust(hspace=0.35, wspace=0.20, left=0.09, right=0.96,
                                    top=0.88, bottom=0.08)

                for ci, tau in enumerate(TAUS):
                    row, col = divmod(ci, 2)
                    ax = axes_page[row, col]
                    plot_panel(ax, gamma, tau, all_data[(gi, ci)], lp_inf,
                               show_title=True, show_xlabel=(row == 1),
                               show_ylabel=(col == 0),
                               xlim=xlim)

                fig_page.suptitle(fr'$\gamma = {gamma:g}$ ($d = {d}$, $N = {N_FIXED}$)',
                                  fontsize=11)

                # Single-column legend in upper-left of tau=0 panel
                axes_page[0, 0].legend(handles=make_legend_handles(),
                                       loc='upper left', ncol=1,
                                       fontsize=6.5, frameon=True, fancybox=True,
                                       edgecolor='#cccccc',
                                       handlelength=1.2, handletextpad=0.4)

                if output_type == 'pdf_page':
                    pdf.savefig(fig_page, dpi=200)
                else:
                    png_path = os.path.join(FIGDIR,
                                            f'fig_free_conv_validation_{GAMMA_NAMES[gamma]}.png')
                    fig_page.savefig(png_path, dpi=300)
                plt.close(fig_page)

    print(f'Saved: {out_pdf}')
    for gamma in GAMMAS:
        print(f'  fig_free_conv_validation_{GAMMA_NAMES[gamma]}.png')


if __name__ == '__main__':
    main()
