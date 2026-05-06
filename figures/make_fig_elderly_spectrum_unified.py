#!/usr/bin/env python3
"""
Unified elderly spectrum figure: single panel with empirical + MC null histograms.

Parameters: d=768, N=200, gamma=3.84, tau*=5.0s.
MC null: B=200 reps of iid OU draws from C0, full 768 eigenvalues per rep.

Output: figures/fig_elderly_spectrum.png
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
    HIST_EMPIRICAL, HIST_MC_NULL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF, MC_NULL_PCTL,
    HIST_ALPHA_EMP, HIST_ALPHA_NULL, CURVE_LW, VLINE_LW, VLINE_STYLE,
)
from style_config import setup_style
setup_style()

FIGDIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
ELDERLY_DA_PATH = os.path.join(FIGDIR, '..', 'experiments', 'sdxl', 'results_da_pcavda',
                                'clip_da_elderly_gs7p5_fc_dt0p0025.npz')
C0_PATH = os.path.join(FIGDIR, '..', 'experiments', 'save_autocovariance',
                        'results', 'elderly_gs7p5_n200_C_tau_full.npz')

N_MC_REPS = 200
ATOM_COLOR = '#7B2D8E'
ZERO_TOL = 1e-8


def mc_null_full(C0, tau, N, n_reps=N_MC_REPS, seed=42):
    """MC null: all eigenvalues of C_sym(tau) under iid OU draws from C0."""
    d = C0.shape[0]
    rng = np.random.RandomState(seed)

    eigvals_C0, eigvecs_C0 = np.linalg.eigh(C0)
    eigvals_C0 = np.maximum(eigvals_C0, 0)
    sqrt_C0 = eigvecs_C0 * np.sqrt(eigvals_C0)[np.newaxis, :]

    rho = np.exp(-tau / 2.0)
    noise_scale = np.sqrt(max(1.0 - np.exp(-tau), 1e-20))

    all_eigs = np.empty((n_reps, d))
    max_eigs = np.empty(n_reps)

    for i in range(n_reps):
        z0 = rng.randn(N, d) @ sqrt_C0.T
        z_noise = rng.randn(N, d) @ sqrt_C0.T
        x_tau = rho * z0 + noise_scale * z_noise

        mu = z0.mean(axis=0)
        z0_c = z0 - mu
        x_tau_c = x_tau - mu

        C_raw = (x_tau_c.T @ z0_c) / N
        C_sym = 0.5 * (C_raw + C_raw.T)

        eigs = np.linalg.eigvalsh(C_sym)
        all_eigs[i] = eigs
        max_eigs[i] = eigs[-1]

        if (i + 1) % 50 == 0:
            print(f'  MC rep {i+1}/{n_reps}')

    return all_eigs, max_eigs


def main():
    # Load real eigenvalues at tau*
    fc_data = np.load(ELDERLY_DA_PATH, allow_pickle=True)
    eigs_raw = fc_data['eigenvalues_full_at_tau_star']
    eigs = np.sort(eigs_raw)[::-1].copy()

    tau_star = 5.0  # dimensionless OU time (= dt * T = 0.0025 * 2000)

    # Load C0
    c0_data = np.load(C0_PATH)
    C0 = c0_data['C0']
    sigma_f_sq = float(c0_data['sigma_f_sq'])
    d = int(c0_data['d'])
    N = int(c0_data['N'])
    gamma = d / N
    sigma = np.sqrt(sigma_f_sq)

    print(f'd={d}, N={N}, gamma={gamma:.2f}, sigma_f^2={sigma_f_sq:.6e}')
    print(f'tau*={tau_star}, DA1={eigs[0]:.4e}')

    # Free-conv edges
    lm_tau, lp_tau = fc_edges(tau_star, sigma, gamma=gamma)
    _, lp_inf = edge_tauinf(sigma, gamma)
    print(f'lambda_+(tau*)={lp_tau:.4e}, lambda_+(inf)={lp_inf:.4e}')

    # MC null
    print(f'Computing MC null ({N_MC_REPS} reps)...')
    mc_all_eigs, mc_max_eigs = mc_null_full(C0, tau_star, N, n_reps=N_MC_REPS)
    mc_95 = float(np.percentile(mc_max_eigs, 95))
    print(f'MC null 95th pctl = {mc_95:.4e}')

    # Free-conv density curve
    pad = 0.3 * (lp_tau - lm_tau)
    lam_grid = np.linspace(lm_tau - pad, lp_tau + pad, 800)
    rho_grid = fc_density(lam_grid, tau_star, sigma, gamma=gamma)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Symlog bins
    linthresh = 5e-4
    def symlog_fn(x):
        return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)
    def inv_symlog_fn(y):
        return np.sign(y) * linthresh * (10**np.abs(y) - 1)

    eig_min = min(eigs.min(), mc_all_eigs.min())
    eig_max = max(eigs.max(), mc_all_eigs.max())
    sl_lo = symlog_fn(eig_min - 0.02 * (eig_max - eig_min))
    sl_hi = symlog_fn(eig_max + 0.02 * (eig_max - eig_min))
    bins_sl = inv_symlog_fn(np.linspace(sl_lo, sl_hi, 80))

    # MC null histogram (darker pink, behind)
    MC_NULL_DARK = '#B94055'
    mc_pooled = mc_all_eigs.ravel()
    ax.hist(mc_pooled, bins=bins_sl, density=True, color=MC_NULL_DARK,
            alpha=HIST_ALPHA_NULL, edgecolor='none', zorder=1, label='MC null')

    # Empirical histogram (light blue, in front)
    ax.hist(eigs, bins=bins_sl, density=True, color=HIST_EMPIRICAL,
            alpha=HIST_ALPHA_EMP, edgecolor='white', linewidth=0.3, zorder=2,
            label='Empirical')

    # Free-conv density (red curve)
    ax.plot(lam_grid, rho_grid, color=FREECONV_CURVE, lw=CURVE_LW, zorder=4)

    # Floor lines
    ax.axvline(lp_tau, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
    ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
    ax.axvline(mc_95, color=MC_NULL_PCTL, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)

    # DA1 marker
    lam1 = eigs[0]
    ax.plot(lam1, ax.get_ylim()[1] * 0.05, marker='v', color=FREECONV_CURVE,
            markersize=5, zorder=6, clip_on=False)
    ax.annotate('DA1', xy=(lam1, 0), fontsize=7, color=FREECONV_CURVE,
                ha='center', va='bottom', fontweight='bold',
                xytext=(lam1, ax.get_ylim()[1] * 0.08))

    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('symlog', linthresh=100)
    ax.set_ylim(bottom=0)

    # Purple atom box at lambda=0
    atom_mass_pred = max(0.0, 1.0 - 2.0 / gamma)
    atom_mc = np.mean(np.abs(mc_all_eigs) < ZERO_TOL)
    atom_empirical = np.sum(np.abs(eigs) < ZERO_TOL) / len(eigs)
    box_text = (f'Atom at $\\lambda=0$\n'
                f'Predicted  $w_0 = {atom_mass_pred:.2f}$\n'
                f'MC null    $w_0 = {atom_mc:.2f}$\n'
                f'Empirical  $w_0 = {atom_empirical:.2f}$')
    ax.annotate(box_text,
                xy=(0, 0), xycoords='data',
                xytext=(0.72, 0.92), textcoords='axes fraction',
                fontsize=6.5, color=ATOM_COLOR, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5',
                          edgecolor=ATOM_COLOR, lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color=ATOM_COLOR, lw=1.5,
                                connectionstyle='arc3,rad=0.3'),
                zorder=8)

    ax.set_xlabel(r'$\lambda$', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(labelsize=7)

    # Legend
    legend_handles = [
        Patch(facecolor=HIST_EMPIRICAL, alpha=HIST_ALPHA_EMP, edgecolor='white',
              label='Empirical eigenvalues'),
        Patch(facecolor=MC_NULL_DARK, alpha=HIST_ALPHA_NULL, edgecolor='none',
              label='MC null eigenvalues'),
        Line2D([], [], color=FREECONV_CURVE, lw=CURVE_LW,
               label='Free-conv. density'),
        Line2D([], [], color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW,
               label=fr'$\lambda_+(\tau^*) = {lp_tau:.2e}$'),
        Line2D([], [], color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW,
               label=fr'$\lambda_+(\infty) = {lp_inf:.2e}$'),
        Line2D([], [], color=MC_NULL_PCTL, ls=VLINE_STYLE, lw=VLINE_LW,
               label=f'MC null 95th pctl = {mc_95:.2e}'),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc='upper left',
              frameon=True, fancybox=True, edgecolor='#cccccc')

    fig.tight_layout()
    out_path = os.path.join(FIGDIR, 'fig_elderly_spectrum.png')
    fig.savefig(out_path, dpi=300)
    print(f'Saved: {out_path}')

    # Summary
    print(f'\nSummary:')
    print(f'  DA1 = {lam1:.4e}')
    print(f'  lambda_+(tau*={tau_star}) = {lp_tau:.4e}')
    print(f'  lambda_+(inf) = {lp_inf:.4e}')
    print(f'  MC null 95th pctl = {mc_95:.4e}')
    print(f'  DA1 / MC_95 = {lam1 / mc_95:.1f}x')
    print(f'  Near-zero eigs: {np.sum(np.abs(eigs) < 1e-8)}/{d}')


if __name__ == '__main__':
    main()
