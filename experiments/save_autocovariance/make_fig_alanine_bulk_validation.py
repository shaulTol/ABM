#!/usr/bin/env python3
"""
Bulk-fit validation figure for alanine C_sym(tau).

4-panel row: histogram of bulk eigenvalues of C_sym(tau) at 4 lags,
with analytic free-convolution density overlay (fitted via one-iteration peel).
Lags chosen to span the MP → symmetric transition and include tau*.

Output:
  results/fig_alanine_bulk_validation.png
  results/fig_alanine_bulk_validation.json
"""
import sys, os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 4 lags spanning the transition:
#   idx 0:  tau=0.1ps   — near MP (e^{-tau/2}=0.95), one-sided
#   idx 6:  tau=1.1ps   — mid-transition (e^{-tau/2}=0.58)
#   idx 10: tau=4.3ps   — nearly symmetric (e^{-tau/2}=0.12)
#   idx 23: tau=362.6ps — tau*, fully symmetric
SELECTED_INDICES = [0, 6, 10, 23]


def peel_fit(eigs_tau, eigs_C0, tau, gamma, sigma_naive):
    """One-iteration peel: identify spikes with naive sigma, refit on bulk."""
    lm_naive, lp_naive = fc_edges(tau, sigma_naive, gamma=gamma)

    spike_mask = eigs_tau > lp_naive
    n_spikes = int(spike_mask.sum())
    spike_vals = eigs_tau[spike_mask]

    tr_C0 = eigs_C0.sum()
    d = len(eigs_C0)
    if n_spikes > 0:
        spike_C0_contribution = eigs_C0[:n_spikes].sum()
        sigma_bulk_sq = (tr_C0 - spike_C0_contribution) / d
    else:
        sigma_bulk_sq = tr_C0 / d
    sigma_bulk = np.sqrt(max(sigma_bulk_sq, 1e-20))

    lm_bulk, lp_bulk = fc_edges(tau, sigma_bulk, gamma=gamma)

    return {
        'n_spikes': n_spikes,
        'spike_vals': spike_vals,
        'spike_indices': np.where(spike_mask)[0].tolist(),
        'sigma_naive_sq': sigma_naive**2,
        'sigma_bulk_sq': float(sigma_bulk_sq),
        'lp_naive': lp_naive,
        'lm_naive': lm_naive,
        'lp_bulk': lp_bulk,
        'lm_bulk': lm_bulk,
        'sigma_bulk': sigma_bulk,
    }


def main():
    data = np.load(os.path.join(RESDIR, 'alanine_C_tau_full.npz'))
    C0 = data['C0']
    lags = data['lags']
    phys_times = data['phys_times']
    eigenvalues = data['eigenvalues']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    N = int(data['N'])

    sigma_naive = np.sqrt(sigma_f_sq)
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    plt.subplots_adjust(wspace=0.32, left=0.05, right=0.97, top=0.82, bottom=0.18)

    json_data = {'gamma': gamma, 'd': d, 'N': N,
                 'sigma_naive_sq': sigma_naive**2, 'panels': []}

    for ax_i, idx in enumerate(SELECTED_INDICES):
        ax = axes[ax_i]
        tau = phys_times[idx]
        lag = lags[idx]
        eigs = eigenvalues[idx].copy()  # descending
        rho_ou = np.exp(-tau / 2.0)

        # Peel fit
        fit = peel_fit(eigs, eigs_C0, tau, gamma, sigma_naive)
        n_spikes = fit['n_spikes']

        # Analytic density on refitted bulk
        lm, lp = fit['lm_bulk'], fit['lp_bulk']
        support_width = lp - lm

        # Density curve — extend slightly beyond support
        pad = 0.25 * support_width
        grid_lo = lm - pad
        grid_hi = lp + pad
        lam_grid = np.linspace(grid_lo, grid_hi, 600)
        rho_grid = fc_density(lam_grid, tau, fit['sigma_bulk'], gamma=gamma)

        # Bulk eigenvalues (those <= lp_bulk)
        bulk_eigs = eigs[eigs <= fit['lp_bulk']]

        # X-axis: zoom on bulk region with some extra room for spike annotations
        x_lo = grid_lo
        x_hi = lp + 2.0 * support_width

        # Histogram bins within bulk region
        bin_lo = min(bulk_eigs.min() if len(bulk_eigs) > 0 else lm, lm) - 0.1 * support_width
        bin_hi = lp + 0.2 * support_width
        bins = np.linspace(bin_lo, bin_hi, 30)

        ax.hist(bulk_eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
                edgecolor='white', linewidth=0.4, zorder=2)

        # Analytic density
        ax.plot(lam_grid, rho_grid, color='#C0392B', lw=2.2, zorder=4)

        # Upper edge
        ax.axvline(fit['lp_bulk'], color='#555555', ls='--', lw=1.2, zorder=3)
        # Lower edge (visible at early tau when one-sided)
        if lm > x_lo + 0.05 * support_width:
            ax.axvline(fit['lm_bulk'], color='#555555', ls='--', lw=1.0, zorder=3,
                       alpha=0.5)

        ax.set_xlim(x_lo, x_hi)

        # Y-limit
        rho_max = rho_grid.max()
        hist_vals, _ = np.histogram(bulk_eigs, bins=bins, density=True)
        hist_max = hist_vals.max() if len(hist_vals) > 0 else rho_max
        y_content_max = max(rho_max, hist_max)
        y_top = y_content_max * 1.35
        ax.set_ylim(0, y_top)

        # Spike markers with annotations
        if n_spikes > 0:
            for j, sv in enumerate(fit['spike_vals']):
                x_pos = lp + (0.4 + j * 0.65) * support_width
                x_pos = min(x_pos, x_hi - 0.05 * support_width)
                y_pos = y_top * (0.90 - j * 0.13)

                ax.plot(x_pos, y_pos, marker='v', color='#C0392B',
                        markersize=6, zorder=5, clip_on=False)

                label = f'{sv:.4f}' if sv > 0.001 else f'{sv:.2e}'
                ax.text(x_pos, y_pos * 1.04, label,
                        fontsize=6.5, color='#C0392B', ha='center', va='bottom',
                        fontweight='bold')

        # Title
        tau_str = f'{tau:.1f}' if tau < 100 else f'{tau:.0f}'
        is_tau_star = abs(tau - 362.6) < 1.0
        title = fr'$\tau = {tau_str}$ ps'
        if is_tau_star:
            title += r'  $(\tau^\star)$'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'$\lambda$', fontsize=9)
        if ax_i == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=7)

        # Annotation box — spike count, sigma_bulk, OU decay
        ann_text = (f'{n_spikes} spike{"s" if n_spikes != 1 else ""}\n'
                    fr'$\sigma^2_{{\mathrm{{bulk}}}} = {fit["sigma_bulk_sq"]:.1e}$'
                    f'\n$e^{{-\\tau/2}} = {rho_ou:.2f}$')
        ax.text(0.97, 0.97, ann_text,
                transform=ax.transAxes, fontsize=6.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

        ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))
        ax.xaxis.get_offset_text().set_fontsize(7)

        json_data['panels'].append({
            'lag_steps': int(lag),
            'tau_ps': float(tau),
            'e_minus_tau_over_2': float(rho_ou),
            'sigma_naive_sq': float(fit['sigma_naive_sq']),
            'sigma_bulk_sq': float(fit['sigma_bulk_sq']),
            'lp_naive': float(fit['lp_naive']),
            'lp_bulk': float(fit['lp_bulk']),
            'lm_bulk': float(fit['lm_bulk']),
            'n_spikes': n_spikes,
            'spike_indices': fit['spike_indices'],
            'spike_values': fit['spike_vals'].tolist(),
            'all_eigenvalues': eigs.tolist(),
        })

    # Shared legend
    legend_handles = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white',
              label='Bulk eigenvalues'),
        Line2D([], [], color='#C0392B', lw=2.2,
               label=r'Free-conv. density ($\sigma^2_{\mathrm{bulk}}$)'),
        Line2D([], [], color='#555555', ls='--', lw=1.2,
               label=r'$\lambda_\pm(\tau)$'),
        Line2D([], [], color='#C0392B', marker='v', ls='none', markersize=6,
               label='Spike eigenvalues'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig_path = os.path.join(RESDIR, 'fig_alanine_bulk_validation.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {fig_path}')

    json_path = os.path.join(RESDIR, 'fig_alanine_bulk_validation.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Saved: {json_path}')

    for p in json_data['panels']:
        print(f"  tau={p['tau_ps']:>7.1f}ps: e^(-tau/2)={p['e_minus_tau_over_2']:.3f}, "
              f"{p['n_spikes']} spikes, sigma_bulk^2={p['sigma_bulk_sq']:.2e}, "
              f"edges=[{p['lm_bulk']:.2e}, {p['lp_bulk']:.2e}]")


if __name__ == '__main__':
    main()
