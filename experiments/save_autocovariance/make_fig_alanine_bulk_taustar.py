#!/usr/bin/env python3
"""
Standalone bulk-fit validation figure at tau* = 362.6 ps for alanine.

Output:
  results/fig_alanine_bulk_taustar.png
  results/fig_alanine_bulk_taustar.json
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
IDX_STAR = 23  # tau=362.6ps


def peel_fit(eigs_tau, eigs_C0, tau, gamma, sigma_naive):
    """One-iteration peel. Returns fit dict + diagnostic info."""
    d = len(eigs_C0)
    tr_C0 = eigs_C0.sum()

    # Naive
    lm_naive, lp_naive = fc_edges(tau, sigma_naive, gamma=gamma)
    spike_mask = eigs_tau > lp_naive
    n_spikes = int(spike_mask.sum())
    spike_vals = eigs_tau[spike_mask]

    # Peel 1
    if n_spikes > 0:
        sigma_bulk_sq_1 = (tr_C0 - eigs_C0[:n_spikes].sum()) / d
    else:
        sigma_bulk_sq_1 = tr_C0 / d
    sigma_bulk_1 = np.sqrt(max(sigma_bulk_sq_1, 1e-20))
    lm1, lp1 = fc_edges(tau, sigma_bulk_1, gamma=gamma)

    # Count below lambda_minus
    n_below_1 = int(np.sum(eigs_tau < lm1))
    below_vals_1 = eigs_tau[eigs_tau < lm1]

    # Peel 2 (diagnostic only)
    spike_mask_2 = eigs_tau > lp1
    n_spikes_2 = int(spike_mask_2.sum())
    if n_spikes_2 > 0:
        sigma_bulk_sq_2 = (tr_C0 - eigs_C0[:n_spikes_2].sum()) / d
    else:
        sigma_bulk_sq_2 = sigma_bulk_sq_1
    sigma_bulk_2 = np.sqrt(max(sigma_bulk_sq_2, 1e-20))
    lm2, lp2 = fc_edges(tau, sigma_bulk_2, gamma=gamma)
    n_below_2 = int(np.sum(eigs_tau < lm2))

    return {
        'n_spikes': n_spikes,
        'spike_vals': spike_vals,
        'sigma_naive_sq': sigma_naive**2,
        'sigma_bulk_sq': float(sigma_bulk_sq_1),
        'sigma_bulk': sigma_bulk_1,
        'lp_naive': lp_naive,
        'lm_naive': lm_naive,
        'lp_bulk': lp1,
        'lm_bulk': lm1,
        'n_below_lm': n_below_1,
        'below_lm_vals': below_vals_1.tolist(),
        # Peel 2 diagnostics
        'peel2_n_spikes': n_spikes_2,
        'peel2_sigma_bulk_sq': float(sigma_bulk_sq_2),
        'peel2_lm': lm2,
        'peel2_lp': lp2,
        'peel2_n_below_lm': n_below_2,
    }


def main():
    data = np.load(os.path.join(RESDIR, 'alanine_C_tau_full.npz'))
    C0 = data['C0']
    phys_times = data['phys_times']
    eigenvalues = data['eigenvalues']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    N = int(data['N'])

    sigma_naive = np.sqrt(sigma_f_sq)
    eigs_C0 = np.linalg.eigvalsh(C0)[::-1].copy()

    tau = phys_times[IDX_STAR]
    eigs = eigenvalues[IDX_STAR].copy()
    fit = peel_fit(eigs, eigs_C0, tau, gamma, sigma_naive)

    lm, lp = fit['lm_bulk'], fit['lp_bulk']
    support_width = lp - lm
    pad = 0.25 * support_width

    # Density curve
    lam_grid = np.linspace(lm - pad, lp + pad, 600)
    rho_grid = fc_density(lam_grid, tau, fit['sigma_bulk'], gamma=gamma)

    # All non-spike eigenvalues (including those below lm)
    bulk_eigs = eigs[eigs <= fit['lp_bulk']]

    # Finer bins: Freedman-Diaconis
    iqr = np.subtract(*np.percentile(bulk_eigs, [75, 25]))
    fd_width = 2.0 * iqr / (len(bulk_eigs) ** (1./3.)) if iqr > 0 else support_width / 15
    bin_lo = min(bulk_eigs.min(), lm) - 0.1 * support_width
    bin_hi = lp + 0.2 * support_width
    n_bins = max(12, min(25, int((bin_hi - bin_lo) / fd_width)))
    bins = np.linspace(bin_lo, bin_hi, n_bins)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.hist(bulk_eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
            edgecolor='white', linewidth=0.4, zorder=2)
    ax.plot(lam_grid, rho_grid, color='#C0392B', lw=2.2, zorder=4)

    # Support edges
    ax.axvline(fit['lp_bulk'], color='#555555', ls='--', lw=1.2, zorder=3)
    ax.axvline(fit['lm_bulk'], color='#555555', ls='--', lw=1.0, zorder=3, alpha=0.6)

    # X range
    x_lo = bin_lo - 0.05 * support_width
    x_hi = lp + 2.2 * support_width
    ax.set_xlim(x_lo, x_hi)

    # Y range
    rho_max = rho_grid.max()
    hist_vals, _ = np.histogram(bulk_eigs, bins=bins, density=True)
    hist_max = hist_vals.max() if len(hist_vals) > 0 else rho_max
    y_top = max(rho_max, hist_max) * 1.35
    ax.set_ylim(0, y_top)

    # Spike markers with DA labels
    spike_labels = [r'DA1 $\to \psi$', r'DA2 $\to \phi$']
    n_spikes = fit['n_spikes']
    for j, sv in enumerate(fit['spike_vals']):
        x_pos = lp + (0.4 + j * 0.7) * support_width
        x_pos = min(x_pos, x_hi - 0.05 * support_width)
        y_pos = y_top * (0.90 - j * 0.16)
        ax.plot(x_pos, y_pos, marker='v', color='#C0392B',
                markersize=7, zorder=5, clip_on=False)
        val_str = f'{sv:.4f}' if sv > 0.001 else f'{sv:.2e}'
        da_label = spike_labels[j] if j < len(spike_labels) else ''
        ax.text(x_pos, y_pos * 1.05, f'{val_str}\n{da_label}',
                fontsize=7, color='#C0392B', ha='center', va='bottom',
                fontweight='bold', linespacing=1.4)

    ax.set_title(r'$\tau^\star = 363$ ps', fontsize=11)
    ax.set_xlabel(r'$\lambda$', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))
    ax.xaxis.get_offset_text().set_fontsize(8)

    # Info box
    ax.text(0.97, 0.97,
            f'{n_spikes} spikes, {fit["n_below_lm"]} below $\\lambda_-$\n'
            fr'$\sigma^2_{{\mathrm{{bulk}}}} = {fit["sigma_bulk_sq"]:.2e}$',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    legend_handles = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white',
              label='Bulk eigenvalues'),
        Line2D([], [], color='#C0392B', lw=2.2,
               label=r'Free-conv. density ($\sigma^2_{\mathrm{bulk}}$)'),
        Line2D([], [], color='#555555', ls='--', lw=1.2,
               label=r'$\lambda_\pm(\tau^\star)$'),
        Line2D([], [], color='#C0392B', marker='v', ls='none', markersize=6,
               label='Spike eigenvalues'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='upper left',
              framealpha=0.9)

    fig.tight_layout()
    fig_path = os.path.join(RESDIR, 'fig_alanine_bulk_taustar.png')
    fig.savefig(fig_path, dpi=200)
    print(f'Saved: {fig_path}')

    # JSON sidecar
    json_data = {
        'tau_ps': float(tau),
        'gamma': gamma,
        'd': d,
        'N': N,
        'sigma_naive_sq': float(fit['sigma_naive_sq']),
        'peel_1': {
            'n_spikes': fit['n_spikes'],
            'spike_values': fit['spike_vals'].tolist(),
            'sigma_bulk_sq': fit['sigma_bulk_sq'],
            'lm': fit['lm_bulk'],
            'lp': fit['lp_bulk'],
            'n_below_lm': fit['n_below_lm'],
            'below_lm_values': fit['below_lm_vals'],
        },
        'peel_2': {
            'n_spikes': fit['peel2_n_spikes'],
            'sigma_bulk_sq': fit['peel2_sigma_bulk_sq'],
            'lm': fit['peel2_lm'],
            'lp': fit['peel2_lp'],
            'n_below_lm': fit['peel2_n_below_lm'],
            'note': 'Peel 2 over-peels (7 spikes, 14 below lm). Peel 1 is the correct stopping point.',
        },
        'n_bins': n_bins,
        'all_eigenvalues': eigs.tolist(),
    }
    json_path = os.path.join(RESDIR, 'fig_alanine_bulk_taustar.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Saved: {json_path}')

    # Summary
    print(f'  Peel 1: {fit["n_spikes"]} spikes, sigma_bulk^2={fit["sigma_bulk_sq"]:.2e}, '
          f'{fit["n_below_lm"]} below lm={fit["lm_bulk"]:.2e}')
    print(f'  Peel 2: {fit["peel2_n_spikes"]} spikes, sigma_bulk^2={fit["peel2_sigma_bulk_sq"]:.2e}, '
          f'{fit["peel2_n_below_lm"]} below lm={fit["peel2_lm"]:.2e} (unstable)')


if __name__ == '__main__':
    main()
