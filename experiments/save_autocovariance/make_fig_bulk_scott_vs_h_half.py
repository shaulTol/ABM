#!/usr/bin/env python3
"""Side-by-side bulk eigenvalue fit: Scott vs h/2 at tau*=362.6ps."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'noise_floor'))

import numpy as np
from free_conv_utils import edges as fc_edges, density as fc_density
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
tau = 362.6
gamma = 45 / 2000


def peel(eigs, eigs_C0, tau, gamma, sigma_f_sq):
    d = len(eigs_C0)
    sigma = np.sqrt(sigma_f_sq)
    lm, lp = fc_edges(tau, sigma, gamma=gamma)
    spike_mask = eigs > lp
    n_spikes = int(spike_mask.sum())
    if n_spikes > 0:
        sigma_bulk_sq = (eigs_C0.sum() - eigs_C0[:n_spikes].sum()) / d
    else:
        sigma_bulk_sq = eigs_C0.sum() / d
    sigma_bulk = np.sqrt(max(sigma_bulk_sq, 1e-20))
    lm1, lp1 = fc_edges(tau, sigma_bulk, gamma=gamma)
    return n_spikes, eigs[spike_mask], sigma_bulk, lm1, lp1


def plot_panel(ax, title, eigs, eigs_C0, sigma_f_sq):
    n_spikes, spike_vals, sigma_bulk, lm, lp = peel(eigs, eigs_C0, tau, gamma, sigma_f_sq)
    support_width = lp - lm
    pad = 0.25 * support_width

    lam_grid = np.linspace(lm - pad, lp + pad, 600)
    rho_grid = fc_density(lam_grid, tau, sigma_bulk, gamma=gamma)

    bulk_eigs = eigs[eigs <= lp]
    iqr = np.subtract(*np.percentile(bulk_eigs, [75, 25]))
    fd_width = 2.0 * iqr / (len(bulk_eigs) ** (1./3.)) if iqr > 0 else support_width / 15
    bin_lo = min(bulk_eigs.min(), lm) - 0.1 * support_width
    bin_hi = lp + 0.2 * support_width
    n_bins = max(12, min(25, int((bin_hi - bin_lo) / fd_width)))
    bins = np.linspace(bin_lo, bin_hi, n_bins)

    ax.hist(bulk_eigs, bins=bins, density=True, color='#7BAFD4', alpha=0.65,
            edgecolor='white', linewidth=0.4, zorder=2)
    ax.plot(lam_grid, rho_grid, color='#C0392B', lw=2.2, zorder=4)
    ax.axvline(lp, color='#555555', ls='--', lw=1.2, zorder=3)
    ax.axvline(lm, color='#555555', ls='--', lw=1.0, zorder=3, alpha=0.6)

    x_lo = bin_lo - 0.05 * support_width
    x_hi = lp + 2.2 * support_width
    ax.set_xlim(x_lo, x_hi)

    rho_max = rho_grid.max()
    hist_vals, _ = np.histogram(bulk_eigs, bins=bins, density=True)
    hist_max = hist_vals.max() if len(hist_vals) > 0 else rho_max
    y_top = max(rho_max, hist_max) * 1.35
    ax.set_ylim(0, y_top)

    # Spike markers — arrows at actual eigenvalue position on x-axis
    for j, sv in enumerate(spike_vals):
        y_pos = y_top * (0.90 - j * 0.16)
        if y_pos < 0:
            continue
        ax.annotate('', xy=(sv, 0), xytext=(sv, y_pos),
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
                    zorder=5)
        val_str = f'{sv:.4f}' if sv > 0.001 else f'{sv:.2e}'
        label = f'DA{j+1}' if j < 5 else ''
        ax.text(sv, y_pos * 1.03, f'{label}\n{val_str}',
                fontsize=7, color='#C0392B', ha='center', va='bottom', fontweight='bold')

    n_neg = int(np.sum(eigs < 0))
    n_below = int(np.sum(eigs < lm))
    ax.text(0.97, 0.97,
            f'{n_spikes} spikes, {n_neg} neg\n'
            f'min={eigs[0]:.1e}',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$\lambda$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))
    ax.xaxis.get_offset_text().set_fontsize(8)


def main():
    scott = np.load(os.path.join(RESDIR, 'alanine_C_tau_full.npz'))
    h2 = np.load(os.path.join(RESDIR, 'alanine_h_half_taustar_eigs.npz'))

    eigs_scott = scott['eigenvalues'][23].copy()
    eigs_h2 = h2['eigenvalues'].copy()
    eigs_C0_scott = np.linalg.eigvalsh(scott['C0'])[::-1].copy()
    eigs_C0_h2 = h2['C0_eigenvalues'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    plot_panel(axes[0], 'Scott ($h = 0.0183$)',
               eigs_scott, eigs_C0_scott, float(scott['sigma_f_sq']))
    plot_panel(axes[1], '$h/2$ ($h = 0.0091$)',
               eigs_h2, eigs_C0_h2, float(h2['sigma_f_sq']))

    axes[0].set_ylabel('Density', fontsize=10)

    legend_handles = [
        Patch(facecolor='#7BAFD4', alpha=0.65, edgecolor='white', label='Bulk eigenvalues'),
        Line2D([], [], color='#C0392B', lw=2.2, label='Free-conv density'),
        Line2D([], [], color='#555555', ls='--', lw=1.2, label=r'$\lambda_\pm$'),
        Line2D([], [], color='#C0392B', marker='v', ls='none', markersize=6, label='Spikes'),
    ]
    axes[1].legend(handles=legend_handles, fontsize=7, loc='upper left', framealpha=0.9)

    fig.suptitle(r'Bulk eigenvalue fit at $\tau^\star = 363$ ps  (alanine, $d{=}45$, $N{=}2000$)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = os.path.join(RESDIR, 'fig_alanine_bulk_scott_vs_h_half.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
