#!/usr/bin/env python3
"""
3-column × 2-row figure for planted spike variants — whitened only.

Columns: 20 sep | 20 var | 10 sep + 10 var
Row 1: spike count #{lambda_j > lambda_+(tau)}
Row 2: rho(tau) = tr(C(tau)) / tr(C(0))

Vertical line: tau where rho < 1/e.
Style matches plot_spike_count_per_direction.py / fig_spike_count_empirical_sigma.png.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'figures'))
from style_config import setup_style, TEXT_WIDTH

setup_style()
matplotlib.rcParams["figure.dpi"] = 120

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data = np.load(os.path.join(OUT_DIR, 'results', 'planted_spikes_variants_whitened.npz'),
               allow_pickle=True)

d = int(data['d'])
N = int(data['N'])
vf = float(data['var_factor'])
tau_max = float(data['tau_max'])
inv_e = 1.0 / np.e

fig, axes = plt.subplots(2, 3, figsize=(14, 6))

for ci in range(3):
    label = str(data[f'cfg{ci}_label'])
    k_sep = int(data[f'cfg{ci}_k_sep'])
    k_var = int(data[f'cfg{ci}_k_var'])
    k_total = k_sep + k_var

    taus = data[f'cfg{ci}_whitened_taus']
    counts = data[f'cfg{ci}_whitened_counts']
    rho_taus = data[f'cfg{ci}_whitened_rho_taus']
    rho_vals = data[f'cfg{ci}_whitened_rho_vals']

    mask = taus > 0
    rho_mask = rho_taus > 0

    # Find tau where rho < 1/e
    tau_1e = None
    for i in range(len(rho_taus)):
        if rho_taus[i] > 0 and rho_vals[i] < inv_e:
            tau_1e = float(rho_taus[i])
            break

    # ── Top row: spike count ──
    ax = axes[0, ci]
    ax.plot(taus[mask], counts[mask], 'o-', color='C0', lw=1.6, ms=5)
    ax.axhline(k_total, color='crimson', ls='--', lw=1.4,
               label=f'planted $k = {k_total}$')
    if tau_1e is not None:
        ax.axvline(tau_1e, color='red', ls=':', lw=1.4, alpha=0.7,
                   label=rf'$\rho < 1/e$ ($\tau$={tau_1e:.1f})')
    ax.set_xscale('log')
    ax.set_title(label, fontweight='bold')
    if ci == 0:
        ax.set_ylabel(r'$\#\{\lambda_j > \lambda_+(\tau)\}$')
    ax.set_ylim(-0.5, k_total + 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Bottom row: rho(tau) ──
    ax = axes[1, ci]
    ax.plot(rho_taus[rho_mask], rho_vals[rho_mask], 'o-', color='tab:red', lw=1.6, ms=5)
    ax.axhline(inv_e, color='gray', ls='--', lw=1.2, alpha=0.7,
               label='$1/e$')
    if tau_1e is not None:
        ax.axvline(tau_1e, color='red', ls=':', lw=1.4, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    if ci == 0:
        ax.set_ylabel(r'$\rho(\tau)$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    rf'Whitened $\hat\sigma_f^2 = \mathrm{{tr}}(\hat\Sigma_f)/d$, '
    rf'variance factor = {vf:.0f}x, $\tau_{{\max}} = {tau_max:.0f}$',
    fontsize=12,
)
fig.tight_layout()

path = os.path.join(OUT_DIR, 'fig_planted_spikes_variants_whitened.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")
