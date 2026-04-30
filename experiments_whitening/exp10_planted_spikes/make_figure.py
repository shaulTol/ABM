#!/usr/bin/env python3
"""
Figure for Exp 10: Planted Spikes — whitened vs original.

2-row layout:
  Row 1: spike count #{lambda_j > lambda_+(tau)} with rho < 1/e vertical line
  Row 2: rho(tau) = tr(C(tau)) / tr(C(0))
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'figures'))
from style_config import setup_style, TEXT_WIDTH

setup_style()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(OUT_DIR, 'results', 'planted_spikes_whitened.json')) as f:
    res_json = json.load(f)
data = np.load(os.path.join(OUT_DIR, 'results', 'planted_spikes_whitened.npz'))

params = res_json['params']
d, k = params['d'], params['k']
dt = params['dt']

taus_w = data['taus_whitened']
taus_o = data['taus_original']
counts_w = data['counts_whitened']
counts_o = data['counts_original']
rho_taus_w = data['rho_taus_whitened']
rho_taus_o = data['rho_taus_original']
rho_vals_w = data['rho_vals_whitened']
rho_vals_o = data['rho_vals_original']

# Find tau where rho first drops below 1/e
threshold = 1.0 / np.e
tau_1e = None
for i in range(len(rho_taus_o)):
    if rho_vals_o[i] < threshold:
        tau_1e = float(rho_taus_o[i])
        break

print(f"Spike count: orig m_hat={counts_o.max()}, whit m_hat={counts_w.max()}")
print(f"rho < 1/e at tau={tau_1e}")

# ── Figure: 2 rows × 1 col ──
linthresh = max(2.0 * dt, 0.05)
fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH * 0.5, 4.0), sharex=True)

c_orig = '#2166ac'
c_whit = '#b2182b'

# ── Row 1: Spike count ──
ax = axes[0]
ax.plot(taus_o, counts_o, 'o-', color=c_orig, lw=1.6, markersize=4,
        label='original', zorder=3)
ax.plot(taus_w, counts_w, 's--', color=c_whit, lw=1.2, markersize=3,
        label='whitened', zorder=3)
ax.axhline(k, color='crimson', linestyle='--', lw=1.4, alpha=0.6,
           label=f'planted $k={k}$')
if tau_1e is not None:
    ax.axvline(tau_1e, color='red', linestyle=':', lw=1.0, alpha=0.5,
               label=fr'$\rho < 1/e$ ($\tau$={tau_1e:.1f})')
ax.set_ylabel(r'$\#\{\lambda_j > \lambda_+(\tau)\}$')
ax.set_ylim(-1, k + 5)
ax.legend(fontsize=6, loc='center right')
ax.text(0.02, 0.95, f'$d={d}$, $k={k}$, $N={params["N"]}$',
        transform=ax.transAxes, fontsize=7, va='top')
ax.grid(True, alpha=0.2)

# ── Row 2: rho(tau) ──
ax = axes[1]
ax.plot(rho_taus_o, rho_vals_o, 'o-', color='tab:red', lw=1.6, markersize=4,
        label='original')
ax.plot(rho_taus_w, rho_vals_w, 's--', color=c_whit, lw=1.2, markersize=3,
        label='whitened', alpha=0.7)
ax.axhline(threshold, color='gray', linestyle='--', lw=1.0, alpha=0.6,
           label='$1/e$')
if tau_1e is not None:
    ax.axvline(tau_1e, color='red', linestyle=':', lw=1.0, alpha=0.5)
ax.set_ylabel(r'$\rho(\tau)$')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.2)

ax.set_xscale('symlog', linthresh=linthresh)
ax.set_xlabel(r'$\tau$')

fig.suptitle('Planted spikes: whitening = no-op\n'
             r'(isotropic $\Sigma_f = \sigma_f^2 I$)', fontsize=9, y=1.01)
fig.tight_layout()
path = os.path.join(OUT_DIR, 'fig_planted_spikes_whitened.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {path}")
