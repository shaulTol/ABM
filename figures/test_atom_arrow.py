#!/usr/bin/env python3
"""Test atom arrow design: floating purple box with predicted/sampled mass."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'noise_floor'))
from free_conv_utils import edges as fc_edges, density as fc_density, edge_tauinf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _spike_fig_style import (
    HIST_EMPIRICAL, FREECONV_CURVE, FLOOR_TAU, FLOOR_INF,
    HIST_ALPHA_EMP, CURVE_LW, VLINE_LW, VLINE_STYLE,
)
from style_config import setup_style
setup_style()

FIGDIR = os.path.dirname(os.path.abspath(__file__))
ATOM_COLOR = '#7B2D8E'
ZERO_TOL = 1e-8

gamma = 2.0
tau = 0.0
N = 500
d = int(gamma * N)
sigma = 1.0

# Simulate
rng = np.random.default_rng(42)
x0 = rng.standard_normal((N, d))
x_tau = x0.copy()
C_raw = (x_tau.T @ x0) / N
C_sym = 0.5 * (C_raw + C_raw.T)
eigs = np.linalg.eigvalsh(C_sym)

# Free-conv
lm, lp = fc_edges(tau, sigma, gamma=gamma)
_, lp_inf = edge_tauinf(sigma, gamma)
pad = 0.15 * (lp - lm)
lam_grid = np.linspace(lm - pad, lp + pad, 600)
rho_grid = fc_density(lam_grid, tau, sigma, gamma=gamma)

# Atom mass: predicted vs sampled
if tau == 0.0:
    atom_pred = max(0.0, 1.0 - 1.0 / gamma)
else:
    atom_pred = max(0.0, 1.0 - 2.0 / gamma)

nonzero_mask = np.abs(eigs) > ZERO_TOL
bulk_eigs = eigs[nonzero_mask]
atom_sampled = np.sum(~nonzero_mask) / len(eigs)
bulk_mass = 1.0 - atom_pred

fig, ax = plt.subplots(figsize=(5, 3.5))

# Histogram of BULK only (zeros removed) — density normalized to bulk mass
eig_min, eig_max = bulk_eigs.min(), bulk_eigs.max()
margin = 0.03 * (eig_max - eig_min)
bins = np.linspace(eig_min - margin, eig_max + margin, 35)
ax.hist(bulk_eigs, bins=bins, density=True, color=HIST_EMPIRICAL,
        alpha=HIST_ALPHA_EMP, edgecolor='none', zorder=2)

# Free-conv density (rescaled by 1/bulk_mass to match bulk-only histogram)
ax.plot(lam_grid, rho_grid / bulk_mass, color=FREECONV_CURVE, lw=CURVE_LW, zorder=4)

# Floor lines
ax.axvline(lp, color=FLOOR_TAU, ls=VLINE_STYLE, lw=VLINE_LW, zorder=5)
ax.axvline(lp_inf, color=FLOOR_INF, ls=VLINE_STYLE, lw=VLINE_LW - 0.2, zorder=5, alpha=0.8)
ax.axvline(0, color='black', ls='-', lw=0.3, alpha=0.2)

ax.set_ylim(bottom=0)
ax.set_title(fr'$\gamma = {gamma:g}$, $\tau = 0$', fontsize=10)
ax.set_xlabel(r'$\lambda$', fontsize=9)
ax.set_ylabel('Density', fontsize=9)

# --- Purple floating box with arrow to lambda=0 ---
box_text = (f'Atom at $\\lambda=0$\n'
            f'Predicted $w_0 = {atom_pred:.2f}$\n'
            f'Sampled  $w_0 = {atom_sampled:.2f}$')

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

out = os.path.join(FIGDIR, 'test_atom_arrow.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
