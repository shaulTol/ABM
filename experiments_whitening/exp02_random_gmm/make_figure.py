#!/usr/bin/env python3
"""
Random GMM scatter: whitened Ŝ_w rank vs MI rank, overlaid with original Ŝ.
Matches style of figures/fig_random_sweep.png.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'figures'))
from style_config import setup_style, SSA_COLOR, MI_COLOR, DIAGONAL_COLOR, TEXT_WIDTH

setup_style()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load whitened results
with open(os.path.join(OUT_DIR, 'results', 'random_sweep_10d_whitened.json')) as f:
    data_w = json.load(f)

# Load original results
ORIG = os.path.join(ROOT, 'experiments', 'ssa_truncation_redef', 'results',
                    'random_sweep_10d_n200.json')
with open(ORIG) as f:
    data_o = json.load(f)

records_w = data_w['records']
records_o = data_o['records']
n = len(records_w)

# Extract SSA values
ssa_w = np.array([r['s_hat_1N_whitened'] for r in records_w])
ssa_o = np.array([r['s_hat_1N'] for r in records_o])
mi = np.array([r['mi'] for r in records_w])

# Ranks
ssa_w_ranks = rankdata(ssa_w)
ssa_o_ranks = rankdata(ssa_o)
mi_ranks = rankdata(mi)

rho_w = data_w['spearman']['rho']
rho_o = data_o['spearman_1N']['rho']

# ── Figure: side-by-side ──
fig, (ax_o, ax_w) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.45))

# Left: original
ax_o.scatter(ssa_o_ranks, mi_ranks, s=25, color=SSA_COLOR,
             edgecolors='k', linewidths=0.4, alpha=0.7, zorder=3)
ax_o.plot([1, n], [1, n], '--', color=DIAGONAL_COLOR, linewidth=0.8)
ax_o.set_xlabel(r'$\hat{S}$ Rank')
ax_o.set_ylabel('MI Rank')
ax_o.set_title('Original')
ax_o.set_xlim(0, n + 1)
ax_o.set_ylim(0, n + 1)
ax_o.set_aspect('equal')
ax_o.text(0.05, 0.95, f'Spearman $\\rho$ = {rho_o:.3f}',
          transform=ax_o.transAxes, va='top', fontsize=8,
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right: whitened
ax_w.scatter(ssa_w_ranks, mi_ranks, s=25, color=MI_COLOR,
             edgecolors='k', linewidths=0.4, alpha=0.7, zorder=3)
ax_w.plot([1, n], [1, n], '--', color=DIAGONAL_COLOR, linewidth=0.8)
ax_w.set_xlabel(r'$\hat{S}_w$ Rank')
ax_w.set_ylabel('MI Rank')
ax_w.set_title('Whitened')
ax_w.set_xlim(0, n + 1)
ax_w.set_ylim(0, n + 1)
ax_w.set_aspect('equal')
ax_w.text(0.05, 0.95, f'Spearman $\\rho$ = {rho_w:.3f}',
          transform=ax_w.transAxes, va='top', fontsize=8,
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Random GMMs (10D, $N$=200)', fontsize=10, y=1.02)
fig.tight_layout()
path = os.path.join(OUT_DIR, 'fig_random_sweep_whitened.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {path}")
