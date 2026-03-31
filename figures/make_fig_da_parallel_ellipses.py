#!/usr/bin/env python3
"""
Generate two DA parallel ellipses figures:
  1. fig_da_parallel_ellipses_2panel.png  — scatter + histograms (main paper)
  2. fig_da_parallel_ellipses_convergence.png — angle + eigenvalues (appendix)
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.metrics.da_directions import extract_da_directions, da_convergence_profile

OUT_DIR = os.path.dirname(__file__)

# ─── GMM Setup ───────────────────────────────────────────────────────────
means = torch.tensor([[0.0, -3.0], [0.0, 3.0]])
covs = torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                      [[25.0, 0.0], [0.0, 1.0]]])
weights = torch.tensor([0.5, 0.5])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

# ─── PCA (analytical) ────────────────────────────────────────────────────
mu_bar = (weights.unsqueeze(1) * means).sum(0)
cov_f = torch.zeros(2, 2)
for i in range(2):
    diff = means[i] - mu_bar
    cov_f += weights[i] * (covs[i] + torch.outer(diff, diff))
pca_evals, pca_evecs = torch.linalg.eigh(cov_f)
pc1 = pca_evecs[:, -1].numpy()

# ─── DA Directions ───────────────────────────────────────────────────────
print("Computing DA directions...")
da_evals, da_evecs, C_tau = extract_da_directions(
    means, covs, weights, lag=800, n_trajectories=10000, dt=0.01,
    device=device, seed=seed)
da1 = da_evecs[:, 0].cpu().numpy()

# ─── Convergence Profile ─────────────────────────────────────────────────
print("Computing convergence profile...")
checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600]
profile = da_convergence_profile(
    means, covs, weights, checkpoints=checkpoints,
    n_trajectories=10000, dt=0.01, device=device, seed=seed,
    reference_direction=torch.tensor([1.0, 0.0]))

# ─── Sample points ───────────────────────────────────────────────────────
torch.manual_seed(seed)
from src.simulation.langevin import LangevinSimulatorD
sim = LangevinSimulatorD(means, covs, weights, dt=0.01, device=device)
samples = sim.sample_initial(2000).cpu().numpy()
origin = mu_bar.numpy()

# ─── FIGURE 1: 2-panel (main paper) ──────────────────────────────────────
fig = plt.figure(figsize=(8, 3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
ax1 = fig.add_subplot(gs[0])
# Right panel: inset-style, vertically squeezed
ax2 = fig.add_axes([0.62, 0.22, 0.34, 0.6])  # [left, bottom, width, height]

# Left: scatter + arrows
ax1.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.35, color='0.55')
arrow_scale = 4.0
for sign in [1, -1]:
    ls = '-' if sign == 1 else '--'
    ax1.annotate('', xy=origin + sign * arrow_scale * pc1, xytext=origin,
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.annotate('', xy=origin + sign * arrow_scale * da1, xytext=origin,
                 arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
ax1.text(origin[0] + arrow_scale * pc1[0] + 0.3,
         origin[1] + arrow_scale * pc1[1] + 0.2,
         'PC1', color='red', fontsize=8, fontweight='bold')
ax1.text(origin[0] + arrow_scale * da1[0] + 0.3,
         origin[1] + arrow_scale * da1[1],
         'DA1', color='tab:blue', fontsize=8, fontweight='bold')
ax1.set_aspect('equal')
ax1.tick_params(axis='both', labelsize=7)

# Right: projected histograms
proj_pc1 = samples @ pc1
proj_da1 = samples @ da1
ax2.hist(proj_pc1, bins=60, alpha=0.5, color='red', label='PC1', density=True)
ax2.hist(proj_da1, bins=60, alpha=0.5, color='tab:blue', label='DA1', density=True)
ax2.set_xlabel('Projected value', fontsize=8)
ax2.set_yticks([])
ax2.legend(fontsize=7, framealpha=0.8)
ax2.tick_params(axis='x', labelsize=7)
for spine in ['top', 'right', 'left']:
    ax2.spines[spine].set_visible(False)

fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
path1 = os.path.join(OUT_DIR, 'fig_da_parallel_ellipses_2panel.png')
fig.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {path1}")

# ─── FIGURE 2: convergence (appendix) ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Left: angle vs tau
ax1.plot(profile['checkpoints'], profile['angles_deg'], 'ko-', markersize=4)
ax1.axhline(y=90, color='tab:blue', linestyle='--', alpha=0.5, label='y-axis (90°)')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='x-axis (0°)')
ax1.set_xlabel(r'Lag $\tau$', fontsize=8)
ax1.set_ylabel('DA1 angle from x-axis (°)', fontsize=8)
ax1.set_xscale('log')
ax1.set_ylim(-5, 100)
ax1.legend(fontsize=7)
ax1.tick_params(labelsize=7)
ax1.grid(True, alpha=0.2)

# Right: eigenvalues vs tau
ax2.plot(profile['checkpoints'], profile['leading_eigenvalues'], 'ko-',
         markersize=4, label=r'$\lambda_1(\tau)$')
if profile['all_eigenvalues'].shape[1] >= 2:
    ax2.plot(profile['checkpoints'], profile['all_eigenvalues'][:, 1],
             's--', color='gray', markersize=3, label=r'$\lambda_2(\tau)$')
ax2.set_xlabel(r'Lag $\tau$', fontsize=8)
ax2.set_ylabel('Eigenvalue', fontsize=8)
ax2.set_xscale('log')
ax2.legend(fontsize=7)
ax2.tick_params(labelsize=7)
ax2.grid(True, alpha=0.2)

plt.tight_layout(pad=0.5)
path2 = os.path.join(OUT_DIR, 'fig_da_parallel_ellipses_convergence.png')
fig.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {path2}")
