"""
Generate fig_da_parallel_ellipses_convergence.png — two panels side by side.
Left: DA1 angle from x-axis vs tau.
Right: lambda_1, lambda_2 (left y-axis) and lambda_1/lambda_2 ratio (right y-axis).
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics.da_directions import da_convergence_profile

# ─── GMM Setup (same as parallel_ellipses.py) ────────────────────────────
means = torch.tensor([[0.0, -3.0],
                       [0.0,  3.0]])
covs = torch.tensor([[[25.0, 0.0],
                       [0.0,  1.0]],
                      [[25.0, 0.0],
                       [0.0,  1.0]]])
weights = torch.tensor([0.5, 0.5])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Compute convergence profile ─────────────────────────────────────────
checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600]
print("Computing DA convergence profile...")
profile = da_convergence_profile(
    means, covs, weights,
    checkpoints=checkpoints,
    n_trajectories=10000, dt=0.01,
    device=device, seed=42,
    reference_direction=torch.tensor([1.0, 0.0]),  # x-axis
)

taus = np.array(profile['checkpoints'])
angles = np.array(profile['angles_deg'])
lam1 = np.array(profile['leading_eigenvalues'])
lam2 = np.array(profile['all_eigenvalues'][:, 1])
# Only compute ratio where lambda_2 > 0 to avoid blow-up
ratio = np.where(lam2 > 0.01, lam1 / lam2, np.nan)

# Use tau=400 as the transition point (where angle jumps and eigenvalues cross)
tau_star = 400
print(f"Marking transition at tau={tau_star}")

# Print all data for reference
for i in range(len(taus)):
    print(f"  tau={taus[i]:6d}  angle={angles[i]:5.1f}  lam1={lam1[i]:7.3f}  lam2={lam2[i]:7.3f}  ratio={ratio[i]:7.2f}")

# ─── Figure ──────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 8})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# LEFT: DA1 angle (left y) + ratio (right y)
ln_angle, = ax1.plot(taus, angles, 'o-', color='k', markersize=3, linewidth=1.2,
                     label=r'DA1 angle ($\degree$)')
ax1.set_xscale('log')
ax1.set_xlabel(r'Lag $\tau$')
ax1.set_ylabel(r'DA1 angle from $x$-axis ($\degree$)')
ax1.set_ylim(-5, 100)
ax1.grid(alpha=0.2, linewidth=0.5)
ax1.spines['top'].set_visible(False)

ax1r = ax1.twinx()
ln_ratio, = ax1r.plot(taus, ratio, ':', color='indianred', linewidth=2.0, marker='d',
                      markersize=3, label=r'$\lambda_1/\lambda_2$')
ax1r.set_ylabel(r'$\lambda_1 / \lambda_2$', color='indianred')
ax1r.tick_params(axis='y', labelcolor='indianred')
ax1r.spines['top'].set_visible(False)

lns_left = [ln_angle, ln_ratio]
ax1.legend(lns_left, [l.get_label() for l in lns_left], fontsize=7, loc='center left')

# RIGHT: eigenvalues only
ax2.plot(taus, lam1, 'o-', color='steelblue', markersize=3, linewidth=1.2,
         label=r'$\lambda_1(\tau)$')
ax2.plot(taus, lam2, 's--', color='steelblue', markersize=3, linewidth=1.2,
         alpha=0.6, label=r'$\lambda_2(\tau)$')
ax2.set_xscale('log')
ax2.set_xlabel(r'Lag $\tau$')
ax2.set_ylabel('Eigenvalue')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(alpha=0.2, linewidth=0.5)

plt.tight_layout(pad=0.5)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_da_parallel_ellipses_convergence.png')
fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close()
print(f"Saved {path}")
