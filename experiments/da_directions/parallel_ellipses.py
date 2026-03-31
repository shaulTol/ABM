#!/usr/bin/env python3
"""
DA Directions Sanity Check: "Parallel Ellipses"

Two equal-weight 2D Gaussians:
    means at (0, -3) and (0, +3), covariance diag(25, 1)
    → Two horizontal ellipses stacked vertically

Expected:
    PC1 ≈ x-axis (max variance from the stretch)
    DA1 ≈ y-axis (diffusion bottleneck is the vertical gap)

4-panel figure:
    1. Scatter with PC1 (red) and DA1 (blue) arrows
    2. Projected histograms onto PC1 vs DA1
    3. Eigenvector angle vs τ (rotation from ~0° to ~90°)
    4. Leading eigenvalue vs τ (exponential decay)

Pass criteria:
    - PC1 within 15° of x-axis
    - DA1 within 15° of y-axis
    - Angle plot shows clear rotation
    - DA1 histogram bimodal
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.metrics.da_directions import extract_da_directions, da_convergence_profile

# ─── GMM Setup ───────────────────────────────────────────────────────────
means = torch.tensor([[0.0, -3.0],
                       [0.0,  3.0]])
covs = torch.tensor([[[25.0, 0.0],
                       [0.0,  1.0]],
                      [[25.0, 0.0],
                       [0.0,  1.0]]])
weights = torch.tensor([0.5, 0.5])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

# ─── PCA (analytical) ────────────────────────────────────────────────────
# Mixture covariance: Σ_f = Σ_i π_i [Σ_i + (μ_i - μ̄)(μ_i - μ̄)ᵀ]
mu_bar = (weights.unsqueeze(1) * means).sum(0)  # (0, 0)
cov_f = torch.zeros(2, 2)
for i in range(2):
    diff = means[i] - mu_bar
    cov_f += weights[i] * (covs[i] + torch.outer(diff, diff))

pca_evals, pca_evecs = torch.linalg.eigh(cov_f)
# eigh returns ascending order; PC1 = last column
pc1 = pca_evecs[:, -1].numpy()
pc1_angle = np.degrees(np.arccos(np.abs(pc1[0])))  # angle from x-axis

print(f"Mixture covariance:\n{cov_f.numpy()}")
print(f"PCA eigenvalues: {pca_evals.numpy()}")
print(f"PC1 direction: {pc1}  (angle from x-axis: {pc1_angle:.1f}°)")

# ─── DA Directions ───────────────────────────────────────────────────────
print("\nComputing DA directions (lag=800, N=10000)...")
da_evals, da_evecs, C_tau = extract_da_directions(
    means, covs, weights,
    lag=800, n_trajectories=10000, dt=0.01,
    device=device, seed=seed,
)

da1 = da_evecs[:, 0].cpu().numpy()  # Leading DA direction
da1_angle_from_y = np.degrees(np.arccos(np.abs(da1[1])))  # angle from y-axis

print(f"DA eigenvalues: {da_evals.cpu().numpy()}")
print(f"DA1 direction: {da1}  (angle from y-axis: {da1_angle_from_y:.1f}°)")

# ─── Convergence Profile ─────────────────────────────────────────────────
print("\nComputing convergence profile...")
checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600]
profile = da_convergence_profile(
    means, covs, weights,
    checkpoints=checkpoints,
    n_trajectories=10000, dt=0.01,
    device=device, seed=seed,
    reference_direction=torch.tensor([1.0, 0.0]),  # x-axis
)

# ─── Sample points for scatter ───────────────────────────────────────────
torch.manual_seed(seed)
from src.simulation.langevin import LangevinSimulatorD
sim = LangevinSimulatorD(means, covs, weights, dt=0.01, device=device)
samples = sim.sample_initial(2000).cpu().numpy()

# ─── Pass criteria ────────────────────────────────────────────────────────
pass_pc1 = pc1_angle < 15.0
pass_da1 = da1_angle_from_y < 15.0

# Check bimodality of DA1 projection via Hartigan's dip or simple peak counting
proj_da1 = samples @ da1
hist_vals, bin_edges = np.histogram(proj_da1, bins=50)
# Simple bimodality check: histogram should have a valley in the middle
mid = len(hist_vals) // 2
left_peak = hist_vals[:mid].max()
right_peak = hist_vals[mid:].max()
valley = hist_vals[mid-3:mid+3].min()
pass_bimodal = valley < 0.5 * min(left_peak, right_peak)

# Check angle rotation: final angle should be > 60° (near 90°)
final_angle = profile['angles_deg'][-1]
pass_rotation = final_angle > 60.0

print(f"\n{'='*50}")
print(f"PASS CRITERIA:")
print(f"  PC1 within 15° of x-axis: {'PASS' if pass_pc1 else 'FAIL'} ({pc1_angle:.1f}°)")
print(f"  DA1 within 15° of y-axis: {'PASS' if pass_da1 else 'FAIL'} ({da1_angle_from_y:.1f}°)")
print(f"  DA1 histogram bimodal:    {'PASS' if pass_bimodal else 'FAIL'}")
print(f"  Angle rotation > 60°:     {'PASS' if pass_rotation else 'FAIL'} ({final_angle:.1f}°)")
all_pass = pass_pc1 and pass_da1 and pass_bimodal and pass_rotation
print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print(f"{'='*50}")

# ─── 4-Panel Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Scatter with arrows
ax = axes[0, 0]
ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, color='gray')
origin = mu_bar.numpy()
arrow_scale = 4.0
ax.annotate('', xy=origin + arrow_scale * pc1, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.annotate('', xy=origin + arrow_scale * da1, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax.annotate('', xy=origin - arrow_scale * pc1, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='--'))
ax.annotate('', xy=origin - arrow_scale * da1, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5, linestyle='--'))
ax.text(origin[0] + arrow_scale * pc1[0] + 0.3, origin[1] + arrow_scale * pc1[1],
        'PC1', color='red', fontsize=12, fontweight='bold')
ax.text(origin[0] + arrow_scale * da1[0] + 0.3, origin[1] + arrow_scale * da1[1],
        'DA1', color='blue', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scatter: PC1 (red) vs DA1 (blue)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Panel 2: Projected histograms
ax = axes[0, 1]
proj_pc1 = samples @ pc1
proj_da1 = samples @ da1
ax.hist(proj_pc1, bins=60, alpha=0.5, color='red', label='proj. onto PC1', density=True)
ax.hist(proj_da1, bins=60, alpha=0.5, color='blue', label='proj. onto DA1', density=True)
ax.set_xlabel('Projected value')
ax.set_ylabel('Density')
ax.set_title('Projected histograms')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Eigenvector angle vs τ
ax = axes[1, 0]
ax.plot(profile['checkpoints'], profile['angles_deg'], 'ko-', markersize=5)
ax.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90° (y-axis)')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='0° (x-axis)')
ax.set_xlabel(r'Lag $\tau$ (diffusion steps)')
ax.set_ylabel('Angle of DA1 from x-axis (°)')
ax.set_title(r'DA1 angle vs $\tau$')
ax.set_xscale('log')
ax.set_ylim(-5, 100)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Leading eigenvalue vs τ
ax = axes[1, 1]
ax.plot(profile['checkpoints'], profile['leading_eigenvalues'], 'ko-', markersize=5)
if profile['all_eigenvalues'].shape[1] >= 2:
    ax.plot(profile['checkpoints'], profile['all_eigenvalues'][:, 1],
            's--', color='gray', markersize=4, label=r'$\lambda_2(\tau)$')
ax.set_xlabel(r'Lag $\tau$ (diffusion steps)')
ax.set_ylabel('Eigenvalue')
ax.set_title(r'Eigenvalues of $C(\tau)$ vs $\tau$')
ax.set_xscale('log')
ax.legend([r'$\lambda_1(\tau)$', r'$\lambda_2(\tau)$'])
ax.grid(True, alpha=0.3)

plt.suptitle('DA Directions Sanity Check: Parallel Ellipses', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(output_dir, 'parallel_ellipses_diagnostic.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {fig_path}")
plt.close()
