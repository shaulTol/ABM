"""
Generate fig_dispersion_vs_fragmentation.png — two 2D scatter panels:
  Left:  Unimodal Gaussian blob (high H, low SSA)
  Right: Parallel-ellipses GMM with DA1/PC1 arrows (low H, high SSA)

Each panel has an SSA / H annotation box in the top-right corner.
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

from style_config import setup_style, TEXT_WIDTH
from src.metrics.da_directions import extract_da_directions
from src.simulation.langevin import LangevinSimulatorD, simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_ssa_score

setup_style()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

# ─── GMM (right panel): parallel ellipses ────────────────────────────────
means = torch.tensor([[0.0, -3.0], [0.0, 3.0]])
covs = torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                      [[25.0, 0.0], [0.0, 1.0]]])
weights = torch.tensor([0.5, 0.5])

# PCA (analytical)
mu_bar = (weights.unsqueeze(1) * means).sum(0)
cov_f = torch.zeros(2, 2)
for i in range(2):
    diff = means[i] - mu_bar
    cov_f += weights[i] * (covs[i] + torch.outer(diff, diff))
pca_evals, pca_evecs = torch.linalg.eigh(cov_f)
pc1 = pca_evecs[:, -1].numpy()

# DA directions
print("Computing DA directions...")
da_evals, da_evecs, C_tau = extract_da_directions(
    means, covs, weights, lag=1200, n_trajectories=10000, dt=0.01,
    device=device, seed=seed)
da1 = da_evecs[:, 0].cpu().numpy()

# Sample points for both panels
N_pts = 2000
torch.manual_seed(seed)
sim = LangevinSimulatorD(means, covs, weights, dt=0.01, device=device)
samples_gmm = sim.sample_initial(N_pts).cpu().numpy()
origin = mu_bar.numpy()

# ─── Unimodal blob (left panel) ──────────────────────────────────────────
# Covariance: 25*I — isotropic blob with same x-variance as the GMM
cov_blob = np.array([[25.0, 0.0], [0.0, 25.0]])
rng = np.random.default_rng(seed)
samples_blob = rng.multivariate_normal([0.0, 0.0], cov_blob, size=N_pts)

# PCA for blob (analytical — eigenvalues of 25*I are both 25)
blob_pca_evals = np.array([25.0, 25.0])
blob_pca_evecs = np.eye(2)  # arbitrary for isotropic; eigh will pick axes
# Use actual eigh for consistency
blob_pca_evals_t, blob_pca_evecs_t = torch.linalg.eigh(
    torch.tensor(cov_blob, dtype=torch.float32))
blob_pc1 = blob_pca_evecs_t[:, -1].numpy()

# DA for blob — use a moderate lag where signal is still present but decaying
# For isotropic Gaussian, ρ(τ) = e^{-τ/2}; at lag=100, dt=0.01, τ=1, ρ=0.61
# The gap should be ~1 at any lag (no preferred direction)
print("Computing DA directions (blob)...")
means_blob_da = torch.tensor([[0.0, 0.0]])
covs_blob_da = torch.tensor([[[25.0, 0.0], [0.0, 25.0]]])
weights_blob_da = torch.tensor([1.0])
blob_da_evals, blob_da_evecs, blob_C_tau = extract_da_directions(
    means_blob_da, covs_blob_da, weights_blob_da, lag=100,
    n_trajectories=10000, dt=0.01, device=device, seed=seed + 500)
blob_da1_raw = blob_da_evecs[:, 0].cpu().numpy()
blob_da_evals_np = blob_da_evals.cpu().numpy()
# For isotropic Gaussian, DA1 direction is arbitrary (gap ≈ 1).
# Rotate ~40° counter-clockwise from the raw direction to avoid overlapping
# the DA1 text label.
rot_angle = np.radians(40)
cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
blob_da1 = np.array([cos_r * blob_da1_raw[0] - sin_r * blob_da1_raw[1],
                     sin_r * blob_da1_raw[0] + cos_r * blob_da1_raw[1]])

# Report blob DA/PCA diagnostics
blob_da1_angle = np.degrees(np.arctan2(blob_da1[1], blob_da1[0]))
blob_pc1_angle = np.degrees(np.arctan2(blob_pc1[1], blob_pc1[0]))
blob_da_gap = float(blob_da_evals_np[0] / blob_da_evals_np[1]) if blob_da_evals_np[1] > 0 else float('inf')
blob_pca_ratio = float(blob_pca_evals_t[-1] / blob_pca_evals_t[-2])

print(f"\nBlob diagnostics:")
print(f"  DA1 angle from x-axis: {blob_da1_angle:.1f}°")
print(f"  PC1 angle from x-axis: {blob_pc1_angle:.1f}°")
print(f"  DA eigenvalues: {blob_da_evals_np}")
print(f"  DA1/DA2 gap ratio: {blob_da_gap:.3f}")
print(f"  PCA eigenvalues: {blob_pca_evals_t.numpy()}")
print(f"  PC1/PC2 ratio: {blob_pca_ratio:.3f}")

# ─── Compute SSA for both ────────────────────────────────────────────────
dt_sim = 0.01
T_sim = 5000
max_lag = 4000
n_traj = 2000

print("Computing SSA (GMM)...")
traj_gmm = simulate_d_trajectories(
    means, covs, weights, n_trajectories=n_traj, T=T_sim, dt=dt_sim,
    device=device, seed=seed + 100)
ssa_gmm, _ = compute_vector_ssa_score(
    traj_gmm.cpu(), max_lag, means, covs, weights, dt=dt_sim, normalize='riemann')

print("Computing SSA (blob)...")
means_blob = torch.tensor([[0.0, 0.0]])
covs_blob_t = torch.tensor([[[25.0, 0.0], [0.0, 25.0]]])
weights_blob = torch.tensor([1.0])
traj_blob = simulate_d_trajectories(
    means_blob, covs_blob_t, weights_blob, n_trajectories=n_traj, T=T_sim, dt=dt_sim,
    device=device, seed=seed + 200)
ssa_blob, _ = compute_vector_ssa_score(
    traj_blob.cpu(), max_lag, means_blob, covs_blob_t, weights_blob, dt=dt_sim, normalize='riemann')

# ─── Compute differential entropy H ──────────────────────────────────────
# Gaussian: H = 0.5 * ln((2*pi*e)^d * det(Sigma))
d = 2
H_blob = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(cov_blob)))

# GMM: no closed form, use Monte Carlo estimate
# H_gmm ≈ -E[log p(x)] with samples from the mixture
n_mc = 100000
rng2 = np.random.default_rng(seed + 300)
labels_mc = rng2.binomial(1, 0.5, n_mc)
means_np = means.numpy()
covs_np = covs.numpy()
x_mc = np.where(labels_mc[:, None] == 0,
                rng2.multivariate_normal(means_np[0], covs_np[0], n_mc),
                rng2.multivariate_normal(means_np[1], covs_np[1], n_mc))
# log p(x) = log(0.5 * N(x|mu0,S0) + 0.5 * N(x|mu1,S1))
from scipy.stats import multivariate_normal as mvn
logp0 = mvn.logpdf(x_mc, means_np[0], covs_np[0])
logp1 = mvn.logpdf(x_mc, means_np[1], covs_np[1])
log_max = np.maximum(logp0, logp1)
log_mixture = log_max + np.log(0.5 * np.exp(logp0 - log_max) + 0.5 * np.exp(logp1 - log_max))
H_gmm = -np.mean(log_mixture)

print(f"\nResults:")
print(f"  Blob:     SSA = {ssa_blob:.3f},  H = {H_blob:.2f} nats")
print(f"  Ellipses: SSA = {ssa_gmm:.3f},  H = {H_gmm:.2f} nats")
print(f"  H_blob > H_gmm: {H_blob > H_gmm}")
print(f"  SSA_blob < SSA_gmm: {ssa_blob < ssa_gmm}")

# ─── FIGURE ──────────────────────────────────────────────────────────────
# Shared symmetric axis range — generous padding around both distributions
lim = 16
xlim = (-lim, lim)
ylim = (-lim, lim)

# Each panel must be square for equal aspect to work without distortion
panel_size = TEXT_WIDTH / 2 - 0.3  # account for labels/padding
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(TEXT_WIDTH, panel_size),
    subplot_kw=dict(aspect='equal', adjustable='box'))

arrow_scale = 4.0
pt_kw = dict(s=1, alpha=0.30, color='0.55', rasterized=True)
contour_kw = dict(colors='0.35', linewidths=0.6, alpha=0.7)

# ─── Analytic density grids ──────────────────────────────────────────────
grid_n = 300
xg = np.linspace(xlim[0], xlim[1], grid_n)
yg = np.linspace(ylim[0], ylim[1], grid_n)
Xg, Yg = np.meshgrid(xg, yg)
pos = np.stack([Xg, Yg], axis=-1)  # (grid_n, grid_n, 2)

# Blob density: N(0, 25*I)
Z_blob = mvn.pdf(pos, mean=[0.0, 0.0], cov=cov_blob)

# GMM density: 0.5*N(mu0, S0) + 0.5*N(mu1, S1)
Z_gmm = 0.5 * mvn.pdf(pos, mean=means_np[0], cov=covs_np[0]) + \
        0.5 * mvn.pdf(pos, mean=means_np[1], cov=covs_np[1])

# Contour levels: 6 evenly-spaced levels from 10% to 90% of peak
n_levels = 6
blob_levels = np.linspace(0.10 * Z_blob.max(), 0.90 * Z_blob.max(), n_levels)
gmm_levels = np.linspace(0.10 * Z_gmm.max(), 0.90 * Z_gmm.max(), n_levels)

# ── Left panel: unimodal blob ──
ax_left.contour(Xg, Yg, Z_blob, levels=blob_levels, **contour_kw)
ax_left.scatter(samples_blob[:, 0], samples_blob[:, 1], **pt_kw)

# DA1/PC1 arrows
origin_blob = np.array([0.0, 0.0])
for sign in [1, -1]:
    ax_left.annotate('', xy=origin_blob + sign * arrow_scale * blob_pc1,
                     xytext=origin_blob,
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_left.annotate('', xy=origin_blob + sign * arrow_scale * blob_da1,
                     xytext=origin_blob,
                     arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
ax_left.text(origin_blob[0] + arrow_scale * blob_pc1[0] + 0.3,
             origin_blob[1] + arrow_scale * blob_pc1[1] + 0.2,
             'PC1', color='red', fontsize=8, fontweight='bold')
ax_left.text(origin_blob[0] + (arrow_scale + 1.5) * blob_da1[0],
             origin_blob[1] + (arrow_scale + 1.5) * blob_da1[1],
             'DA1', color='tab:blue', fontsize=8, fontweight='bold')

ax_left.set_xlim(xlim)
ax_left.set_ylim(ylim)
ax_left.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_left.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_left.set_xlabel(r'$x_1$')
ax_left.set_ylabel(r'$x_2$')

# Annotation box
bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7',
                  alpha=0.85)
ax_left.text(0.97, 0.97,
             f'SSA = {ssa_blob:.3f}\n$H$ = {H_blob:.2f} nats',
             transform=ax_left.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right',
             bbox=bbox_props)

# ── Right panel: parallel ellipses with DA1/PC1 arrows ──
ax_right.contour(Xg, Yg, Z_gmm, levels=gmm_levels, **contour_kw)
ax_right.scatter(samples_gmm[:, 0], samples_gmm[:, 1], **pt_kw)

for sign in [1, -1]:
    ax_right.annotate('', xy=origin + sign * arrow_scale * pc1, xytext=origin,
                      arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_right.annotate('', xy=origin + sign * arrow_scale * da1, xytext=origin,
                      arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
ax_right.text(origin[0] + arrow_scale * pc1[0] + 0.3,
              origin[1] + arrow_scale * pc1[1] + 0.2,
              'PC1', color='red', fontsize=8, fontweight='bold')
ax_right.text(origin[0] + arrow_scale * da1[0] + 0.3,
              origin[1] + arrow_scale * da1[1],
              'DA1', color='tab:blue', fontsize=8, fontweight='bold')

ax_right.set_xlim(xlim)
ax_right.set_ylim(ylim)
ax_right.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_right.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_right.set_xlabel(r'$x_1$')
ax_right.set_ylabel(r'$x_2$')

ax_right.text(0.97, 0.97,
              f'SSA = {ssa_gmm:.3f}\n$H$ = {H_gmm:.2f} nats',
              transform=ax_right.transAxes, fontsize=7,
              verticalalignment='top', horizontalalignment='right',
              bbox=bbox_props)

fig.tight_layout(pad=0.5)
path = os.path.join(OUT_DIR, 'fig_dispersion_vs_fragmentation.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved {path}")
