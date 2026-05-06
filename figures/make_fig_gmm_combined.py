#!/usr/bin/env python3
"""
Combined figure: 4 GMM sweep panels (left 2×2) + 2 dispersion panels (right column).
Layout: 3 columns × 2 rows.
  [Separation] [Variance]  [Blob (isotropic)]
  [Weight]     [N modes]   [Bimodal GMM]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from style_config import setup_style, TEXT_WIDTH
setup_style()

import torch
from src.metrics.da_directions import extract_da_directions
from src.simulation.langevin import LangevinSimulatorD, simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa
from scipy.stats import multivariate_normal as mvn

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════════════════════════
# Left 2×2: GMM sweep curves (from saved JSON)
# ══════════════════════════════════════════════════════════════════════════════
RESULTS_PATH = os.path.join(FIG_DIR, '..', 'experiments', 'ssa_truncation_redef',
                            'results', 'gmm_10d_n200_all_sweeps_T200.json')
FLOOR_KEY = '1N'

with open(RESULTS_PATH) as f:
    sweep_data = json.load(f)

sweep_configs = [
    ('sep_10d', 'Separation', 'Separation'),
    ('var_10d', 'Variance', 'Variance'),
    ('weight_10d', 'Weight', 'Weight'),
    ('nmodes_10d', 'Number of modes', 'Mode count'),
]
CLIP = {'sep_10d': (None, 6.0), 'var_10d': (0.75, None)}

# ══════════════════════════════════════════════════════════════════════════════
# Right column: dispersion vs fragmentation (recompute DA/PCA/samples)
# ══════════════════════════════════════════════════════════════════════════════
seed = 42
N_pts = 2000

# GMM: parallel ellipses
means = torch.tensor([[0.0, -3.0], [0.0, 3.0]])
covs = torch.tensor([[[25.0, 0.0], [0.0, 1.0]],
                      [[25.0, 0.0], [0.0, 1.0]]])
weights = torch.tensor([0.5, 0.5])

mu_bar = (weights.unsqueeze(1) * means).sum(0)
cov_f = torch.zeros(2, 2)
for i in range(2):
    diff = means[i] - mu_bar
    cov_f += weights[i] * (covs[i] + torch.outer(diff, diff))
pca_evals, pca_evecs = torch.linalg.eigh(cov_f)
pc1 = pca_evecs[:, -1].numpy()

DA_LAG_GMM = 1200
DA_DT = 0.01
print("Computing DA directions (GMM)...")
da_evals, da_evecs, C_tau = extract_da_directions(
    means, covs, weights, lag=DA_LAG_GMM, n_trajectories=10000, dt=DA_DT,
    device=device, seed=seed)
da1 = da_evecs[:, 0].cpu().numpy()

torch.manual_seed(seed)
sim = LangevinSimulatorD(means, covs, weights, dt=0.01, device=device)
samples_gmm = sim.sample_initial(N_pts).cpu().numpy()
origin = mu_bar.numpy()

# Blob: isotropic Gaussian
cov_blob = np.array([[25.0, 0.0], [0.0, 25.0]])
rng = np.random.default_rng(seed)
samples_blob = rng.multivariate_normal([0.0, 0.0], cov_blob, size=N_pts)

blob_pca_evals_t, blob_pca_evecs_t = torch.linalg.eigh(
    torch.tensor(cov_blob, dtype=torch.float32))
blob_pc1 = blob_pca_evecs_t[:, -1].numpy()

print("Computing DA directions (blob)...")
means_blob_da = torch.tensor([[0.0, 0.0]])
covs_blob_da = torch.tensor([[[25.0, 0.0], [0.0, 25.0]]])
weights_blob_da = torch.tensor([1.0])
blob_da_evals, blob_da_evecs, blob_C_tau = extract_da_directions(
    means_blob_da, covs_blob_da, weights_blob_da, lag=100,
    n_trajectories=10000, dt=0.01, device=device, seed=seed + 500)
blob_da1_raw = blob_da_evecs[:, 0].cpu().numpy()

# Rotate blob DA1 ~40° to avoid label overlap (direction is arbitrary for isotropic)
rot_angle = np.radians(40)
cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
blob_da1 = np.array([cos_r * blob_da1_raw[0] - sin_r * blob_da1_raw[1],
                     sin_r * blob_da1_raw[0] + cos_r * blob_da1_raw[1]])

# Load pre-computed SSA values
npz = np.load(os.path.join(FIG_DIR, 'dispersion_rho_sq_dt01.npz'))
ssa_blob = float(npz['ssa_blob'])
ssa_gmm = float(npz['ssa_gmm'])

# Entropy
d = 2
H_blob = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(cov_blob)))
means_np = means.numpy()
covs_np = covs.numpy()
n_mc = 100000
rng2 = np.random.default_rng(seed + 300)
labels_mc = rng2.binomial(1, 0.5, n_mc)
x_mc = np.where(labels_mc[:, None] == 0,
                rng2.multivariate_normal(means_np[0], covs_np[0], n_mc),
                rng2.multivariate_normal(means_np[1], covs_np[1], n_mc))
logp0 = mvn.logpdf(x_mc, means_np[0], covs_np[0])
logp1 = mvn.logpdf(x_mc, means_np[1], covs_np[1])
log_max = np.maximum(logp0, logp1)
log_mixture = log_max + np.log(0.5 * np.exp(logp0 - log_max) + 0.5 * np.exp(logp1 - log_max))
H_gmm = -np.mean(log_mixture)

print(f"Blob: SSA={ssa_blob:.3f}, H={H_blob:.2f}")
print(f"GMM:  SSA={ssa_gmm:.3f}, H={H_gmm:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig_w = TEXT_WIDTH * 1.8  # wider than single-column
fig_h = fig_w * 0.55

fig = plt.figure(figsize=(fig_w, fig_h))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1],
             wspace=0.22, hspace=0.25)

# ── Sweep panels (left 2×2) ──────────────────────────────────────────────────
sweep_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
panel_labels = ['(a)', '(b)', '(c)', '(d)']

for idx, (config, xlabel, title) in enumerate(sweep_configs):
    row, col = sweep_positions[idx]
    ax = fig.add_subplot(gs[row, col])

    sw = sweep_data['sweeps'][config]
    params = np.array(sw['param_values'])
    ssa_arr = np.array(sw[f's_hat_{FLOOR_KEY}'])
    mi_arr = np.array(sw['mi'])

    if config in CLIP:
        lo, hi = CLIP[config]
        mask = np.ones(len(params), dtype=bool)
        if lo is not None:
            mask &= params >= lo
        if hi is not None:
            mask &= params <= hi
        params = params[mask]
        ssa_arr = ssa_arr[:, mask]
        mi_arr = mi_arr[:, mask]

    ssa_mean = ssa_arr.mean(axis=0)
    ssa_sd = ssa_arr.std(axis=0)
    mi_mean = mi_arr.mean(axis=0)

    ax.plot(params, ssa_mean, '-', color='tab:blue', linewidth=2, label='SSA')
    ax.fill_between(params, ssa_mean - ssa_sd, ssa_mean + ssa_sd,
                    color='tab:blue', alpha=0.25, label=r'$\pm 1$ SD')
    ax.set_xlabel(xlabel, fontsize=10, labelpad=3)
    ax.tick_params(axis='x', labelsize=8, pad=1)

    if col == 0:
        ax.set_ylabel('SSA', color='tab:blue', fontsize=10, labelpad=3)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=8, pad=1)

    # MI on twin axis
    ax2 = ax.twinx()
    ax2.plot(params, mi_mean, 's--', color='tab:red', markersize=3,
             linewidth=1.5, label='MI')
    if col == 1:
        ax2.set_ylabel('MI (nats)', color='tab:red', fontsize=10, labelpad=3)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=8, pad=1)

    if config == 'nmodes_10d':
        from matplotlib.ticker import MultipleLocator
        ax2.yaxis.set_major_locator(MultipleLocator(0.4))
        ax.set_xticks([1, 2, 3, 4, 5])
        for ki in [3, 4]:
            ax.annotate(r'$\dagger$', xy=(params[ki], ssa_mean[ki]),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=12, color='tab:blue', ha='center', weight='bold')

    if idx == 2:  # panel (c) — weight
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
                  fontsize=9, framealpha=0.8, handlelength=2)

    # Panel label — (b) and (c) top-right, others top-left
    if idx in [1, 2]:  # (b) variance, (c) weight
        ax.text(0.97, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='right')
    else:
        ax.text(0.03, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left')

# ── Dispersion panels (right column) ─────────────────────────────────────────
lim = 15
xlim_d = (-lim, lim)
ylim_d = (-lim, lim)
arrow_scale = 4.0
pt_kw = dict(s=1, alpha=0.30, color='0.55', rasterized=True)
contour_kw = dict(colors='0.35', linewidths=0.6, alpha=0.7)
bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.85)

# Density grids
grid_n = 300
xg = np.linspace(xlim_d[0], xlim_d[1], grid_n)
yg = np.linspace(ylim_d[0], ylim_d[1], grid_n)
Xg, Yg = np.meshgrid(xg, yg)
pos = np.stack([Xg, Yg], axis=-1)
Z_blob = mvn.pdf(pos, mean=[0.0, 0.0], cov=cov_blob)
Z_gmm = 0.5 * mvn.pdf(pos, mean=means_np[0], cov=covs_np[0]) + \
        0.5 * mvn.pdf(pos, mean=means_np[1], cov=covs_np[1])
n_levels = 6
blob_levels = np.linspace(0.10 * Z_blob.max(), 0.90 * Z_blob.max(), n_levels)
gmm_levels = np.linspace(0.10 * Z_gmm.max(), 0.90 * Z_gmm.max(), n_levels)

# Top-right: blob
ax_blob = fig.add_subplot(gs[0, 2])
ax_blob.contour(Xg, Yg, Z_blob, levels=blob_levels, **contour_kw)
ax_blob.scatter(samples_blob[:, 0], samples_blob[:, 1], **pt_kw)
origin_blob = np.array([0.0, 0.0])
for sign in [1, -1]:
    ax_blob.annotate('', xy=origin_blob + sign * arrow_scale * blob_pc1,
                     xytext=origin_blob,
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_blob.annotate('', xy=origin_blob + sign * arrow_scale * blob_da1,
                     xytext=origin_blob,
                     arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
ax_blob.set_xlim(xlim_d); ax_blob.set_ylim(ylim_d)
ax_blob.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_blob.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_blob.set_aspect('equal')
ax_blob.text(0.97, 0.97, f'SSA = {ssa_blob:.3f}\n$H$ = {H_blob:.2f} nats',
             transform=ax_blob.transAxes, fontsize=9,
             va='top', ha='right', bbox=bbox_props)
ax_blob.text(0.03, 0.95, '(e)', transform=ax_blob.transAxes,
             fontsize=11, fontweight='bold', va='top', ha='left')

# Bottom-right: bimodal GMM
ax_gmm = fig.add_subplot(gs[1, 2])
ax_gmm.contour(Xg, Yg, Z_gmm, levels=gmm_levels, **contour_kw)
ax_gmm.scatter(samples_gmm[:, 0], samples_gmm[:, 1], **pt_kw)
for sign in [1, -1]:
    ax_gmm.annotate('', xy=origin + sign * arrow_scale * pc1, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_gmm.annotate('', xy=origin + sign * arrow_scale * da1, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2))
ax_gmm.set_xlim(xlim_d); ax_gmm.set_ylim(ylim_d)
ax_gmm.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_gmm.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_gmm.set_aspect('equal')
ax_gmm.text(0.97, 0.97, f'SSA = {ssa_gmm:.3f}\n$H$ = {H_gmm:.2f} nats',
            transform=ax_gmm.transAxes, fontsize=9,
            va='top', ha='right', bbox=bbox_props)
ax_gmm.text(0.03, 0.95, '(f)', transform=ax_gmm.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')

# Arrow legend in bottom-right of (f)
from matplotlib.lines import Line2D
arrow_legend = [
    Line2D([0], [0], color='tab:blue', lw=2, label='DA1'),
    Line2D([0], [0], color='red', lw=2, label='PC1'),
]
ax_gmm.legend(handles=arrow_legend, loc='lower right', fontsize=9,
              framealpha=0.85, edgecolor='0.7')

# Save
for ext in ['png', 'pdf']:
    path = os.path.join(FIG_DIR, f'fig_gmm_combined.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {path}")
plt.close(fig)
