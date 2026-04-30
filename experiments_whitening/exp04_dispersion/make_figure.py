#!/usr/bin/env python3
"""
Replicate fig_dispersion_vs_fragmentation.png with whitened SSA annotated.
Same 2-panel scatter+contour+DA1/PC1 layout, adding SSA_w line to boxes.
Pure numpy — no torch needed.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'figures'))
from style_config import setup_style, TEXT_WIDTH

setup_style()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
seed = 42

# Load whitened results
with open(os.path.join(OUT_DIR, 'results', 'dispersion_whitened.json')) as f:
    res = json.load(f)

ssa_gmm_orig = res['gmm_original']['ssa_original']
ssa_gmm_whit = res['gmm_whitened']['ssa_whitened']
ssa_blob_orig = res['blob_original']['ssa_original']
ssa_blob_whit = res['blob_whitened']['ssa_whitened']

# ─── GMM: parallel ellipses ──────────────────────────────────────────────
means_np = np.array([[0.0, -3.0], [0.0, 3.0]])
covs_np = np.array([[[25.0, 0.0], [0.0, 1.0]],
                     [[25.0, 0.0], [0.0, 1.0]]])
w_gmm = np.array([0.5, 0.5])

mu_bar = (w_gmm[:, None] * means_np).sum(0)
cov_f = np.zeros((2, 2))
for i in range(2):
    diff = means_np[i] - mu_bar
    cov_f += w_gmm[i] * (covs_np[i] + np.outer(diff, diff))

# PCA (analytic)
pca_evals, pca_evecs = np.linalg.eigh(cov_f)
pc1 = pca_evecs[:, -1]  # largest eigenvalue

# DA1 is along y-axis for this symmetric 2-mode GMM
da1 = np.array([0.0, 1.0])

# Sample GMM
rng = np.random.default_rng(seed)
labels = rng.binomial(1, 0.5, 2000)
samples_gmm = np.where(labels[:, None] == 0,
                        rng.multivariate_normal(means_np[0], covs_np[0], 2000),
                        rng.multivariate_normal(means_np[1], covs_np[1], 2000))

# ─── Blob ─────────────────────────────────────────────────────────────────
cov_blob = np.array([[25.0, 0.0], [0.0, 25.0]])
samples_blob = rng.multivariate_normal([0.0, 0.0], cov_blob, size=2000)
blob_pc1 = np.array([0.0, 1.0])  # arbitrary for isotropic
# DA1 rotated ~40° from arbitrary direction (matching original)
rot_angle = np.radians(40)
blob_da1 = np.array([np.cos(rot_angle + np.pi/4), np.sin(rot_angle + np.pi/4)])
blob_da1 /= np.linalg.norm(blob_da1)

# ─── Entropy ──────────────────────────────────────────────────────────────
d = 2
H_blob = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(cov_blob)))
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

# ─── FIGURE ───────────────────────────────────────────────────────────────
lim = 16
xlim = (-lim, lim)
ylim = (-lim, lim)

panel_size = TEXT_WIDTH / 2 - 0.3
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(TEXT_WIDTH, panel_size),
    subplot_kw=dict(aspect='equal', adjustable='box'))

arrow_scale = 4.0
pt_kw = dict(s=1, alpha=0.30, color='0.55', rasterized=True)
contour_kw = dict(colors='0.35', linewidths=0.6, alpha=0.7)

# Density grids
grid_n = 300
xg = np.linspace(xlim[0], xlim[1], grid_n)
yg = np.linspace(ylim[0], ylim[1], grid_n)
Xg, Yg = np.meshgrid(xg, yg)
pos = np.stack([Xg, Yg], axis=-1)

Z_blob = mvn.pdf(pos, mean=[0.0, 0.0], cov=cov_blob)
Z_gmm = 0.5 * mvn.pdf(pos, mean=means_np[0], cov=covs_np[0]) + \
        0.5 * mvn.pdf(pos, mean=means_np[1], cov=covs_np[1])

n_levels = 6
blob_levels = np.linspace(0.10 * Z_blob.max(), 0.90 * Z_blob.max(), n_levels)
gmm_levels = np.linspace(0.10 * Z_gmm.max(), 0.90 * Z_gmm.max(), n_levels)

# ── Left: blob ──
ax_left.contour(Xg, Yg, Z_blob, levels=blob_levels, **contour_kw)
ax_left.scatter(samples_blob[:, 0], samples_blob[:, 1], **pt_kw)

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

ax_left.set_xlim(xlim); ax_left.set_ylim(ylim)
ax_left.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_left.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_left.set_xlabel(r'$x_1$'); ax_left.set_ylabel(r'$x_2$')

bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.85)
ax_left.text(0.97, 0.97,
             f'SSA = {ssa_blob_orig:.3f}\n'
             f'SSA$_w$ = {ssa_blob_whit:.3f}\n'
             f'$H$ = {H_blob:.2f} nats',
             transform=ax_left.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right',
             bbox=bbox_props)

# ── Right: parallel ellipses ──
origin = mu_bar
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

ax_right.set_xlim(xlim); ax_right.set_ylim(ylim)
ax_right.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_right.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax_right.set_xlabel(r'$x_1$'); ax_right.set_ylabel(r'$x_2$')

ax_right.text(0.97, 0.97,
              f'SSA = {ssa_gmm_orig:.3f}\n'
              f'SSA$_w$ = {ssa_gmm_whit:.3f}\n'
              f'$H$ = {H_gmm:.2f} nats',
              transform=ax_right.transAxes, fontsize=7,
              verticalalignment='top', horizontalalignment='right',
              bbox=bbox_props)

fig.tight_layout(pad=0.5)
path = os.path.join(OUT_DIR, 'fig_dispersion_whitened.png')
fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {path}")
