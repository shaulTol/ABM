#!/usr/bin/env python3
"""
Generate 2-panel figure showing DA/TICA/PC directions as arrows on Ramachandran density.

Panel 1: DA1, TICA1, PC1 arrows (slow modes — ψ-dominant)
Panel 2: DA2, TICA2, PC2 arrows (second modes — φ-dominant)

Arrows show correlation vectors (r(ψ), r(φ)) for each method, overlaid on the
Ramachandran density background.

Saves to figures/fig_alanine_direction_arrows.png.
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Style ──
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})

EXPDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(project_root, 'figures')


def main():
    # ── Load data ──
    kde = np.load(os.path.join(EXPDIR, 'results', 'alanine_da_results_kde_convergence.npz'),
                  allow_pickle=True)
    gmm = np.load(os.path.join(EXPDIR, 'results', 'alanine_da_results_gmm_only.npz'),
                  allow_pickle=True)

    phi = np.degrees(kde['gt_phi'])
    psi = np.degrees(kde['gt_psi'])

    # DA eigenvectors at τ*=362.6ps (idx=23)
    da_evecs = kde['conv_top_k_eigenvectors'][23]  # (45, 5)

    # Load distances for projection
    thesis_data = os.path.expanduser(
        '~/github/thesis/modality_score/autocorrelation-mixture-analysis/'
        'experiments/da_directions/alanine_dipeptide/data')
    local_data = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')

    all_dists = None
    for dd in [thesis_data, local_data]:
        dist_f = os.path.join(dd, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
        if os.path.exists(dist_f):
            dist_data = np.load(dist_f)
            all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
            break
    if all_dists is None:
        raise FileNotFoundError("Alanine distance data not found")

    data_mean = all_dists.mean(axis=0)
    Xc = all_dists - data_mean

    # ── Projections ──
    da1_proj = Xc @ da_evecs[:, 0]
    da2_proj = Xc @ da_evecs[:, 1]
    tica1_proj = gmm['tica_proj'][:, 0]
    tica2_proj = gmm['tica_proj'][:, 1]

    # PCA from pre-computed file (consistent with other figures)
    pc1_proj = gmm['pca_proj'][:, 0]
    pc2_proj = gmm['pca_proj'][:, 1]

    # ── Correlation vectors (r(ψ), r(φ)) from paper Table 1 ──
    # Signed correlations computed from projections; |r| values match table.
    # Eigenvector sign is arbitrary — flip all methods to point in consistent
    # direction: positive r for their dominant dihedral.
    computed = {
        'DA1': da1_proj, 'TICA1': tica1_proj, 'PCA1': pc1_proj,
        'DA2': da2_proj, 'TICA2': tica2_proj, 'PCA2': pc2_proj,
    }
    # Which dihedral is dominant for each method (for sign convention)
    dominant = {'DA1': 'psi', 'TICA1': 'psi', 'PCA1': 'psi',
                'DA2': 'phi', 'TICA2': 'phi', 'PCA2': 'phi'}
    corr_vecs = {}
    for name, proj in computed.items():
        r_psi = pearsonr(proj, psi)[0]
        r_phi = pearsonr(proj, phi)[0]
        # Flip sign so dominant correlation is positive
        dom = dominant[name]
        if (dom == 'psi' and r_psi < 0) or (dom == 'phi' and r_phi < 0):
            r_psi, r_phi = -r_psi, -r_phi
        corr_vecs[name] = (r_psi, r_phi)
        print(f"  {name}: r(ψ)={r_psi:+.4f}, r(φ)={r_phi:+.4f}, |r|={np.sqrt(r_psi**2+r_phi**2):.4f}")

    # ── 2D density background ──
    from scipy.ndimage import gaussian_filter
    bins = 180
    H, xedges, yedges = np.histogram2d(psi, phi, bins=bins,
                                         range=[[-180, 180], [-180, 180]])
    H = H.T  # transpose so y=phi, x=psi
    H_smooth = gaussian_filter(H, sigma=2.0)
    # Log-scale for better contrast
    H_log = np.log1p(H_smooth)

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    method_colors = {
        'DA1': '#d62728', 'DA2': '#d62728',
        'TICA1': '#1f77b4', 'TICA2': '#1f77b4',
        'PCA1': '#2ca02c', 'PCA2': '#2ca02c',
    }
    arrow_scale = 150  # scale correlation vector to degrees for arrow length

    # Label offsets (dx, dy in points) to avoid overlap — hand-tuned per panel
    label_offsets = {
        'DA1': (8, 8), 'TICA1': (-8, -14), 'PCA1': (8, -14),
        'DA2': (8, 8), 'TICA2': (-12, -10), 'PCA2': (12, -10),
    }

    panels = [
        ('First components', ['DA1', 'TICA1', 'PCA1']),
        ('Second components', ['DA2', 'TICA2', 'PCA2']),
    ]

    for panel_idx, (title, method_names) in enumerate(panels):
        ax = axes[panel_idx]

        # Density background — filled contours
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        levels = np.linspace(0, H_log.max(), 20)
        ax.contourf(x_centers, y_centers, H_log, levels=levels,
                     cmap='bone_r', alpha=0.85)
        ax.contour(x_centers, y_centers, H_log, levels=levels[::3],
                    colors='#888888', linewidths=0.3, alpha=0.5)

        # Arrows from geometric center
        cx, cy = 0.0, 0.0

        for name in method_names:
            r_psi, r_phi = corr_vecs[name]
            color = method_colors[name]

            mag = np.sqrt(r_psi**2 + r_phi**2)
            dx = r_psi * arrow_scale
            dy = r_phi * arrow_scale

            ax.annotate('', xy=(cx + dx, cy + dy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                        mutation_scale=15))

            # Label at arrow tip with per-method offset
            lox, loy = label_offsets[name]
            label = f'{name}\n|r($\\psi$)|={abs(r_psi):.2f}, |r($\\phi$)|={abs(r_phi):.2f}'
            ax.annotate(label, xy=(cx + dx, cy + dy),
                        fontsize=7.5, fontweight='bold', color=color,
                        ha='left' if lox >= 0 else 'right', va='bottom' if loy >= 0 else 'top',
                        textcoords='offset points', xycoords='data',
                        xytext=(lox, loy))

        ax.set_xlabel(r'$\psi$ (deg)')
        ax.set_ylabel(r'$\phi$ (deg)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.xaxis.set_major_locator(MultipleLocator(90))
        ax.yaxis.set_major_locator(MultipleLocator(90))
        ax.set_title(title)

        # Origin marker
        ax.plot(cx, cy, 'k+', markersize=10, markeredgewidth=2)

    fig.tight_layout()
    outpath = os.path.join(FIGDIR, 'fig_alanine_direction_arrows.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
