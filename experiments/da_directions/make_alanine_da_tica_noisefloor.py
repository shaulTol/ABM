#!/usr/bin/env python3
"""
Generate alanine DA vs TICA Ramachandran figure using free-conv noise-floor τ*.

Uses KDE-DA eigenvectors at τ*(m=2) = 362.6ps (free-conv λ₂ crossing).
1x2 layout: DA1, TICA1 on Ramachandran (ψ, φ).

Saves to figures/fig_alanine_da_tica.png.
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

# ── Style (matching make_paper_figures.py) ──
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
    # DA from KDE convergence grid
    kde = np.load(os.path.join(EXPDIR, 'results', 'alanine_da_results_kde_convergence.npz'),
                  allow_pickle=True)
    # TICA from original GMM file (TICA is score-independent, no need to recompute)
    gmm = np.load(os.path.join(EXPDIR, 'results', 'alanine_da_results_gmm_only.npz'),
                  allow_pickle=True)

    phi = np.degrees(kde['gt_phi'])
    psi = np.degrees(kde['gt_psi'])

    phys_t = kde['conv_phys_times']

    # Free-conv noise-floor τ*(m=2) = 362.6ps → idx=23 (last before λ₂ crosses floor)
    tau_star_idx = 23
    tau_star_ps = phys_t[tau_star_idx]
    print(f"Free-conv noise-floor τ*(m=2): lag={kde['conv_checkpoints'][tau_star_idx]}, t={tau_star_ps:.1f}ps")

    # DA eigenvectors at noise-floor τ* (from KDE)
    da_evecs = kde['conv_top_k_eigenvectors'][tau_star_idx]  # (45, 5)
    da1_vec = da_evecs[:, 0]

    # TICA1 projection (lag=10, from original GMM computation)
    tica1_proj = gmm['tica_proj'][:, 0]

    # Load distances for projection
    all_dists = None
    thesis_data = os.path.expanduser(
        '~/github/thesis/modality_score/autocorrelation-mixture-analysis/'
        'experiments/da_directions/alanine_dipeptide/data')
    local_data = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')

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

    da1_proj = Xc @ da1_vec

    # DA2 projection
    da2_vec = da_evecs[:, 1]
    da2_proj = Xc @ da2_vec

    # TICA2 (from GMM file)
    tica2_proj = gmm['tica_proj'][:, 1]

    # Correlations
    r_da1_psi = pearsonr(da1_proj, psi)[0]
    r_tica1_psi = pearsonr(tica1_proj, psi)[0]

    print(f"\nCorrelations at noise-floor τ*={tau_star_ps:.1f}ps:")
    print(f"  DA1:   |r(ψ)|={abs(r_da1_psi):.4f}  |r(φ)|={abs(pearsonr(da1_proj, phi)[0]):.4f}")
    print(f"  DA2:   |r(ψ)|={abs(pearsonr(da2_proj, psi)[0]):.4f}  |r(φ)|={abs(pearsonr(da2_proj, phi)[0]):.4f}")
    print(f"  TICA1: |r(ψ)|={abs(pearsonr(tica1_proj, psi)[0]):.4f}  |r(φ)|={abs(pearsonr(tica1_proj, phi)[0]):.4f}")
    print(f"  TICA2: |r(ψ)|={abs(pearsonr(tica2_proj, psi)[0]):.4f}  |r(φ)|={abs(pearsonr(tica2_proj, phi)[0]):.4f}")

    # No sign flip — keep natural DA/TICA directions

    # PC1 from pre-computed file (sign flipped for consistent visual direction)
    pc1_proj = -gmm['pca_proj'][:, 0]

    # Subsample for scatter plots (750k is too many points)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(phi), size=50000, replace=False)

    # Shift periodic boundary: cut at ψ = -110° (low-density region)
    psi_cut = -110.0
    psi_plot = ((psi - psi_cut) % 360) + psi_cut

    # ── Ramachandran figure (three panels: DA1, TICA1, PC1) ──
    from matplotlib.ticker import FixedLocator
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (da1_proj, 'DA1 proj.'),
        (tica1_proj, 'TICA1 proj.'),
        (pc1_proj, 'PC1 proj.'),
    ]

    for ax, (proj, cb_label) in zip(axes, panels):
        plot_vals = proj[idx]
        vmin, vmax = np.percentile(plot_vals, [2, 98])
        sc = ax.scatter(
            psi_plot[idx], phi[idx], c=plot_vals, cmap='viridis',
            s=0.5, alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(cb_label)
        ax.set_xlabel(r'$\psi$ (deg)')
        ax.set_ylabel(r'$\phi$ (deg)')
        ax.set_xlim(psi_cut, psi_cut + 360)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(FixedLocator([-90, 0, 90, 180]))
        ax.yaxis.set_major_locator(MultipleLocator(90))

    fig.tight_layout()
    outpath = os.path.join(FIGDIR, 'fig_alanine_da_tica.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved {outpath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
