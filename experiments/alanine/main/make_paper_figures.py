"""
Generate paper figures for DA vs TICA comparison.

FIGURE 1: Alanine dipeptide (2x2)
FIGURE 2: Chignolin (1x3)

Requires:
  - results/alanine_da_results_gmm_only.npz (job 44317525)
  - results/chignolin_da_results_gmm_only.npz (PDB-updated dRMSD)
  - alanine_dipeptide/data/alanine-dipeptide-3x250ns-heavy-atom-distances.npz
"""
import numpy as np
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

DA_COLOR = '#1f77b4'    # blue
TICA_COLOR = '#ff7f0e'  # orange
PCA_COLOR = '#7f7f7f'   # gray
PCA2_COLOR = '#aaaaaa'  # lighter gray

NEURIPS_FULL = 7.0   # full page width


# ========================================================================
# FIGURE 1 — Alanine Dipeptide Ramachandran (two panels)
# ========================================================================
def make_fig_alanine():
    OUTDIR = os.path.expanduser('~/github/thesis/figures')
    os.makedirs(OUTDIR, exist_ok=True)

    ala = np.load('results/alanine_da_results_gmm_only.npz', allow_pickle=True)
    dt = float(ala['dt'])

    # Ground truth dihedrals (radians -> degrees)
    phi = np.degrees(ala['gt_phi'])
    psi = np.degrees(ala['gt_psi'])

    # DA1 projection at optimal lag
    da1_proj = ala['da_gmm_proj'][:, 0]

    # TICA1 projection
    tica1_proj = ala['tica_proj'][:, 0]

    # Gap profiles (for separate figure)
    cps = ala['conv_checkpoints']
    phys_t = cps * dt
    gap12 = ala['conv_gap_profile']
    gap23 = ala['conv_gap23_profile']
    opt1 = int(ala['conv_optimal_idx'])
    opt2 = int(ala['conv_optimal_idx_da2'])

    # Subsample for scatter plots (750k is too many points)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(phi), size=50000, replace=False)

    # ── Ramachandran figure (two panels) ──
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: DA1 (sign flipped for visual comparison with TICA1)
    ax = axes[0]
    da1_flipped = -da1_proj[idx]
    vmin, vmax = np.percentile(da1_flipped, [2, 98])
    sc = ax.scatter(
        phi[idx], psi[idx], c=da1_flipped, cmap='viridis',
        s=0.5, alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('DA1 proj.')
    ax.text(0.02, 0.02, 'sign flipped\nfor comparison',
            transform=ax.transAxes, fontsize=7, va='bottom', color='0.4')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))

    # Right: TICA1
    ax = axes[1]
    vmin2, vmax2 = np.percentile(tica1_proj[idx], [2, 98])
    sc2 = ax.scatter(
        phi[idx], psi[idx], c=tica1_proj[idx], cmap='viridis',
        s=0.5, alpha=0.3, vmin=vmin2, vmax=vmax2, rasterized=True,
    )
    cb2 = fig.colorbar(sc2, ax=ax, shrink=0.8, pad=0.02)
    cb2.set_label('TICA1 proj.')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))

    fig.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig_alanine_da_tica.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved {outpath}")
    plt.close(fig)

    # ── Eigenvalue gap figure (separate) ──
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(phys_t, gap12, 'o-', color=DA_COLOR, markersize=3, linewidth=1.2,
            label=r'$\lambda_1/\lambda_2$ (DA1)')
    ax.plot(phys_t, gap23, 's-', color=TICA_COLOR, markersize=3, linewidth=1.2,
            label=r'$\lambda_2/\lambda_3$ (DA2)')
    ax.axvline(phys_t[opt1], color=DA_COLOR, linestyle='--', alpha=0.6,
               linewidth=1)
    ax.axvline(phys_t[opt2], color=TICA_COLOR, linestyle='--', alpha=0.6,
               linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Physical time (Langevin units)')
    ax.set_ylabel('Eigenvalue gap')
    ax.legend(loc='upper left')

    fig.tight_layout()
    outpath2 = os.path.join(OUTDIR, 'fig_alanine_eigenvalue_gap.png')
    fig.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f"Saved {outpath2}")
    plt.close(fig)


# ========================================================================
# FIGURE 2 — Chignolin
# ========================================================================
def make_fig_chignolin():
    chi = np.load('results/chignolin_da_results_gmm_only.npz', allow_pickle=True)

    drmsd = chi['gt_rmsd']
    da1 = chi['da_gmm_proj'][:, 0]
    tica1 = chi['tica_proj'][:, 0]
    pca1 = chi['pca_proj'][:, 0]

    r_da1 = abs(np.corrcoef(da1, drmsd)[0, 1])
    r_tica1 = abs(np.corrcoef(tica1, drmsd)[0, 1])
    r_pca1 = abs(np.corrcoef(pca1, drmsd)[0, 1])
    print(f"Chignolin: DA1={r_da1:.4f}, TICA1={r_tica1:.4f}, PCA1={r_pca1:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(NEURIPS_FULL, 2.4))

    # Subsample for scatter (1.87M too many)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(drmsd), size=50000, replace=False)

    # (a) DA1 vs dRMSD
    ax = axes[0]
    ax.scatter(da1[idx], drmsd[idx], s=0.3, alpha=0.3, color=DA_COLOR,
               rasterized=True)
    ax.set_xlabel('DA1 projection')
    ax.set_ylabel(r'dRMSD to native ($\AA$)')

    # (b) TICA1 vs dRMSD
    ax = axes[1]
    ax.scatter(tica1[idx], drmsd[idx], s=0.3, alpha=0.3, color=TICA_COLOR,
               rasterized=True)
    ax.set_xlabel('TICA1 projection')
    ax.set_ylabel(r'dRMSD to native ($\AA$)')

    # (c) Bar chart
    ax = axes[2]
    methods = ['DA1', 'TICA1', 'PCA1']
    vals = [r_da1, r_tica1, r_pca1]
    colors = [DA_COLOR, TICA_COLOR, PCA_COLOR]
    ax.bar(methods, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel(r'$|r|$ with dRMSD')
    ax.set_ylim(0.7, 1.02)

    fig.tight_layout(w_pad=1.5)
    fig.savefig('figures/fig_chignolin_da_tica.png', dpi=300, bbox_inches='tight')
    print("Saved figures/fig_chignolin_da_tica.png")
    plt.close(fig)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('figures', exist_ok=True)
    make_fig_alanine()
    make_fig_chignolin()
    print("Done.")
