"""Render the consolidated 2-panel Ramachandran figure for the two baselines.

Matches style of figures/fig_alanine_da_tica.png:
- viridis colormap, 2-98 percentile vmin/vmax
- 50k-frame subsample, s=0.5, alpha=0.3
- xlabel phi (deg), ylabel psi (deg), MultipleLocator(90)
- (a)/(b) labels, no titles, footnotesize-ready
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_alanine

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})


def main():
    sk = np.load(os.path.join(RESDIR, 'sparse_kmeans.npz'))
    sm = np.load(os.path.join(RESDIR, 'spectral_map.npz'))
    _, phi, psi = load_alanine()
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)

    sk_score = sk['score_all']
    sm1 = sm['embed'][:, 0]

    rng = np.random.RandomState(42)
    idx = rng.choice(len(phi_deg), size=50000, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    v = sk_score[idx]
    vmin, vmax = np.percentile(v, [2, 98])
    sc = ax.scatter(phi_deg[idx], psi_deg[idx], c=v, cmap='viridis',
                    s=0.5, alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('sparse K-means proj.')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))
    ax.text(0.02, 0.96, '(a)', transform=ax.transAxes, fontsize=11,
            va='top', ha='left', fontweight='bold')

    ax = axes[1]
    v = sm1[idx]
    vmin, vmax = np.percentile(v, [2, 98])
    sc2 = ax.scatter(phi_deg[idx], psi_deg[idx], c=v, cmap='viridis',
                     s=0.5, alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True)
    cb2 = fig.colorbar(sc2, ax=ax, shrink=0.8, pad=0.02)
    cb2.set_label('Spectral Map SM1')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))
    ax.text(0.02, 0.96, '(b)', transform=ax.transAxes, fontsize=11,
            va='top', ha='left', fontweight='bold')

    fig.tight_layout()
    out = os.path.join(RESDIR,
                       'fig_alanine_baselines_sparse_kmeans_spectralmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
