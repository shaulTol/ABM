#!/usr/bin/env python
"""
Regenerate all synthetic dataset figures with real computed data.

Output files (must match LaTeX \includegraphics paths):
  figures/fig_random_sweep.png
  figures/fig_gmm_sweeps_curves.png
  figures/fig_dispersion_vs_fragmentation.png
  figures/fig_da_parallel_ellipses_2panel.png
  figures/fig_gmm_sweeps_snapshots.png
"""
import os
import sys
import json
import numpy as np
import torch

# Add src to path for SSA computation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import setup_style, SSA_COLOR, MI_COLOR, SCATTER_COLOR, DIAGONAL_COLOR, TEXT_WIDTH

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FIG_DIR, 'data')


def load_sweep_data(config_name):
    """Load sweep data from figures/data/ (n5000 preferred) or experiments/gmm_sweep/results/."""
    for suffix in ['_n5000_data.json', '_n1000_data.json']:
        path = os.path.join(DATA_DIR, f'{config_name}{suffix}')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # Fallback to experiments dir
    path = os.path.join(REPO_ROOT, 'experiments', 'gmm_sweep', 'results', f'{config_name}_data.json')
    with open(path) as f:
        return json.load(f)


def load_random_sweep():
    """Load random GMM sweep data."""
    path = os.path.join(REPO_ROOT, 'experiments', 'gmm_sweep', 'results', 'random_sweep_data.json')
    with open(path) as f:
        return json.load(f)


# =============================================================================
# FIGURE 1: Rank correlation scatter
# =============================================================================
def make_fig_random_sweep():
    from scipy import stats as sp_stats
    setup_style()

    data = load_random_sweep()
    dim_data = data['2d']
    records = dim_data['records']
    ssa_values = np.array([r['ssa'] for r in records])
    mi_values = np.array([r['mi'] for r in records])

    ssa_ranks = sp_stats.rankdata(ssa_values)
    mi_ranks = sp_stats.rankdata(mi_values)
    rho, _ = sp_stats.spearmanr(ssa_values, mi_values)

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.8))
    ax.scatter(ssa_ranks, mi_ranks, c=SCATTER_COLOR, s=22, alpha=0.7,
               edgecolors='white', linewidths=0.3, zorder=3)
    ax.plot([0, 52], [0, 52], color=DIAGONAL_COLOR, linestyle='--',
            linewidth=0.8, zorder=1)
    ax.text(0.05, 0.95, f'Spearman $\\rho = {rho:.2f}$',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))
    ax.set_xlabel('SSA Rank')
    ax.set_ylabel('MI Rank')
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 52)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_random_sweep.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# FIGURE 2: 2x2 GMM parameter sweeps (dual-axis SSA/MI)
# =============================================================================
def make_fig_gmm_sweeps_curves():
    setup_style()

    sweep_configs = [
        ('sep_2d', 'Separation ($\\sigma$ units)', 'Separation'),
        ('var_2d', 'Variance', 'Variance'),
        ('weight_2d', 'Weight $\\pi_1$', 'Weight asymmetry'),
        ('nmodes_fixdist_2d', 'Number of modes', 'Mode count'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.4))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for idx, (ax, (config, xlabel, title)) in enumerate(zip(axes_flat, sweep_configs)):
        data = load_sweep_data(config)
        pv = np.array(data['param_values'])
        ssa = np.array(data['ssa_values'])
        mi = np.array(data['mi_values'])

        is_discrete = (config == 'nmodes_fixdist_2d')
        every = max(1, len(pv) // 6)

        if is_discrete:
            ax.plot(pv, ssa, color=SSA_COLOR, marker='o', markersize=4.5,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=3)
        else:
            ax.plot(pv, ssa, color=SSA_COLOR, linewidth=1.3, zorder=3)
            ax.plot(pv[::every], ssa[::every], color=SSA_COLOR, marker='o',
                    markersize=4, markeredgecolor='white', markeredgewidth=0.3,
                    linestyle='none', zorder=4)

        ax.set_ylabel('SSA', color=SSA_COLOR, fontsize=8)
        ax.tick_params(axis='y', labelcolor=SSA_COLOR, labelsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9)

        # Add some padding to y limits
        ssa_pad = (ssa.max() - ssa.min()) * 0.1
        ax.set_ylim(ssa.min() - ssa_pad, ssa.max() + ssa_pad)

        ax2 = ax.twinx()
        if is_discrete:
            ax2.plot(pv, mi, color=MI_COLOR, marker='s', markersize=4.5,
                     markeredgecolor='white', markeredgewidth=0.4, zorder=3)
        else:
            ax2.plot(pv, mi, color=MI_COLOR, linewidth=1.3, zorder=3)
            ax2.plot(pv[::every], mi[::every], color=MI_COLOR, marker='s',
                     markersize=3.5, markeredgecolor='white', markeredgewidth=0.3,
                     linestyle='none', zorder=4)

        ax2.set_ylabel('MI (nats)', color=MI_COLOR, fontsize=8)
        ax2.tick_params(axis='y', labelcolor=MI_COLOR, labelsize=7)
        mi_pad = (mi.max() - mi.min()) * 0.1
        ax2.set_ylim(mi.min() - mi_pad, mi.max() + mi_pad)

        ax.spines['left'].set_color(SSA_COLOR)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['right'].set_color(MI_COLOR)
        ax.spines['right'].set_linewidth(0.8)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_color(SSA_COLOR)
        ax2.spines['left'].set_linewidth(0.8)
        ax2.spines['right'].set_color(MI_COLOR)
        ax2.spines['right'].set_linewidth(0.8)

    lines = [
        plt.Line2D([0], [0], color=SSA_COLOR, marker='o', markersize=4,
                   markeredgecolor='white', markeredgewidth=0.3, label='SSA'),
        plt.Line2D([0], [0], color=MI_COLOR, marker='s', markersize=3.5,
                   markeredgecolor='white', markeredgewidth=0.3, label='MI'),
    ]
    axes[0, 0].legend(handles=lines, loc='center right', fontsize=7,
                      framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout(w_pad=1.8, h_pad=1.2)
    path = os.path.join(FIG_DIR, 'fig_gmm_sweeps_curves.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# FIGURE 3: Dispersion vs fragmentation (1D, with computed SSA + entropy)
# =============================================================================
def compute_1d_ssa(means, variances, weights, dt=0.01, T=1000, N=10000, max_lag=500):
    """Compute SSA for a 1D GMM using the exact same code path as the 10D experiments.

    Uses simulate_d_trajectories + compute_vector_ssa_score with trace normalization,
    matching compute.py's compute_ssa_and_mi defaults but with d=1.
    """
    from src.simulation.langevin import simulate_d_trajectories
    from src.metrics.autocorrelation import compute_vector_ssa_score

    # Reshape to d=1 format: means (k,1), covs (k,1,1), weights (k,)
    k = len(means)
    means_t = torch.tensor(means, dtype=torch.float64).reshape(k, 1)
    covs_t = torch.tensor(variances, dtype=torch.float64).reshape(k, 1, 1)
    weights_t = torch.tensor(weights, dtype=torch.float64)

    torch.manual_seed(42)
    trajectories = simulate_d_trajectories(
        means_t, covs_t, weights_t,
        n_trajectories=N, T=T, dt=dt,
        device=torch.device('cpu'))

    S, _ = compute_vector_ssa_score(
        trajectories, max_lag,
        gmm_means=means_t, gmm_covs=covs_t, gmm_weights=weights_t,
        normalization='trace')
    return S


def compute_1d_entropy(means, variances, weights):
    """Compute differential entropy of a 1D GMM via numerical integration."""
    from scipy import integrate
    from scipy.stats import norm

    def gmm_pdf(x):
        p = 0.0
        for m, v, w in zip(means, variances, weights):
            p += w * norm.pdf(x, loc=m, scale=np.sqrt(v))
        return p

    def neg_p_log_p(x):
        p = gmm_pdf(x)
        if p < 1e-30:
            return 0.0
        return -p * np.log(p)

    # Integrate over a wide range
    max_std = np.sqrt(max(variances))
    lo = min(means) - 8 * max_std
    hi = max(means) + 8 * max_std
    H, _ = integrate.quad(neg_p_log_p, lo, hi, limit=200)
    return H


def make_fig_dispersion_vs_fragmentation():
    setup_style()
    np.random.seed(123)

    # Three 1D distributions:
    # (a) Isotropic Gaussian, sigma^2=1
    # (b) Bimodal GMM, means +/-4, unit variance
    # (c) Matched-variance Gaussian, sigma^2=17

    configs = [
        ('', [-4.0, 4.0], [1.0, 1.0], [0.5, 0.5], MI_COLOR),
        ('', [0.0], [17.0], [1.0], '#4daf4a'),
    ]

    # Compute SSA and entropy for each
    print("Computing 1D SSA and entropy for dispersion figure...")
    results = []
    for name, means, variances, weights, color in configs:
        print(f"  {name.replace(chr(10), ' ')}...", end=' ', flush=True)
        ssa = compute_1d_ssa(means, variances, weights, dt=0.01, T=1000, N=10000, max_lag=500)
        H = compute_1d_entropy(means, variances, weights)
        results.append((ssa, H))
        print(f"SSA={ssa:.2f}, H={H:.2f} nats")

    # Sample for histograms
    n = 5000
    labels = np.random.binomial(1, 0.5, n)
    gmm = np.where(labels == 0, np.random.normal(-4, 1, n), np.random.normal(4, 1, n))
    matched = np.random.normal(0, np.sqrt(17), n)

    samples_list = [gmm, matched]

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH * 0.5, 2.4), sharex=True)
    bins = np.linspace(-14, 14, 60)
    hist_kw = dict(bins=bins, density=True, alpha=0.75, edgecolor='white', linewidth=0.3)

    for ax, (name, means, variances, weights, color), (ssa, H), data, label in zip(
            axes, configs, results, samples_list, ['(a)', '(b)']):
        ax.hist(data, color=color, **hist_kw)
        ax.set_xlim(-14, 14)
        ax.tick_params(labelsize=6)
        ax.text(0.97, 0.95, f'SSA$\\,={ssa:.2f}$\n$H={H:.2f}$ nats',
                transform=ax.transAxes, fontsize=6.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='top')

    axes[-1].set_xlabel('$x$', fontsize=7)
    fig.supylabel('Density', fontsize=8, x=0.02)

    plt.tight_layout(h_pad=0.5)
    path = os.path.join(FIG_DIR, 'fig_dispersion_vs_fragmentation.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# FIGURE 4: Parallel ellipses (scatter with arrows)
# =============================================================================
def make_fig_da_parallel_ellipses():
    setup_style()
    np.random.seed(42)

    n = 3000
    labels = np.random.binomial(1, 0.5, n)
    x1 = np.where(labels == 0, np.random.normal(0, 5, n), np.random.normal(0, 5, n))
    x2 = np.where(labels == 0, np.random.normal(-3, 1, n), np.random.normal(3, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    ax.scatter(x1, x2, s=0.8, alpha=0.12, c='#777777', rasterized=True)

    # DA1 arrow (vertical = inter-mode direction)
    ax.annotate('', xy=(0, 6.2), xytext=(0, -6.2),
                arrowprops=dict(arrowstyle='->', color=SSA_COLOR, lw=2.5))
    ax.text(0.9, 5.5, 'DA1', color=SSA_COLOR, fontsize=10, fontweight='bold')

    # PC1 arrow (horizontal = max variance direction)
    ax.annotate('', xy=(14, 0), xytext=(-14, 0),
                arrowprops=dict(arrowstyle='->', color=MI_COLOR, lw=2.5))
    ax.text(12, -1.5, 'PC1', color=MI_COLOR, fontsize=10, fontweight='bold')

    ax.set_xlim(-18, 18)
    ax.set_ylim(-8, 8)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_da_parallel_ellipses_2panel.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# APPENDIX: GMM distribution snapshots grid
# =============================================================================
def make_fig_gmm_sweeps_snapshots():
    setup_style()
    from scipy.stats import multivariate_normal
    from matplotlib.colors import LinearSegmentedColormap

    def make_gmm_density(means, covs, weights, xrange, yrange, ngrid=600):
        x = np.linspace(*xrange, ngrid)
        y = np.linspace(*yrange, ngrid)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Z = np.zeros_like(X)
        for mu, cov, w in zip(means, covs, weights):
            Z += w * multivariate_normal.pdf(pos, mean=mu, cov=cov)
        return X, Y, Z

    sep_vals = [1.0, 2.5, 4.0, 5.5, 7.0]
    sep_configs = []
    for s in sep_vals:
        means = [[-s/2, 0], [s/2, 0]]
        covs = [np.eye(2), np.eye(2)]
        sep_configs.append((means, covs, [0.5, 0.5], f'$s={s:.1f}$'))

    var_vals = [0.3, 1.0, 2.0, 3.5, 5.0]
    var_configs = []
    for v in var_vals:
        means = [[-2, 0], [2, 0]]
        covs = [v * np.eye(2), v * np.eye(2)]
        var_configs.append((means, covs, [0.5, 0.5], f'$v={v:.1f}$'))

    wt_vals = [0.5, 0.62, 0.75, 0.85, 0.95]
    wt_configs = []
    for w in wt_vals:
        means = [[-2, 0], [2, 0]]
        covs = [np.eye(2), np.eye(2)]
        wt_configs.append((means, covs, [w, 1-w], f'$\\pi_1={w:.2f}$'))

    mc_vals = [1, 2, 4, 6, 8]
    mc_configs = []
    for k in mc_vals:
        if k == 1:
            means = [[0, 0]]
            covs = [0.5 * np.eye(2)]
            weights = [1.0]
        else:
            angles = np.linspace(0, 2*np.pi, k, endpoint=False)
            r = 4.0
            means = [[r * np.cos(a), r * np.sin(a)] for a in angles]
            covs = [0.5 * np.eye(2)] * k
            weights = [1/k] * k
        mc_configs.append((means, covs, weights, f'$k={k}$'))

    all_rows = [
        ('Separation', sep_configs),
        ('Variance', var_configs),
        ('Weight', wt_configs),
        ('Mode count', mc_configs),
    ]

    cmap = LinearSegmentedColormap.from_list('clean_blue',
        ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])

    fig, axes = plt.subplots(4, 5, figsize=(10.0, 10.0))

    for row_idx, (row_title, configs) in enumerate(all_rows):
        for col_idx, (means, covs, weights, label) in enumerate(configs):
            ax = axes[row_idx, col_idx]

            all_means = np.array(means)
            max_std = np.sqrt(max(np.max(c) for c in covs))
            max_extent = max(np.abs(all_means).max() + 3.5 * max_std, 4)
            xr = (-max_extent, max_extent)
            yr = (-max_extent, max_extent)

            X, Y, Z = make_gmm_density(means, covs, weights, xr, yr, ngrid=600)

            Z_sqrt = np.sqrt(Z)
            Z_sqrt[Z_sqrt < Z_sqrt.max() * 0.01] = 0
            Z_norm = Z_sqrt / Z_sqrt.max()

            extent = [xr[0], xr[1], yr[0], yr[1]]
            ax.imshow(Z_norm, extent=extent, origin='lower', cmap=cmap,
                      aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
            ax.contour(X, Y, Z, levels=5, colors='#08519c', linewidths=0.5, alpha=0.4)

            ax.set_xlim(xr)
            ax.set_ylim(yr)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color('#cccccc')

            ax.set_xlabel(label, fontsize=12, labelpad=3)

        axes[row_idx, 0].set_ylabel(row_title, fontsize=13, fontweight='bold', labelpad=6)

    plt.tight_layout(w_pad=0.5, h_pad=1.0)
    path = os.path.join(FIG_DIR, 'fig_gmm_sweeps_snapshots.png')
    plt.savefig(path, dpi=900)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# FIGURE 6: Alanine DA vs TICA Ramachandran
# =============================================================================
def make_fig_alanine_da_tica():
    """DA1 vs TICA1 Ramachandran using free-conv noise-floor τ*=362.6ps."""
    setup_style()
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    from matplotlib.ticker import MultipleLocator
    from scipy.stats import pearsonr

    exp_dir = os.path.join(REPO_ROOT, 'experiments', 'da_directions')

    # DA from KDE convergence grid (free-conv τ*=362.6ps, idx=23)
    kde_path = os.path.join(exp_dir, 'results', 'alanine_da_results_kde_convergence.npz')
    kde = np.load(kde_path, allow_pickle=True)

    # TICA from original GMM file (TICA is score-independent)
    gmm_path = os.path.join(exp_dir, 'results', 'alanine_da_results_gmm_only.npz')
    gmm = np.load(gmm_path, allow_pickle=True)

    phi = np.degrees(kde['gt_phi'])
    psi = np.degrees(kde['gt_psi'])
    tica1_proj = gmm['tica_proj'][:, 0]

    # DA1 eigenvector at free-conv noise-floor τ*(m=2) = 362.6ps (idx=23)
    da1_vec = kde['conv_top_k_eigenvectors'][23][:, 0]

    # Load distances for projection
    all_dists = None
    for dd in [os.path.expanduser(
            '~/github/thesis/modality_score/autocorrelation-mixture-analysis/'
            'experiments/da_directions/alanine_dipeptide/data'),
            os.path.join(exp_dir, 'alanine_dipeptide', 'data')]:
        dist_f = os.path.join(dd, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
        if os.path.exists(dist_f):
            dist_data = np.load(dist_f)
            all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
            break
    if all_dists is None:
        raise FileNotFoundError("Alanine distance data not found")

    da1_proj = (all_dists - all_dists.mean(axis=0)) @ da1_vec

    # Subsample for scatter plots (750k is too many points)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(phi), size=50000, replace=False)

    # ── Ramachandran figure (two panels) ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Left: DA1
    ax = axes[0]
    da1_plot = da1_proj[idx]
    vmin, vmax = np.percentile(da1_plot, [2, 98])
    sc = ax.scatter(
        psi[idx], phi[idx], c=da1_plot, cmap='viridis',
        s=0.5, alpha=0.3, vmin=vmin, vmax=vmax, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('DA1 proj.')
    ax.set_xlabel(r'$\psi$ (deg)')
    ax.set_ylabel(r'$\phi$ (deg)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))

    # Right: TICA1
    ax = axes[1]
    vmin2, vmax2 = np.percentile(tica1_proj[idx], [2, 98])
    sc2 = ax.scatter(
        psi[idx], phi[idx], c=tica1_proj[idx], cmap='viridis',
        s=0.5, alpha=0.3, vmin=vmin2, vmax=vmax2, rasterized=True,
    )
    cb2 = fig.colorbar(sc2, ax=ax, shrink=0.8, pad=0.02)
    cb2.set_label('TICA1 proj.')
    ax.set_xlabel(r'$\psi$ (deg)')
    ax.set_ylabel(r'$\phi$ (deg)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.xaxis.set_major_locator(MultipleLocator(90))
    ax.yaxis.set_major_locator(MultipleLocator(90))

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_alanine_da_tica.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='Only regenerate specific figure (random, sweeps, dispersion, ellipses, snapshots)')
    args = parser.parse_args()

    figures = {
        'random': make_fig_random_sweep,
        'sweeps': make_fig_gmm_sweeps_curves,
        'dispersion': make_fig_dispersion_vs_fragmentation,
        'ellipses': make_fig_da_parallel_ellipses,
        'snapshots': make_fig_gmm_sweeps_snapshots,
        'alanine': make_fig_alanine_da_tica,
    }

    if args.only:
        if args.only in figures:
            figures[args.only]()
        else:
            print(f"Unknown figure: {args.only}. Choose from: {list(figures.keys())}")
    else:
        for name, func in figures.items():
            print(f"\n{'='*60}")
            print(f"Generating: {name}")
            print(f"{'='*60}")
            func()

    print("\nDone!")
