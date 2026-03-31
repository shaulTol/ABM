#!/usr/bin/env python
"""Generate paper-ready composite figures from SSA vs MI sweep data."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data directory (Run 6 final results)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'results', 'run6_2026-03-03_final', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'run6_2026-03-03_final', 'images', 'paper_figures')

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})

# Display names for each config
TITLES = {
    'sep_2d': 'Varying Separation (2D)',
    'var_2d': 'Varying Variance (2D)',
    'weight_2d': 'Varying Weight (2D)',
    'nmodes_fixdist_2d': 'Varying #Modes (2D, d=4)',
    'nmodes_2d': 'Varying #Modes (2D, R=3)',
    'sep_10d_all': 'Sep. All Dims',
    'sep_10d_half': 'Sep. Dims 0-4',
    'sep_10d_dim0': 'Sep. Dim 0 Only',
    'var_10d_all': 'Var. All Dims',
    'var_10d_half': 'Var. Dims 0-4',
    'var_10d_dim0': 'Var. Dim 0 Only',
    'weight_10d': 'Varying Weight (10D)',
    'nmodes_fixdist_10d': 'Varying #Modes (10D, d=4)',
}

XLABELS = {
    'sep_2d': 'Separation',
    'var_2d': 'Variance',
    'weight_2d': 'Weight of Mode 1',
    'nmodes_fixdist_2d': 'Number of Modes',
    'nmodes_2d': 'Number of Modes',
    'sep_10d_all': 'Separation (per dim)',
    'sep_10d_half': 'Separation (dims 0-4)',
    'sep_10d_dim0': 'Separation (dim 0)',
    'var_10d_all': 'Variance',
    'var_10d_half': 'Variance (dims 0-4)',
    'var_10d_dim0': 'Variance (dim 0)',
    'weight_10d': 'Weight of Mode 1',
    'nmodes_fixdist_10d': 'Number of Modes',
}


def load_data(config_name):
    """Load JSON data for a sweep config."""
    path = os.path.join(DATA_DIR, f'{config_name}_data.json')
    with open(path) as f:
        return json.load(f)


def _dual_axis_panel(ax, param_values, ssa_values, mi_values, xlabel, title):
    """Draw a dual-axis SSA/MI panel on the given axes."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel('SSA', color='tab:blue')
    ax.plot(param_values, ssa_values, 'o-', color='tab:blue', markersize=4, label='SSA')
    ax.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax.twinx()
    ax2.set_ylabel('MI (nats)', color='tab:red')
    ax2.plot(param_values, mi_values, 's--', color='tab:red', markersize=4, label='MI')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax.set_title(title)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)


def make_composite(config_names, nrows, ncols, figsize, out_name, suptitle=None):
    """Generic composite figure builder."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    for i, name in enumerate(config_names):
        data = load_data(name)
        _dual_axis_panel(
            axes_flat[i],
            data['param_values'],
            data['ssa_values'],
            data['mi_values'],
            XLABELS.get(name, ''),
            TITLES.get(name, name),
        )

    # Hide unused axes
    for i in range(len(config_names), len(axes_flat)):
        axes_flat[i].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_random_scatter():
    """Figure 5: rank-correlation scatter for random GMMs."""
    random_path = os.path.join(os.path.dirname(__file__), 'results', 'random_sweep_data.json')
    if not os.path.exists(random_path):
        print(f"Skipping random sweep figure — {random_path} not found (run run_random_sweep.py first)")
        return

    with open(random_path) as f:
        random_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    from scipy.stats import rankdata

    for ax, dim_key, label in [(axes[0], '2d', '2D'), (axes[1], '10d', '10D')]:
        sweep = random_data[dim_key]
        records = sweep['records']
        ssa_ranks = rankdata([r['ssa'] for r in records])
        mi_ranks = rankdata([r['mi'] for r in records])
        n = len(records)

        ax.scatter(ssa_ranks, mi_ranks, s=30, color='tab:blue',
                   edgecolors='k', linewidths=0.5, alpha=0.7)
        ax.plot([1, n], [1, n], '--', color='gray', linewidth=1, label='Perfect correlation')

        rho = sweep['spearman_rho']
        pval = sweep['spearman_pval']
        ax.set_xlabel('SSA Rank')
        ax.set_ylabel('MI Rank')
        ax.set_title(f'Random GMMs ({label})')
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)
        ax.set_aspect('equal')
        ax.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}\np = {pval:.2e}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_random_sweep.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Figure 1: 2D basics (2×2)
    make_composite(
        ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d'],
        nrows=2, ncols=2, figsize=(10, 8),
        out_name='fig_2d_basics.png',
    )

    # Figure 2: 10D separation (1×3)
    make_composite(
        ['sep_10d_all', 'sep_10d_half', 'sep_10d_dim0'],
        nrows=1, ncols=3, figsize=(12, 4),
        out_name='fig_10d_separation.png',
    )

    # Figure 3: 10D variance (1×3)
    make_composite(
        ['var_10d_all', 'var_10d_half', 'var_10d_dim0'],
        nrows=1, ncols=3, figsize=(12, 4),
        out_name='fig_10d_variance.png',
    )

    # Figure 4: 10D weight and modes (1×2)
    make_composite(
        ['weight_10d', 'nmodes_fixdist_10d'],
        nrows=1, ncols=2, figsize=(8, 4),
        out_name='fig_10d_weight_and_modes.png',
    )

    # Figure 5: Random sweep scatter
    plot_random_scatter()

    print("\nDone! All paper figures saved.")


if __name__ == '__main__':
    main()
