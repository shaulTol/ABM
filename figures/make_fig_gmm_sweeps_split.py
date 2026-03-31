#!/usr/bin/env python
"""
Generate split GMM sweep figures:
  fig_gmm_sweeps_curves.png    — 4 line plots (main paper)
  fig_gmm_sweeps_snapshots.png — 4 filmstrip panels (appendix)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import multivariate_normal

# Paths
SWEEP_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'experiments', 'gmm_sweep')
DATA_DIR = os.path.join(SWEEP_DIR, 'results', 'run6_2026-03-03_final', 'data')
FIG_DIR = os.path.dirname(__file__)


def load_data(config_name):
    for suffix in ['_n5000_data.json', '_n1000_data.json']:
        local_path = os.path.join(FIG_DIR, 'data', f'{config_name}{suffix}')
        if os.path.exists(local_path):
            path = local_path
            break
    else:
        path = os.path.join(DATA_DIR, f'{config_name}_data.json')
    with open(path) as f:
        return json.load(f)


# ── Curves figure ────────────────────────────────────────────────────────

def draw_sweep_panel(ax, config_name, xlabel, title, show_left_y, show_right_y, show_legend):
    data = load_data(config_name)
    pv = data['param_values']
    ssa = data['ssa_values']
    mi = data['mi_values']

    ax.plot(pv, ssa, 'o-', color='tab:blue', markersize=3, linewidth=1.5, label='SSA')
    ax.tick_params(axis='x', labelsize=7, pad=1)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)

    if show_left_y:
        ax.set_ylabel('SSA', color='tab:blue', fontsize=8, labelpad=2)
        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=7, pad=1)
    else:
        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=7, pad=1)
        ax.set_ylabel('')

    ax2 = ax.twinx()
    ax2.plot(pv, mi, 's--', color='tab:red', markersize=3, linewidth=1.5, label='MI')

    if show_right_y:
        ax2.set_ylabel('MI (nats)', color='tab:red', fontsize=8, labelpad=2)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=7, pad=1)
    else:
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=7, pad=1)
        ax2.set_ylabel('')

    if show_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=7,
                  framealpha=0.8, handlelength=1.5)

    return ax, ax2


def make_curves_figure():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    sweep_configs = [
        ('sep_2d', 'Separation', 'Separation'),
        ('var_2d', 'Variance', 'Variance'),
        ('weight_2d', r'Weight $\pi_1$', 'Weight'),
        ('nmodes_fixdist_2d', 'Number of modes', 'Mode count'),
    ]

    for col, (config, xlabel, title) in enumerate(sweep_configs):
        show_left = (col == 0)
        show_right = (col == 3)
        show_legend = (col == 0)
        draw_sweep_panel(axes[col], config, xlabel, title,
                         show_left, show_right, show_legend)

    plt.tight_layout(pad=0.5)
    path = os.path.join(FIG_DIR, 'fig_gmm_sweeps_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


# ── Snapshots figure ─────────────────────────────────────────────────────

def gmm_density_grid(means, covs, weights, xlim, ylim, ngrid=200):
    x = np.linspace(xlim[0], xlim[1], ngrid)
    y = np.linspace(ylim[0], ylim[1], ngrid)
    X, Y = np.meshgrid(x, y)
    pos = np.stack([X, Y], axis=-1)
    Z = np.zeros_like(X)
    for mu, cov, w in zip(means, covs, weights):
        Z += w * multivariate_normal.pdf(pos, mean=mu, cov=cov)
    return X, Y, Z


def draw_filmstrip(fig, gs_cell, snapshots, cmap_values, param_values_str):
    n = len(snapshots)
    all_means = np.concatenate([s[0] for s in snapshots], axis=0)
    all_stds = np.array([np.sqrt(np.max(np.diag(c))) for s in snapshots for c in s[1]])
    margin = max(all_stds.max() * 3, 2.0)
    xmin, xmax = all_means[:, 0].min() - margin, all_means[:, 0].max() + margin
    ymin, ymax = all_means[:, 1].min() - margin, all_means[:, 1].max() + margin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin) / 2
    xlim = (cx - span, cx + span)
    ylim = (cy - span, cy + span)
    color = '#4a90d9'
    inner_gs = GridSpecFromSubplotSpec(1, n, subplot_spec=gs_cell, wspace=0.05)
    for i, ((means, covs, weights), cv) in enumerate(zip(snapshots, cmap_values)):
        ax = fig.add_subplot(inner_gs[i])
        X, Y, Z = gmm_density_grid(means, covs, weights, xlim, ylim, ngrid=120)
        peak = Z.max()
        if peak > 0:
            levels = [0.10 * peak, 0.30 * peak, 0.60 * peak]
            ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=1.0, alpha=0.8)
            ax.contourf(X, Y, Z, levels=[levels[0], peak * 1.1], colors=[color], alpha=0.2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.text(0.5, -0.08, param_values_str[i], transform=ax.transAxes,
                ha='center', va='top', fontsize=6, color='0.4')
        if i > 0:
            ax.axvline(xlim[0], color='0.7', linewidth=0.5, clip_on=False)


def build_separation_snapshots():
    sep_values = [1, 2.5, 4, 5.5, 7]
    snapshots = []
    for sep in sep_values:
        means = np.array([[0, 0], [sep, 0]], dtype=float)
        covs = np.array([np.eye(2), np.eye(2)])
        weights = np.array([0.5, 0.5])
        snapshots.append((means, covs, weights))
    return snapshots, np.linspace(0.3, 1.0, len(sep_values)), [f's={v}' for v in sep_values]


def build_variance_snapshots():
    var_values = [0.3, 1, 2, 3.5, 5]
    snapshots = []
    for var in var_values:
        means = np.array([[0, 0], [4, 0]], dtype=float)
        covs = np.array([var * np.eye(2), var * np.eye(2)])
        weights = np.array([0.5, 0.5])
        snapshots.append((means, covs, weights))
    return snapshots, np.linspace(0.3, 1.0, len(var_values)), [f'v={v}' for v in var_values]


def build_weight_snapshots():
    w_values = [0.5, 0.62, 0.75, 0.85, 0.95]
    snapshots = []
    for w in w_values:
        means = np.array([[0, 0], [4, 0]], dtype=float)
        covs = np.array([np.eye(2), np.eye(2)])
        weights = np.array([w, 1 - w])
        snapshots.append((means, covs, weights))
    return snapshots, np.linspace(0.3, 1.0, len(w_values)), [f'π={v}' for v in w_values]


def build_modecount_snapshots():
    k_values = [1, 2, 4, 6, 8]
    distance = 4.0
    variance = 0.5
    snapshots = []
    for k in k_values:
        if k == 1:
            means = np.array([[0, 0]], dtype=float)
            covs = np.array([variance * np.eye(2)])
            weights = np.array([1.0])
        else:
            radius = distance / (2 * np.sin(np.pi / k))
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
            means = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            covs = np.array([variance * np.eye(2)] * k)
            weights = np.ones(k) / k
        snapshots.append((means, covs, weights))
    return snapshots, np.linspace(0.3, 1.0, len(k_values)), [f'k={v}' for v in k_values]


def draw_strip_manual(fig, snapshots, cmap_values, param_labels, title, row, col):
    """Draw a titled strip of 5 contour plots at a specific 2x2 grid position."""
    n = len(snapshots)

    # Compute per-snapshot limits so each panel zooms to its own content
    per_snap_lims = []
    for (means, covs, weights) in snapshots:
        stds = np.array([np.sqrt(np.max(np.diag(c))) for c in covs])
        margin = max(stds.max() * 2.5, 1.5)
        xmin_s = means[:, 0].min() - margin
        xmax_s = means[:, 0].max() + margin
        ymin_s = means[:, 1].min() - margin
        ymax_s = means[:, 1].max() + margin
        cx, cy = (xmin_s + xmax_s) / 2, (ymin_s + ymax_s) / 2
        sp = max(xmax_s - xmin_s, ymax_s - ymin_s) / 2
        per_snap_lims.append(((cx - sp, cx + sp), (cy - sp, cy + sp)))

    # Light-to-dark blue colormap for filled contours
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list(
        'custom_blues', ['#e8f0fb', '#a8c8e8', '#5a9bd5', '#2a6db5', '#154577'])

    # Manual positioning
    x_offset = 0.03 + col * 0.50
    quad_w = 0.46
    panel_w = quad_w / n * 0.95
    panel_h = panel_w  # square panels
    gap = quad_w / n * 0.05
    row_height = panel_h + 0.10  # panel + title + label space
    y_offset = 1.0 - (row + 1) * row_height - row * 0.03

    # Title
    fig.text(x_offset + quad_w / 2, y_offset + panel_h + 0.03,
             title, fontsize=10, fontweight='bold', ha='center', va='bottom')

    for i, ((means, covs, weights), cv) in enumerate(zip(snapshots, cmap_values)):
        xlim, ylim = per_snap_lims[i]
        left = x_offset + i * (panel_w + gap)
        ax = fig.add_axes([left, y_offset, panel_w, panel_h])
        X, Y, Z = gmm_density_grid(means, covs, weights, xlim, ylim, ngrid=200)
        peak = Z.max()
        if peak > 0:
            # 10 filled contour levels for smooth gradient
            n_levels = 10
            levels_fill = np.linspace(0.02 * peak, peak * 1.01, n_levels + 1)
            ax.contourf(X, Y, Z, levels=levels_fill, cmap=blues, alpha=0.85)
            # 6 contour lines for definition
            levels_line = np.linspace(0.05 * peak, 0.90 * peak, 6)
            ax.contour(X, Y, Z, levels=levels_line, colors=['#1a5276'],
                       linewidths=1.3, alpha=0.7)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.text(0.5, -0.10, param_labels[i], transform=ax.transAxes,
                ha='center', va='top', fontsize=7, color='0.4')
        if i > 0:
            ax.axvline(xlim[0], color='0.7', linewidth=0.5, clip_on=False)


def make_snapshots_figure():
    fig = plt.figure(figsize=(10, 7))

    builders = [
        build_separation_snapshots,
        build_variance_snapshots,
        build_weight_snapshots,
        build_modecount_snapshots,
    ]
    titles = ['Separation', 'Variance', 'Weight', 'Mode count']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (r, c), (builder, title) in zip(positions, zip(builders, titles)):
        snapshots, cmap_values, labels = builder()
        draw_strip_manual(fig, snapshots, cmap_values, labels, title, r, c)

    path = os.path.join(FIG_DIR, 'fig_gmm_sweeps_snapshots.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == '__main__':
    make_curves_figure()
    make_snapshots_figure()
