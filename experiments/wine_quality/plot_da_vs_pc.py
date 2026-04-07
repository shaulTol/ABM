#!/usr/bin/env python3
"""Scatter plot: DA1 vs PC1 projections, colored by red/white wine."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import urllib.request

EXPDIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free SO2', 'total SO2', 'density',
    'pH', 'sulphates', 'alcohol'
]

BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'


def load_wine():
    data_dir = os.path.join(EXPDIR, 'data')
    all_X, all_labels = [], []
    for name, label in [('winequality-red.csv', 0), ('winequality-white.csv', 1)]:
        fpath = os.path.join(data_dir, name)
        if not os.path.exists(fpath):
            os.makedirs(data_dir, exist_ok=True)
            urllib.request.urlretrieve(f'{BASE_URL}/{name}', fpath)
        with open(fpath) as f:
            f.readline()
            rows = []
            for line in f:
                vals = line.strip().split(';')
                if len(vals) >= 12:
                    rows.append([float(v) for v in vals])
        data = np.array(rows)
        all_X.append(data[:, :11])
        all_labels.append(np.full(len(data), label, dtype=int))
    return np.vstack(all_X), np.concatenate(all_labels)


def main():
    # Load results
    with open(os.path.join(EXPDIR, 'results_dt001.json')) as f:
        results = json.load(f)

    pc1 = np.array(results['pca']['pc1_loadings'])
    da1 = np.array(results['da']['da1_loadings'])

    # Load and standardize data
    X_raw, labels = load_wine()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    is_red = labels == 0
    is_white = labels == 1

    proj_pc1 = X @ pc1
    proj_da1 = X @ da1

    # ── Figure 1: 2D scatter DA1 vs PC1 ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(proj_pc1[is_white], proj_da1[is_white], s=3, alpha=0.3,
               c='gold', label=f'White (n={is_white.sum()})')
    ax.scatter(proj_pc1[is_red], proj_da1[is_red], s=3, alpha=0.3,
               c='firebrick', label=f'Red (n={is_red.sum()})')
    ax.set_xlabel('PC1 projection (SO2-dominated)', fontsize=12)
    ax.set_ylabel('DA1 projection (chlorides/sulphates)', fontsize=12)
    cos_sim = abs(np.dot(pc1, da1))
    ax.set_title(f'DA1 vs PC1 projections  (|cos|={cos_sim:.3f})', fontsize=14)
    ax.legend(fontsize=11, markerscale=5)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, 'scatter_da1_vs_pc1.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved scatter_da1_vs_pc1.png')

    # ── Figure 2: Marginal histograms with scatter (joint plot style) ──
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    ax_main.scatter(proj_pc1[is_white], proj_da1[is_white], s=2, alpha=0.2, c='gold')
    ax_main.scatter(proj_pc1[is_red], proj_da1[is_red], s=2, alpha=0.2, c='firebrick')
    ax_main.set_xlabel('PC1 projection', fontsize=12)
    ax_main.set_ylabel('DA1 projection', fontsize=12)

    # Top marginal (PC1)
    ax_top.hist(proj_pc1[is_white], bins=60, alpha=0.6, color='gold', density=True, label='White')
    ax_top.hist(proj_pc1[is_red], bins=60, alpha=0.6, color='firebrick', density=True, label='Red')
    ax_top.set_ylabel('Density')
    ax_top.set_title('PC1: bimodal (red/white separated)', fontsize=12)
    ax_top.legend(fontsize=9)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right marginal (DA1)
    ax_right.hist(proj_da1[is_white], bins=60, alpha=0.6, color='gold', density=True,
                  orientation='horizontal')
    ax_right.hist(proj_da1[is_red], bins=60, alpha=0.6, color='firebrick', density=True,
                  orientation='horizontal')
    ax_right.set_xlabel('Density')
    ax_right.set_title('DA1: heavy tail\n(not bimodal)', fontsize=12)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle(f'Wine Quality: PC1 vs DA1  (|cos|={cos_sim:.3f})', fontsize=14, y=0.98)
    fig.savefig(os.path.join(EXPDIR, 'joint_da1_vs_pc1.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved joint_da1_vs_pc1.png')

    # ── Figure 3: Feature loading comparison with interpretation ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    d = len(FEATURE_NAMES)
    x_pos = np.arange(d)

    # PC1 loadings
    colors_pc1 = ['steelblue' if v >= 0 else 'lightcoral' for v in pc1]
    ax1.barh(x_pos, pc1, color=colors_pc1, alpha=0.8)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax1.set_xlabel('Loading', fontsize=11)
    ax1.set_title('PC1 (max variance)\nSO2 + residual sugar', fontsize=12)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.invert_yaxis()

    # DA1 loadings
    colors_da1 = ['coral' if v >= 0 else 'lightblue' for v in da1]
    ax2.barh(x_pos, da1, color=colors_da1, alpha=0.8)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax2.set_xlabel('Loading', fontsize=11)
    ax2.set_title('DA1 (slowest mixing)\nchlorides + sulphates', fontsize=12)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, 'loadings_horizontal.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved loadings_horizontal.png')

    # Print interpretation
    print(f'\n=== Interpretation ===')
    print(f'PC1 finds: SO2 axis (bimodal red/white split, |r_pb|=0.825)')
    print(f'DA1 finds: chlorides+sulphates axis (slow mixing due to heavy tail, |r_pb|=0.671)')
    print(f'|cos(PC1, DA1)| = {cos_sim:.3f} — different directions!')
    print(f'DA1 has LOWER SSA (0.088 vs 0.111) — slow mixing ≠ bimodality')
    print(f'The chlorides tail creates a "trap" that mixes slowly but isn\'t a separate mode')


if __name__ == '__main__':
    main()
