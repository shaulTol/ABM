#!/usr/bin/env python3
"""Random GMM scatter: Ŝ rank vs MI rank, 10D N=200, 1/N floor."""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.plotting.style_config import setup_style, SSA_COLOR, DIAGONAL_COLOR, TEXT_WIDTH

setup_style()

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'random_sweep_10d_n200.json')
OUT_DIR = os.path.dirname(__file__)


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    records = data['records']
    n = len(records)

    ssa_ranks = rankdata([r['s_hat_1N'] for r in records])
    mi_ranks = rankdata([r['mi'] for r in records])
    rho = data['spearman_1N']['rho']
    pval = data['spearman_1N']['pval']

    fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH * 0.45, TEXT_WIDTH * 0.45))

    ax.scatter(ssa_ranks, mi_ranks, s=25, color=SSA_COLOR,
               edgecolors='k', linewidths=0.4, alpha=0.7, zorder=3)
    ax.plot([1, n], [1, n], '--', color=DIAGONAL_COLOR, linewidth=0.8)

    ax.set_xlabel(r'$\hat{S}$ Rank')
    ax.set_ylabel('MI Rank')
    ax.set_title(f'Random GMMs (10D, $N$=200)')
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    ax.set_aspect('equal')
    ax.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}\n$p$ = {pval:.1e}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        out = os.path.join(OUT_DIR, f'fig_random_sweep.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
