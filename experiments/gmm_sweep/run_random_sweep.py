#!/usr/bin/env python
"""Random GMM sweep: compute SSA and MI for random mixtures, measure rank correlation."""

import os
import sys
import json
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mixture_builders import build_random_gmm
from compute import compute_ssa_and_mi


def run_random_sweep(dim, n_seeds=50, device='cpu'):
    """Run random GMM sweep for a given dimensionality."""
    records = []
    for seed in tqdm(range(n_seeds), desc=f'random_{dim}d'):
        gm, meta = build_random_gmm(dim, seed)
        result = compute_ssa_and_mi(gm, device=device)
        records.append({
            'seed': seed,
            'k': meta['k'],
            'separation': meta['separation'],
            'variance': meta['variance'],
            'weights': meta['weights'],
            'ssa': result['ssa'],
            'mi': result['mi'],
        })

    ssa_vals = [r['ssa'] for r in records]
    mi_vals = [r['mi'] for r in records]
    rho, pval = spearmanr(ssa_vals, mi_vals)

    return {
        'dim': dim,
        'n_seeds': n_seeds,
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'records': records,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_random = {}
    for dim in [2, 10]:
        print(f"\n=== Random sweep {dim}D ===")
        sweep = run_random_sweep(dim, n_seeds=50, device=device)
        print(f"  Spearman rho = {sweep['spearman_rho']:.4f} (p = {sweep['spearman_pval']:.4e})")
        all_random[f'{dim}d'] = sweep

    out_path = os.path.join(results_dir, 'random_sweep_data.json')
    with open(out_path, 'w') as f:
        json.dump(all_random, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
