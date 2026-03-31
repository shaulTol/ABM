#!/usr/bin/env python
"""SSA vs MI Parameter Sweep Analysis"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sweep_configs import SWEEP_CONFIGS
from compute import compute_ssa_and_mi
from divergence import detect_divergence
from plotting import plot_sweep, plot_divergence_distributions, plot_summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for config in SWEEP_CONFIGS:
        print(f"\n=== {config['description']} ===")
        ssa_values, mi_values = [], []
        mixtures = []  # store for divergence plots

        for p in tqdm(config['param_values'], desc=config['name']):
            gm = config['build_mixture'](p)
            result = compute_ssa_and_mi(gm, device=device)
            ssa_values.append(result['ssa'])
            mi_values.append(result['mi'])
            mixtures.append(gm)

        # Save intermediate results as JSON
        json_path = os.path.join(results_dir, f"{config['name']}_data.json")
        with open(json_path, 'w') as f:
            json.dump({
                'param_values': config['param_values'].tolist() if hasattr(config['param_values'], 'tolist') else list(config['param_values']),
                'ssa_values': ssa_values,
                'mi_values': mi_values,
            }, f, indent=2)

        # Detect divergence
        divs = detect_divergence(config['param_values'], ssa_values, mi_values)
        print(f"  Found {len(divs)} divergence region(s)")

        # Plot sweep
        plot_sweep(config['name'], config['description'], config['param_name'],
                   config['param_values'], ssa_values, mi_values, divs, results_dir)

        # Plot distributions at divergence points
        for i, div in enumerate(divs):
            si, ei = div['start_idx'], div['end_idx']
            before_idx = max(0, si - 1)
            after_idx = min(len(mixtures) - 1, ei + 1)
            plot_divergence_distributions(
                config['name'], config['dim'],
                mixtures[before_idx], mixtures[after_idx],
                config['param_values'][before_idx], config['param_values'][after_idx],
                config['param_name'], i, results_dir
            )

        all_results.append({
            'config': config,
            'ssa_values': ssa_values,
            'mi_values': mi_values,
            'divergences': divs,
        })

    # Summary plot
    plot_summary(all_results, results_dir)

    print(f"\n=== Done! All figures saved to {results_dir}/ ===")
    print(f"Total figures: {len(os.listdir(results_dir))} files")


if __name__ == '__main__':
    main()
