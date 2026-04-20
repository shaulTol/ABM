#!/usr/bin/env python3
"""
GMM sweep repeatability: rerun 4 sweeps with 10 seeds each.

For each sweep config and parameter value, we run 10 independent seeds
(controlling both torch and numpy RNG). This gives mean ± SD bands for
the SSA curve. MI is analytical (deterministic given GMM params), so
only SSA needs replication.

Selected sweeps (covering 2D/10D, separation/variance/weight):
  1. var_10d_all  — flagship (SSA↓ while MI flat)
  2. sep_2d       — basic 2D separation sweep
  3. weight_2d    — weight asymmetry
  4. sep_10d_all  — 10D separation sweep

Output:
  results_gmm_repeatability.json  — all raw SSA values per (config, param, seed)
  SUMMARY.md
"""
import sys
import os
import json
import time
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments', 'gmm_sweep'))

import numpy as np
import torch

from sweep_configs import SWEEP_CONFIGS
from compute import compute_ssa_and_mi

EXPDIR = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 10
SELECTED = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']  # paper Fig 1 panels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajectories', type=int, default=10000,
                        help='Trajectories per run (default 10000, use 2000 for fast)')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Take every Nth parameter value (default 1 = all)')
    args = parser.parse_args()

    langevin_params = dict(
        n_samples_ssa=2000,
        n_samples_mi=5000,
        n_trajectories=args.n_trajectories,
        T=1000,
        max_lag=500,
        dt=0.01,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'GMM Sweep Repeatability Analysis')
    print(f'Device: {device}, Seeds: {N_SEEDS}, n_traj: {args.n_trajectories}')
    print(f'Selected sweeps: {SELECTED}')
    print('=' * 70)

    # Find selected configs
    configs = {c['name']: c for c in SWEEP_CONFIGS}
    results = {}

    for sweep_name in SELECTED:
        cfg = configs[sweep_name]
        param_values = list(cfg['param_values'][::args.subsample])
        n_params = len(param_values)
        print(f'\n--- {sweep_name}: {cfg["description"]} ({n_params} points × {N_SEEDS} seeds) ---')

        ssa_all = np.zeros((n_params, N_SEEDS))
        mi_all = np.zeros((n_params, N_SEEDS))

        t0 = time.time()
        for pi, p in enumerate(param_values):
            for si in range(N_SEEDS):
                # Seed everything
                seed = 1000 * si + 42
                torch.manual_seed(seed)
                np.random.seed(seed)

                gm = cfg['build_mixture'](p)
                result = compute_ssa_and_mi(gm, device=device, **langevin_params)
                ssa_all[pi, si] = result['ssa']
                mi_all[pi, si] = result['mi']

            ssa_mean = ssa_all[pi].mean()
            ssa_sd = ssa_all[pi].std()
            cv = ssa_sd / ssa_mean if ssa_mean > 0 else 0
            if pi % 5 == 0 or pi == n_params - 1:
                print(f'  [{pi+1:>2}/{n_params}] param={p:.3f}: '
                      f'SSA={ssa_mean:.4f}±{ssa_sd:.4f} (CV={cv:.3f})')

        elapsed = time.time() - t0
        print(f'  Done in {elapsed:.0f}s')

        # Store results
        results[sweep_name] = {
            'description': cfg['description'],
            'param_name': cfg['param_name'],
            'param_values': [float(p) for p in param_values],
            'ssa_per_seed': ssa_all.tolist(),  # (n_params, N_SEEDS)
            'mi_per_seed': mi_all.tolist(),
            'ssa_mean': ssa_all.mean(axis=1).tolist(),
            'ssa_sd': ssa_all.std(axis=1).tolist(),
            'ssa_cv': (ssa_all.std(axis=1) / np.maximum(ssa_all.mean(axis=1), 1e-10)).tolist(),
            'mi_mean': mi_all.mean(axis=1).tolist(),
            'mi_sd': mi_all.std(axis=1).tolist(),
        }

    # Save JSON
    out_path = os.path.join(EXPDIR, 'results_gmm_repeatability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Print summary
    print(f'\n{"=" * 70}')
    print('SUMMARY: Coefficient of Variation (CV = SD/mean) across seeds')
    print(f'{"=" * 70}')
    for sweep_name in SELECTED:
        r = results[sweep_name]
        cvs = np.array(r['ssa_cv'])
        print(f'  {sweep_name:>15}: CV mean={cvs.mean():.4f}, '
              f'max={cvs.max():.4f}, median={np.median(cvs):.4f}')

    # Write SUMMARY.md
    write_summary(results, args.n_trajectories)


def write_summary(results, n_trajectories):
    lines = []
    lines.append('# Task 3: GMM Sweep Repeatability')
    lines.append('')
    lines.append('## Method')
    lines.append('')
    lines.append(f'Each of 4 sweep configurations was rerun with {N_SEEDS} independent ')
    lines.append('random seeds. The seeds control both torch and numpy RNG, affecting ')
    lines.append('Langevin trajectory initial conditions and noise. MI is analytical ')
    lines.append('(deterministic given GMM parameters) so only SSA has stochastic ')
    lines.append('variation.')
    lines.append('')
    lines.append(f'Langevin parameters: n_trajectories={n_trajectories:,}, T=1,000, dt=0.01, ')
    lines.append(f'max_lag=500, n_samples=2,000.')
    lines.append('')
    lines.append('## Selected Sweeps')
    lines.append('')

    for sweep_name in SELECTED:
        r = results[sweep_name]
        lines.append(f'- **{sweep_name}**: {r["description"]}')

    lines.append('')
    lines.append('## Results: Coefficient of Variation')
    lines.append('')
    lines.append('CV = SD/mean across 10 seeds, summarized over all parameter values:')
    lines.append('')
    lines.append('| Sweep | CV mean | CV median | CV max | Max SD |')
    lines.append('|-------|---------|-----------|--------|--------|')

    for sweep_name in SELECTED:
        r = results[sweep_name]
        cvs = np.array(r['ssa_cv'])
        sds = np.array(r['ssa_sd'])
        lines.append(f'| {sweep_name} | {cvs.mean():.4f} | {np.median(cvs):.4f} '
                     f'| {cvs.max():.4f} | {sds.max():.4f} |')

    lines.append('')
    lines.append('## Assessment')
    lines.append('')

    all_cvs = []
    for sweep_name in SELECTED:
        all_cvs.extend(results[sweep_name]['ssa_cv'])
    all_cvs = np.array(all_cvs)

    if all_cvs.max() < 0.05:
        lines.append(f'All CVs < 5%. SSA estimates are highly reproducible ')
        lines.append(f'across seeds. The sweep curves in the paper are representative ')
        lines.append(f'of the population mean within narrow error bands.')
    elif all_cvs.max() < 0.10:
        lines.append(f'Most CVs < 5%, max CV = {all_cvs.max():.3f}. SSA estimates ')
        lines.append(f'are reproducible with moderate variability at some parameter ')
        lines.append(f'values. The qualitative sweep trends are robust.')
    else:
        lines.append(f'Some CVs exceed 10% (max = {all_cvs.max():.3f}). ')
        lines.append(f'These parameter values should be examined for high variability.')

    summary_path = os.path.join(EXPDIR, 'SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {summary_path}')


if __name__ == '__main__':
    main()
