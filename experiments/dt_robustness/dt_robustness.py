#!/usr/bin/env python3
"""
Δt-Robustness Sweep: reproduce 4 GMM sweeps at dt=0.005 (halved from dt=0.01).

Physical time T_phys = T*dt = 10.0 is held fixed:
  - Original: dt=0.01, T=1000, max_lag=500
  - This run:  dt=0.005, T=2000, max_lag=1000

Uses independent seeds (offset 137) so ±1 SD bands are honest.
"""
import sys
import os
import json
import time
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'experiments', 'gmm_sweep'))

import numpy as np
import torch

from sweep_configs import SWEEP_CONFIGS
from compute import compute_ssa_and_mi

EXPDIR = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 10
SELECTED = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajectories', type=int, default=10000)
    parser.add_argument('--dt', type=float, default=0.005,
                        help='Euler-Maruyama step size (default 0.005)')
    args = parser.parse_args()

    # Physical time matched: T_phys = T*dt = 10.0
    T = int(10.0 / args.dt)
    max_lag = int(5.0 / args.dt)

    langevin_params = dict(
        n_samples_ssa=2000,
        n_samples_mi=5000,
        n_trajectories=args.n_trajectories,
        T=T,
        max_lag=max_lag,
        dt=args.dt,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Δt-Robustness Sweep')
    print(f'Device: {device}, dt={args.dt}, T={T}, max_lag={max_lag}')
    print(f'Physical time: {T*args.dt:.1f}, Physical max_lag: {max_lag*args.dt:.1f}')
    print(f'Seeds: {N_SEEDS}, n_traj: {args.n_trajectories}')
    print(f'Selected sweeps: {SELECTED}')
    print('=' * 70)

    configs = {c['name']: c for c in SWEEP_CONFIGS}
    results = {}
    total_t0 = time.time()

    for sweep_name in SELECTED:
        cfg = configs[sweep_name]
        param_values = list(cfg['param_values'])
        n_params = len(param_values)
        print(f'\n--- {sweep_name}: {cfg["description"]} ({n_params} points × {N_SEEDS} seeds) ---')

        ssa_all = np.zeros((n_params, N_SEEDS))
        mi_all = np.zeros((n_params, N_SEEDS))

        t0 = time.time()
        for pi, p in enumerate(param_values):
            for si in range(N_SEEDS):
                # Independent seeds (offset 137, different from original offset 42)
                seed = 1000 * si + 137
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

        results[sweep_name] = {
            'description': cfg['description'],
            'param_name': cfg['param_name'],
            'param_values': [float(p) for p in param_values],
            'ssa_per_seed': ssa_all.tolist(),
            'mi_per_seed': mi_all.tolist(),
            'ssa_mean': ssa_all.mean(axis=1).tolist(),
            'ssa_sd': ssa_all.std(axis=1).tolist(),
            'mi_mean': mi_all.mean(axis=1).tolist(),
            'mi_sd': mi_all.std(axis=1).tolist(),
            'dt': args.dt,
            'T': T,
            'max_lag': max_lag,
        }

    total_elapsed = time.time() - total_t0

    # Save results
    out_path = os.path.join(EXPDIR, f'results_dt_{args.dt}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Sanity check: compare with dt=0.01 results if available
    ref_path = os.path.join(EXPDIR, '..', 'robustness', 'task3_gmm_repeatability',
                            'results_gmm_repeatability.json')
    if os.path.exists(ref_path):
        print(f'\n{"=" * 70}')
        print('SANITY CHECK: Comparison with dt=0.01')
        print(f'{"=" * 70}')
        with open(ref_path) as f:
            ref = json.load(f)

        n_flags = 0
        max_diff = 0.0
        for sweep_name in SELECTED:
            if sweep_name not in ref:
                continue
            r_new = results[sweep_name]
            r_ref = ref[sweep_name]
            # Match parameter values
            new_params = np.array(r_new['param_values'])
            ref_params = np.array(r_ref['param_values'])
            new_mean = np.array(r_new['ssa_mean'])
            ref_mean = np.array(r_ref['ssa_mean'])
            ref_sd = np.array(r_ref['ssa_sd'])

            # Point-by-point comparison (only matching params)
            for i, p in enumerate(new_params):
                idx = np.argmin(np.abs(ref_params - p))
                if abs(ref_params[idx] - p) < 0.01:
                    diff = abs(new_mean[i] - ref_mean[idx])
                    max_diff = max(max_diff, diff)
                    if ref_sd[idx] > 0 and diff > 2 * ref_sd[idx]:
                        n_flags += 1
                        print(f'  FLAG: {sweep_name} param={p:.3f}: '
                              f'dt=0.005 SSA={new_mean[i]:.4f}, '
                              f'dt=0.01 SSA={ref_mean[idx]:.4f}±{ref_sd[idx]:.4f} '
                              f'(diff={diff:.4f} > 2σ={2*ref_sd[idx]:.4f})')

        print(f'\n  Ranking discrepancies (>2σ): {n_flags}')
        print(f'  Max absolute SSA difference: {max_diff:.5f}')

    # Summary
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    print(f'  Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)')
    print(f'  dt={args.dt}, T={T}, max_lag={max_lag}, n_traj={args.n_trajectories}')
    print(f'  Results: {out_path}')


if __name__ == '__main__':
    main()
