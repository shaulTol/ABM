#!/usr/bin/env python3
"""
GMM dt-convergence study: fixed T_phys=100, halving dt each time.

dt=0.05 → T=2000, max_lag=2000
dt=0.025 → T=4000, max_lag=4000
dt=0.0125 → T=8000, max_lag=8000

Saves separate JSON per dt value. 10 seeds each.
"""
import sys
import os
import json
import time
import argparse
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'gmm_sweep'))

from experiments.gmm_sweep.sweep_configs import SWEEP_CONFIGS
from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa

from ssa_redef import find_tau_star_from_rho_sq

N_SEEDS = 10
N_TRAJECTORIES = 10000
T_PHYS = 100.0
N_SAMPLES_MI = 5000

SWEEP_NAMES = ['sep_2d', 'var_2d', 'weight_2d', 'nmodes_fixdist_2d']


def get_sweep_config(name):
    for cfg in SWEEP_CONFIGS:
        if cfg['name'] == name:
            return cfg
    raise ValueError(f"Unknown sweep: {name}")


def compute_rho_sq_profile(gm, device, seed, T, dt, max_lag):
    torch.manual_seed(seed)
    np.random.seed(seed)

    means = gm.means.to(device)
    covs = gm.covs.to(device)
    weights = gm.weights.to(device)

    trajectories = simulate_d_trajectories(
        means, covs, weights,
        n_trajectories=N_TRAJECTORIES, T=T, dt=dt, device=device,
    )

    rho, mu_f, cov_f = compute_vector_autocorrelation(
        trajectories, max_lag,
        gmm_means=means, gmm_covs=covs, gmm_weights=weights,
        demean=True, normalization='trace',
    )

    rho_np = rho.cpu().numpy()
    rho_sq = rho_np ** 2

    d = means.shape[1]
    var_f = torch.trace(cov_f).item() / d
    lags = np.arange(max_lag + 1) * dt

    return rho_sq, lags, var_f


def run_single_sweep(sweep_name, device, dt, T, max_lag):
    cfg = get_sweep_config(sweep_name)
    param_values = cfg['param_values']
    d = cfg.get('dim', 2)

    print(f"\n{'='*60}")
    print(f"Sweep: {sweep_name} (dt={dt}, T={T}, max_lag={max_lag})")
    print(f"{'='*60}")

    results = {
        'sweep_name': sweep_name,
        'param_name': cfg['param_name'],
        'param_values': [float(p) for p in param_values],
        'dim': d,
        'n_seeds': N_SEEDS,
        'n_traj': N_TRAJECTORIES,
        'T': T,
        'dt': dt,
        'max_lag': max_lag,
        'T_phys': T_PHYS,
        'seeds': [],
    }

    for seed in range(N_SEEDS):
        seed_data = {
            'seed': seed,
            's_hat': [],
            'tau_star': [],
            'L_star': [],
            'mi': [],
        }

        for pi, pval in enumerate(param_values):
            t0 = time.time()
            gm = cfg['build_mixture'](pval)

            torch.manual_seed(seed * 1000 + 99)
            try:
                mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
            except Exception:
                mi = float('nan')

            rho_sq, lags, var_f = compute_rho_sq_profile(gm, device, seed * 1000 + pi, T, dt, max_lag)

            tau_star_idx, tau_star = find_tau_star_from_rho_sq(
                rho_sq, lags, dt, var_f, d, N_TRAJECTORIES
            )

            rho_sq_nozero = rho_sq[1:]
            lags_nozero = lags[1:]
            mask_trunc = lags_nozero <= tau_star
            rho_sq_trunc = rho_sq_nozero[mask_trunc]

            s_hat = aggregate_ssa(rho_sq_trunc, dt=dt, normalize='riemann')
            L_star = int(mask_trunc.sum())

            seed_data['s_hat'].append(float(s_hat))
            seed_data['tau_star'].append(float(tau_star))
            seed_data['L_star'].append(L_star)
            seed_data['mi'].append(float(mi))

            elapsed = time.time() - t0
            if pi % 5 == 0:
                print(f"  seed={seed}, param[{pi}]={pval:.3f}: "
                      f"S_hat={s_hat:.4f}, τ*={tau_star:.2f} (L*={L_star}/{max_lag}), "
                      f"MI={mi:.4f} ({elapsed:.1f}s)")

        results['seeds'].append(seed_data)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True, help="Time step")
    args = parser.parse_args()

    dt = args.dt
    T = int(T_PHYS / dt)
    max_lag = T

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"dt={dt}, T={T}, max_lag={max_lag}, T_phys={T_PHYS}")

    results_dir = os.path.join(BASEDIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    dt_str = f"{dt}".replace('.', 'p')
    out_path = os.path.join(results_dir, f'gmm_sweeps_redef_dt{dt_str}.json')

    all_results = {'dt': dt, 'T': T, 'T_phys': T_PHYS}
    t_total = time.time()

    for sweep_name in SWEEP_NAMES:
        t0 = time.time()
        result = run_single_sweep(sweep_name, device, dt, T, max_lag)
        elapsed = time.time() - t0
        print(f"\n  {sweep_name} done in {elapsed:.0f}s")
        all_results[sweep_name] = result

        all_results['total_time_s'] = time.time() - t_total
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Checkpoint saved to {out_path}")

    total_time = time.time() - t_total
    all_results['total_time_s'] = total_time
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
