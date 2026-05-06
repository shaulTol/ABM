#!/usr/bin/env python3
"""
GMM nmodes sweep in 10D, N=200 — matches SDXL regime (gamma=0.05).
Two dt values: 0.01 and 0.005, both T_phys=100s.
10 seeds, multiple noise floors.
"""
import sys, os, json, time, argparse
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'gmm_sweep'))
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'noise_floor'))

from mixture_builders import build_line_fixed_distance
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation
from free_conv_utils import edges as freeconv_edges

# ── Parameters ──
D = 10
N_TRAJ = 200
N_SEEDS = 10
SEPARATION = 4.0
K_VALUES = [1, 2, 3, 4, 5]
N_SAMPLES_MI = 5000
T_PHYS = 100.0  # physical time in seconds


def compute_floors(sigma_f, gamma, N, d):
    """Compute all noise floors."""
    floors = {
        '1/N': 1.0 / N,
        '1/(Nd)': 1.0 / (N * d),
    }
    try:
        _, lp_inf = freeconv_edges(100.0, sigma=sigma_f, gamma=gamma)
        floors['free_conv'] = (lp_inf / (sigma_f ** 2)) ** 2
    except Exception:
        floors['free_conv'] = None
    return floors


def compute_shat(rho_sq, lags, dt, floor):
    """Compute Ŝ = Δt × Σρ² truncated at floor."""
    below = rho_sq < floor
    if floor > 0 and below.any():
        trunc_idx = np.argmax(below)
        tau_star = lags[trunc_idx]
        s_hat = dt * rho_sq[:trunc_idx].sum()
    else:
        tau_star = lags[-1]
        s_hat = dt * rho_sq.sum()
    return float(s_hat), float(tau_star)


def run_sweep(dt, device):
    T = int(T_PHYS / dt)
    MAX_LAG = T
    gamma = D / N_TRAJ

    print(f"\n{'='*60}")
    print(f"dt={dt}, T={T}, T_phys={T_PHYS}s, d={D}, N={N_TRAJ}, gamma={gamma}")
    print(f"{'='*60}", flush=True)

    results = {
        'dt': dt, 'T': T, 'T_phys': T_PHYS,
        'd': D, 'N': N_TRAJ, 'gamma': gamma,
        'separation': SEPARATION, 'K_values': K_VALUES,
        'n_seeds': N_SEEDS,
        'seeds': [],
    }

    for seed in range(N_SEEDS):
        seed_data = {
            'seed': seed,
            's_hat_1N': [], 's_hat_1Nd': [], 's_hat_freeconv': [], 's_hat_nofloor': [],
            'tau_star_1N': [], 'tau_star_1Nd': [], 'tau_star_freeconv': [], 'tau_star_nofloor': [],
            'mi': [],
        }

        for K in K_VALUES:
            t0 = time.time()
            gm = build_line_fixed_distance(K, D, SEPARATION)

            torch.manual_seed(seed * 1000 + K)
            np.random.seed(seed * 1000 + K)

            means = gm.means.to(device)
            covs = gm.covs.to(device)
            weights = gm.weights.to(device)

            # MI
            torch.manual_seed(seed * 1000 + K + 99)
            try:
                mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
            except Exception:
                mi = float('nan')

            # Simulate
            torch.manual_seed(seed * 1000 + K)
            np.random.seed(seed * 1000 + K)
            trajectories = simulate_d_trajectories(
                means, covs, weights,
                n_trajectories=N_TRAJ, T=T, dt=dt, device=device,
            )

            rho, mu_f, cov_f = compute_vector_autocorrelation(
                trajectories, MAX_LAG,
                gmm_means=means, gmm_covs=covs, gmm_weights=weights,
                demean=True, normalization='trace',
            )

            rho_sq = (rho.cpu().numpy() ** 2)[1:]  # exclude lag 0
            lags = np.arange(1, MAX_LAG + 1) * dt
            var_f = torch.trace(cov_f).item() / D
            sigma_f = np.sqrt(var_f)

            # All floors
            floors = compute_floors(sigma_f, gamma, N_TRAJ, D)

            s1N, t1N = compute_shat(rho_sq, lags, dt, floors['1/N'])
            s1Nd, t1Nd = compute_shat(rho_sq, lags, dt, floors['1/(Nd)'])
            sno, tno = compute_shat(rho_sq, lags, dt, 0.0)

            if floors['free_conv'] is not None:
                sfc, tfc = compute_shat(rho_sq, lags, dt, floors['free_conv'])
            else:
                sfc, tfc = float('nan'), float('nan')

            seed_data['s_hat_1N'].append(s1N)
            seed_data['s_hat_1Nd'].append(s1Nd)
            seed_data['s_hat_freeconv'].append(sfc)
            seed_data['s_hat_nofloor'].append(sno)
            seed_data['tau_star_1N'].append(t1N)
            seed_data['tau_star_1Nd'].append(t1Nd)
            seed_data['tau_star_freeconv'].append(tfc)
            seed_data['tau_star_nofloor'].append(tno)
            seed_data['mi'].append(mi)

            elapsed = time.time() - t0
            if seed == 0:
                print(f"  K={K}: Ŝ(1/N)={s1N:.3f} Ŝ(1/Nd)={s1Nd:.3f} "
                      f"Ŝ(fc)={sfc:.3f} Ŝ(no)={sno:.3f} MI={mi:.3f} ({elapsed:.1f}s)",
                      flush=True)

        results['seeds'].append(seed_data)
        print(f"  Seed {seed} done", flush=True)

    # Summary stats
    print(f"\n--- Summary (dt={dt}) ---")
    for fi, fname in enumerate(['1N', '1Nd', 'freeconv', 'nofloor']):
        key = f's_hat_{fname}'
        arr = np.array([s[key] for s in results['seeds']])  # (n_seeds, n_K)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        mono = all(np.diff(means) > 0)
        print(f"  {fname:10s}: " + " ".join(f"K={K}:{means[i]:.3f}±{stds[i]:.3f}" for i, K in enumerate(K_VALUES))
              + f"  {'MONO' if mono else 'NOT MONO'}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=None, help='Single dt to run (default: both 0.01 and 0.005)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results_dir = os.path.join(CODEDIR, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    dt_values = [args.dt] if args.dt else [0.01, 0.005]
    t_total = time.time()

    all_results = {}
    for dt in dt_values:
        all_results[f'dt_{dt}'] = run_sweep(dt, device)

    all_results['total_time_s'] = time.time() - t_total

    out_path = os.path.join(results_dir, 'gmm_10d_n200_nmodes.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {all_results['total_time_s']:.0f}s")


if __name__ == '__main__':
    main()
