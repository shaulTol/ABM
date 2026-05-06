#!/usr/bin/env python3
"""
Δt-Robustness for 10D N=200 GMM sweeps.
Runs all 4 sweeps at dt=0.005 (half of dt=0.01), holding T_phys=100s fixed.
Saves raw ρ² for post-hoc floor analysis.

  dt=0.01:  T=10000, MAX_LAG=10000 (from all_sweeps job)
  dt=0.005: T=20000, MAX_LAG=20000 (this job)

Uses independent seeds (offset 137) so ±1 SD bands are honest.
"""
import sys, os, json, time
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'gmm_sweep'))

from mixture_builders import build_bimodal_nd, build_line_fixed_distance
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation

# ── Parameters ──
D = 10
N_TRAJ = 200
N_SEEDS = 10
DT = 0.005          # half of the original 0.01
T_PHYS = 100.0
T = int(T_PHYS / DT)  # 20000
MAX_LAG = T
N_SAMPLES_MI = 5000
SEED_OFFSET = 137    # different from all_sweeps (which uses si*10000+pi)

FLOOR_1N = 1.0 / N_TRAJ
FLOOR_1Nd = 1.0 / (N_TRAJ * D)


# ── Same sweep definitions as all_sweeps ──

def build_sep(sep):
    seps = [0.0] * D; seps[0] = sep
    return build_bimodal_nd(dim=D, separation_per_dim=seps, variance_per_dim=[1.0] * D)

def build_var(var):
    seps = [0.0] * D; seps[0] = 4.0
    return build_bimodal_nd(dim=D, separation_per_dim=seps, variance_per_dim=[var] * D)

def build_weight(w1):
    means = torch.zeros(2, D); means[1, 0] = 4.0
    covs = torch.eye(D).unsqueeze(0).expand(2, -1, -1).clone()
    weights = torch.tensor([w1, 1.0 - w1])
    from src.core.gaussian_mixture import GaussianMixture
    return GaussianMixture(means, covs, weights, device='cpu')

def build_nmodes(k):
    return build_line_fixed_distance(n_modes=int(k), distance=4.0, variance=0.5, dim=D)


SWEEPS = {
    'sep_10d': {
        'description': '10D Bimodal: Varying Separation (dim 0)',
        'param_name': 'Separation',
        'param_values': np.linspace(0.5, 8.0, 30).tolist(),
        'build': build_sep,
    },
    'var_10d': {
        'description': '10D Bimodal: Varying Variance',
        'param_name': 'Variance',
        'param_values': np.linspace(0.1, 5.0, 30).tolist(),
        'build': build_var,
    },
    'weight_10d': {
        'description': '10D Bimodal: Varying Weight',
        'param_name': 'Weight of Mode 1',
        'param_values': np.linspace(0.5, 0.95, 20).tolist(),
        'build': build_weight,
    },
    'nmodes_10d': {
        'description': '10D Line: Varying Modes (distance=4)',
        'param_name': 'Number of Modes',
        'param_values': [1, 2, 3, 4, 5],
        'build': build_nmodes,
    },
}


def compute_shat(rho_sq, dt, floor):
    if floor > 0:
        below = rho_sq < floor
        if below.any():
            trunc_idx = np.argmax(below)
            return float(dt * rho_sq[:trunc_idx].sum()), int(trunc_idx)
    return float(dt * rho_sq.sum()), len(rho_sq)


def run_one_point(gm, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    means = gm.means.to(device)
    covs = gm.covs.to(device)
    weights = gm.weights.to(device)

    torch.manual_seed(seed + 99999)
    try:
        mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
    except Exception:
        mi = float('nan')

    torch.manual_seed(seed)
    np.random.seed(seed)
    trajectories = simulate_d_trajectories(
        means, covs, weights,
        n_trajectories=N_TRAJ, T=T, dt=DT, device=device,
    )

    rho, mu_f, cov_f = compute_vector_autocorrelation(
        trajectories, MAX_LAG,
        gmm_means=means, gmm_covs=covs, gmm_weights=weights,
        demean=True, normalization='trace',
    )

    rho_sq = (rho.cpu().numpy() ** 2)[1:]
    return rho_sq.astype(np.float32), float(mi)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"dt-ROBUSTNESS: d={D}, N={N_TRAJ}, dt={DT}, T={T}, T_phys={T_PHYS}s")
    print(f"Floors: 1/N={FLOOR_1N}, 1/(Nd)={FLOOR_1Nd}")
    print(f"Seeds: {N_SEEDS} (offset={SEED_OFFSET})")

    results_dir = os.path.join(CODEDIR, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    npz_data = {}
    all_results = {
        'd': D, 'N': N_TRAJ, 'dt': DT, 'T': T, 'T_phys': T_PHYS,
        'floors': {'1/N': FLOOR_1N, '1/(Nd)': FLOOR_1Nd},
        'n_seeds': N_SEEDS, 'seed_offset': SEED_OFFSET,
        'sweeps': {},
    }

    t_total = time.time()

    for sweep_name, cfg in SWEEPS.items():
        params = cfg['param_values']
        n_params = len(params)
        print(f"\n{'='*60}")
        print(f"{sweep_name}: {cfg['description']} ({n_params} points x {N_SEEDS} seeds)")
        print(f"{'='*60}", flush=True)

        rho_sq_all = np.zeros((N_SEEDS, n_params, MAX_LAG), dtype=np.float32)
        mi_all = np.zeros((N_SEEDS, n_params), dtype=np.float32)

        for si in range(N_SEEDS):
            for pi, p in enumerate(params):
                t0 = time.time()
                gm = cfg['build'](p)
                seed_val = si * 10000 + pi + SEED_OFFSET
                rho_sq, mi = run_one_point(gm, seed_val, device)
                rho_sq_all[si, pi, :] = rho_sq
                mi_all[si, pi] = mi

                if si == 0:
                    s_1N, _ = compute_shat(rho_sq, DT, FLOOR_1N)
                    s_1Nd, _ = compute_shat(rho_sq, DT, FLOOR_1Nd)
                    elapsed = time.time() - t0
                    print(f"  p={p:.2f}: S(1/N)={s_1N:.3f} S(1/Nd)={s_1Nd:.3f} "
                          f"MI={mi:.3f} ({elapsed:.1f}s)", flush=True)

            print(f"  Seed {si} done", flush=True)

        # Compute Ŝ for both floors
        sweep_result = {
            'description': cfg['description'],
            'param_name': cfg['param_name'],
            'param_values': [float(p) for p in params],
            's_hat_1N': [], 's_hat_1Nd': [],
            'tau_star_1N': [], 'tau_star_1Nd': [],
            'mi': mi_all.tolist(),
        }

        for si in range(N_SEEDS):
            row_1N, row_1Nd, t_1N, t_1Nd = [], [], [], []
            for pi in range(n_params):
                s1, i1 = compute_shat(rho_sq_all[si, pi], DT, FLOOR_1N)
                s2, i2 = compute_shat(rho_sq_all[si, pi], DT, FLOOR_1Nd)
                row_1N.append(s1); t_1N.append(i1 * DT)
                row_1Nd.append(s2); t_1Nd.append(i2 * DT)
            sweep_result['s_hat_1N'].append(row_1N)
            sweep_result['s_hat_1Nd'].append(row_1Nd)
            sweep_result['tau_star_1N'].append(t_1N)
            sweep_result['tau_star_1Nd'].append(t_1Nd)

        for fname, fkey in [('1/N', '1N'), ('1/(Nd)', '1Nd')]:
            arr = np.array(sweep_result[f's_hat_{fkey}'])
            means = arr.mean(axis=0)
            print(f"  {fname}: Mean S range [{means.min():.3f}, {means.max():.3f}]")

        all_results['sweeps'][sweep_name] = sweep_result
        npz_data[f'{sweep_name}_rho_sq'] = rho_sq_all
        npz_data[f'{sweep_name}_mi'] = mi_all
        npz_data[f'{sweep_name}_params'] = np.array(params)

    all_results['total_time_s'] = time.time() - t_total

    out_json = os.path.join(results_dir, 'gmm_10d_n200_dt_robustness.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON to {out_json}")

    out_npz = os.path.join(results_dir, 'gmm_10d_n200_dt_robustness_rho_sq.npz')
    np.savez_compressed(out_npz, dt=DT, d=D, N=N_TRAJ, **npz_data)
    print(f"Saved raw rho_sq to {out_npz}")
    print(f"Total time: {all_results['total_time_s']:.0f}s")


if __name__ == '__main__':
    main()
