#!/usr/bin/env python3
"""
Exp 1: GMM Sweeps — whitened variant.

Replicates experiments/ssa_truncation_redef/code/run_gmm_10d_all_sweeps.py
with whitening applied to each sweep point.

For each GMM config:
  1. Compute analytic Sigma_f → WhiteningTransform
  2. Whiten samples and score
  3. Run Langevin in whitened space with var_f ≈ 1
  4. Compute SSA with trace normalization + 1/N floor truncation
  5. Compare with original SSA

Parameters match the paper: d=10, N=200, dt=0.01, T_phys=200s, 10 seeds.
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'gmm_sweep'))

import torch
from mixture_builders import build_bimodal_nd, build_line_fixed_distance
from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import LangevinSimulatorD
from src.metrics.autocorrelation import (
    compute_stationary_moments, compute_vector_autocorrelation, aggregate_ssa,
)
from experiments_whitening.shared.whitening import WhiteningTransform

# ── Parameters (match original) ──
D = 10
N_TRAJ = 200
N_SEEDS = 10
DT = 0.01
T_PHYS = 200.0
T = int(T_PHYS / DT)  # 20000
MAX_LAG = T
N_SAMPLES_MI = 5000

FLOOR_1N = 1.0 / N_TRAJ


# ── Sweep definitions (same as original) ──
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
    """Compute Ŝ = Δt × Σρ² truncated at floor."""
    if floor > 0:
        below = rho_sq < floor
        if below.any():
            trunc_idx = np.argmax(below)
            return float(dt * rho_sq[:trunc_idx].sum()), int(trunc_idx)
    return float(dt * rho_sq.sum()), len(rho_sq)


def run_one_point_whitened(gm, seed, device):
    """Run whitened simulation for one GMM config, return (rho_sq, mi, diagnostics)."""
    means = gm.means
    covs = gm.covs
    weights = gm.weights

    # MI (coordinate-invariant, no whitening needed)
    torch.manual_seed(seed + 99999)
    try:
        mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
    except Exception:
        mi = float('nan')

    # Analytic Sigma_f
    mu_f, Sigma_f = compute_stationary_moments(means, covs, weights)
    mu_f_np = mu_f.cpu().numpy()
    Sigma_f_np = Sigma_f.cpu().numpy()

    # Whitening transform
    wt = WhiteningTransform(analytic_cov=Sigma_f_np, mu=mu_f_np)

    # Build original-space simulator for score function
    sim_orig = LangevinSimulatorD(means, covs, weights, dt=DT, device=device)

    # Sample initial conditions
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = sim_orig.sample_initial(N_TRAJ).cpu().numpy()

    # Whiten samples
    x0_tilde = wt.whiten(x0)
    x0_tilde_t = torch.tensor(x0_tilde, dtype=torch.float32, device=device)

    # Build whitened score
    W_inv_t = torch.tensor(wt.W_inv, dtype=torch.float32, device=device)
    mu_t = torch.tensor(wt.mu, dtype=torch.float32, device=device)

    def score_tilde(x_tilde):
        x = x_tilde @ W_inv_t.T + mu_t
        grad = sim_orig.grad_log_pdf(x)
        return grad @ W_inv_t

    # Simulate in whitened space
    var_f_tilde = wt.sigma_tilde_sq  # ≈ 1
    sigma_tilde = float(np.sqrt(var_f_tilde))
    sqrt_dt = float(np.sqrt(DT))

    # Pre-allocate trajectory storage
    traj = torch.zeros(N_TRAJ, T + 1, D, device=device)
    traj[:, 0, :] = x0_tilde_t

    x = x0_tilde_t
    for t in range(T):
        grad = score_tilde(x)
        drift = 0.5 * var_f_tilde * DT * grad
        noise = sigma_tilde * sqrt_dt * torch.randn_like(x)
        x = x + drift + noise
        traj[:, t + 1, :] = x

    # Compute autocorrelation in whitened space
    # Use unit-Gaussian "GMM" for mu_f/cov_f (whitened moments)
    gmm_means_w = torch.zeros(1, D, device=device)
    gmm_covs_w = torch.eye(D, device=device).unsqueeze(0)
    gmm_weights_w = torch.ones(1, device=device)

    rho, _, _ = compute_vector_autocorrelation(
        traj, MAX_LAG, gmm_means_w, gmm_covs_w, gmm_weights_w,
        demean=True, normalization='trace')

    rho_sq = (rho.cpu().numpy() ** 2)[1:]
    return rho_sq.astype(np.float32), float(mi), wt.diagnostics()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"d={D}, N={N_TRAJ}, dt={DT}, T={T}, T_phys={T_PHYS}s (WHITENED)")
    print(f"Floor: 1/N={FLOOR_1N}")
    print(f"Seeds: {N_SEEDS}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    npz_data = {}
    all_results = {
        'd': D, 'N': N_TRAJ, 'dt': DT, 'T': T, 'T_phys': T_PHYS,
        'floor': FLOOR_1N,
        'n_seeds': N_SEEDS,
        'whitened': True,
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
        diag_list = []

        for si in range(N_SEEDS):
            for pi, p in enumerate(params):
                t0 = time.time()
                gm = cfg['build'](p)
                seed_val = si * 10000 + pi
                rho_sq, mi, diag = run_one_point_whitened(gm, seed_val, device)
                rho_sq_all[si, pi, :] = rho_sq
                mi_all[si, pi] = mi

                if si == 0:
                    s_1N, _ = compute_shat(rho_sq, DT, FLOOR_1N)
                    elapsed = time.time() - t0
                    print(f"  p={p:.2f}: S_whit(1/N)={s_1N:.3f} MI={mi:.3f} "
                          f"sig_tilde={diag['sigma_tilde_sq']:.4f} ({elapsed:.1f}s)",
                          flush=True)
                    if pi == 0:
                        diag_list.append(diag)

            print(f"  Seed {si} done", flush=True)

        # Compute Ŝ from raw ρ²
        sweep_result = {
            'description': cfg['description'],
            'param_name': cfg['param_name'],
            'param_values': [float(p) for p in params],
            's_hat_1N': [],
            'tau_star_1N': [],
            'mi': mi_all.tolist(),
        }

        for si in range(N_SEEDS):
            row_1N, t_1N = [], []
            for pi in range(n_params):
                s1, i1 = compute_shat(rho_sq_all[si, pi], DT, FLOOR_1N)
                row_1N.append(s1); t_1N.append(i1 * DT)
            sweep_result['s_hat_1N'].append(row_1N)
            sweep_result['tau_star_1N'].append(t_1N)

        arr = np.array(sweep_result['s_hat_1N'])
        means_s = arr.mean(axis=0)
        print(f"  1/N: Mean S range [{means_s.min():.3f}, {means_s.max():.3f}]")

        all_results['sweeps'][sweep_name] = sweep_result
        npz_data[f'{sweep_name}_rho_sq'] = rho_sq_all
        npz_data[f'{sweep_name}_mi'] = mi_all
        npz_data[f'{sweep_name}_params'] = np.array(params)

    all_results['total_time_s'] = time.time() - t_total

    # Save JSON
    out_json = os.path.join(results_dir, 'gmm_10d_whitened_all_sweeps.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON to {out_json}")

    # Save raw ρ²
    out_npz = os.path.join(results_dir, 'gmm_10d_whitened_all_sweeps_rho_sq.npz')
    np.savez_compressed(out_npz, dt=DT, d=D, N=N_TRAJ, **npz_data)
    print(f"Saved raw rho_sq to {out_npz}")
    print(f"Total time: {all_results['total_time_s']:.0f}s")


if __name__ == '__main__':
    main()
