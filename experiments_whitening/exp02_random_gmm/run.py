#!/usr/bin/env python3
"""
Exp 2: Random GMM Sweep — whitened variant.

50 random bimodal GMMs in 10D, N=200. Same structure as
experiments/ssa_truncation_redef/code/run_random_sweep_10d.py but with
whitening applied to each GMM instance.
"""
import sys, os, json, time
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import torch
from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import LangevinSimulatorD
from src.metrics.autocorrelation import (
    compute_stationary_moments, compute_vector_autocorrelation, aggregate_ssa,
)
from experiments_whitening.shared.whitening import WhiteningTransform

# ── Parameters ──
D = 10
N_TRAJ = 200
DT = 0.01
T_PHYS = 200.0
T = int(T_PHYS / DT)
MAX_LAG = T
FLOOR_1N = 1.0 / N_TRAJ
N_SAMPLES_MI = 5000
N_RANDOM = 50


def build_random_gmm_10d(seed):
    rng = np.random.RandomState(seed)
    separation = rng.uniform(1.0, 8.0)
    variance = rng.uniform(0.5, 3.0)
    w1 = rng.uniform(0.5, 0.95)

    means = torch.zeros(2, D)
    means[1, 0] = separation
    covs = (variance * torch.eye(D)).unsqueeze(0).expand(2, -1, -1).clone()
    weights = torch.tensor([w1, 1.0 - w1], dtype=torch.float32)
    gm = GaussianMixture(means, covs, weights, device='cpu')
    meta = {'k': 2, 'separation': float(separation),
            'variance': float(variance), 'w1': float(w1)}
    return gm, meta


def compute_shat(rho_sq, dt, floor):
    if floor > 0:
        below = rho_sq < floor
        if below.any():
            trunc_idx = np.argmax(below)
            return float(dt * rho_sq[:trunc_idx].sum()), int(trunc_idx)
    return float(dt * rho_sq.sum()), len(rho_sq)


def run_one_whitened(gm, seed, device):
    """Run whitened simulation, return (rho_sq, mi, diagnostics)."""
    means = gm.means
    covs = gm.covs
    weights = gm.weights

    torch.manual_seed(seed + 99999)
    try:
        mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
    except Exception:
        mi = float('nan')

    mu_f, Sigma_f = compute_stationary_moments(means, covs, weights)
    mu_f_np = mu_f.cpu().numpy()
    Sigma_f_np = Sigma_f.cpu().numpy()
    wt = WhiteningTransform(analytic_cov=Sigma_f_np, mu=mu_f_np)

    sim_orig = LangevinSimulatorD(means, covs, weights, dt=DT, device=device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = sim_orig.sample_initial(N_TRAJ).cpu().numpy()
    x0_tilde = wt.whiten(x0)
    x0_tilde_t = torch.tensor(x0_tilde, dtype=torch.float32, device=device)

    W_inv_t = torch.tensor(wt.W_inv, dtype=torch.float32, device=device)
    mu_t = torch.tensor(wt.mu, dtype=torch.float32, device=device)

    def score_tilde(x_tilde):
        x = x_tilde @ W_inv_t.T + mu_t
        grad = sim_orig.grad_log_pdf(x)
        return grad @ W_inv_t

    var_f_tilde = wt.sigma_tilde_sq
    sigma_tilde = float(np.sqrt(var_f_tilde))
    sqrt_dt = float(np.sqrt(DT))

    traj = torch.zeros(N_TRAJ, T + 1, D, device=device)
    traj[:, 0, :] = x0_tilde_t
    x = x0_tilde_t
    for t in range(T):
        grad = score_tilde(x)
        drift = 0.5 * var_f_tilde * DT * grad
        noise = sigma_tilde * sqrt_dt * torch.randn_like(x)
        x = x + drift + noise
        traj[:, t + 1, :] = x

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
    print(f"Random GMMs: {N_RANDOM}")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    rho_sq_all = np.zeros((N_RANDOM, MAX_LAG), dtype=np.float32)
    records = []
    t_total = time.time()

    for seed in range(N_RANDOM):
        t0 = time.time()
        gm, meta = build_random_gmm_10d(seed)
        rho_sq, mi, diag = run_one_whitened(gm, seed, device)
        rho_sq_all[seed, :] = rho_sq

        s_1N, _ = compute_shat(rho_sq, DT, FLOOR_1N)

        records.append({
            'seed': seed, **meta,
            's_hat_1N_whitened': s_1N, 'mi': mi,
            'sigma_tilde_sq': diag['sigma_tilde_sq'],
        })

        elapsed = time.time() - t0
        print(f"  [{seed:2d}/{N_RANDOM}] sep={meta['separation']:.2f} var={meta['variance']:.2f} "
              f"w1={meta['w1']:.2f} | S_whit={s_1N:.3f} MI={mi:.3f} ({elapsed:.1f}s)",
              flush=True)

    # Spearman
    mi_vals = [r['mi'] for r in records]
    s_vals = [r['s_hat_1N_whitened'] for r in records]
    rho_sp, p_sp = spearmanr(s_vals, mi_vals)
    print(f"\nSpearman(S_whit, MI) = {rho_sp:.4f} (p = {p_sp:.2e})")

    total_time = time.time() - t_total
    result = {
        'd': D, 'N': N_TRAJ, 'dt': DT, 'T': T, 'T_phys': T_PHYS,
        'floor': FLOOR_1N,
        'n_random': N_RANDOM,
        'whitened': True,
        'spearman': {'rho': float(rho_sp), 'pval': float(p_sp)},
        'records': records,
        'total_time_s': total_time,
    }

    out_json = os.path.join(results_dir, 'random_sweep_10d_whitened.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON to {out_json}")

    out_npz = os.path.join(results_dir, 'random_sweep_10d_whitened_rho_sq.npz')
    np.savez_compressed(out_npz, rho_sq=rho_sq_all, dt=DT, d=D, N=N_TRAJ)
    print(f"Saved raw rho_sq to {out_npz}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
