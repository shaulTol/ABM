#!/usr/bin/env python3
"""
Random GMM sweep in 10D, N=200 — Spearman(Ŝ, MI) with multiple floors.
50 random bimodal GMMs, separation in dim 0.
Saves raw ρ² as .npz.
"""
import sys, os, json, time
import torch
import numpy as np
from scipy.stats import spearmanr

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..')
sys.path.insert(0, REPODIR)

from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation

# ── Parameters ──
D = 10
N_TRAJ = 200
DT = 0.01
T_PHYS = 200.0
T = int(T_PHYS / DT)  # 20000
MAX_LAG = T
FLOOR_1N = 1.0 / N_TRAJ           # 0.005
FLOOR_1Nd = 1.0 / (N_TRAJ * D)    # 5e-4
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"d={D}, N={N_TRAJ}, dt={DT}, T={T}, T_phys={T_PHYS}s")
    print(f"Floors: 1/N={FLOOR_1N}, 1/(Nd)={FLOOR_1Nd}")
    print(f"Random GMMs: {N_RANDOM}")

    results_dir = os.path.join(CODEDIR, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    rho_sq_all = np.zeros((N_RANDOM, MAX_LAG), dtype=np.float32)
    records = []
    t_total = time.time()

    for seed in range(N_RANDOM):
        t0 = time.time()
        gm, meta = build_random_gmm_10d(seed)

        means = gm.means.to(device)
        covs = gm.covs.to(device)
        weights = gm.weights.to(device)

        # MI
        torch.manual_seed(seed + 99999)
        try:
            mi = gm.mutual_information(n_samples=N_SAMPLES_MI).item()
        except Exception:
            mi = float('nan')

        # Simulate
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
        rho_sq_all[seed, :] = rho_sq.astype(np.float32)

        s_1N, _ = compute_shat(rho_sq, DT, FLOOR_1N)
        s_1Nd, _ = compute_shat(rho_sq, DT, FLOOR_1Nd)

        records.append({
            'seed': seed, **meta,
            's_hat_1N': s_1N, 's_hat_1Nd': s_1Nd, 'mi': mi,
        })

        elapsed = time.time() - t0
        print(f"  [{seed:2d}/{N_RANDOM}] sep={meta['separation']:.2f} var={meta['variance']:.2f} "
              f"w1={meta['w1']:.2f} | S(1/N)={s_1N:.3f} S(1/Nd)={s_1Nd:.3f} MI={mi:.3f} ({elapsed:.1f}s)",
              flush=True)

    # Spearman for both floors
    mi_vals = [r['mi'] for r in records]
    rho_1N, p_1N = spearmanr([r['s_hat_1N'] for r in records], mi_vals)
    rho_1Nd, p_1Nd = spearmanr([r['s_hat_1Nd'] for r in records], mi_vals)
    print(f"\nSpearman(S_1/N, MI) = {rho_1N:.4f} (p = {p_1N:.2e})")
    print(f"Spearman(S_1/Nd, MI) = {rho_1Nd:.4f} (p = {p_1Nd:.2e})")

    total_time = time.time() - t_total
    result = {
        'd': D, 'N': N_TRAJ, 'dt': DT, 'T': T, 'T_phys': T_PHYS,
        'floors': {'1/N': FLOOR_1N, '1/(Nd)': FLOOR_1Nd},
        'n_random': N_RANDOM,
        'spearman_1N': {'rho': float(rho_1N), 'pval': float(p_1N)},
        'spearman_1Nd': {'rho': float(rho_1Nd), 'pval': float(p_1Nd)},
        'records': records,
        'total_time_s': total_time,
    }

    # Save JSON
    out_json = os.path.join(results_dir, 'random_sweep_10d_n200_T200.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON to {out_json}")

    # Save raw ρ²
    out_npz = os.path.join(results_dir, 'random_sweep_10d_n200_T200_rho_sq.npz')
    np.savez_compressed(out_npz, rho_sq=rho_sq_all, dt=DT, d=D, N=N_TRAJ)
    print(f"Saved raw rho_sq to {out_npz}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
