#!/usr/bin/env python3
"""Quick test: nmodes ρ² profiles with different noise floors."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gmm_sweep'))

from experiments.gmm_sweep.sweep_configs import SWEEP_CONFIGS
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation

cfg = [c for c in SWEEP_CONFIGS if c['name'] == 'nmodes_fixdist_2d'][0]
d = cfg.get('dim', 2)

device = 'cpu'
DT = 0.01
T = 5000
MAX_LAG = 5000
N_TRAJ = 10000
seed = 0

print(f"d={d}, N={N_TRAJ}, T={T}, dt={DT}, T_phys={T*DT}s")
print(f"Floors: 1/N={1/N_TRAJ:.6f}, 1/(Nd)={1/(N_TRAJ*d):.6f}")
print(flush=True)

for K in [1, 2, 3, 4, 5]:
    pval = float(K)
    gm = cfg['build_mixture'](pval)

    torch.manual_seed(seed * 1000 + int(K) - 1)
    np.random.seed(seed * 1000 + int(K) - 1)

    means = gm.means.to(device)
    covs = gm.covs.to(device)
    weights = gm.weights.to(device)

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
    lags = np.arange(1, MAX_LAG + 1) * DT

    print(f"K={K}:")
    for t_check in [1, 5, 10, 25, 50]:
        idx = int(t_check / DT) - 1
        if idx < len(rho_sq):
            print(f"  rho_sq(t={t_check}s) = {rho_sq[idx]:.8f}")

    floors = {
        '1/N': 1.0 / N_TRAJ,
        '1/(Nd)': 1.0 / (N_TRAJ * d),
        'no_floor': 0.0,
    }
    for fname, floor in floors.items():
        below = rho_sq < floor
        if floor > 0 and below.any():
            trunc_idx = np.argmax(below)
            tau_star = lags[trunc_idx]
            s_hat = DT * rho_sq[:trunc_idx].sum()
        else:
            tau_star = lags[-1]
            s_hat = DT * rho_sq.sum()
        print(f"  {fname:10s}: tau*={tau_star:6.1f}s, Shat={s_hat:.4f}")
    print(flush=True)
