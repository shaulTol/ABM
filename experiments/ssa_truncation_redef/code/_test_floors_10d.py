#!/usr/bin/env python3
"""Quick test: nmodes 10D, N=200, different noise floors."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gmm_sweep'))

from mixture_builders import build_line_fixed_distance
from src.core.gaussian_mixture import GaussianMixture
from src.simulation.langevin import simulate_d_trajectories
from src.metrics.autocorrelation import compute_vector_autocorrelation

# Free-conv edges
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'noise_floor'))
from free_conv_utils import edges as freeconv_edges

d = 10
N_TRAJ = 200
DT = 0.005
T = 20000   # T_phys = 100s
MAX_LAG = 20000
device = 'cpu'
seed = 42
separation = 4.0
gamma = d / N_TRAJ

print(f"d={d}, N={N_TRAJ}, gamma={gamma:.4f}")
print(f"T={T}, dt={DT}, T_phys={T*DT:.0f}s, max_lag={MAX_LAG}")
print(f"Floors: 1/N={1/N_TRAJ:.6f}, 1/(Nd)={1/(N_TRAJ*d):.6f}")
print(flush=True)

for K in [1, 2, 3, 4, 5]:
    t0 = time.time()
    # Build K-mode line mixture in 10D
    gm = build_line_fixed_distance(K, d, separation)

    torch.manual_seed(seed * 1000 + K)
    np.random.seed(seed * 1000 + K)

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
    var_f = torch.trace(cov_f).item() / d
    sigma_f = np.sqrt(var_f)

    elapsed = time.time() - t0
    print(f"\nK={K} (elapsed={elapsed:.1f}s, sigma_f={sigma_f:.4f}):")

    # rho_sq profile at key timepoints
    for t_check in [1, 5, 10, 25, 50, 100]:
        idx = int(t_check / DT) - 1
        if idx < len(rho_sq):
            print(f"  rho_sq(t={t_check:3d}s) = {rho_sq[idx]:.8f}")

    # Free-conv floor for rho_sq
    # lambda_+ at tau->inf, then rho_sq_floor = (lambda_+ / sigma_f^2)^2
    try:
        _, lp_inf = freeconv_edges(100.0, sigma=sigma_f, gamma=gamma)
        rho_sq_freeconv = (lp_inf / (sigma_f**2))**2
    except Exception as e:
        rho_sq_freeconv = None
        print(f"  free-conv error: {e}")

    floors = {
        '1/N': 1.0 / N_TRAJ,
        '1/(Nd)': 1.0 / (N_TRAJ * d),
        'no_floor': 0.0,
    }
    if rho_sq_freeconv is not None:
        floors['free_conv'] = rho_sq_freeconv
        print(f"  free_conv floor = {rho_sq_freeconv:.8f}")

    for fname, floor in sorted(floors.items(), key=lambda x: -x[1]):
        below = rho_sq < floor
        if floor > 0 and below.any():
            trunc_idx = np.argmax(below)
            tau_star = lags[trunc_idx]
            s_hat = DT * rho_sq[:trunc_idx].sum()
        else:
            tau_star = lags[-1]
            s_hat = DT * rho_sq.sum()
        print(f"  {fname:12s} (floor={floor:.2e}): tau*={tau_star:6.1f}s, Shat={s_hat:.4f}")

print(flush=True)
