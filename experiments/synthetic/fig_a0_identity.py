#!/usr/bin/env python3
"""
Appendix A.2 Figure: Why not A_0 = I_d?

Shows that using A_0 = I_d makes SSA scale-dependent: rescaling the same
density by c changes SSA. The canonical choice A_0 = sigma_f^2 I_d
preserves it.

Uses the new SSA definition: Ŝ = dt × Σ ρ(ℓ)² (Riemann sum).

Layout: 3 contour panels of rescaled GMMs (c=0.5, 1, 2).
SSA scores are printed to stdout (not annotated on figure).

Usage: conda run -n ssa python make_fig_a0_identity.py
"""
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.plotting.style_config import setup_style, TEXT_WIDTH

from src.simulation.langevin import LangevinSimulatorD
from src.metrics.autocorrelation import compute_vector_autocorrelation, aggregate_ssa

setup_style()

OUT_DIR = os.path.dirname(__file__)

# ── Base GMM: symmetric bimodal in 2D ──────────────────────────────────
base_means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
base_covs = torch.tensor([[[0.5, 0.0], [0.0, 0.5]],
                           [[0.5, 0.0], [0.0, 0.5]]])
base_weights = torch.tensor([0.5, 0.5])

scales = [0.5, 1.0, 2.0]

N_TRAJ = 2000
T_STEPS = 5000
DT = 0.05
MAX_LAG = 2000
SEED = 42


def make_scaled_gmm(c):
    return c * base_means, c**2 * base_covs, base_weights.clone()


def simulate_identity_diffusion(means, covs, weights, n_traj, T, dt, max_lag, seed):
    torch.manual_seed(seed)
    sim = LangevinSimulatorD(means, covs, weights, dt=dt)
    x = sim.sample_initial(n_traj)
    trajectory = torch.zeros(n_traj, T + 1, sim.d)
    trajectory[:, 0] = x
    sqrt_dt = np.sqrt(dt)
    for t in range(T):
        grad = sim.grad_log_pdf(x)
        drift = 0.5 * dt * grad
        noise = sqrt_dt * torch.randn_like(x)
        x = x + drift + noise
        trajectory[:, t + 1] = x
    rho, _, _ = compute_vector_autocorrelation(
        trajectory, max_lag, means, covs, weights, normalization='trace')
    rho_sq = (rho[1:]**2).numpy()
    ssa = aggregate_ssa(rho_sq, dt=dt, normalize='riemann')
    return ssa


def simulate_canonical_diffusion(means, covs, weights, n_traj, T, dt, max_lag, seed):
    torch.manual_seed(seed)
    sim = LangevinSimulatorD(means, covs, weights, dt=dt)
    x0 = sim.sample_initial(n_traj)
    trajectory = sim.simulate(x0, T)
    rho, _, _ = compute_vector_autocorrelation(
        trajectory, max_lag, means, covs, weights, normalization='trace')
    rho_sq = (rho[1:]**2).numpy()
    ssa = aggregate_ssa(rho_sq, dt=dt, normalize='riemann')
    return ssa


def eval_gmm_2d(means, covs, weights, xlim, ylim, ngrid=200):
    x = np.linspace(xlim[0], xlim[1], ngrid)
    y = np.linspace(ylim[0], ylim[1], ngrid)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    sim = LangevinSimulatorD(means, covs, weights, dt=0.01)
    Z = sim.pdf(pts).numpy().reshape(ngrid, ngrid)
    return X, Y, Z


# ── Compute ─────────────────────────────────────────────────────────────
print(f"Parameters: N_traj={N_TRAJ}, T={T_STEPS}, dt={DT}, max_lag={MAX_LAG}")
print(f"Physical time: T*dt={T_STEPS*DT:.1f}, max_lag*dt={MAX_LAG*DT:.1f}")
print(f"SSA definition: Riemann sum (Ŝ = dt × Σ ρ²)")
print()

gmm_data = []
print(f"{'c':>5s}  {'σ_f²':>8s}  {'Ŝ(I_d)':>10s}  {'Ŝ(σ_f²I)':>10s}")
for i, c in enumerate(scales):
    means_c, covs_c, weights_c = make_scaled_gmm(c)
    sim = LangevinSimulatorD(means_c, covs_c, weights_c, dt=DT)
    sigma_f2 = sim.var_f

    ssa_id = simulate_identity_diffusion(
        means_c, covs_c, weights_c, N_TRAJ, T_STEPS, DT, MAX_LAG, SEED)
    ssa_can = simulate_canonical_diffusion(
        means_c, covs_c, weights_c, N_TRAJ, T_STEPS, DT, MAX_LAG, SEED)

    print(f"{c:5.1f}  {sigma_f2:8.3f}  {ssa_id:10.3f}  {ssa_can:10.3f}")
    gmm_data.append((means_c, covs_c, weights_c, sigma_f2, ssa_id, ssa_can))

# ── Figure: 3 contour panels ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.0))
plt.subplots_adjust(wspace=0.25)

LIM_X, LIM_Y = 8.0, 5.0
cmap = 'YlOrRd'

for i, c in enumerate(scales):
    ax = axes[i]
    means_c, covs_c, weights_c, sigma_f2, ssa_id, ssa_can = gmm_data[i]
    X, Y, Z = eval_gmm_2d(means_c, covs_c, weights_c, (-LIM_X, LIM_X), (-LIM_Y, LIM_Y))
    ax.contourf(X, Y, Z, levels=15, cmap=cmap)
    ax.set_xlim(-LIM_X, LIM_X)
    ax.set_ylim(-LIM_Y, LIM_Y)
    ax.set_aspect('equal')
    ax.set_title(f'$c = {c}$', fontsize=9)
    if i == 0:
        ax.set_ylabel('$x_2$')
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('$x_1$')

fig.subplots_adjust(bottom=0.15)

outpath = os.path.join(OUT_DIR, 'fig_a0_identity.png')
fig.savefig(outpath, dpi=300)
print(f"\nSaved: {outpath}")
plt.close(fig)
