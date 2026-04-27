"""
Validate free-convolution density vs empirical eigenvalues and naive MP.

Produces a 5×4 panel figure:
  - 5 rows: different gamma values (0.1, 0.25, 0.5, 0.75, 1.0)
  - 4 columns: tau = 0, 50, 200, 1000 (step counts with dt=0.01)
  - sigma = 1 for all panels

Each panel shows:
  - Light pink histogram: empirical eigenvalues from OU Langevin simulation
  - Red solid: naive MP density (sigma^2 <- rho(tau)^2)
  - Green dashed: free-convolution analytic density (Theorem 1)
  - Green dotted vertical: analytic lambda_+
  - Blue dotted vertical: largest sampled eigenvalue

All panels share the same x-axis scale.

Usage:
    python3 plot_free_conv_validation.py
    python3 plot_free_conv_validation.py --dt 0.01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "noise_floor"))
sys.path.insert(0, str(ROOT / "figures"))

from free_conv_utils import density, edges  # noqa: E402
from style_config import setup_style, SSA_COLOR, MI_COLOR, TEXT_WIDTH  # noqa: E402

setup_style()

OUT_DIR = ROOT / "figures" / "mp_generalization"


def simulate_langevin(d, N, dt, step_taus, seed):
    """Run OU Langevin and return eigenvalues of C_sym(tau) at each lag."""
    np.random.seed(seed)
    x = np.random.randn(N, d)
    x0 = x.copy()

    snapshots = {0: x0.copy()}
    current = 0
    for target in sorted(set(step_taus) - {0}):
        steps = target - current
        for _ in range(steps):
            x = x - 0.5 * x * dt + np.sqrt(dt) * np.random.randn(N, d)
        current = target
        snapshots[target] = x.copy()

    eigs_by_tau = {}
    for tau_steps in step_taus:
        x_tau = snapshots[tau_steps]
        C = x_tau.T @ x0 / N
        C_sym = 0.5 * (C + C.T)
        eigs_by_tau[tau_steps] = np.linalg.eigvalsh(C_sym)
    return eigs_by_tau


def run(dt=0.01, n_trials=3, save_path=None):
    sigma = 1.0
    step_taus = [0, 50, 200, 1000]
    gammas = [0.1, 0.25, 0.5, 0.75, 1.0]

    # Multiple (d, N) at same gamma, pooled for smoother histograms
    base_dims = [100, 200, 400]

    # Pre-compute all eigenvalues
    all_data = {}  # (gamma_idx, tau_steps) -> eigs array
    for gi, gamma in enumerate(gammas):
        configs = [(d, int(round(d / gamma))) for d in base_dims]
        eigs_by_tau = {t: [] for t in step_taus}
        for d, N in configs:
            for trial in range(n_trials):
                eigs = simulate_langevin(d, N, dt, step_taus, seed=42 + trial + d + gi * 1000)
                for t in step_taus:
                    eigs_by_tau[t].append(eigs[t])
        for t in step_taus:
            all_data[(gi, t)] = np.concatenate(eigs_by_tau[t])

    # Find global x-axis range
    global_min = min(all_data[k].min() for k in all_data)
    global_max = max(all_data[k].max() for k in all_data)
    x_margin = 0.05 * (global_max - global_min)
    xlim = (global_min - x_margin, global_max + x_margin)

    n_rows = len(gammas)
    n_cols = len(step_taus)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.9),
                             sharex=True)

    for gi, gamma in enumerate(gammas):
        for ci, tau_steps in enumerate(step_taus):
            ax = axes[gi, ci]
            eigs = all_data[(gi, tau_steps)]
            tau_phys = tau_steps * dt
            rho_tau = np.exp(-tau_phys / 2)

            # Light pink histogram
            ax.hist(eigs, bins=100, density=True, alpha=0.5, color='#b3e6b3',
                    edgecolor='none')

            # Naive MP (red solid) — uses marginal sigma^2, independent of tau
            eff = sigma ** 2
            if True:
                lm_mp = eff * (1 - np.sqrt(gamma)) ** 2
                lp_mp = eff * (1 + np.sqrt(gamma)) ** 2
                xs = np.linspace(max(lm_mp * 0.8, xlim[0]), min(lp_mp * 1.1, xlim[1]), 400)
                ys = np.zeros_like(xs)
                mask = (xs > lm_mp) & (xs < lp_mp)
                ys[mask] = (np.sqrt((lp_mp - xs[mask]) * (xs[mask] - lm_mp))
                            / (2 * np.pi * gamma * eff * np.maximum(xs[mask], 1e-30)))
                ax.plot(xs, ys, '-', color=MI_COLOR, lw=1.2)

            # Free-conv analytic (green dashed)
            if tau_phys == 0:
                lm_fc = sigma ** 2 * (1 - np.sqrt(gamma)) ** 2
                lp_fc = sigma ** 2 * (1 + np.sqrt(gamma)) ** 2
            else:
                lm_fc, lp_fc = edges(tau_phys, sigma, gamma=gamma)
            lam_grid = np.linspace(lm_fc, lp_fc, 500)
            rho_grid = density(lam_grid, tau_phys, sigma, gamma=gamma)
            ax.plot(lam_grid, rho_grid, '-', color=SSA_COLOR, lw=1.4)

            # Analytic λ+ (gray dashed)
            ax.axvline(lp_fc, color='#555555', ls='--', lw=1.0, alpha=0.8)

            # Largest sampled eigenvalue (blue dotted)
            eig_max = eigs.max()
            ax.axvline(eig_max, color=SSA_COLOR, ls=':', lw=1.0, alpha=0.8)

            ax.axvline(0, color='black', ls='-', lw=0.3, alpha=0.3)

            # Labels
            if gi == 0:
                ax.set_title(rf'$\tau = {tau_phys:.0f}$' if tau_phys >= 1
                             else rf'$\tau = {tau_phys}$', fontsize=9)
            if gi == n_rows - 1:
                ax.set_xlabel(r'$\lambda$')
            if ci == 0:
                ax.set_ylabel(rf'$\gamma = {gamma}$', fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.set_xlim(xlim)

    # Single legend next to suptitle
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#b3e6b3', alpha=0.5, label='Empirical'),
        Line2D([], [], color=MI_COLOR, ls='-', lw=1.2, label='MP'),
        Line2D([], [], color=SSA_COLOR, ls='-', lw=1.4, label='Free conv.'),
        Line2D([], [], color='#555555', ls='--', lw=1.0, alpha=0.8, label=r'$\lambda_+^{\mathrm{th}}$'),
        Line2D([], [], color=SSA_COLOR, ls=':', lw=1.0, alpha=0.8, label=r'$\lambda_1^{\mathrm{emp}}$'),
    ]

    fig.subplots_adjust(hspace=0.15, wspace=0.15, top=0.91, bottom=0.06, left=0.08, right=0.98)
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.53, 0.995),
               fontsize=7, ncol=5, handlelength=1.5, columnspacing=1.0, frameon=False)

    if save_path is None:
        save_path = OUT_DIR / "fig_free_conv_validation.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    run(dt=args.dt, n_trials=args.n_trials, save_path=args.save)
