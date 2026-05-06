#!/usr/bin/env python3
"""
Trajectory bootstrap for GMM nmodes sweep (K=1..5, 10 seeds, N=200, d=10).
Resamples trajectories with replacement, recomputes Ŝ end-to-end.
"""
import sys, os, time
import torch
import numpy as np

CODEDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR = os.path.join(CODEDIR, '..', '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'gmm_sweep'))

from mixture_builders import build_line_fixed_distance
from src.simulation.langevin import simulate_d_trajectories

D = 10; N = 200; DT = 0.01; T_PHYS = 200.0; T = int(T_PHYS / DT)
K_VALUES = [1, 2, 3, 4, 5]; N_SEEDS = 10; B = 10000
FLOOR = 1.0 / N  # 0.005


def per_traj_autocorr_fft(centered):
    """Per-trajectory unnormalized autocorrelation via FFT. Returns (N, T) for lags 1..T."""
    Nn, Tp1, d = centered.shape
    # Zero-pad to avoid circular correlation
    padded = torch.zeros(Nn, 2 * Tp1, d, device=centered.device)
    padded[:, :Tp1, :] = centered
    # FFT, power spectrum, IFFT — summed over d dimensions
    F = torch.fft.rfft(padded, dim=1)  # (N, Tp1+1, d)
    power = (F * F.conj()).real  # (N, Tp1+1, d)
    autocorr_full = torch.fft.irfft(power, n=2 * Tp1, dim=1)  # (N, 2*Tp1, d)
    # Sum over d, take lags 1..T (skip lag 0)
    autocorr = autocorr_full[:, 1:Tp1, :].sum(dim=2)  # (N, T)
    return autocorr  # a_n(τ) = Σ_t Σ_d Y_{n,t,d} * Y_{n,t+τ,d} (unnormalized)


def compute_shat(rho_sq, dt, floor):
    below = rho_sq < floor
    if below.any():
        idx = np.argmax(below)
        if idx == 0:
            return 0.0
        return float(dt * rho_sq[:idx].sum())
    return float(dt * rho_sq.sum())


def main():
    device = 'cpu'
    print(f"D={D}, N={N}, dt={DT}, T_phys={T_PHYS}, K={K_VALUES}, seeds={N_SEEDS}, B={B}")

    # Check seeding: seed_val = si*10000 + ki (param index), so different across K
    print("\nSeeding: seed_val = si*10000 + param_idx → trajectories NOT shared across K")
    print("→ Bootstrap each K independently\n")

    # Process each (seed, K): simulate, compute per-traj autocorr, bootstrap
    shat_point = np.zeros((N_SEEDS, len(K_VALUES)))
    shat_boots = np.zeros((N_SEEDS, len(K_VALUES), B))
    rng = np.random.default_rng(42)

    t0_all = time.time()
    for si in range(N_SEEDS):
        for ki, K in enumerate(K_VALUES):
            seed_val = si * 10000 + ki
            gm = build_line_fixed_distance(n_modes=K, distance=4.0, variance=0.5, dim=D)
            means = gm.means.to(device)
            covs = gm.covs.to(device)
            weights = gm.weights.to(device)
            mu_f = (weights.unsqueeze(1) * means).sum(0)

            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            traj = simulate_d_trajectories(means, covs, weights,
                                           n_trajectories=N, T=T, dt=DT, device=device)
            # (N, T+1, d)
            centered = traj - mu_f.unsqueeze(0).unsqueeze(0)

            # Per-trajectory C(0) contribution: v_n = (1/(T+1)) Σ_t ||Y_{n,t}||²
            v_n = (centered ** 2).sum(dim=(1, 2)).numpy() / (T + 1)  # (N,)

            # Per-trajectory autocorrelation via FFT: a_n(τ) for τ=1..T
            a_n = per_traj_autocorr_fft(centered).numpy()  # (N, T)

            # Point estimate: ρ(τ) = mean(a_n(τ)) / ((T+1-τ) * mean(v_n))
            denom_tau = np.arange(T, 0, -1, dtype=np.float64)  # T+1-τ for τ=1..T → T, T-1, ..., 1
            rho = a_n.mean(axis=0) / (denom_tau * v_n.mean() + 1e-30)
            rho_sq = rho ** 2
            shat_point[si, ki] = compute_shat(rho_sq, DT, FLOOR)

            # Bootstrap: resample trajectory indices
            for b in range(B):
                idx = rng.choice(N, size=N, replace=True)
                rho_b = a_n[idx].mean(axis=0) / (denom_tau * v_n[idx].mean() + 1e-30)
                shat_boots[si, ki, b] = compute_shat(rho_b ** 2, DT, FLOOR)

        elapsed = time.time() - t0_all
        print(f"  Seed {si+1}/{N_SEEDS} done ({elapsed:.0f}s)")

    # ── Results ──
    # Pool across seeds: average Ŝ across seeds for each bootstrap rep
    shat_avg_boots = shat_boots.mean(axis=0)  # (K, B) — seed-averaged per boot rep
    shat_mean = shat_point.mean(axis=0)       # (K,)
    shat_se = shat_avg_boots.std(axis=1)      # (K,)

    print(f"\n{'K':>3} {'Ŝ':>10} {'SE':>10} {'Pr[Ŝ_K > Ŝ_{K-1}]':>20}")
    print("-" * 50)
    for ki, K in enumerate(K_VALUES):
        if ki == 0:
            prob_str = "—"
        else:
            prob = (shat_avg_boots[ki] > shat_avg_boots[ki - 1]).mean()
            prob_str = f"{prob:.4f}"
        print(f"  {K:>3d} {shat_mean[ki]:>10.4f} {shat_se[ki]:>10.4f} {prob_str:>20}")

    # Save
    out_path = os.path.join(REPODIR, 'experiments/ssa_truncation_redef/results/nmodes_traj_bootstrap.npz')
    np.savez(out_path, K_values=K_VALUES, shat_point=shat_point, shat_boots=shat_boots,
             shat_mean=shat_mean, shat_se=shat_se, B=B, N=N, D=D, dt=DT, T_phys=T_PHYS, floor=FLOOR)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
