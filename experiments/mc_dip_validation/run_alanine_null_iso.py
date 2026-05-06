#!/usr/bin/env python3
"""
MC Null Test for Alanine DA Directions — ISOTROPIC Gaussian null.

Compares observed DA eigenvalues against an isotropic Gaussian null.
The null model samples from N(0, sigma_f^2 * I_d) where
sigma_f^2 = tr(C0) / d — the scalar diffusion coefficient.

This differs from run_alanine_null.py which uses N(0, C0) (matched covariance).

Each array task runs REPS_PER_TASK replicates (default 10).
SLURM_ARRAY_TASK_ID determines the seed offset.

Params: d=45, N=2000 KDE centers, dt=0.1, h=0.0091.
"""
import sys
import os
import time
import argparse
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.kde.kde_score import KDEScore
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import extract_da_directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=None,
                        help='Array task ID (default: from SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--reps_per_task', type=int, default=10)
    parser.add_argument('--N', type=int, default=2000, help='Number of KDE centers')
    parser.add_argument('--lag', type=int, default=100000,
                        help='Lag in steps (phys_lag = lag * dt ps)')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--h', type=float, default=0.0091, help='KDE bandwidth (h_scott/2)')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--n_eigvals', type=int, default=10,
                        help='Number of eigenvalues to store')
    parser.add_argument('--ctau_path', type=str,
                        default='../save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz',
                        help='Path to pre-saved C(τ) npz with C0 matrix (for sigma_f^2)')
    parser.add_argument('--output_prefix', type=str, default='alanine_null_iso_lag100k',
                        help='Prefix for output .npz filename')
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task {task_id}: device={device}, reps={args.reps_per_task}, "
          f"N={args.N}, lag_steps={args.lag}, phys_lag_ps={args.lag * args.dt}, "
          f"dt={args.dt}, h={args.h}")

    # --- Compute sigma_f^2 = tr(C0) / d from pre-saved C(0) ---
    print(f"Loading C(0) from {args.ctau_path}...")
    ctau_data = np.load(args.ctau_path)
    C0 = np.array(ctau_data['C0'], dtype=np.float64)  # (45, 45)
    d = int(ctau_data['d'])
    sigma_f_sq = np.trace(C0) / d
    sigma_f = np.sqrt(sigma_f_sq)
    print(f"  C0 shape: {C0.shape}, d={d}")
    print(f"  sigma_f_sq = tr(C0)/d = {sigma_f_sq:.6e}")
    print(f"  sigma_f = {sigma_f:.6e}")
    print(f"  NULL: N(0, sigma_f^2 * I_{d}) — ISOTROPIC")

    # --- Run null replicates ---
    null_evals_all = np.zeros((args.reps_per_task, args.n_eigvals))
    # Seed offset 20000 to avoid collision with existing nulls (which use 5000+)
    seed_offset = 20000 + task_id * args.reps_per_task

    for rep in range(args.reps_per_task):
        seed = seed_offset + rep
        t0 = time.time()
        print(f"\n  Rep {rep+1}/{args.reps_per_task} (seed={seed})...")

        # 1. Sample null data: x_null ~ N(0, sigma_f^2 * I_d) — ISOTROPIC
        rng = np.random.RandomState(seed)
        x_null_np = sigma_f * rng.randn(args.N, d)  # (N, 45)
        x_null = torch.from_numpy(x_null_np).float()

        # 2. Build KDE score on null data
        kde = KDEScore(x_null, bandwidth=args.h, device=device, batch_size=500)

        # 3. Build simulator
        sim = GenericLangevinSimulator(
            score_fn=kde.score,
            samples=x_null,
            dt=args.dt,
            device=device,
        )

        # 4. Extract DA directions
        evals_b, _, _ = extract_da_directions(
            simulator=sim,
            lag=args.lag,
            n_trajectories=args.N,
            dt=args.dt,
            device=device,
            seed=seed + 100000,
        )
        evals_np = evals_b.cpu().numpy()
        null_evals_all[rep, :] = evals_np[:args.n_eigvals]

        elapsed = time.time() - t0
        print(f"    λ₁={evals_np[0]:.6f}, λ₂={evals_np[1]:.6f}, "
              f"gap={evals_np[0]/max(evals_np[1],1e-12):.2f}, "
              f"time={elapsed:.0f}s")

    # --- Save ---
    out_path = os.path.join(args.output_dir, f'{args.output_prefix}_task{task_id:03d}.npz')
    np.savez(out_path,
             null_eigenvalues=null_evals_all,
             task_id=task_id,
             seed_offset=seed_offset,
             reps_per_task=args.reps_per_task,
             N=args.N, lag=args.lag, dt=args.dt, h=args.h, d=d,
             n_eigvals=args.n_eigvals,
             sigma_f_sq=sigma_f_sq,
             null_type='isotropic')
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
