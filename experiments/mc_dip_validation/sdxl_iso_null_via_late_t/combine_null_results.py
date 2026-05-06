#!/usr/bin/env python3
"""Concatenate per-task .npz files from the SDXL anisotropic-null array job.

Reads results/task_*.npz and writes results/sdxl_null_anisotropic_combined.npz
with stacked z0 (N_big, 768) and z_tau (N_big, 768).
"""
import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output', type=str,
                        default='results/sdxl_null_anisotropic_combined.npz')
    parser.add_argument('--expected_N_big', type=int, default=800,
                        help='Expected total trajectories; warn if mismatch.')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, 'task_*.npz')))
    if not files:
        raise SystemExit(f"No task_*.npz files found in {args.input_dir}")
    print(f"Found {len(files)} task files")

    z0s, z_taus = [], []
    task_ids, seeds, lam_maxes = [], [], []
    for f in files:
        d = np.load(f)
        z0s.append(d['z0'])
        z_taus.append(d['z_tau'])
        task_ids.append(int(d['task_id']))
        seeds.append(int(d['seed']))
        lam_maxes.append(float(d['lam_max_subset']))
        print(f"  {os.path.basename(f)}: z0={d['z0'].shape} z_tau={d['z_tau'].shape} "
              f"lam_max_subset={float(d['lam_max_subset']):.4e}")

    z0 = np.concatenate(z0s, axis=0)
    z_tau = np.concatenate(z_taus, axis=0)
    print(f"\nCombined: z0={z0.shape}, z_tau={z_tau.shape}")
    if z0.shape[0] != args.expected_N_big:
        print(f"WARNING: N_big = {z0.shape[0]} != expected {args.expected_N_big}")

    # Reference-frame check: any task with subset lam_max way above the mean?
    lam_arr = np.array(lam_maxes)
    print(f"\nPer-task subset lambda_max:")
    print(f"  mean = {lam_arr.mean():.4e}, std = {lam_arr.std():.4e}")
    print(f"  min  = {lam_arr.min():.4e}, max = {lam_arr.max():.4e}")

    np.savez_compressed(
        args.output,
        z0=z0, z_tau=z_tau,
        task_ids=np.array(task_ids),
        seeds=np.array(seeds),
        per_task_lam_max=lam_arr,
        N_big=np.array(z0.shape[0]),
    )
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
