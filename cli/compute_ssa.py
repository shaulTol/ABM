#!/usr/bin/env python3
"""
Command line: SSA (Ŝ) from a saved C(τ) NPZ.

    python -m cli.compute_ssa <C_tau.npz> [--bootstrap B]

Implementation: src/pipeline/compute_ssa.py
"""
import os, sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import compute_ssa, bootstrap_ssa


def main():
    p = argparse.ArgumentParser(description='SSA (Ŝ) from a saved C(τ) NPZ.')
    p.add_argument('path', help='NPZ produced by src.pipeline.save_lags')
    p.add_argument('--bootstrap', type=int, nargs='?', const=2000, metavar='B',
                   help='trajectory bootstrap with B replicates (default 2000)')
    args = p.parse_args()

    result = compute_ssa(args.path)
    print(f"Ŝ = {result['s_hat']:.4f}  (τ* = {result['tau_star']:.3f}, "
          f"{result['n_lags_used']} lags used"
          f"{'' if result['converged'] else '; 1/N floor not reached, lower bound'})")

    if args.bootstrap:
        boot = bootstrap_ssa(args.path, B=args.bootstrap)
        print(f"Bootstrap (B={args.bootstrap}): Ŝ = {boot['s_hat']:.4f} ± {boot['se']:.4f} "
              f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")


if __name__ == '__main__':
    main()
