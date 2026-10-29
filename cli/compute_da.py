#!/usr/bin/env python3
"""
Command line: DA directions (+ optional matched Monte Carlo null) from a saved C(τ) NPZ.

    python -m cli.compute_da <C_tau.npz> [--significance B]

Implementation: src/pipeline/compute_da.py
"""
import os, sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import compute_da, da_significance, n_metastable


def main():
    p = argparse.ArgumentParser(description='DA directions from a saved C(τ) NPZ.')
    p.add_argument('path', help='NPZ produced by src.pipeline.save_lags')
    p.add_argument('--significance', type=int, nargs='?', const=100, metavar='B',
                   help='matched-Gaussian Monte Carlo null with B replicates (default 100)')
    args = p.parse_args()

    nm = n_metastable(args.path)
    print(f"m̂ = {nm['m_hat']} metastable direction(s)")
    if nm['m_hat'] == 0:
        return
    result = compute_da(args.path, tau_idx=nm['tau_star_idx'])
    print(f"DA at τ* = {result['tau_star']:.3f}")
    print(f"  λ₁ = {result['eigenvalues'][0]:.4e}  "
          f"(gap = {result['gap_ratio']:.1f}× floor)")
    if len(result['eigenvalues']) > 1:
        print(f"  λ₂ = {result['eigenvalues'][1]:.4e}")
    print(f"  Floor λ₊ = {result['noise_floor']:.4e}")

    if args.significance:
        B = args.significance
        sig = da_significance(args.path, B=B, tau_idx=nm['tau_star_idx'], m=nm['m_hat'])
        print(f"\nMatched-Gaussian MC null (B={B}, Bonferroni over m̂={nm['m_hat']}):")
        for i, (ev, p_val) in enumerate(zip(sig['observed_evals'], sig['p_values'])):
            print(f"  DA{i+1}: λ={ev:.4e}, p={p_val:.4f}")


if __name__ == '__main__':
    main()
