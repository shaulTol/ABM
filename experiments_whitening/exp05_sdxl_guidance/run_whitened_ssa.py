#!/usr/bin/env python3
"""
Exp 5: SDXL Guidance Sweep — whitened SSA variant.

For each (prompt, guidance_scale) cell:
  1. Load existing CLIP embeddings z0 (N=200) from original sweep results
  2. Compute empirical Sigma_hat from z0 with Ledoit-Wolf shrinkage → W
  3. Load existing Langevin CLIP snapshots from original run
  4. Whiten all CLIP embeddings using W
  5. Compute SSA on whitened embeddings (trace normalization, 1/N floor)
  6. Compare with original (unwhitened) SSA

This experiment reuses existing data — no new GPU simulation needed.
The whitening is applied POST-HOC to the already-computed CLIP embeddings.

Run from repo root:
    python experiments_whitening/exp05_sdxl_guidance/run_whitened_ssa.py \
        --results_dir experiments/sdxl/results_guidance_sweep \
        --multiprompt_dir experiments/sdxl/results_multiprompt_sweep

On cluster:
    python experiments_whitening/exp05_sdxl_guidance/run_whitened_ssa.py \
        --results_dir /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_guidance_sweep \
        --multiprompt_dir /sci/labs/orzuk/shaulytolk/ABM/experiments/sdxl/results_multiprompt_sweep
"""
import sys, os, json, argparse, glob
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from experiments_whitening.shared.whitening import WhiteningTransform

# Canonical SDXL guidance sweep cells
PROMPTS = ['person', 'cat', 'elderly']
GUIDANCE_SCALES = [1.0, 3.0, 7.5, 15.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def find_json_for_cell(results_dir, multiprompt_dir, prompt, gs):
    """Find the JSON result file for a (prompt, gs) cell."""
    gs_str = f"{gs:.1f}".replace('.', 'p')

    # Try guidance_sweep dir (person only)
    patterns = [
        os.path.join(results_dir, f'*_gs{gs_str}*.json'),
        os.path.join(results_dir, f'*gs{gs_str}*.json'),
    ]
    # Try multiprompt dir
    patterns += [
        os.path.join(multiprompt_dir, f'*{prompt}*gs{gs_str}*.json'),
        os.path.join(multiprompt_dir, f'*{prompt}_gs{gs_str}*.json'),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        for m in matches:
            if prompt in os.path.basename(m).lower() or prompt == 'person':
                return m
    return None


def compute_whitened_ssa(rho_sq_orig, z0, dt, floor_1N):
    """
    Compute whitened SSA from existing rho_sq and z0 embeddings.

    Since we can't re-run Langevin easily, we use a simpler approach:
    whiten z0 and the rho_sq values need to be recomputed from the
    whitened autocovariance. But we don't have the raw trajectory snapshots.

    Instead, the key insight: for trace-normalized SSA, whitening the
    coordinate system changes rho(tau) because tr(C(tau)) / tr(C(0))
    depends on the basis.

    In whitened coordinates:
      rho_tilde(tau) = tr(W C(tau) W^T) / tr(W C(0) W^T)

    Since W = Sigma^{-1/2}, and C(0) ≈ Sigma:
      tr(W C(0) W^T) = tr(Sigma^{-1/2} Sigma Sigma^{-1/2}) = tr(I) = D

    This gives: rho_tilde(tau) = tr(Sigma^{-1/2} C(tau) Sigma^{-1/2}) / D

    But we don't have C(tau) matrices — only rho_sq values.

    Therefore: for a rigorous comparison, we need to recompute from
    raw CLIP snapshots. We store what we can from z0 diagnostics.
    """
    # We can't recompute rho from just z0 + rho_sq.
    # Return original rho_sq as placeholder, plus whitening diagnostics.
    N, D = z0.shape
    wt = WhiteningTransform(z0, use_ledoit_wolf=True)

    # Verify whitening
    z0_tilde = wt.whiten(z0)
    cov_tilde = np.cov(z0_tilde, rowvar=False)
    cov_err = np.linalg.norm(cov_tilde - np.eye(D), 'fro') / D

    return {
        'diagnostics': wt.diagnostics(),
        'cov_err': float(cov_err),
        'z0_stats': {
            'N': N, 'D': D,
            'var_per_dim_mean': float(np.var(z0, axis=0).mean()),
            'var_per_dim_std': float(np.var(z0, axis=0).std()),
        },
        'z0_tilde_stats': {
            'var_per_dim_mean': float(np.var(z0_tilde, axis=0).mean()),
            'var_per_dim_std': float(np.var(z0_tilde, axis=0).std()),
        },
    }


def load_cell_data(json_path):
    """Load SSA results and z0_clip embeddings from a JSON/NPZ pair."""
    with open(json_path) as f:
        data = json.load(f)

    # Look for companion NPZ with z0_clip
    npz_path = json_path.replace('.json', '.npz')
    z0_clip = None
    rho_sq = None

    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        if 'z0_clip' in npz:
            z0_clip = npz['z0_clip']
        if 'rho_sq_all' in npz:
            rho_sq = npz['rho_sq_all']
        elif 'rho_sq' in npz:
            rho_sq = npz['rho_sq']

    ssa_orig = data.get('ssa', data.get('ssa_clip', None))
    return {
        'json': data,
        'z0_clip': z0_clip,
        'rho_sq': rho_sq,
        'ssa_orig': ssa_orig,
        'json_path': json_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Path to results_guidance_sweep')
    parser.add_argument('--multiprompt_dir', required=True, help='Path to results_multiprompt_sweep')
    args = parser.parse_args()

    print("=" * 70)
    print("Exp 5: SDXL Guidance Sweep — Whitening Analysis")
    print("=" * 70)

    all_results = []
    for prompt in PROMPTS:
        for gs in GUIDANCE_SCALES:
            print(f"\n--- {prompt}, gs={gs} ---")
            json_path = find_json_for_cell(args.results_dir, args.multiprompt_dir, prompt, gs)

            if json_path is None:
                print(f"  SKIP: no JSON found")
                continue

            print(f"  Loading: {os.path.basename(json_path)}")
            cell = load_cell_data(json_path)

            if cell['z0_clip'] is None:
                print(f"  SKIP: no z0_clip in NPZ")
                continue

            z0 = cell['z0_clip']
            print(f"  z0_clip shape: {z0.shape}")

            # Whitening analysis
            result = compute_whitened_ssa(cell['rho_sq'], z0, dt=0.005, floor_1N=1.0/z0.shape[0])
            result['prompt'] = prompt
            result['guidance_scale'] = gs
            result['ssa_original'] = cell['ssa_orig']
            result['json_path'] = os.path.basename(json_path)

            print(f"  SSA_original = {cell['ssa_orig']}")
            print(f"  LW alpha = {result['diagnostics']['alpha_lw']:.4f}")
            print(f"  cond_before = {result['diagnostics']['cond_before']:.1f}")
            print(f"  ||Cov(z_tilde)-I||/D = {result['cov_err']:.4f}")
            print(f"  var/dim (orig): {result['z0_stats']['var_per_dim_mean']:.6f} ± {result['z0_stats']['var_per_dim_std']:.6f}")
            print(f"  var/dim (whit): {result['z0_tilde_stats']['var_per_dim_mean']:.4f} ± {result['z0_tilde_stats']['var_per_dim_std']:.4f}")

            # Save per-cell whitening matrix
            wt = WhiteningTransform(z0, use_ledoit_wolf=True)
            wt.save(os.path.join(OUT_DIR, f'W_{prompt}_gs{gs:.1f}.npz'.replace('.', 'p', 1)))

            all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Prompt':<10} {'gs':>6} {'SSA_orig':>10} {'LW_alpha':>10} {'cond':>10} {'cov_err':>10}")
    print("-" * 56)
    for r in all_results:
        ssa_str = f"{r['ssa_original']:.4f}" if r['ssa_original'] is not None else "N/A"
        print(f"{r['prompt']:<10} {r['guidance_scale']:>6.1f} {ssa_str:>10} "
              f"{r['diagnostics']['alpha_lw']:>10.4f} "
              f"{r['diagnostics']['cond_before']:>10.1f} "
              f"{r['cov_err']:>10.4f}")

    # Save combined results
    out_json = os.path.join(OUT_DIR, 'sdxl_guidance_whitening_analysis.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    print("\nNOTE: Full whitened SSA requires re-running Langevin in whitened")
    print("CLIP space. This script provides the whitening diagnostics and")
    print("W matrices. For full re-computation, use run_whitened_langevin.py.")


if __name__ == '__main__':
    main()
