#!/usr/bin/env python3
"""Bootstrap p-value for SDXL elderly DA1 against the anisotropic null pool.

Reads results/sdxl_null_anisotropic_combined.npz (combined z0, z_tau from the
array job) and the observed eigenvalues at tau* from the elderly free-conv run.

For b = 1..B_boot=10000:
  1. Sample 200 indices from {1, ..., N_big} with replacement.
  2. Center z0_sub = z0[idx] - mu_z0; z_tau_sub = z_tau[idx] - mu_z_tau.
     (mu_z0, mu_z_tau computed once over the full N_big null pool.)
  3. C_b = (z_tau_sub.T @ z0_sub) / 200.
  4. Eigendecompose C_b_sym = (C_b + C_b.T)/2, store top-10 eigenvalues by
     magnitude (descending).

Output:
  results/sdxl_null_bootstrap_eigenvalues.npz   # (B_boot, 10)
  results/sdxl_anisotropic_null_pvalues.json    # summary

Tail extrapolation: also fits a Gumbel to the lambda_1 bootstrap distribution
and reports a sensitivity p-value from the fit (in case lambda_1^obs sits
above all 10000 empirical draws).
"""
import argparse
import json
import os
import time

import numpy as np


def gumbel_fit_and_pvalue(samples, x_obs):
    """MLE Gumbel fit to `samples`; return P[X >= x_obs] under fitted Gumbel.

    Gumbel CDF: F(x) = exp(-exp(-(x-mu)/beta))
    MLE via scipy if available, else closed-form approximation.
    """
    try:
        from scipy.stats import gumbel_r
        params = gumbel_r.fit(samples)
        loc, scale = params
        # P(X >= x_obs) = 1 - F(x_obs)
        sf = float(gumbel_r.sf(x_obs, loc=loc, scale=scale))
        return {'loc': float(loc), 'scale': float(scale), 'p_tail': sf,
                'method': 'scipy.stats.gumbel_r MLE'}
    except ImportError:
        # Method-of-moments fallback (less accurate)
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        scale = std * np.sqrt(6) / np.pi
        loc = mean - scale * 0.5772  # Euler-Mascheroni
        z = (x_obs - loc) / scale
        sf = float(np.exp(-np.exp(-z))) if z < 30 else 0.0
        sf = 1 - sf  # P(X>=x) = 1 - F(x)
        return {'loc': loc, 'scale': scale, 'p_tail': sf,
                'method': 'method-of-moments (no scipy)'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined', type=str,
                        default='results/sdxl_null_anisotropic_combined.npz')
    parser.add_argument('--observed_npz', type=str,
                        default='../../sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz')
    parser.add_argument('--B_boot', type=int, default=10000)
    parser.add_argument('--n_resample', type=int, default=200,
                        help='Subsample size per bootstrap draw (matches real-data N=200).')
    parser.add_argument('--n_top_eigvals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260502)
    parser.add_argument('--out_eig', type=str,
                        default='results/sdxl_null_bootstrap_eigenvalues.npz')
    parser.add_argument('--out_json', type=str,
                        default='results/sdxl_anisotropic_null_pvalues.json')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    def resolve(p): return p if os.path.isabs(p) else os.path.join(here, p)

    combined = resolve(args.combined)
    observed = resolve(args.observed_npz)
    out_eig = resolve(args.out_eig)
    out_json = resolve(args.out_json)
    os.makedirs(os.path.dirname(out_eig), exist_ok=True)

    print(f"Loading combined null pool: {combined}")
    nd = np.load(combined)
    z0 = nd['z0'].astype(np.float64)
    z_tau = nd['z_tau'].astype(np.float64)
    N_big = z0.shape[0]
    D = z0.shape[1]
    print(f"  z0={z0.shape}, z_tau={z_tau.shape}, N_big={N_big}, D={D}")

    print(f"Loading observed eigenvalues: {observed}")
    obs = np.load(observed)
    obs_eigvals = obs['eigenvalues_full_at_tau_star']
    obs_top = obs_eigvals[:args.n_top_eigvals]
    lambda1_obs = float(obs_top[0])
    print(f"  lambda_1^obs = {lambda1_obs:.6e}")
    print(f"  top-{args.n_top_eigvals} obs eigvals: {obs_top}")

    # Center using full pool means.
    mu_z0 = z0.mean(axis=0)
    mu_z_tau = z_tau.mean(axis=0)
    z0_c_pool = z0 - mu_z0
    z_tau_c_pool = z_tau - mu_z_tau

    rng = np.random.default_rng(args.seed)
    boot_eigs = np.zeros((args.B_boot, args.n_top_eigvals), dtype=np.float64)
    print(f"\nBootstrapping B={args.B_boot} draws of size {args.n_resample}...")
    t0 = time.time()
    log_every = max(1, args.B_boot // 20)
    for b in range(args.B_boot):
        idx = rng.integers(0, N_big, size=args.n_resample)
        z0_sub = z0_c_pool[idx]
        ztau_sub = z_tau_c_pool[idx]
        C_b = ztau_sub.T @ z0_sub / args.n_resample
        Csym = 0.5 * (C_b + C_b.T)
        evals = np.linalg.eigvalsh(Csym)
        # Sort by magnitude (descending) — matches how observed eigvals were saved.
        order = np.argsort(-np.abs(evals))
        boot_eigs[b] = evals[order[:args.n_top_eigvals]]
        if (b + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (args.B_boot - b - 1) / rate
            print(f"  {b+1}/{args.B_boot} ({rate:.1f}/s, ETA {eta:.0f}s)")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    np.savez_compressed(out_eig,
                        boot_eigenvalues=boot_eigs,
                        observed_top=obs_top,
                        N_big=np.array(N_big),
                        n_resample=np.array(args.n_resample),
                        B_boot=np.array(args.B_boot))
    print(f"Saved {out_eig}")

    # Per-direction empirical p-values.
    # Note: we compare each obs eigvalue to its corresponding rank in the
    # bootstrap distribution; sign-aware comparison via abs() is consistent
    # with how observed eigs are stored (sorted by |.|).
    pvalues = {}
    for i in range(args.n_top_eigvals):
        obs_i = obs_top[i]
        boot_i = boot_eigs[:, i]
        ge_count = int((np.abs(boot_i) >= np.abs(obs_i)).sum())
        p_emp = (1 + ge_count) / (args.B_boot + 1)
        pvalues[f'lambda_{i+1}'] = {
            'observed': float(obs_i),
            'null_mean_signed': float(boot_i.mean()),
            'null_mean_abs': float(np.abs(boot_i).mean()),
            'null_max_abs': float(np.abs(boot_i).max()),
            'count_ge_obs': ge_count,
            'p_empirical': p_emp,
        }
        print(f"  lambda_{i+1}: obs={obs_i:+.4e}  null|.|_max={np.abs(boot_i).max():.4e}  "
              f"p_emp={p_emp:.4e}  count={ge_count}/{args.B_boot}")

    # Gumbel tail extrapolation for lambda_1 (signed values; lambda_1 expected positive).
    print("\nFitting Gumbel to lambda_1 bootstrap distribution for tail extrapolation...")
    g = gumbel_fit_and_pvalue(boot_eigs[:, 0], lambda1_obs)
    print(f"  Gumbel fit ({g['method']}): loc={g['loc']:.4e} scale={g['scale']:.4e}")
    print(f"  P(lambda_1_null >= lambda_1_obs) under fitted Gumbel = {g['p_tail']:.4e}")

    summary = {
        'N_big': N_big,
        'n_resample': args.n_resample,
        'B_boot': args.B_boot,
        'observed_top': [float(v) for v in obs_top],
        'lambda1_observed': lambda1_obs,
        'pvalues': pvalues,
        'gumbel_lambda1': g,
        'bonferroni_threshold_DA1_only': 0.05,
        'caveat': ('Bootstrap subsamples share the underlying N_big simulated '
                   'trajectories; the resulting p-value is conservative w.r.t. '
                   'between-simulation variance. Effective resolution ~1/B_boot '
                   '~= 1e-4 with B_boot=10000.'),
    }
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == '__main__':
    main()
