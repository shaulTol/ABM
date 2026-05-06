#!/usr/bin/env python3
"""
Merge all MC null + dip test results into a summary table and 1D histogram figures.

Reads:
  results/ellipses_null_results.npz   (from run_ellipses_null.py)
  results/alanine_null_task*.npz      (from run_alanine_null.py, merged)
  results/sdxl_null_task*.npz         (from run_sdxl_null.py, merged)
  results/alanine_dip_results.npz     (from run_dip_tests.py)
  results/sdxl_dip_results.npz        (from run_dip_tests.py)

Plus observed eigenvalues from canonical data files.

Outputs:
  results/mc_dip_summary.csv
  results/mc_dip_summary.txt
  results/hist_*.png (1D histograms with Gaussian overlay)
"""
import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def load_observed_alanine():
    """Load observed eigenvalues from h/2 C(τ) at τ=10,000ps."""
    ctau = np.load(os.path.join(project_root,
        'experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz'))
    evals = ctau['eigenvalues'][-1]  # last lag, (45,)
    # Sort descending (eigenvalues stored ascending from eigh)
    evals_desc = np.sort(evals)[::-1].copy()
    return evals_desc


def load_observed_sdxl():
    """Load observed eigenvalues from canonical SDXL elderly DA run."""
    npz = np.load(os.path.join(project_root,
        'experiments/sdxl/results_da_pcavda/clip_da_elderly_gs7p5_fc_dt0p0025.npz'))
    return npz['eigenvalues_full_at_tau_star']  # (768,) descending


def merge_null_files(pattern):
    """Merge array job results into single array."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    all_evals = []
    for f in files:
        d = np.load(f)
        all_evals.append(d['null_eigenvalues'])
    return np.concatenate(all_evals, axis=0)  # (B, n_eigvals)


def compute_pvalue(obs_i, null_i):
    """One-sided MC p-value: P(null >= obs)."""
    B = len(null_i)
    return (1 + np.sum(null_i >= obs_i)) / (1 + B)


def plot_histogram(proj, direction_name, experiment, output_dir, gaussian_overlay=True):
    """Plot 1D histogram of projection with Gaussian overlay."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(proj, bins=80, density=True, alpha=0.6, color='steelblue',
            edgecolor='none', label='Data')

    if gaussian_overlay:
        mu, sigma = proj.mean(), proj.std()
        x = np.linspace(proj.min(), proj.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=1.5,
                label=f'N({mu:.2f}, {sigma:.2f}²)')

    ax.set_xlabel(f'Projection onto {direction_name}')
    ax.set_ylabel('Density')
    ax.set_title(f'{experiment}: {direction_name}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f'hist_{experiment.lower().replace(" ", "_")}_{direction_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    results_dir = args.results_dir
    rows = []

    # ================================================================
    # 1. Parallel Ellipses
    # ================================================================
    ell_path = os.path.join(results_dir, 'ellipses_null_results.npz')
    if os.path.exists(ell_path):
        print("=== Parallel Ellipses ===")
        ell = np.load(ell_path)

        rows.append({
            'Experiment': 'Parallel ellipses',
            'Direction': 'DA1',
            'lambda_obs': ell['obs_eigenvalues'][0],
            'MC_p': ell['p_values'][0],
            'Dip_stat': float(ell['dip_da1']),
            'Dip_p': float(ell['pval_da1']),
            'Bonf_alpha': 0.05,
            'Pass': (ell['p_values'][0] < 0.05) and (float(ell['pval_da1']) < 0.05),
        })
        rows.append({
            'Experiment': 'Parallel ellipses',
            'Direction': 'PC1 (ctrl)',
            'lambda_obs': np.nan,
            'MC_p': np.nan,
            'Dip_stat': float(ell['dip_pc1']),
            'Dip_p': float(ell['pval_pc1']),
            'Bonf_alpha': 0.05,
            'Pass': None,  # control
        })

        # Histograms
        plot_histogram(ell['proj_da1'], 'DA1', 'Ellipses', results_dir)
        plot_histogram(ell['proj_pc1'], 'PC1', 'Ellipses', results_dir)
    else:
        print(f"Missing {ell_path}")

    # ================================================================
    # 2. Alanine
    # ================================================================
    print("\n=== Alanine ===")
    obs_ala = load_observed_alanine()
    print(f"  Observed λ₁={obs_ala[0]:.6f}, λ₂={obs_ala[1]:.6f}")

    # Merge null
    null_ala = merge_null_files(os.path.join(results_dir, 'alanine_null_task*.npz'))
    ala_dip_path = os.path.join(results_dir, 'alanine_dip_results.npz')

    if null_ala is not None:
        B = null_ala.shape[0]
        print(f"  Null replicates: B={B}")
        for i, name in enumerate(['DA1', 'DA2']):
            if i < null_ala.shape[1]:
                p = compute_pvalue(obs_ala[i], null_ala[:, i])
                print(f"  {name}: obs λ={obs_ala[i]:.6f}, null median={np.median(null_ala[:,i]):.6f}, p={p:.4f}")

                row = {
                    'Experiment': 'Alanine',
                    'Direction': name,
                    'lambda_obs': obs_ala[i],
                    'MC_p': p,
                    'Dip_stat': np.nan,
                    'Dip_p': np.nan,
                    'Bonf_alpha': 0.025,
                    'Pass': None,
                }
                # Fill in dip if available
                if os.path.exists(ala_dip_path):
                    dip_data = np.load(ala_dip_path)
                    dip_key = f'dip_{name.lower()}'
                    pval_key = f'pval_{name.lower()}'
                    if dip_key in dip_data:
                        row['Dip_stat'] = float(dip_data[dip_key])
                        row['Dip_p'] = float(dip_data[pval_key])
                        row['Pass'] = (p < 0.025) and (float(dip_data[pval_key]) < 0.025)
                rows.append(row)
    else:
        print("  No null results found (run cluster jobs first)")

    # Alanine PC1 control (dip only)
    if os.path.exists(ala_dip_path):
        dip_data = np.load(ala_dip_path)
        rows.append({
            'Experiment': 'Alanine',
            'Direction': 'PC1 (ctrl)',
            'lambda_obs': np.nan,
            'MC_p': np.nan,
            'Dip_stat': float(dip_data['dip_pc1']),
            'Dip_p': float(dip_data['pval_pc1']),
            'Bonf_alpha': 0.025,
            'Pass': None,
        })

        # Histograms
        plot_histogram(dip_data['proj_da1'], 'DA1', 'Alanine', results_dir)
        plot_histogram(dip_data['proj_da2'], 'DA2', 'Alanine', results_dir)
        plot_histogram(dip_data['proj_pc1'], 'PC1', 'Alanine', results_dir)

    # ================================================================
    # 3. SDXL Elderly
    # ================================================================
    print("\n=== SDXL Elderly ===")
    obs_sdxl = load_observed_sdxl()
    print(f"  Observed λ₁={obs_sdxl[0]:.6f}")

    null_sdxl = merge_null_files(os.path.join(results_dir, 'sdxl_null_task*.npz'))

    sdxl_dip_path = os.path.join(results_dir, 'sdxl_dip_results.npz')

    sdxl_mc_p = np.nan
    if null_sdxl is not None:
        B = null_sdxl.shape[0]
        print(f"  Null replicates: B={B}")
        sdxl_mc_p = compute_pvalue(obs_sdxl[0], null_sdxl[:, 0])
        print(f"  DA1: obs λ₁={obs_sdxl[0]:.6f}, null median={np.median(null_sdxl[:,0]):.6f}, p={sdxl_mc_p:.4f}")
    else:
        print("  No null results found (run cluster jobs first)")

    sdxl_dip_stat = np.nan
    sdxl_dip_p = np.nan
    if os.path.exists(sdxl_dip_path):
        sdxl_dip = np.load(sdxl_dip_path)
        sdxl_dip_stat = float(sdxl_dip['dip_da1'])
        sdxl_dip_p = float(sdxl_dip['pval_da1'])

        # Histograms
        plot_histogram(sdxl_dip['da1_scores'], 'DA1', 'SDXL_elderly', results_dir)
        plot_histogram(sdxl_dip['pc1_scores'], 'PC1', 'SDXL_elderly', results_dir)

    rows.append({
        'Experiment': 'SDXL elderly',
        'Direction': 'DA1',
        'lambda_obs': obs_sdxl[0],
        'MC_p': sdxl_mc_p,
        'Dip_stat': sdxl_dip_stat,
        'Dip_p': sdxl_dip_p,
        'Bonf_alpha': 0.05,
        'Pass': (not np.isnan(sdxl_mc_p) and sdxl_mc_p < 0.05 and
                 not np.isnan(sdxl_dip_p) and sdxl_dip_p < 0.05)
                if not (np.isnan(sdxl_mc_p) and np.isnan(sdxl_dip_p)) else None,
    })

    # ================================================================
    # Print and save table
    # ================================================================
    print(f"\n{'='*90}")
    header = f"{'Experiment':<20} {'Direction':<12} {'λ_obs':>10} {'MC p':>10} {'Dip stat':>10} {'Dip p':>10} {'α':>6} {'Pass':>6}"
    print(header)
    print('-' * 90)

    for r in rows:
        lam = f"{r['lambda_obs']:.6f}" if not np.isnan(r['lambda_obs']) else 'N/A'
        mc_p = f"{r['MC_p']:.4f}" if not np.isnan(r['MC_p']) else 'N/A'
        dip_s = f"{r['Dip_stat']:.4f}" if not np.isnan(r['Dip_stat']) else 'N/A'
        dip_p = f"{r['Dip_p']:.2e}" if not np.isnan(r['Dip_p']) else 'N/A'
        alpha = f"{r['Bonf_alpha']:.3f}"
        passed = 'Yes' if r['Pass'] is True else ('No' if r['Pass'] is False else 'ctrl')
        print(f"{r['Experiment']:<20} {r['Direction']:<12} {lam:>10} {mc_p:>10} {dip_s:>10} {dip_p:>10} {alpha:>6} {passed:>6}")

    print(f"{'='*90}")

    # Save CSV
    csv_path = os.path.join(results_dir, 'mc_dip_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('Experiment,Direction,lambda_obs,MC_p,Dip_stat,Dip_p,Bonf_alpha,Pass\n')
        for r in rows:
            lam = f"{r['lambda_obs']:.6f}" if not np.isnan(r['lambda_obs']) else ''
            mc_p = f"{r['MC_p']:.4f}" if not np.isnan(r['MC_p']) else ''
            dip_s = f"{r['Dip_stat']:.4f}" if not np.isnan(r['Dip_stat']) else ''
            dip_p = f"{r['Dip_p']:.2e}" if not np.isnan(r['Dip_p']) else ''
            passed = str(r['Pass']) if r['Pass'] is not None else 'ctrl'
            f.write(f"{r['Experiment']},{r['Direction']},{lam},{mc_p},{dip_s},{dip_p},{r['Bonf_alpha']},{passed}\n")
    print(f"\nSaved {csv_path}")


if __name__ == '__main__':
    main()
