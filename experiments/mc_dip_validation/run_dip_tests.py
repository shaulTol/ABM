#!/usr/bin/env python3
"""
Dip Tests on DA/PCA Projections for Alanine and SDXL.

Runs Hartigan's dip test to check whether 1D projections onto DA/PCA
directions are non-unimodal. LOCAL script.

Alanine: Projects 750k MD frames onto DA1/DA2/PC1 from h/2 C(τ) at τ=10,000ps.
SDXL elderly: Uses pre-computed DA1 projections (N=200) from the canonical run.
"""
import sys
import os
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import diptest


def run_alanine_dip(results_dir):
    """Dip test on alanine DA1, DA2, and PC1 projections."""
    # Load C(τ) data with eigenvectors
    ctau_path = os.path.join(project_root,
        'experiments/save_autocovariance/results/alanine_C_tau_full_h_half_10k.npz')
    ctau = np.load(ctau_path)

    # Find τ=10,000ps (last lag)
    phys_times = ctau['phys_times']
    lags = ctau['lags']
    C_tau_all = ctau['C_tau']  # (38, 45, 45)
    C0 = ctau['C0']  # (45, 45)

    # Use last lag (τ=10,000ps)
    tau_idx = len(phys_times) - 1
    tau_ps = phys_times[tau_idx]
    print(f"Alanine: using τ={tau_ps:.0f}ps (lag index {tau_idx})")

    C_tau = C_tau_all[tau_idx]  # (45, 45)
    C_tau_sym = 0.5 * (C_tau + C_tau.T)

    # Eigendecompose C(τ) for DA directions
    evals, evecs = np.linalg.eigh(C_tau_sym)
    idx_desc = np.argsort(evals)[::-1].copy()
    evals_desc = evals[idx_desc]
    evecs_desc = evecs[:, idx_desc]

    da1 = evecs_desc[:, 0]
    da2 = evecs_desc[:, 1]

    # PCA: eigendecompose C(0)
    C0_sym = 0.5 * (C0 + C0.T).astype(np.float64)
    pca_evals, pca_evecs = np.linalg.eigh(C0_sym)
    pca_idx = np.argsort(pca_evals)[::-1].copy()
    pc1 = pca_evecs[:, pca_idx[0]]

    # Load alanine data (750k frames, 45D)
    print("Loading alanine data...")
    import mdshare
    local_filename = mdshare.fetch(
        'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
        working_directory='data'
    )
    with np.load(local_filename) as fh:
        data = np.concatenate([fh[f'arr_{i}'] for i in range(3)])  # (750000, 45)

    gt_phi = ctau['gt_phi']  # (750000,)
    gt_psi = ctau['gt_psi']  # (750000,)

    # Sign-flip DA1 to match TICA1 convention (positive correlation with psi)
    proj_da1 = data @ da1
    if np.corrcoef(proj_da1, gt_psi)[0, 1] < 0:
        da1 = -da1
        proj_da1 = -proj_da1

    proj_da2 = data @ da2
    if np.corrcoef(proj_da2, gt_phi)[0, 1] < 0:
        da2 = -da2
        proj_da2 = -proj_da2

    proj_pc1 = data @ pc1

    # Dip tests
    # Bonferroni correction for 2 DA directions: α = 0.025
    dip_da1, pval_da1 = diptest.diptest(proj_da1)
    dip_da2, pval_da2 = diptest.diptest(proj_da2)
    dip_pc1, pval_pc1 = diptest.diptest(proj_pc1)

    # Correlations with CVs
    r_da1_psi = np.abs(np.corrcoef(proj_da1, gt_psi)[0, 1])
    r_da1_phi = np.abs(np.corrcoef(proj_da1, gt_phi)[0, 1])
    r_da2_psi = np.abs(np.corrcoef(proj_da2, gt_psi)[0, 1])
    r_da2_phi = np.abs(np.corrcoef(proj_da2, gt_phi)[0, 1])

    print(f"\nAlanine Dip Tests (N={data.shape[0]}):")
    print(f"  DA1: dip={dip_da1:.4f}, p={pval_da1:.2e}, |r(ψ)|={r_da1_psi:.3f}")
    print(f"  DA2: dip={dip_da2:.4f}, p={pval_da2:.2e}, |r(φ)|={r_da2_phi:.3f}")
    print(f"  PC1 (control): dip={dip_pc1:.4f}, p={pval_pc1:.2e}")

    # Save
    out_path = os.path.join(results_dir, 'alanine_dip_results.npz')
    np.savez(out_path,
             da1=da1, da2=da2, pc1=pc1,
             eigenvalues_da=evals_desc[:5],
             eigenvalues_pca=pca_evals[pca_idx[:5]],
             dip_da1=dip_da1, pval_da1=pval_da1,
             dip_da2=dip_da2, pval_da2=pval_da2,
             dip_pc1=dip_pc1, pval_pc1=pval_pc1,
             proj_da1=proj_da1, proj_da2=proj_da2, proj_pc1=proj_pc1,
             r_da1_psi=r_da1_psi, r_da1_phi=r_da1_phi,
             r_da2_psi=r_da2_psi, r_da2_phi=r_da2_phi,
             tau_ps=tau_ps, N=data.shape[0], d=data.shape[1])
    print(f"Saved {out_path}")

    return {
        'dip_da1': dip_da1, 'pval_da1': pval_da1,
        'dip_da2': dip_da2, 'pval_da2': pval_da2,
        'dip_pc1': dip_pc1, 'pval_pc1': pval_pc1,
    }


def run_sdxl_dip(results_dir):
    """Dip test on SDXL elderly DA1/PC1 projections."""
    # Load pre-computed projections from figure script
    indices_path = os.path.join(project_root,
        'experiments/sdxl/results_da_pcavda/fig_elderly_da_vs_pca_freeconv_taustar_indices.npz')
    idx = np.load(indices_path)

    da1_scores = idx['da1_scores']  # (200,)
    pc1_scores = idx['pc1_scores']  # (200,)

    print(f"\nSDXL Elderly Dip Tests (N={len(da1_scores)}):")

    dip_da1, pval_da1 = diptest.diptest(da1_scores)
    dip_pc1, pval_pc1 = diptest.diptest(pc1_scores)

    print(f"  DA1: dip={dip_da1:.4f}, p={pval_da1:.2e}")
    print(f"  PC1 (control): dip={dip_pc1:.4f}, p={pval_pc1:.2e}")

    # Also check DA2 if available
    da2_scores = idx.get('da2_scores', None)
    dip_da2 = pval_da2 = None
    if da2_scores is not None:
        dip_da2, pval_da2 = diptest.diptest(da2_scores)
        print(f"  DA2: dip={dip_da2:.4f}, p={pval_da2:.2e}")

    # Save
    out_path = os.path.join(results_dir, 'sdxl_dip_results.npz')
    save_dict = dict(
        da1_scores=da1_scores, pc1_scores=pc1_scores,
        dip_da1=dip_da1, pval_da1=pval_da1,
        dip_pc1=dip_pc1, pval_pc1=pval_pc1,
        N=len(da1_scores),
    )
    if da2_scores is not None:
        save_dict.update(da2_scores=da2_scores, dip_da2=dip_da2, pval_da2=pval_da2)
    np.savez(os.path.join(results_dir, 'sdxl_dip_results.npz'), **save_dict)
    print(f"Saved {os.path.join(results_dir, 'sdxl_dip_results.npz')}")

    return {
        'dip_da1': dip_da1, 'pval_da1': pval_da1,
        'dip_pc1': dip_pc1, 'pval_pc1': pval_pc1,
        'dip_da2': dip_da2, 'pval_da2': pval_da2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--skip_alanine', action='store_true')
    parser.add_argument('--skip_sdxl', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_alanine:
        run_alanine_dip(args.output_dir)

    if not args.skip_sdxl:
        run_sdxl_dip(args.output_dir)


if __name__ == '__main__':
    main()
