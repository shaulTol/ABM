#!/usr/bin/env python3
"""
Table 2: Alanine dipeptide DA lag sensitivity.

Reports DA1/DA2 correlations with ground-truth dihedrals (ψ, φ) across
a range of lags τ, plus eigenvalue gap ratios and DA1 angular stability.

Reads: results/alanine_C_tau_full_h_half_10k.npz + raw alanine data.
Output: prints LaTeX table rows for Table 2.
"""
import sys, os
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.noise_floor.free_conv_utils import edges

EXPDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(EXPDIR, 'results', 'alanine_C_tau_full_h_half_10k.npz')
DATA_DIR = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')


def load_alanine_gt():
    """Load ground-truth dihedrals."""
    dihed_file = os.path.join(DATA_DIR,
                              'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
    dist_file = os.path.join(DATA_DIR,
                             'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed = np.load(dihed_file)
    dists = np.load(dist_file)
    all_diheds = np.concatenate([dihed[f'arr_{i}'] for i in range(3)])
    all_dists = np.concatenate([dists[f'arr_{i}'] for i in range(3)])
    return all_dists, all_diheds[:, 0], all_diheds[:, 1]  # features, phi, psi


def main():
    data = np.load(DATAPATH)
    C_tau_all = data['C_tau_all']       # (n_lags, d, d)
    eigenvalues = data['eigenvalues']   # (n_lags, d)
    phys_times = data['phys_times']
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    d = int(data['d'])
    sigma = np.sqrt(sigma_f_sq)

    all_dists, gt_phi, gt_psi = load_alanine_gt()

    # Reference DA1 at the largest lag
    ref_idx = len(phys_times) - 1
    evals_ref, evecs_ref = np.linalg.eigh(C_tau_all[ref_idx])
    order_ref = np.argsort(evals_ref)[::-1].copy()
    da1_ref = evecs_ref[:, order_ref[0]]

    print(f"{'tau_ps':>10} {'lam1':>10} {'lam2':>10} {'lp':>10} "
          f"{'gap1':>6} {'gap2':>6} "
          f"{'DA1_psi':>8} {'DA1_phi':>8} {'DA2_psi':>8} {'DA2_phi':>8} "
          f"{'angle':>6}")
    print("-" * 105)

    for i in range(len(phys_times)):
        tau = phys_times[i]
        _, lp = edges(tau, sigma, gamma=gamma)

        evals, evecs = np.linalg.eigh(C_tau_all[i])
        order = np.argsort(evals)[::-1].copy()
        evals = evals[order]
        evecs = evecs[:, order]

        gap1 = evals[0] / lp if lp > 0 else float('inf')
        gap2 = evals[1] / lp if lp > 0 else float('inf')

        # Project data onto DA1 and DA2
        proj1 = all_dists @ evecs[:, 0]
        proj2 = all_dists @ evecs[:, 1]

        r_da1_psi = abs(pearsonr(proj1, gt_psi)[0])
        r_da1_phi = abs(pearsonr(proj1, gt_phi)[0])
        r_da2_psi = abs(pearsonr(proj2, gt_psi)[0])
        r_da2_phi = abs(pearsonr(proj2, gt_phi)[0])

        # Angle to reference DA1
        cos_angle = abs(np.dot(evecs[:, 0], da1_ref))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        print(f"{tau:10.1f} {evals[0]:10.4e} {evals[1]:10.4e} {lp:10.4e} "
              f"{gap1:6.1f} {gap2:6.1f} "
              f"{r_da1_psi:8.3f} {r_da1_phi:8.3f} {r_da2_psi:8.3f} {r_da2_phi:8.3f} "
              f"{angle_deg:6.1f}")


if __name__ == '__main__':
    main()
