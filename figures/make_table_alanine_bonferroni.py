#!/usr/bin/env python3
"""
Bonferroni table for alanine eigenvalues at tau*=10000ps.

Reports lambda_i^obs for i=1..5, lambda_+(tau*), ratio, and pass/fail.
hat_m = maximal i such that ALL of lambda_1,...,lambda_i exceed lambda_+(tau*).

Output: figures/table_alanine_bonferroni.{md,csv}
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'noise_floor'))
from free_conv_utils import edges as fc_edges

FIGDIR = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(FIGDIR, '..', 'experiments', 'save_autocovariance',
                        'results', 'alanine_C_tau_full_h_half_10k.npz')

TAU_STAR_IDX = 37  # tau = 10000 ps


def main():
    data = np.load(DATAPATH)
    eigenvalues = data['eigenvalues']  # (38, 45), sorted descending
    sigma_f_sq = float(data['sigma_f_sq'])
    gamma = float(data['gamma'])
    phys_times = data['phys_times']

    sigma = np.sqrt(sigma_f_sq)
    tau_ps = phys_times[TAU_STAR_IDX]
    eigs = eigenvalues[TAU_STAR_IDX]  # 45 values, descending

    _, lp_tau = fc_edges(tau_ps, sigma, gamma=gamma)

    print(f'tau* = {tau_ps:.0f} ps')
    print(f'sigma_f^2 = {sigma_f_sq:.4e}')
    print(f'gamma = {gamma}')
    print(f'lambda_+(tau*) = {lp_tau:.4e}')
    print()

    # Determine hat_m
    hat_m = 0
    for i in range(5):
        if eigs[i] > lp_tau:
            hat_m = i + 1
        else:
            break

    # Build table
    rows = []
    for i in range(5):
        lam_obs = eigs[i]
        ratio = lam_obs / lp_tau
        passes = lam_obs > lp_tau
        rows.append((i + 1, lam_obs, lp_tau, ratio, passes))

    # Markdown
    md_lines = [
        f'| $i$ | $\\lambda_i^{{\\mathrm{{obs}}}}$ | $\\lambda_+(\\tau^*)$ | Ratio | Pass |',
        '|:---:|:---:|:---:|:---:|:---:|',
    ]
    for i, lam_obs, lp, ratio, passes in rows:
        pass_str = 'yes' if passes else 'no'
        md_lines.append(f'| {i} | {lam_obs:.4e} | {lp:.4e} | {ratio:.1f} | {pass_str} |')
    md_lines.append(f'\n$\\hat{{m}} = {hat_m}$')

    md_path = os.path.join(FIGDIR, 'table_alanine_bonferroni.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'Saved: {md_path}')

    # CSV
    csv_lines = ['i,lambda_obs,lambda_plus_tau,ratio,pass']
    for i, lam_obs, lp, ratio, passes in rows:
        csv_lines.append(f'{i},{lam_obs:.6e},{lp:.6e},{ratio:.2f},{"yes" if passes else "no"}')
    csv_lines.append(f'# hat_m = {hat_m}')

    csv_path = os.path.join(FIGDIR, 'table_alanine_bonferroni.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f'Saved: {csv_path}')

    # Print
    print()
    for line in md_lines:
        print(line)


if __name__ == '__main__':
    main()
