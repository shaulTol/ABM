#!/usr/bin/env python3
"""
Alanine dipeptide DA lag sensitivity analysis.

Uses pre-computed eigenvectors from the KDE gap profile (28 geometric
checkpoints) and the noise floor analysis. No rerunning of Langevin.

Reports:
1. Top-2 eigenvalues, noise floor bounds, DA1/DA2 correlations with
   ground truth (ψ, φ), and angle to DA1 at τ*=130 ps — at each lag.
2. Bootstrap SEs for the four headline correlations at τ*=130 ps
   (resampling frames from the 750k alanine dataset).

Output:
  results_alanine_lag_sensitivity.csv
  results_alanine_correlation_bootstrap.csv
  SUMMARY.md
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import json
import csv
import numpy as np
from scipy.stats import pearsonr

EXPDIR = os.path.dirname(os.path.abspath(__file__))
DA_DIR = os.path.join(project_root, 'experiments', 'alanine', 'main')
RESDIR = os.path.join(DA_DIR, 'results')

N_BOOT = 5000
RNG = np.random.RandomState(42)


def load_alanine_gt():
    """Load ground truth dihedrals from GMM results file."""
    gmm_file = os.path.join(RESDIR, 'alanine_da_results_gmm_only.npz')
    gmm = np.load(gmm_file, allow_pickle=True)
    return gmm['gt_phi'], gmm['gt_psi']


def load_alanine_features():
    """Load 45D pairwise distance features."""
    thesis_data = os.path.expanduser(
        '~/github/thesis/modality_score/autocorrelation-mixture-analysis/'
        'experiments/da_directions/alanine_dipeptide/data')
    local_data = os.path.join(DA_DIR, 'alanine_dipeptide', 'data')

    for dd in [thesis_data, local_data]:
        dist_f = os.path.join(dd, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
        if os.path.exists(dist_f):
            break
    else:
        raise FileNotFoundError('Alanine distance data not found')

    dist_data = np.load(dist_f)
    all_dists = np.concatenate([dist_data[f'arr_{i}'] for i in range(3)])
    return all_dists


def main():
    print('Alanine Dipeptide DA Lag Sensitivity Analysis')
    print('=' * 70)

    # ── Load data ──
    phi, psi = load_alanine_gt()
    all_dists = load_alanine_features()
    data_mean = all_dists.mean(axis=0)
    all_centered = all_dists - data_mean
    N = len(phi)
    print(f'Loaded: {N:,} frames, 45D features')

    # ── Load KDE gap profile (eigenvectors at 28 checkpoints) ──
    kde = np.load(os.path.join(RESDIR, 'alanine_kde_gap_profile.npz'), allow_pickle=True)
    phys_times = kde['phys_times']
    all_evals = kde['all_eigenvalues']       # (28, 45)
    lead_evecs = kde['leading_eigenvectors']  # (28, 45) — DA1 only

    # We need DA2 as well — check if top_k are stored
    # The KDE gap profile only stores leading_eigenvectors (DA1).
    # Use GMM file for top-k eigenvectors.
    gmm = np.load(os.path.join(RESDIR, 'alanine_da_results_gmm_only.npz'), allow_pickle=True)
    gmm_dt = float(gmm['dt'])
    gmm_cp = gmm['conv_checkpoints']
    gmm_phys = gmm_cp * gmm_dt
    gmm_topk = gmm['conv_top_k_eigenvectors']  # (30, 45, 5)
    gmm_evals = gmm['conv_eigenvalues']         # (30, 45)
    gmm_gap12 = gmm['conv_gap_profile']
    gmm_gap23 = gmm['conv_gap23_profile']

    # ── Load noise floor ──
    nf_file = os.path.join(project_root, 'experiments', 'noise_floor',
                           'alanine', 'noise_floor_alanine.json')
    with open(nf_file) as f:
        nf = json.load(f)
    nf_phys = np.array(nf['phys_times'])
    nf_floor_upper = np.array(nf['floor_upper'])   # conservative
    nf_floor_lower = np.array(nf['floor_lower'])   # aggressive

    # ── Select lags ──
    # Target: 70, 100, 130, 160, 200, 260 ps
    # Map to nearest GMM checkpoint (denser near these ranges)
    target_lags_ps = [70, 100, 130, 180, 260]
    ref_lag_ps = 130  # reference lag for angle computation

    print(f'\nTarget lags: {target_lags_ps} ps')
    print(f'Reference lag: {ref_lag_ps} ps')

    # Find nearest GMM checkpoints
    selected = []
    for tgt in target_lags_ps:
        idx = np.argmin(np.abs(gmm_phys - tgt))
        selected.append(idx)
    print(f'Selected GMM checkpoint indices: {selected}')
    print(f'Actual physical times: {[f"{gmm_phys[i]:.1f}" for i in selected]} ps')

    # Reference DA1 at τ*=130 ps
    ref_idx = np.argmin(np.abs(gmm_phys - ref_lag_ps))
    ref_da1 = gmm_topk[ref_idx, :, 0]
    ref_da1 /= np.linalg.norm(ref_da1)
    print(f'Reference: idx={ref_idx}, phys={gmm_phys[ref_idx]:.1f} ps')

    # Also get noise floor at each checkpoint (match phys times)
    def get_noise_floor(phys_t):
        idx = np.argmin(np.abs(nf_phys - phys_t))
        return nf_floor_upper[idx], nf_floor_lower[idx]

    # ── Compute correlations at each lag ──
    print(f'\n{"τ (ps)":>8} {"λ1":>10} {"λ2":>10} {"floor+":>8} {"floor-":>8} '
          f'{"DA1|r(ψ)|":>10} {"DA1|r(φ)|":>10} {"DA2|r(ψ)|":>10} {"DA2|r(φ)|":>10} '
          f'{"∠(DA1,ref)":>10} {"gap12":>7}')
    print('-' * 110)

    lag_rows = []
    for i, idx in enumerate(selected):
        phys_t = gmm_phys[idx]
        evals = gmm_evals[idx]
        # Sort descending
        sort_idx = np.argsort(evals)[::-1].copy()
        evals_sorted = evals[sort_idx]
        lam1, lam2 = evals_sorted[0], evals_sorted[1]

        da1 = gmm_topk[idx, :, 0]
        da2 = gmm_topk[idx, :, 1]
        da1 /= np.linalg.norm(da1)
        da2 /= np.linalg.norm(da2)

        proj_da1 = all_centered @ da1
        proj_da2 = all_centered @ da2

        r_da1_psi = abs(pearsonr(proj_da1, psi)[0])
        r_da1_phi = abs(pearsonr(proj_da1, phi)[0])
        r_da2_psi = abs(pearsonr(proj_da2, psi)[0])
        r_da2_phi = abs(pearsonr(proj_da2, phi)[0])

        # Angle to reference DA1
        cos_angle = abs(np.dot(da1, ref_da1))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))

        gap12 = gmm_gap12[idx]
        floor_upper, floor_lower = get_noise_floor(phys_t)

        # Check: is λ2 above conservative floor?
        lam2_above_floor = lam2 > floor_upper

        print(f'{phys_t:>8.1f} {lam1:>10.6f} {lam2:>10.6f} {floor_upper:>8.6f} {floor_lower:>8.6f} '
              f'{r_da1_psi:>10.4f} {r_da1_phi:>10.4f} {r_da2_psi:>10.4f} {r_da2_phi:>10.4f} '
              f'{angle_deg:>10.1f} {gap12:>7.1f}'
              f'{"" if lam2_above_floor else "  ← λ2 below floor"}')

        lag_rows.append({
            'tau_ps': phys_t,
            'lambda_1': lam1,
            'lambda_2': lam2,
            'floor_conservative': floor_upper,
            'floor_aggressive': floor_lower,
            'da1_r_psi': r_da1_psi,
            'da1_r_phi': r_da1_phi,
            'da2_r_psi': r_da2_psi,
            'da2_r_phi': r_da2_phi,
            'angle_da1_ref_deg': angle_deg,
            'gap12': gap12,
            'lambda2_above_conservative_floor': lam2_above_floor,
        })

    # Save lag sensitivity CSV
    csv_path = os.path.join(EXPDIR, 'results_alanine_lag_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=lag_rows[0].keys())
        writer.writeheader()
        writer.writerows(lag_rows)
    print(f'\nSaved: {csv_path}')

    # ── Bootstrap at reference lag ──
    print(f'\nBootstrap at τ*={gmm_phys[ref_idx]:.1f} ps ({N_BOOT} replicates)...')

    da1_ref = gmm_topk[ref_idx, :, 0]
    da2_ref = gmm_topk[ref_idx, :, 1]
    da1_ref /= np.linalg.norm(da1_ref)
    da2_ref /= np.linalg.norm(da2_ref)

    proj_da1_all = all_centered @ da1_ref
    proj_da2_all = all_centered @ da2_ref

    # Point estimates
    r_da1_psi_pt = abs(pearsonr(proj_da1_all, psi)[0])
    r_da1_phi_pt = abs(pearsonr(proj_da1_all, phi)[0])
    r_da2_psi_pt = abs(pearsonr(proj_da2_all, psi)[0])
    r_da2_phi_pt = abs(pearsonr(proj_da2_all, phi)[0])

    # Bootstrap: resample frames
    boot_r = np.zeros((N_BOOT, 4))
    for b in range(N_BOOT):
        idx_b = RNG.randint(0, N, size=N)
        boot_r[b, 0] = abs(pearsonr(proj_da1_all[idx_b], psi[idx_b])[0])
        boot_r[b, 1] = abs(pearsonr(proj_da1_all[idx_b], phi[idx_b])[0])
        boot_r[b, 2] = abs(pearsonr(proj_da2_all[idx_b], psi[idx_b])[0])
        boot_r[b, 3] = abs(pearsonr(proj_da2_all[idx_b], phi[idx_b])[0])

    labels = ['DA1 |r(ψ)|', 'DA1 |r(φ)|', 'DA2 |r(ψ)|', 'DA2 |r(φ)|']
    points = [r_da1_psi_pt, r_da1_phi_pt, r_da2_psi_pt, r_da2_phi_pt]

    print(f'\n{"Metric":>15} {"Point":>8} {"SE":>8} {"95% CI":>20}')
    print('-' * 55)
    boot_rows = []
    for j, (lab, pt) in enumerate(zip(labels, points)):
        se = boot_r[:, j].std()
        ci_lo, ci_hi = np.percentile(boot_r[:, j], [2.5, 97.5])
        print(f'{lab:>15} {pt:>8.4f} {se:>8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]')
        boot_rows.append({
            'metric': lab,
            'point_estimate': pt,
            'se': se,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'tau_ps': gmm_phys[ref_idx],
        })

    boot_csv = os.path.join(EXPDIR, 'results_alanine_correlation_bootstrap.csv')
    with open(boot_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=boot_rows[0].keys())
        writer.writeheader()
        writer.writerows(boot_rows)
    print(f'\nSaved: {boot_csv}')

    # ── Write SUMMARY.md ──
    write_summary(lag_rows, boot_rows, gmm_phys[ref_idx])
    print('Done.')


def write_summary(lag_rows, boot_rows, ref_tau):
    lines = []
    lines.append('# Task 2: Alanine Dipeptide DA Lag Sensitivity')
    lines.append('')
    lines.append('## Method')
    lines.append('')
    lines.append(f'Pre-computed eigenvectors from the KDE-based convergence profile ')
    lines.append(f'(28 geometric checkpoints, dt=0.1 ps, N=2000 trajectories, ')
    lines.append(f'Scott bandwidth h=0.018). Eigenvectors from GMM-DA profile ')
    lines.append(f'(30 checkpoints, dt=0.0001, N=5000) used for DA1/DA2 at each lag.')
    lines.append(f'')
    lines.append(f'Noise floor bounds from Appendix H: conservative ')
    lines.append(f'λ⁺_floor = σ²_f · (1+√(d/N))² and aggressive ')
    lines.append(f'λ⁻_floor = σ²_f · (1-√(d/N))².')
    lines.append(f'')
    lines.append(f'Bootstrap SEs at τ*={ref_tau:.1f} ps: 5000 replicates, ')
    lines.append(f'resampling the 750,000 alanine frames with replacement.')
    lines.append('')
    lines.append('## Lag Sensitivity Table')
    lines.append('')
    lines.append('| τ (ps) | λ₁ | λ₂ | floor⁺ | gap | DA1 \\|r(ψ)\\| | DA1 \\|r(φ)\\| | DA2 \\|r(ψ)\\| | DA2 \\|r(φ)\\| | ∠(DA1,ref) | λ₂>floor? |')
    lines.append('|--------|-----|-----|--------|-----|-----------|-----------|-----------|-----------|------------|-----------|')
    for r in lag_rows:
        floor_ok = 'Yes' if r['lambda2_above_conservative_floor'] else '**No**'
        lines.append(
            f'| {r["tau_ps"]:.1f} '
            f'| {r["lambda_1"]:.6f} '
            f'| {r["lambda_2"]:.6f} '
            f'| {r["floor_conservative"]:.6f} '
            f'| {r["gap12"]:.1f}× '
            f'| {r["da1_r_psi"]:.4f} '
            f'| {r["da1_r_phi"]:.4f} '
            f'| {r["da2_r_psi"]:.4f} '
            f'| {r["da2_r_phi"]:.4f} '
            f'| {r["angle_da1_ref_deg"]:.1f}° '
            f'| {floor_ok} |')

    lines.append('')
    lines.append('## Bootstrap SEs at τ*={:.1f} ps'.format(ref_tau))
    lines.append('')
    lines.append('| Metric | Point estimate | SE | 95% CI |')
    lines.append('|--------|---------------|-----|--------|')
    for r in boot_rows:
        lines.append(f'| {r["metric"]} | {r["point_estimate"]:.4f} '
                     f'| {r["se"]:.4f} | [{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}] |')

    lines.append('')
    lines.append('## Assessment')
    lines.append('')

    # Check ordering stability
    psi_dominant = all(r['da1_r_psi'] > r['da1_r_phi'] for r in lag_rows)
    phi_dominant = all(r['da2_r_phi'] > r['da2_r_psi'] for r in lag_rows)

    lines.append(f'**(a) ψ-slowest, φ-second ordering stable?** '
                 f'{"Yes" if psi_dominant and phi_dominant else "No"} — ')
    if psi_dominant:
        lines.append(f'DA1 tracks ψ (|r(ψ)| > |r(φ)|) at all tested lags.')
    else:
        lines.append(f'DA1 does NOT consistently track ψ at all lags.')
    if phi_dominant:
        lines.append(f'DA2 tracks φ (|r(φ)| > |r(ψ)|) at all tested lags.')
    else:
        lines.append(f'DA2 does NOT consistently track φ at all lags.')

    lines.append('')
    # Check noise floor
    failed_lags = [r for r in lag_rows if not r['lambda2_above_conservative_floor']]
    if failed_lags:
        lines.append(f'**(b) Conservative noise floor failures:** '
                     f'{len(failed_lags)} lag(s) have λ₂ below the conservative floor:')
        for r in failed_lags:
            lines.append(f'  - τ = {r["tau_ps"]:.1f} ps: '
                         f'λ₂ = {r["lambda_2"]:.6f} < floor = {r["floor_conservative"]:.6f}')
    else:
        lines.append(f'**(b)** All tested lags have λ₂ above the conservative noise floor.')

    # DA1 stability
    angles = [r['angle_da1_ref_deg'] for r in lag_rows]
    lines.append(f'')
    lines.append(f'**(c) DA1 directional stability:** angles from reference '
                 f'(τ*={ref_tau:.1f} ps) range from {min(angles):.1f}° to {max(angles):.1f}°.')

    summary_path = os.path.join(EXPDIR, 'SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
