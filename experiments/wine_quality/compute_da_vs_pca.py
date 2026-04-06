#!/usr/bin/env python3
"""
DA vs PCA on Wine Quality dataset (combined red + white).

Demonstrates that DA finds the red/white wine separation direction
that PCA misses, because the bimodal split isn't the max-variance axis.

Usage:
  python compute_da_vs_pca.py [--dt DT] [--bw-mult BW_MULT] [--suffix SUFFIX]
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import json
import time
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, pointbiserialr
import types

# Patch broken __init__.py
sys.modules['src.kde.autocorrelation_score'] = types.ModuleType('src.kde.autocorrelation_score')
sys.modules['src.kde.autocorrelation_score'].autocorrelation_score = None

from src.kde.kde_score import KDEScore
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import extract_da_directions
from src.metrics.autocorrelation import compute_ssa_score

EXPDIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free SO2', 'total SO2', 'density',
    'pH', 'sulphates', 'alcohol'
]

BASE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality'


def download_data():
    """Download and combine red + white wine datasets."""
    data_dir = os.path.join(EXPDIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    all_X = []
    all_labels = []

    for name, label in [('winequality-red.csv', 0), ('winequality-white.csv', 1)]:
        fpath = os.path.join(data_dir, name)
        if not os.path.exists(fpath):
            url = f'{BASE_URL}/{name}'
            print(f'  Downloading {url}...')
            urllib.request.urlretrieve(url, fpath)

        # CSV is semicolon-separated
        with open(fpath) as f:
            header = f.readline()  # skip header
            rows = []
            for line in f:
                vals = line.strip().split(';')
                if len(vals) >= 12:
                    rows.append([float(v) for v in vals])
        data = np.array(rows)
        # First 11 columns are features, last is quality (drop it)
        all_X.append(data[:, :11])
        all_labels.append(np.full(len(data), label, dtype=int))
        print(f'  {name}: {len(data)} samples')

    X = np.vstack(all_X)
    labels = np.concatenate(all_labels)  # 0=red, 1=white
    return X, labels


def sarle_bimodality_coefficient(x):
    """Sarle's bimodality coefficient: BC = (skew^2 + 1) / kurtosis."""
    from scipy.stats import skew, kurtosis
    n = len(x)
    g = skew(x)
    k = kurtosis(x, fisher=False)  # excess=False → Pearson kurtosis
    return (g**2 + 1) / k


def dip_test_pvalue(x):
    """Hartigan's dip test. Returns (dip_stat, p_value)."""
    try:
        import diptest
        dip, pval = diptest.diptest(x)
        return dip, pval
    except ImportError:
        print('  WARNING: diptest not installed, skipping dip test')
        return None, None


def dt_stability_test(kde, X_t, device):
    """Find largest stable dt."""
    best_dt = None
    for test_dt in [0.1, 0.01, 0.001]:
        sim_test = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_t, dt=test_dt, device=device)
        torch.manual_seed(0)
        x_test = sim_test.sample_initial(200)
        init_norms = x_test.norm(dim=1)
        stable = True
        for step in range(1000):
            x_test = sim_test.step(x_test)
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f'  dt={test_dt}: UNSTABLE (NaN/Inf at step {step})')
                stable = False
                break
        if stable:
            final_norms = x_test.norm(dim=1)
            ratio = (final_norms / init_norms).max().item()
            if ratio > 10:
                print(f'  dt={test_dt}: UNSTABLE (norm ratio {ratio:.1f}x)')
            else:
                print(f'  dt={test_dt}: STABLE (max norm ratio {ratio:.1f}x, '
                      f'sigma_f^2={sim_test.var_f:.4f})')
                if best_dt is None:
                    best_dt = test_dt
    return best_dt or 0.001


def compute_1d_ssa(proj_1d, n_traj=5000, T_steps=500, device=None):
    """Compute SSA for a 1D projection using KDE-based canonical diffusion."""
    device = device or torch.device('cpu')
    proj_t = torch.tensor(proj_1d[:, None], dtype=torch.float32, device=device)

    kde_1d = KDEScore(proj_t, bandwidth='scott', device=device, batch_size=5000)
    sim_1d = GenericLangevinSimulator(
        score_fn=kde_1d.score, samples=proj_t, dt=0.01, device=device)

    torch.manual_seed(99)
    x0 = sim_1d.sample_initial(n_traj)
    trajectory = [x0.squeeze(-1)]
    x = x0
    for _ in range(T_steps):
        x = sim_1d.step(x)
        trajectory.append(x.squeeze(-1))
    traj_tensor = torch.stack(trajectory, dim=1)

    # Center by global mean
    mu = proj_t.mean()
    traj_tensor = traj_tensor - mu

    ssa, rho = compute_ssa_score(traj_tensor, max_lag=T_steps)
    return ssa, rho.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=None, help='Override dt (skip stability test)')
    parser.add_argument('--bw-mult', type=float, default=1.0, help='Bandwidth multiplier on Scott')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output files')
    args = parser.parse_args()

    device = torch.device('cpu')
    t_start = time.time()
    suffix = f'_{args.suffix}' if args.suffix else ''

    print(f'=== Config: dt={args.dt}, bw_mult={args.bw_mult}, suffix="{args.suffix}" ===')

    # ── 1. Load data ──
    print('[1/7] Loading Wine Quality dataset...')
    X_raw, labels = download_data()
    N, d = X_raw.shape
    n_red = (labels == 0).sum()
    n_white = (labels == 1).sum()
    print(f'  Combined: {N} samples ({n_red} red, {n_white} white), {d} features')

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    is_red = labels == 0
    is_white = labels == 1

    # ── 2. PCA ──
    print('\n[2/7] Computing PCA...')
    pca = PCA(n_components=d)
    pca.fit(X)
    pc1 = pca.components_[0]
    print(f'  PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}')
    print(f'  PC1 loadings:')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'    {name:20s}: {pc1[i]:+.3f}')

    # ── 3. Build KDE scorer ──
    print('\n[3/7] Building KDE scorer...')
    np.random.seed(42)
    center_idx = np.random.choice(N, 2000, replace=False)
    kde_scott = KDEScore(X_t[center_idx], bandwidth='scott', device=device, batch_size=2000)
    h_scott = kde_scott.h
    bw_mult = args.bw_mult
    h_used = h_scott * bw_mult
    kde = KDEScore(X_t[center_idx], bandwidth=h_used, device=device, batch_size=2000)
    print(f'  KDE: h_scott={h_scott:.4f}, h_used={h_used:.4f} ({bw_mult}x Scott), '
          f'2000 centers in {d}D')

    if args.dt is not None:
        best_dt = args.dt
        print(f'  Using specified dt={best_dt}')
    else:
        print('\n  dt stability test:')
        best_dt = dt_stability_test(kde, X_t, device)
        print(f'  -> Using dt={best_dt}')

    # ── 4. DA convergence profile ──
    print('\n[4/7] Computing DA convergence profile...')
    sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_t, dt=best_dt, device=device)
    print(f'  sigma_f^2 = {sim.var_f:.4f}')

    n_traj = 5000
    # Physical times: 0.1, 0.2, 0.4, ..., 204.8
    # Convert to steps based on dt
    phys_times = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
    checkpoints = [max(1, int(pt / best_dt)) for pt in phys_times]
    # Remove duplicates (can happen at large dt)
    checkpoints = sorted(set(checkpoints))
    print(f'  Checkpoints (steps): {checkpoints}')
    print(f'  Physical times: {[cp * best_dt for cp in checkpoints]}')

    all_eigenvalues = []
    all_gap12 = []
    all_da1_vecs = []

    for cp in checkpoints:
        t0 = time.time()
        evals, evecs, _ = extract_da_directions(
            simulator=sim, lag=cp, n_trajectories=n_traj,
            dt=best_dt, device=device, seed=42)
        evals_np = evals.numpy()
        all_eigenvalues.append(evals_np)
        gap = evals_np[0] / evals_np[1] if evals_np[1] > 0 else float('inf')
        all_gap12.append(gap)
        all_da1_vecs.append(evecs[:, 0].numpy())
        print(f'  tau={cp:4d}: lambda1={evals_np[0]:.4f}, gap12={gap:.2f} [{time.time()-t0:.1f}s]')

    all_eigenvalues = np.array(all_eigenvalues)
    all_gap12 = np.array(all_gap12)

    opt_idx = np.argmax(all_gap12)
    opt_tau = checkpoints[opt_idx]
    da1 = all_da1_vecs[opt_idx]
    print(f'\n  Optimal tau={opt_tau} (gap12={all_gap12[opt_idx]:.2f})')
    print(f'  DA1 loadings:')
    for i, name in enumerate(FEATURE_NAMES):
        print(f'    {name:20s}: {da1[i]:+.3f}')

    # ── 5. Direction comparison ──
    print('\n[5/7] Direction comparison...')
    cos_sim = abs(np.dot(pc1, da1))
    print(f'  |cos(PC1, DA1)| = {cos_sim:.4f}')

    proj_pc1 = X @ pc1
    proj_da1 = X @ da1

    # Point-biserial correlation with red/white
    r_pc1_label, p_pc1 = pointbiserialr(labels, proj_pc1)
    r_da1_label, p_da1 = pointbiserialr(labels, proj_da1)
    print(f'  Point-biserial r(PC1, red/white) = {r_pc1_label:.4f} (p={p_pc1:.2e})')
    print(f'  Point-biserial r(DA1, red/white) = {r_da1_label:.4f} (p={p_da1:.2e})')

    # Bimodality coefficient
    bc_pc1 = sarle_bimodality_coefficient(proj_pc1)
    bc_da1 = sarle_bimodality_coefficient(proj_da1)
    print(f'  Bimodality coeff: PC1={bc_pc1:.4f}, DA1={bc_da1:.4f}')

    # Dip test
    dip_pc1, dip_p_pc1 = dip_test_pvalue(proj_pc1.astype(np.float64))
    dip_da1, dip_p_da1 = dip_test_pvalue(proj_da1.astype(np.float64))
    if dip_pc1 is not None:
        print(f'  Dip test: PC1 dip={dip_pc1:.4f} (p={dip_p_pc1:.4f}), '
              f'DA1 dip={dip_da1:.4f} (p={dip_p_da1:.4f})')

    # ── 6. SSA comparison ──
    print('\n[6/7] Computing SSA on 1D projections...')
    t0 = time.time()
    ssa_pc1, rho_pc1 = compute_1d_ssa(proj_pc1, n_traj=5000, T_steps=500, device=device)
    print(f'  SSA(PC1 projection) = {ssa_pc1:.4f} [{time.time()-t0:.1f}s]')

    t0 = time.time()
    ssa_da1, rho_da1 = compute_1d_ssa(proj_da1, n_traj=5000, T_steps=500, device=device)
    print(f'  SSA(DA1 projection) = {ssa_da1:.4f} [{time.time()-t0:.1f}s]')

    # ── 7. Figures ──
    print('\n[7/7] Generating figures...')

    # --- Figure 1: Projected histograms ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, proj, label, ssa_val, rpb in [
        (axes[0], proj_pc1, 'PC1', ssa_pc1, r_pc1_label),
        (axes[1], proj_da1, 'DA1', ssa_da1, r_da1_label),
    ]:
        ax.hist(proj[is_red], bins=60, alpha=0.6, label='Red wine', color='firebrick', density=True)
        ax.hist(proj[is_white], bins=60, alpha=0.6, label='White wine', color='gold', density=True)
        ax.set_title(f'Projection onto {label}  (SSA={ssa_val:.4f}, r_pb={abs(rpb):.3f})',
                     fontsize=14)
        ax.set_xlabel(f'{label} projection', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, f'projected_histograms{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved projected_histograms{suffix}.png')

    # --- Figure 2: Loading comparison ---
    fig, ax = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(d)
    width = 0.35
    ax.bar(x_pos - width/2, pc1, width, label='PC1', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, da1, width, label='DA1', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Loading', fontsize=12)
    ax.set_title(f'Feature Loadings: PC1 vs DA1  (|cos|={cos_sim:.3f})', fontsize=14)
    ax.legend(fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, f'loading_comparison{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved loading_comparison{suffix}.png')

    # --- Figure 3: Eigenvalue gap profile ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(checkpoints, all_gap12, 'o-', color='coral', linewidth=2, markersize=6)
    ax.axvline(opt_tau, color='gray', linestyle='--', alpha=0.5,
               label=f'optimal tau={opt_tau}')
    ax.set_xlabel('Lag (tau)', fontsize=12)
    ax.set_ylabel('Gap ratio lambda_1 / lambda_2', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_title('Eigenvalue Gap Profile', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, f'eigenvalue_gap_profile{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved eigenvalue_gap_profile{suffix}.png')

    # --- Figure 4: Eigenvalue decay ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(4, d)):
        ax.plot(checkpoints, all_eigenvalues[:, i], 'o-', label=f'lambda_{i+1}', linewidth=2)
    ax.set_xlabel('Lag (tau)', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_title('Eigenvalue Decay vs Lag', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, f'eigenvalue_decay{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved eigenvalue_decay{suffix}.png')

    # ── Save results JSON ──
    results = {
        'dataset': 'wine_quality_combined',
        'n_samples': int(N),
        'n_red': int(n_red),
        'n_white': int(n_white),
        'n_features': int(d),
        'feature_names': FEATURE_NAMES,
        'pca': {
            'pc1_loadings': pc1.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        },
        'da': {
            'da1_loadings': da1.tolist(),
            'optimal_tau': int(opt_tau),
            'optimal_gap12': float(all_gap12[opt_idx]),
            'dt': float(best_dt),
            'n_trajectories': n_traj,
            'kde_bandwidth': float(h_used),
            'kde_bandwidth_scott': float(h_scott),
            'kde_bandwidth_multiplier': bw_mult,
            'kde_n_centers': 2000,
            'sigma_f_sq': float(sim.var_f),
        },
        'comparison': {
            'cosine_similarity_abs': float(cos_sim),
            'ssa_pc1': float(ssa_pc1),
            'ssa_da1': float(ssa_da1),
            'pointbiserial_pc1': float(r_pc1_label),
            'pointbiserial_da1': float(r_da1_label),
            'bimodality_coeff_pc1': float(bc_pc1),
            'bimodality_coeff_da1': float(bc_da1),
            'dip_stat_pc1': float(dip_pc1) if dip_pc1 is not None else None,
            'dip_pval_pc1': float(dip_p_pc1) if dip_p_pc1 is not None else None,
            'dip_stat_da1': float(dip_da1) if dip_da1 is not None else None,
            'dip_pval_da1': float(dip_p_da1) if dip_p_da1 is not None else None,
        },
        'eigenvalue_profile': {
            'checkpoints': checkpoints,
            'gap12': all_gap12.tolist(),
            'eigenvalues_top4': all_eigenvalues[:, :4].tolist(),
        },
        'elapsed_seconds': time.time() - t_start,
    }

    with open(os.path.join(EXPDIR, f'results{suffix}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved results{suffix}.json')

    # ── Summary ──
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'  |cos(PC1, DA1)|        = {cos_sim:.4f}')
    print(f'  SSA(PC1 projection)    = {ssa_pc1:.4f}')
    print(f'  SSA(DA1 projection)    = {ssa_da1:.4f}')
    print(f'  |r_pb|(PC1, red/white) = {abs(r_pc1_label):.4f}')
    print(f'  |r_pb|(DA1, red/white) = {abs(r_da1_label):.4f}')
    print(f'  BC(PC1)                = {bc_pc1:.4f}')
    print(f'  BC(DA1)                = {bc_da1:.4f}')
    if dip_pc1 is not None:
        print(f'  Dip(PC1)               = {dip_pc1:.4f} (p={dip_p_pc1:.4f})')
        print(f'  Dip(DA1)               = {dip_da1:.4f} (p={dip_p_da1:.4f})')

    print(f'\nPC1 dominant loadings:')
    pc1_order = np.argsort(np.abs(pc1))[::-1].copy()
    for i in pc1_order[:4]:
        print(f'    {FEATURE_NAMES[i]:20s}: {pc1[i]:+.3f}')
    print(f'\nDA1 dominant loadings:')
    da1_order = np.argsort(np.abs(da1))[::-1].copy()
    for i in da1_order[:4]:
        print(f'    {FEATURE_NAMES[i]:20s}: {da1[i]:+.3f}')

    print(f'\nOptimal lag: tau={opt_tau} (gap12={all_gap12[opt_idx]:.2f})')
    print(f'Total time: {(time.time()-t_start)/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
