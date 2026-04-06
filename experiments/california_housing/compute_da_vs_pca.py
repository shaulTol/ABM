#!/usr/bin/env python3
"""
DA vs PCA on California Housing dataset.

Demonstrates that DA finds a meaningful subpopulation separation direction
that PCA misses, because the separation axis isn't the max-variance axis.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# Patch broken __init__.py before importing
import importlib
import types
# Create a dummy module to satisfy the missing import in src.kde.__init__
sys.modules['src.kde.autocorrelation_score'] = types.ModuleType('src.kde.autocorrelation_score')
sys.modules['src.kde.autocorrelation_score'].autocorrelation_score = None

from src.kde.kde_score import KDEScore
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import extract_da_directions
from src.metrics.autocorrelation import compute_ssa_score

EXPDIR = os.path.dirname(os.path.abspath(__file__))

# ─── Feature names ───
FEATURE_NAMES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]


def dt_stability_test(kde, X_sub_t, device):
    """Find largest stable dt."""
    best_dt = None
    for test_dt in [0.1, 0.01, 0.001]:
        sim_test = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_sub_t, dt=test_dt, device=device)
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


def compute_1d_ssa(proj_1d, n_traj=10000, T_steps=1000, device=None):
    """Compute SSA for a 1D projection using KDE-based canonical diffusion."""
    device = device or torch.device('cpu')
    proj_t = torch.tensor(proj_1d[:, None], dtype=torch.float32, device=device)

    # 1D KDE
    kde_1d = KDEScore(proj_t, bandwidth='scott', device=device, batch_size=5000)

    # 1D simulator
    sim_1d = GenericLangevinSimulator(
        score_fn=kde_1d.score, samples=proj_t, dt=0.01, device=device)

    # Simulate trajectories
    torch.manual_seed(99)
    x0 = sim_1d.sample_initial(n_traj)  # (N, 1)
    trajectory = [x0.squeeze(-1)]
    x = x0
    for _ in range(T_steps):
        x = sim_1d.step(x)
        trajectory.append(x.squeeze(-1))
    traj_tensor = torch.stack(trajectory, dim=1)  # (N, T+1)

    # Center by global mean
    mu = proj_t.mean()
    traj_tensor = traj_tensor - mu

    ssa, rho = compute_ssa_score(traj_tensor, max_lag=T_steps)
    return ssa, rho.numpy()


def main():
    device = torch.device('cpu')
    t_start = time.time()

    # ── 1. Load, log-transform skewed features, standardize ──
    print('[1/7] Loading California Housing dataset...')
    housing = fetch_california_housing()
    X_raw = housing.data  # (20640, 8)
    y = housing.target    # median house value
    N, d = X_raw.shape
    print(f'  {N} samples, {d} features')

    # Log-transform positive right-skewed features
    LOG_FEATURES = [0, 2, 3, 4, 5]  # MedInc, AveRooms, AveBedrms, Population, AveOccup
    X_pre = X_raw.copy()
    for i in LOG_FEATURES:
        X_pre[:, i] = np.log1p(X_pre[:, i])  # log(1+x) to handle near-zero
    print(f'  Log-transformed: {[FEATURE_NAMES[i] for i in LOG_FEATURES]}')

    scaler = StandardScaler()
    X = scaler.fit_transform(X_pre)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # ── 2. PCA ──
    print('\n[2/7] Computing PCA...')
    pca = PCA(n_components=d)
    pca.fit(X)
    pc1 = pca.components_[0]  # (8,)
    print(f'  PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}')
    print(f'  PC1 loadings: {dict(zip(FEATURE_NAMES, [f"{v:.3f}" for v in pc1]))}')

    # ── 3. Build KDE scorer ──
    print('\n[3/7] Building KDE scorer...')
    np.random.seed(42)
    # Use 2000 centers for tractable score evaluation
    center_idx = np.random.choice(N, 2000, replace=False)
    # First get Scott's bandwidth, then use 2x for smoother score
    kde_scott = KDEScore(X_t[center_idx], bandwidth='scott', device=device, batch_size=2000)
    h_scott = kde_scott.h
    bw_mult = 2.0
    h_used = h_scott * bw_mult
    kde = KDEScore(X_t[center_idx], bandwidth=h_used, device=device, batch_size=2000)
    print(f'  KDE: h_scott={h_scott:.4f}, h_used={h_used:.4f} ({bw_mult}x Scott), 2000 centers in {d}D')

    # dt stability test
    print('\n  dt stability test:')
    best_dt = dt_stability_test(kde, X_t, device)
    print(f'  -> Using dt={best_dt}')

    # ── 4. DA convergence profile ──
    print('\n[4/7] Computing DA convergence profile...')
    sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_t, dt=best_dt, device=device)
    print(f'  sigma_f^2 = {sim.var_f:.4f}')

    n_traj = 5000
    checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

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

    all_eigenvalues = np.array(all_eigenvalues)  # (n_cp, d)
    all_gap12 = np.array(all_gap12)

    # Select optimal lag
    opt_idx = np.argmax(all_gap12)
    opt_tau = checkpoints[opt_idx]
    da1 = all_da1_vecs[opt_idx]
    print(f'\n  Optimal tau={opt_tau} (gap12={all_gap12[opt_idx]:.2f})')
    print(f'  DA1 loadings: {dict(zip(FEATURE_NAMES, [f"{v:.3f}" for v in da1]))}')

    # ── 5. Direction comparison ──
    print('\n[5/7] Direction comparison...')
    cos_sim = abs(np.dot(pc1, da1))
    print(f'  |cos(PC1, DA1)| = {cos_sim:.4f}')

    # Project data
    proj_pc1 = X @ pc1
    proj_da1 = X @ da1

    # Correlations with target and features
    r_pc1_target = pearsonr(proj_pc1, y)[0]
    r_da1_target = pearsonr(proj_da1, y)[0]
    print(f'  Correlation with house value: PC1 r={r_pc1_target:.3f}, DA1 r={r_da1_target:.3f}')

    # Median income split for coloring
    med_inc_raw = X_raw[:, 0]  # MedInc is feature 0
    income_high = med_inc_raw > np.median(med_inc_raw)

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

    for ax, proj, label, ssa_val in [
        (axes[0], proj_pc1, 'PC1', ssa_pc1),
        (axes[1], proj_da1, 'DA1', ssa_da1),
    ]:
        ax.hist(proj[~income_high], bins=80, alpha=0.6, label='Low income', color='steelblue', density=True)
        ax.hist(proj[income_high], bins=80, alpha=0.6, label='High income', color='coral', density=True)
        ax.set_title(f'Projection onto {label}  (SSA = {ssa_val:.4f})', fontsize=14)
        ax.set_xlabel(f'{label} projection', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, 'projected_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved projected_histograms.png')

    # --- Figure 2: Loading comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(d)
    width = 0.35
    ax.bar(x_pos - width/2, pc1, width, label='PC1', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, da1, width, label='DA1', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Loading', fontsize=12)
    ax.set_title(f'Feature Loadings: PC1 vs DA1  (|cos| = {cos_sim:.3f})', fontsize=14)
    ax.legend(fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, 'loading_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved loading_comparison.png')

    # --- Figure 3: Eigenvalue gap profile ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(checkpoints, all_gap12, 'o-', color='coral', linewidth=2, markersize=6)
    ax1.axvline(opt_tau, color='gray', linestyle='--', alpha=0.5,
                label=f'optimal tau={opt_tau}')
    ax1.set_xlabel('Lag (tau)', fontsize=12)
    ax1.set_ylabel('Gap ratio lambda_1 / lambda_2', fontsize=12)
    ax1.set_xscale('log', base=2)
    ax1.set_title('Eigenvalue Gap Profile', fontsize=14)
    ax1.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(EXPDIR, 'eigenvalue_gap_profile.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved eigenvalue_gap_profile.png')

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
    fig.savefig(os.path.join(EXPDIR, 'eigenvalue_decay.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved eigenvalue_decay.png')

    # ── Save results JSON ──
    results = {
        'dataset': 'california_housing',
        'n_samples': int(N),
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
            'corr_pc1_house_value': float(r_pc1_target),
            'corr_da1_house_value': float(r_da1_target),
        },
        'eigenvalue_profile': {
            'checkpoints': checkpoints,
            'gap12': all_gap12.tolist(),
            'lambda1': all_eigenvalues[:, 0].tolist(),
            'lambda2': all_eigenvalues[:, 1].tolist(),
        },
        'elapsed_seconds': time.time() - t_start,
    }

    with open(os.path.join(EXPDIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('  Saved results.json')

    # ── Summary ──
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'  |cos(PC1, DA1)|     = {cos_sim:.4f}')
    print(f'  SSA(PC1 projection) = {ssa_pc1:.4f}')
    print(f'  SSA(DA1 projection) = {ssa_da1:.4f}')
    print(f'  r(PC1, house value) = {r_pc1_target:.3f}')
    print(f'  r(DA1, house value) = {r_da1_target:.3f}')
    print(f'\nPC1 dominant loadings:')
    pc1_order = np.argsort(np.abs(pc1))[::-1]
    for i in pc1_order[:4]:
        print(f'    {FEATURE_NAMES[i]:12s}: {pc1[i]:+.3f}')
    print(f'\nDA1 dominant loadings:')
    da1_order = np.argsort(np.abs(da1))[::-1].copy()
    for i in da1_order[:4]:
        print(f'    {FEATURE_NAMES[i]:12s}: {da1[i]:+.3f}')
    print(f'\nOptimal lag: tau={opt_tau} (gap12={all_gap12[opt_idx]:.2f})')
    print(f'Total time: {(time.time()-t_start)/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    main()
