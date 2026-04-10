#!/usr/bin/env python3
"""
Extend DA convergence profile for Dry Bean dataset to larger lags.

The original run topped out at lag=32768 with gap=6.42, still accelerating.
This script continues to lag=262144 to find where the gap plateaus.

Usage:
  conda run -n ssa python compute_extended_da.py
"""
import sys, os, types, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/github/ssa-paper'))
sys.modules['src.kde.autocorrelation_score'] = types.ModuleType('src.kde.autocorrelation_score')
sys.modules['src.kde.autocorrelation_score'].autocorrelation_score = None

from src.kde.kde_score import KDEScore
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.metrics.da_directions import da_convergence_profile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EXPDIR = os.path.dirname(os.path.abspath(__file__))


def load_dry_bean():
    """Load from npz (cluster-safe, no pandas dependency)."""
    npz_path = os.path.join(EXPDIR, 'data', 'dry_bean.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        return d['X'].astype(float), d['labels'], list(d['feature_cols'])
    # Fallback to xlsx (local only)
    import pandas as pd
    path = os.path.join(EXPDIR, 'data', 'DryBeanDataset', 'Dry_Bean_Dataset.xlsx')
    df = pd.read_excel(path)
    feature_cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                    'Extent', 'Solidity', 'roundness', 'Compactness',
                    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
    X = df[feature_cols].values.astype(float)
    labels_str = df['Class'].values
    return X, labels_str, feature_cols


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    t_start = time.time()

    # Load and subsample (same seed as original)
    print('[1/4] Loading Dry Bean dataset...')
    X_full, labels_str, feature_cols = load_dry_bean()
    np.random.seed(42)
    idx = np.random.choice(len(X_full), 2000, replace=False)
    X_raw = X_full[idx]
    labels = labels_str[idx]
    print(f'  N={len(X_raw)}, d={len(feature_cols)}')

    # Standardize + PCA to 8D (same as original)
    print('[2/4] Standardizing + PCA...')
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_raw)
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_s)
    X_t = torch.tensor(X_pca, dtype=torch.float32, device=device)
    pc1_pca = pca.components_[0]  # PC1 in PCA space is just e_1
    pc1_ref = torch.zeros(8, device=device)
    pc1_ref[0] = 1.0
    print(f'  PCA: {X_s.shape[1]}D -> {X_pca.shape[1]}D, explained var: {pca.explained_variance_ratio_.sum():.3f}')

    # Build KDE scorer
    print('[3/4] Building KDE scorer + stability test...')
    kde = KDEScore(X_t, bandwidth='scott', device=device, batch_size=2000)
    print(f'  KDE: h_scott={kde.h:.4f}, {len(X_t)} centers in {X_pca.shape[1]}D')

    # Stability test
    for test_dt in [0.1, 0.01, 0.001]:
        sim_test = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_t, dt=test_dt, device=device)
        torch.manual_seed(0)
        x_test = sim_test.sample_initial(200)
        stable = True
        for step in range(1000):
            x_test = sim_test.step(x_test)
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f'  dt={test_dt}: UNSTABLE at step {step}')
                stable = False
                break
        if stable:
            print(f'  dt={test_dt}: STABLE (var_f={sim_test.var_f:.4f})')
            best_dt = test_dt
            break
    else:
        best_dt = 0.001

    print(f'  Using dt={best_dt}')

    # DA convergence profile with extended checkpoints
    print('[4/4] Computing DA convergence profile...')
    sim = GenericLangevinSimulator(
        score_fn=kde.score, samples=X_t, dt=best_dt, device=device)

    # Powers of 2 from 1 to 262144
    checkpoints = [2**i for i in range(19)]  # 1, 2, ..., 262144
    print(f'  Checkpoints: {checkpoints}')
    print(f'  Max lag: {checkpoints[-1]} steps = {checkpoints[-1]*best_dt:.1f} time units')

    profile = da_convergence_profile(
        simulator=sim,
        checkpoints=checkpoints,
        n_trajectories=2000,
        dt=best_dt,
        device=device,
        seed=42,
        reference_direction=pc1_ref,
    )

    # Print results
    print(f'\n{"Lag":>8s}  {"Gap":>8s}  {"Angle":>7s}  {"λ1":>10s}  {"λ2":>10s}')
    for i, cp in enumerate(profile['checkpoints']):
        gap = profile['gap_profile'][i]
        angle = profile['angles_deg'][i]
        l1 = profile['all_eigenvalues'][i, 0]
        l2 = profile['all_eigenvalues'][i, 1]
        print(f'{cp:>8d}  {gap:>8.3f}  {angle:>6.1f}°  {l1:>10.6f}  {l2:>10.6f}')

    opt_idx = profile['optimal_idx']
    opt_lag = profile['checkpoints'][opt_idx]
    opt_gap = profile['gap_profile'][opt_idx]
    print(f'\nOptimal lag: {opt_lag} (gap={opt_gap:.2f})')
    print(f'DA1 (PCA space): {profile["optimal_eigenvectors"][:, 0]}')

    # Back-project DA1 to original feature space
    da1_pca = profile['optimal_eigenvectors'][:, 0]
    da1_orig = pca.components_.T @ da1_pca
    da1_orig = da1_orig / np.linalg.norm(da1_orig)
    print(f'\nDA1 top loadings (original features):')
    order = np.argsort(np.abs(da1_orig))[::-1].copy()
    for i in order[:5]:
        print(f'  {feature_cols[i]:25s}: {da1_orig[i]:+.4f}')

    cos_da1_pc1 = abs(np.dot(da1_orig, pca.components_[0]))
    print(f'\n|cos(DA1, PC1)| in original space: {cos_da1_pc1:.4f}')

    # Save extended profile
    outpath = os.path.join(EXPDIR, 'winner', 'dry_bean_full_profile_extended.npz')
    np.savez(outpath,
             checkpoints=profile['checkpoints'],
             gap_profile=profile['gap_profile'],
             all_eigenvalues=profile['all_eigenvalues'],
             leading_eigenvectors=profile['leading_eigenvectors'],
             angles_deg=profile['angles_deg'],
             optimal_eigenvectors=profile['optimal_eigenvectors'],
             optimal_eigenvalues=profile['optimal_eigenvalues'])
    print(f'\nSaved: {outpath}')

    # Update results JSON
    results_path = os.path.join(EXPDIR, 'winner', 'dry_bean_full_results.json')
    with open(results_path) as f:
        results = json.load(f)
    results['gap_at_optimal_extended'] = float(opt_gap)
    results['optimal_lag_extended'] = int(opt_lag)
    results['gap_converges_extended'] = bool(
        profile['gap_profile'][-1] / profile['gap_profile'][-2] < 1.1)
    results['da1_extended'] = da1_pca.tolist()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Updated: {results_path}')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
