#!/usr/bin/env python3
"""
DA vs TICA comparison — heavy computation script.

Computes TICA, PCA, GMM-DA, KDE-DA, and convergence profiles for alanine
dipeptide or chignolin. Saves all results to an npz file for plotting in
the companion notebook DA_vs_TICA_comparison.ipynb.

Usage:
    python compute_da_comparison.py --system alanine
    python compute_da_comparison.py --system chignolin
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import argparse
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')

from itertools import combinations
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture

from src.metrics.da_directions import extract_da_directions, da_convergence_profile
from src.metrics.autocorrelation import compute_stationary_moments
from src.simulation.langevin import LangevinSimulatorD
from src.simulation.generic_langevin import GenericLangevinSimulator
from src.kde.kde_score import KDEScore

# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════
CONFIGS = {
    'alanine': dict(
        dt=0.0001,            # stable for sigma_f^2 ~ 0.09 in 45D
        da_max_phys=2000.0,   # generous upper bound; peak gap selects optimal lag
        gmm_k=4,
        tica_lag=10,          # 10 frames = 10 ps
        n_sub=10000,          # subsample for GMM
        n_traj_gmm=5000,     # DA trajectories (GMM — fast)
        n_traj_kde=2000,     # DA trajectories (KDE — slow)
        n_kde_centers=2000,
    ),
    'chignolin': dict(
        dt=0.001,             # stable for sigma_f^2 ~ 5.8 in 45D
        da_max_phys=500.0,    # generous upper bound; adaptive stopping finds convergence
        gmm_k=4,
        tica_lag=1,           # 1 frame = 100 ps
        n_sub=10000,
        n_traj_gmm=5000,
        n_traj_kde=2000,
        n_kde_centers=2000,
    ),
}

EXPDIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(EXPDIR, 'results')
os.makedirs(RESDIR, exist_ok=True)


def _save_results(system, cfg, d, sigma_f_sq,
                  tica_evals, tica_evecs, tica_proj, tica_rank,
                  pca_evals, pca_evecs, pca_proj,
                  da_gmm_evals, da_gmm_evecs, da_gmm_proj,
                  da_kde_evals, da_kde_evecs, da_kde_proj,
                  profile, kde_h=0.0, kde_n_centers=0, suffix='',
                  conv_gt_corr=None, conv_tica1_angle=None,
                  conv_da2_gt_corr=None, conv_tica2_angle=None):
    """Save all results to npz. KDE arrays can be None (gmm-only checkpoint)."""
    n_save = 5
    save_dict = {
        # TICA
        'tica_evals': tica_evals[:n_save].astype(np.float64),
        'tica_evecs': tica_evecs[:, :n_save].astype(np.float64),
        'tica_proj': tica_proj[:, :n_save].astype(np.float32),
        'tica_rank': np.array(tica_rank),
        # PCA
        'pca_evals': pca_evals[:n_save].astype(np.float64),
        'pca_evecs': pca_evecs[:, :n_save].astype(np.float64),
        'pca_proj': pca_proj[:, :n_save].astype(np.float32),
        # DA (GMM)
        'da_gmm_evals': da_gmm_evals[:n_save].cpu().numpy().astype(np.float64),
        'da_gmm_evecs': da_gmm_evecs[:, :n_save].cpu().numpy().astype(np.float64),
        'da_gmm_proj': da_gmm_proj[:, :n_save].astype(np.float32),
        # Convergence
        'conv_checkpoints': profile['checkpoints'],
        'conv_angles': profile['angles_deg'],
        'conv_eigenvalues': profile['all_eigenvalues'],
        'conv_leading_eigenvectors': profile['leading_eigenvectors'],
        # Config
        'conv_gap_profile': profile.get('gap_profile', np.array([])),
        'conv_optimal_idx': np.int64(profile.get('optimal_idx', 0)),
        'conv_optimal_eigenvalues': profile.get('optimal_eigenvalues', np.array([])),
        'conv_optimal_eigenvectors': profile.get('optimal_eigenvectors', np.array([])),
        'dt': np.float64(cfg['dt']),
        'da_max_phys': np.float64(cfg['da_max_phys']),
        'n_traj_gmm': np.int64(cfg['n_traj_gmm']),
        'n_traj_kde': np.int64(cfg['n_traj_kde']),
        'gmm_k': np.int64(cfg['gmm_k']),
        'sigma_f_sq': np.float64(sigma_f_sq),
        'kde_h': np.float64(kde_h),
        'kde_n_centers': np.int64(kde_n_centers),
        'tica_lag': np.int64(cfg['tica_lag']),
        'frame_interval_ps': np.int64(d['frame_interval_ps']),
    }

    # DA1 convergence diagnostics
    if conv_gt_corr is not None:
        for k, v in conv_gt_corr.items():
            save_dict[f'conv_da1_r_{k}'] = np.array(v, dtype=np.float64)
    if conv_tica1_angle is not None:
        save_dict['conv_da1_tica1_angle'] = np.array(conv_tica1_angle, dtype=np.float64)

    # DA2 convergence diagnostics + optimal lag
    if conv_da2_gt_corr is not None:
        for k, v in conv_da2_gt_corr.items():
            save_dict[f'conv_da2_r_{k}'] = np.array(v, dtype=np.float64)
    if conv_tica2_angle is not None:
        save_dict['conv_da2_tica2_angle'] = np.array(conv_tica2_angle, dtype=np.float64)

    # DA2 gap and optimal lag from profile
    if 'gap23_profile' in profile:
        save_dict['conv_gap23_profile'] = profile['gap23_profile']
        save_dict['conv_optimal_idx_da2'] = np.int64(profile.get('optimal_idx_da2', 0))
    if 'optimal_eigenvalues_da2' in profile:
        save_dict['conv_optimal_eigenvalues_da2'] = profile['optimal_eigenvalues_da2']
        save_dict['conv_optimal_eigenvectors_da2'] = profile['optimal_eigenvectors_da2']
    if 'top_k_eigenvectors' in profile:
        save_dict['conv_top_k_eigenvectors'] = profile['top_k_eigenvectors']

    # DA (KDE) — may be None for gmm-only checkpoint
    if da_kde_evals is not None:
        save_dict['da_kde_evals'] = da_kde_evals[:n_save].cpu().numpy().astype(np.float64)
        save_dict['da_kde_evecs'] = da_kde_evecs[:, :n_save].cpu().numpy().astype(np.float64)
        save_dict['da_kde_proj'] = da_kde_proj[:, :n_save].astype(np.float32)
        save_dict['has_kde'] = np.bool_(True)
    else:
        save_dict['has_kde'] = np.bool_(False)

    # Ground truth arrays
    for k, v in d['ground_truth'].items():
        save_dict[f'gt_{k}'] = v.astype(np.float32)

    out_path = os.path.join(RESDIR, f'{system}_da_results{suffix}.npz')
    np.savez_compressed(out_path, **save_dict)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path} ({size_mb:.1f} MB)')
    return out_path


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════
def load_alanine():
    """Load alanine dipeptide: 45D heavy-atom distances + (phi, psi) ground truth."""
    data_dir = os.path.join(EXPDIR, 'alanine_dipeptide', 'data')

    dist_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    dihed_file = os.path.join(data_dir, 'alanine-dipeptide-3x250ns-backbone-dihedrals.npz')

    if not os.path.exists(dist_file) or not os.path.exists(dihed_file):
        # Fallback: try mdshare download
        import mdshare
        os.makedirs(data_dir, exist_ok=True)
        dist_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
            working_directory=data_dir)
        dihed_file = mdshare.fetch(
            'alanine-dipeptide-3x250ns-backbone-dihedrals.npz',
            working_directory=data_dir)

    dist_data = np.load(dist_file)
    dihed_data = np.load(dihed_file)

    dists_list = [dist_data[f'arr_{i}'] for i in range(3)]
    diheds_list = [dihed_data[f'arr_{i}'] for i in range(3)]

    all_dists = np.concatenate(dists_list)      # (750000, 45)
    all_diheds = np.concatenate(diheds_list)     # (750000, 2)

    return dict(
        all_dists=all_dists,
        ground_truth={'phi': all_diheds[:, 0], 'psi': all_diheds[:, 1]},
        traj_list=dists_list,
        frame_interval_ps=1,
    )


def load_chignolin():
    """Load chignolin: 45D Ca pairwise distances + dRMSD ground truth.

    Uses pre-computed chignolin_da_data.npz which contains both the pooled
    distances/RMSD and the per-trajectory distance arrays (needed for TICA).
    Generate this file locally with: python compute_da_comparison.py --prepare-chignolin
    """
    # Try the combined file first (contains per-trajectory splits for TICA)
    combined_file = os.path.join(EXPDIR, 'chignolin', 'data', 'chignolin_da_data.npz')
    if os.path.exists(combined_file):
        print(f'  Loading combined file: {combined_file}')
        data = np.load(combined_file, allow_pickle=True)
        all_dists = data['all_dists']
        dist_rmsd = data['dist_rmsd']
        traj_lengths = data['traj_lengths']
        # Reconstruct traj_list from concatenated + lengths
        all_traj_dists = data['all_traj_dists']
        traj_list = []
        offset = 0
        for length in traj_lengths:
            traj_list.append(all_traj_dists[offset:offset+length])
            offset += length
        return dict(
            all_dists=all_dists,
            ground_truth={'rmsd': dist_rmsd},
            traj_list=traj_list,
            frame_interval_ps=100,
        )

    # Fallback: load from raw coords (requires coords_nowater/)
    npz = np.load(os.path.join(EXPDIR, 'chignolin', 'data', 'chignolin_ca_distances.npz'))
    all_dists = npz['all_dists']
    dist_rmsd = npz['dist_rmsd']

    data_dir = os.path.join(EXPDIR, 'chignolin', 'data', 'coords_nowater')
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    ca_in_protein = np.array([8, 29, 50, 64, 76, 91, 105, 112, 126, 150])
    ca_pairs = list(combinations(range(10), 2))
    pair_arr = np.array(ca_pairs)

    traj_list = []
    for f in files:
        arr = np.load(os.path.join(data_dir, f))
        ca_coords = arr[:, ca_in_protein, :]
        diff = ca_coords[:, pair_arr[:, 0], :] - ca_coords[:, pair_arr[:, 1], :]
        traj_list.append(np.linalg.norm(diff, axis=-1))

    return dict(
        all_dists=all_dists,
        ground_truth={'rmsd': dist_rmsd},
        traj_list=traj_list,
        frame_interval_ps=100,
    )


def prepare_chignolin():
    """Pre-compute chignolin per-trajectory distances into a single npz.

    Run locally (where coords_nowater/ exists) to create chignolin_da_data.npz,
    then upload to cluster. This avoids uploading 3.7GB of raw coords.
    """
    print('Preparing chignolin data...')
    npz = np.load(os.path.join(EXPDIR, 'chignolin', 'data', 'chignolin_ca_distances.npz'))
    all_dists = npz['all_dists']
    dist_rmsd = npz['dist_rmsd']

    data_dir = os.path.join(EXPDIR, 'chignolin', 'data', 'coords_nowater')
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    ca_in_protein = np.array([8, 29, 50, 64, 76, 91, 105, 112, 126, 150])
    ca_pairs = list(combinations(range(10), 2))
    pair_arr = np.array(ca_pairs)

    traj_list = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f'  Processing trajectory {i}/{len(files)}...')
        arr = np.load(os.path.join(data_dir, f))
        ca_coords = arr[:, ca_in_protein, :]
        diff = ca_coords[:, pair_arr[:, 0], :] - ca_coords[:, pair_arr[:, 1], :]
        traj_list.append(np.linalg.norm(diff, axis=-1).astype(np.float32))

    traj_lengths = np.array([len(t) for t in traj_list])
    all_traj_dists = np.concatenate(traj_list, axis=0)

    out_path = os.path.join(EXPDIR, 'chignolin', 'data', 'chignolin_da_data.npz')
    np.savez_compressed(out_path,
                        all_dists=all_dists,
                        dist_rmsd=dist_rmsd,
                        all_traj_dists=all_traj_dists,
                        traj_lengths=traj_lengths)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'Saved: {out_path} ({size_mb:.1f} MB)')
    print(f'  {len(traj_list)} trajectories, {all_traj_dists.shape[0]:,} total frames')


# ════════════════════════════════════════════════════════════════════════
# TICA
# ════════════════════════════════════════════════════════════════════════
def compute_tica(traj_list, all_dists, tica_lag, epsilon=1e-6):
    """Compute TICA via whitened eigenproblem."""
    dim = all_dists.shape[1]
    data_mean = all_dists.mean(axis=0)

    C0 = np.zeros((dim, dim))
    Ctau = np.zeros((dim, dim))
    n_c0 = n_pairs = 0

    for traj in traj_list:
        c = traj - data_mean
        T = len(traj)
        C0 += c.T @ c;  n_c0 += T
        Ctau += c[:T-tica_lag].T @ c[tica_lag:];  n_pairs += T - tica_lag

    C0 /= n_c0;  Ctau /= n_pairs
    Ctau_sym = (Ctau + Ctau.T) / 2

    ev0, U0 = scipy_eigh(C0)
    eps = max(epsilon, -ev0.min() + 1e-16) if ev0.min() < 0 else epsilon
    keep = np.abs(ev0) >= eps
    rank = keep.sum()

    W = U0[:, keep] @ np.diag(1.0 / np.sqrt(ev0[keep]))
    Cw = W.T @ Ctau_sym @ W
    Cw = (Cw + Cw.T) / 2
    tica_evals, tica_evecs_w = scipy_eigh(Cw)
    tica_evecs = W @ tica_evecs_w

    idx = np.argsort(tica_evals)[::-1]
    tica_evals = tica_evals[idx]
    tica_evecs = tica_evecs[:, idx]

    all_centered = all_dists - data_mean
    tica_proj = all_centered @ tica_evecs

    return tica_evals, tica_evecs, tica_proj, rank, C0, data_mean


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main(system, skip_kde=False):
    cfg = CONFIGS[system]
    device = torch.device('cpu')
    t_start = time.time()

    da_max_steps = int(cfg['da_max_phys'] / cfg['dt'])
    print(f'{"="*70}')
    print(f'DA comparison: {system}')
    print(f'  dt={cfg["dt"]}, max_phys_t={cfg["da_max_phys"]}, '
          f'max_steps={da_max_steps:,}, n_traj_gmm={cfg["n_traj_gmm"]}, n_traj_kde={cfg["n_traj_kde"]}')
    print(f'{"="*70}')

    # ── 1. Load data ──
    print('\n[1/7] Loading data...')
    if system == 'alanine':
        d = load_alanine()
    else:
        d = load_chignolin()
    all_dists = d['all_dists']
    N_total, dim = all_dists.shape
    print(f'  {N_total:,} frames, {dim}D features')

    # ── 2. TICA ──
    print(f'\n[2/7] Computing TICA (lag={cfg["tica_lag"]})...')
    t0 = time.time()
    tica_evals, tica_evecs, tica_proj, tica_rank, C0, data_mean = compute_tica(
        d['traj_list'], all_dists, cfg['tica_lag'])
    all_centered = all_dists - data_mean
    print(f'  Whitening rank: {tica_rank}/{dim}')
    dt_ps = d['frame_interval_ps']
    for i in range(5):
        lam = tica_evals[i]
        tau = -cfg['tica_lag'] * dt_ps / np.log(lam) if 0 < lam < 1 else np.inf
        gt_strs = [f'|r({k})|={abs(pearsonr(tica_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  TICA{i+1}: lambda={lam:.4f}, tau={tau:.1f}ps, {", ".join(gt_strs)}')
    print(f'  [{time.time()-t0:.1f}s]')

    # ── 3. PCA ──
    print(f'\n[3/7] Computing PCA...')
    pca_evals, pca_evecs = np.linalg.eigh(C0)
    idx = np.argsort(pca_evals)[::-1]
    pca_evals = pca_evals[idx]
    pca_evecs = pca_evecs[:, idx]
    pca_proj = all_centered @ pca_evecs
    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(pca_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  PC{i+1}: var={100*pca_evals[i]/pca_evals.sum():.1f}%, {", ".join(gt_strs)}')

    # ── 4. GMM fit + simulator ──
    print(f'\n[4/7] Fitting GMM(k={cfg["gmm_k"]})...')
    np.random.seed(42)
    sub_idx = np.random.choice(N_total, cfg['n_sub'], replace=False)
    X_sub = all_dists[sub_idx]
    X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=device)

    gmm = GaussianMixture(n_components=cfg['gmm_k'], covariance_type='full',
                          n_init=3, max_iter=500, random_state=42)
    gmm.fit(X_sub)
    gmm_means = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
    gmm_covs = torch.tensor(gmm.covariances_, dtype=torch.float32, device=device)
    gmm_weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=device)

    gmm_sim = LangevinSimulatorD(gmm_means, gmm_covs, gmm_weights, dt=cfg['dt'], device=device)
    sigma_f_sq = float(gmm_sim.var_f)
    noise_drift = (sigma_f_sq * cfg['dt'])**0.5 / (sigma_f_sq * cfg['dt'] / 2)
    print(f'  sigma_f^2 = {sigma_f_sq:.4f}, noise/drift = {noise_drift:.1f}x')
    print(f'  weights = {gmm.weights_.round(3)}')

    # Quick stability test
    torch.manual_seed(0)
    x_test = gmm_sim.sample_initial(100)
    for _ in range(1000):
        x_test = gmm_sim.step(x_test)
    n_nan = torch.isnan(x_test).any(dim=1).sum().item()
    print(f'  Stability test: {n_nan} NaN / 100 -> {"STABLE" if n_nan == 0 else "UNSTABLE!"}')

    # ── 5. GMM-DA via convergence profile + peak gap selection ──
    print(f'\n[5/8] DA convergence profile (GMM, max_steps={da_max_steps:,})...')
    t0 = time.time()
    pca1 = torch.tensor(pca_evecs[:, 0], dtype=torch.float32, device=device)

    # ~30 geometric checkpoints up to generous max
    phys_times = np.geomspace(0.1, cfg['da_max_phys'], 30)
    checkpoints = sorted(set(max(1, int(pt / cfg['dt'])) for pt in phys_times))

    print(f'  {len(checkpoints)} checkpoints, max={checkpoints[-1]:,} steps')
    print(f'  Lag selection: peak eigenvalue gap lambda1/lambda2')

    profile = da_convergence_profile(
        gmm_means, gmm_covs, gmm_weights,
        checkpoints=checkpoints,
        n_trajectories=cfg['n_traj_gmm'],
        dt=cfg['dt'], seed=42,
        reference_direction=pca1,
    )

    # Use gap-selected DA directions
    opt_idx = profile['optimal_idx']
    opt_step = profile['checkpoints'][opt_idx]
    opt_phys = opt_step * cfg['dt']
    da_gmm_evals = torch.from_numpy(profile['optimal_eigenvalues']).float()
    da_gmm_evecs = torch.from_numpy(profile['optimal_eigenvectors']).float()
    da_gmm_proj = all_centered @ da_gmm_evecs.numpy()
    elapsed_gmm = time.time() - t0

    print(f'  Gap12 profile: {", ".join(f"{g:.1f}" for g in profile["gap_profile"])}')
    print(f'  DA1 peak gap at phys_t={opt_phys:.1f} (step {opt_step:,}, '
          f'checkpoint {opt_idx+1}/{len(profile["checkpoints"])})')
    print(f'  angle(DA1, PCA1) = {profile["angles_deg"][opt_idx]:.1f} deg at tau*')

    gap12 = da_gmm_evals[0].item() / da_gmm_evals[1].item() if da_gmm_evals[1].item() > 0 else float('inf')
    gap13 = da_gmm_evals[0].item() / da_gmm_evals[2].item() if da_gmm_evals[2].item() > 0 else float('inf')
    print(f'  Top-3 eigenvalues at DA1 lag: {da_gmm_evals[:3].numpy()}')
    print(f'  Eigenvalue gap lambda1/lambda2 = {gap12:.1f}x, lambda1/lambda3 = {gap13:.1f}x')
    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(da_gmm_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  DA-GMM{i+1} (at DA1 lag): {", ".join(gt_strs)}')

    # ── DA2 at its own optimal lag (peak λ₂/λ₃) ──
    opt_idx2 = profile['optimal_idx_da2']
    opt_step2 = profile['checkpoints'][opt_idx2]
    opt_phys2 = opt_step2 * cfg['dt']
    da2_evals = torch.from_numpy(profile['optimal_eigenvalues_da2']).float()
    da2_evecs = torch.from_numpy(profile['optimal_eigenvectors_da2']).float()
    da2_proj = all_centered @ da2_evecs.numpy()

    print(f'\n  Gap23 profile: {", ".join(f"{g:.1f}" for g in profile["gap23_profile"])}')
    print(f'  DA2 peak gap (lam2/lam3) at phys_t={opt_phys2:.1f} (step {opt_step2:,}, '
          f'checkpoint {opt_idx2+1}/{len(profile["checkpoints"])})')
    print(f'  Top-3 eigenvalues at DA2 lag: {da2_evals[:3].numpy()}')
    da2_gap23 = da2_evals[1].item() / da2_evals[2].item() if da2_evals[2].item() > 0 else float('inf')
    print(f'  lambda2/lambda3 = {da2_gap23:.1f}x at DA2 lag')
    for i in range(3):
        gt_strs = [f'|r({k})|={abs(pearsonr(da2_proj[:, i], v)[0]):.3f}'
                   for k, v in d['ground_truth'].items()]
        print(f'  DA-GMM{i+1} (at DA2 lag): {", ".join(gt_strs)}')

    print(f'  [{elapsed_gmm:.1f}s]')

    # ── 5b. DA1 & DA2 convergence diagnostics at each checkpoint ──
    print(f'\n[6/8] DA1 & DA2 convergence diagnostics...')
    tica1 = torch.tensor(tica_evecs[:, 0], dtype=torch.float32, device=device)
    tica1_norm = tica1 / tica1.norm()
    tica2 = torch.tensor(tica_evecs[:, 1], dtype=torch.float32, device=device)
    tica2_norm = tica2 / tica2.norm()

    conv_gt_corr = {k: [] for k in d['ground_truth']}  # |r(DA1, gt)| at each checkpoint
    conv_tica1_angle = []  # angle(DA1, TICA1) at each checkpoint
    conv_da2_gt_corr = {k: [] for k in d['ground_truth']}  # |r(DA2, gt)| at each checkpoint
    conv_tica2_angle = []  # angle(DA2, TICA2) at each checkpoint

    for cp_i in range(len(profile['checkpoints'])):
        # DA1 diagnostics
        da1_vec = profile['leading_eigenvectors'][cp_i]
        da1_t = torch.tensor(da1_vec, dtype=torch.float32, device=device)
        da1_proj = all_centered @ da1_vec
        for k, v in d['ground_truth'].items():
            conv_gt_corr[k].append(abs(pearsonr(da1_proj, v)[0]))
        cos_a = torch.dot(da1_t, tica1_norm).abs().clamp(-1, 1)
        conv_tica1_angle.append((torch.acos(cos_a) * 180.0 / np.pi).item())

        # DA2 diagnostics (from top_k_eigenvectors)
        da2_vec = profile['top_k_eigenvectors'][cp_i][:, 1]  # (d,) second eigenvector
        da2_t = torch.tensor(da2_vec, dtype=torch.float32, device=device)
        da2_proj = all_centered @ da2_vec
        for k, v in d['ground_truth'].items():
            conv_da2_gt_corr[k].append(abs(pearsonr(da2_proj, v)[0]))
        cos_a2 = torch.dot(da2_t, tica2_norm).abs().clamp(-1, 1)
        conv_tica2_angle.append((torch.acos(cos_a2) * 180.0 / np.pi).item())

    phys_times_actual = profile['checkpoints'] * cfg['dt']
    print(f'  DA1 convergence (angle to TICA1):')
    for cp_i in range(0, len(phys_times_actual), max(1, len(phys_times_actual)//8)):
        gt_strs = [f'|r({k})|={conv_gt_corr[k][cp_i]:.3f}' for k in d['ground_truth']]
        print(f'    phys_t={phys_times_actual[cp_i]:.1f}: '
              f'angle(TICA1)={conv_tica1_angle[cp_i]:.1f}°, '
              f'gap12={profile["gap_profile"][cp_i]:.1f}x, '
              f'{", ".join(gt_strs)}')

    print(f'  DA2 convergence (angle to TICA2):')
    for cp_i in range(0, len(phys_times_actual), max(1, len(phys_times_actual)//8)):
        gt_strs = [f'|r({k})|={conv_da2_gt_corr[k][cp_i]:.3f}' for k in d['ground_truth']]
        print(f'    phys_t={phys_times_actual[cp_i]:.1f}: '
              f'angle(TICA2)={conv_tica2_angle[cp_i]:.1f}°, '
              f'gap23={profile["gap23_profile"][cp_i]:.1f}x, '
              f'{", ".join(gt_strs)}')

    # ── 5c. Save GMM results (checkpoint before slow KDE) ──
    print(f'\n[7/8] Saving GMM checkpoint...')
    _save_results(system, cfg, d, sigma_f_sq,
                  tica_evals, tica_evecs, tica_proj, tica_rank,
                  pca_evals, pca_evecs, pca_proj,
                  da_gmm_evals, da_gmm_evecs, da_gmm_proj,
                  None, None, None,  # KDE not done yet
                  profile, kde_h=0.0, kde_n_centers=cfg['n_kde_centers'],
                  suffix='_gmm_only',
                  conv_gt_corr=conv_gt_corr, conv_tica1_angle=conv_tica1_angle,
                  conv_da2_gt_corr=conv_da2_gt_corr, conv_tica2_angle=conv_tica2_angle)

    # ── 5d. KDE-DA (use gap-selected lag from GMM) ──
    if skip_kde:
        print(f'\n[8/8] Skipping KDE (--skip-kde flag)')
    else:
        kde_lag_steps = int(opt_step)  # use GMM gap-selected lag
        print(f'\n[8/8] Extracting DA directions (KDE, lag={kde_lag_steps:,} from GMM gap selection, N={cfg["n_traj_kde"]})...')
        t0 = time.time()
        center_idx = np.random.choice(cfg['n_sub'], cfg['n_kde_centers'], replace=False)
        kde = KDEScore(X_sub_t[center_idx], bandwidth='scott', device=device, batch_size=1000)
        kde_sim = GenericLangevinSimulator(
            score_fn=kde.score, samples=X_sub_t, dt=cfg['dt'], device=device)
        print(f'  KDE: h={kde.h:.4f}, {cfg["n_kde_centers"]} centers')
        print(f'  sigma_f^2 (empirical) = {kde_sim.var_f:.4f}')

        da_kde_evals, da_kde_evecs, _ = extract_da_directions(
            simulator=kde_sim,
            lag=kde_lag_steps, n_trajectories=cfg['n_traj_kde'], dt=cfg['dt'],
            device=device, seed=42)
        da_kde_proj = all_centered @ da_kde_evecs.numpy()
        elapsed_kde = time.time() - t0

        for i in range(3):
            gt_strs = [f'|r({k})|={abs(pearsonr(da_kde_proj[:, i], v)[0]):.3f}'
                       for k, v in d['ground_truth'].items()]
            print(f'  DA-KDE{i+1}: {", ".join(gt_strs)}')

        # GMM <-> KDE agreement
        for i in range(2):
            r_cross = abs(pearsonr(da_gmm_proj[:, i], da_kde_proj[:, i])[0])
            print(f'  DA{i+1} GMM<->KDE: |r|={r_cross:.3f}')
        print(f'  [{elapsed_kde:.1f}s]')

        # Save final results (with KDE)
        print(f'\nSaving final results (with KDE)...')
        _save_results(system, cfg, d, sigma_f_sq,
                      tica_evals, tica_evecs, tica_proj, tica_rank,
                      pca_evals, pca_evecs, pca_proj,
                      da_gmm_evals, da_gmm_evecs, da_gmm_proj,
                      da_kde_evals, da_kde_evecs, da_kde_proj,
                      profile, kde_h=float(kde.h), kde_n_centers=cfg['n_kde_centers'],
                      conv_gt_corr=conv_gt_corr, conv_tica1_angle=conv_tica1_angle,
                      conv_da2_gt_corr=conv_da2_gt_corr, conv_tica2_angle=conv_tica2_angle)

    elapsed_total = time.time() - t_start
    print(f'\nTotal time: {elapsed_total/60:.1f} min')
    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute DA vs TICA comparison')
    parser.add_argument('--system', choices=list(CONFIGS.keys()),
                        help='Which system to compute')
    parser.add_argument('--skip-kde', action='store_true',
                        help='Skip KDE-DA (GMM-only)')
    parser.add_argument('--prepare-chignolin', action='store_true',
                        help='Pre-compute chignolin per-trajectory distances into a single npz')
    args = parser.parse_args()

    if args.prepare_chignolin:
        prepare_chignolin()
    elif args.system:
        main(args.system, skip_kde=args.skip_kde)
    else:
        parser.error('Either --system or --prepare-chignolin is required')
