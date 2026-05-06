#!/usr/bin/env python3
"""
Trajectory-averaged SSA + bootstrap CI + DA significance pipeline.

Combines three stages:
  1. Recompute SSA using trajectory-averaged autocovariance estimator
  2. Bootstrap confidence intervals (B=2000, trajectory resampling)
  3. DA significance: spike counting + Silverman modality test + text probes

Usage (SSA recompute):
    python traj_averaged_pipeline.py --checkpoint path/to/ckpt.npz \
        --dt 0.005 --name my_experiment --output_dir results

Usage (bootstrap, no GPU needed):
    python traj_averaged_pipeline.py --mode bootstrap \
        --clip_traj z_clip_traj_elderly.npz --name elderly_gs7p5 --B 2000

Usage (DA significance):
    python traj_averaged_pipeline.py --mode da_significance \
        --clip_traj z_clip_traj_elderly.npz --dt 0.0025 --T 2000 \
        --name elderly_gs7p5 --output_dir results
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# Path setup: own directory + project root (2 levels up)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.noise_floor.free_conv_utils import edges as fc_edges


# ============================================================
# Stage 1: Trajectory-averaged SSA recomputation
# ============================================================

def compute_snapshot_lags(T, schedule="log_and_fixed"):
    """Generate snapshot lag values (same as run_ssa_da_clip.py)."""
    if schedule == "log_and_fixed":
        log_lags = [2**i for i in range(int(np.log2(T)) + 1) if 2**i <= T]
        if T not in log_lags:
            log_lags.append(T)
        fixed_lags = list(range(20, T + 1, 20))
        snapshot_lags = sorted(set(log_lags + fixed_lags))
    else:
        snapshot_lags = sorted(set(int(round(x)) for x in np.linspace(10, T, 20)))
        if T not in snapshot_lags:
            snapshot_lags.append(T)
    return snapshot_lags


def load_checkpoint(ckpt_path):
    """Load Langevin checkpoint, return x0 and snapshots dict."""
    data = np.load(ckpt_path, allow_pickle=True)
    x0 = data["x0_flat"].astype(np.float32)
    snapshot_lags = data["snapshot_lags"].tolist()
    var_f = float(data["var_f"])

    snapshots = {}
    for lag in snapshot_lags:
        key = f"snapshot_{lag}"
        if key in data:
            snapshots[lag] = data[key].astype(np.float32)
    return x0, snapshots, snapshot_lags, var_f


def embed_trajectories(x0_np, snapshots, snapshot_lags, args):
    """CLIP-embed x0 and all snapshots."""
    import torch
    from sdxl_components import RealCLIPEmbedder
    from diffusers import AutoencoderKL

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading VAE for CLIP embedding...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()

    embedder = RealCLIPEmbedder(
        vae, clip_model_name="openai/clip-vit-large-patch14",
        device=device, vae_batch_size=4, clip_batch_size=8
    )

    latent_shape = (4, 128, 128) if x0_np.shape[1] == 65536 else (4, 64, 64)
    print(f"Latent shape: {latent_shape}, embedding {len(snapshot_lags)+1} snapshots...")

    t_start = time.time()
    z0_clip = embedder.embed(x0_np, latent_shape=latent_shape)

    z_clip_snapshots = {}
    for i, lag in enumerate(snapshot_lags):
        snap = snapshots[lag]
        z_clip_snapshots[lag] = embedder.embed(snap, latent_shape=latent_shape)
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i+1}/{len(snapshot_lags)} snapshots")

    embed_time = time.time() - t_start
    print(f"Embedding complete in {embed_time:.1f}s")
    return z0_clip, z_clip_snapshots, embed_time


def compute_trajectory_averaged_ssa(z0_clip, z_clip_snapshots, snapshot_lags, dt, T):
    """Compute SSA using trajectory-averaged autocovariance estimator.

    For fixed-20 lags: pools all (s, s+τ) pairs from the fixed-20 grid.
    For log-binary lags: falls back to simple-pair (z0 vs z_tau).

    Returns dict with rho_sq arrays, SSA values, lag info.
    """
    from src.metrics.autocorrelation import aggregate_ssa

    N, D = z0_clip.shape
    gamma = D / N

    # Fixed-20 grid lags
    fixed20_lags = sorted([l for l in snapshot_lags if l % 20 == 0 and l > 0])
    log_lags = sorted([l for l in snapshot_lags if l not in fixed20_lags and l > 0])

    # Build full trajectory array for fixed-20 grid: (N, n_fixed, D)
    fixed20_with_zero = [0] + fixed20_lags
    n_times = len(fixed20_with_zero)

    z_traj = np.zeros((N, n_times, D), dtype=np.float32)
    z_traj[:, 0, :] = z0_clip
    for i, lag in enumerate(fixed20_lags):
        z_traj[:, i + 1, :] = z_clip_snapshots[lag]

    # Global mean centering
    mu_global = z_traj.mean(axis=(0, 1), keepdims=True)
    z_traj_c = z_traj - mu_global

    # C_traj(0)
    C0_traj = (z_traj_c ** 2).sum() / (N * n_times)

    # Simple-pair C(0) for comparison
    mu_simple = z0_clip.mean(axis=0)
    z0_c = z0_clip - mu_simple
    C0_simple = (z0_c ** 2).sum() / N

    print(f"  C0_traj={C0_traj:.6f}, C0_simple={C0_simple:.6f}, "
          f"ratio={C0_traj/C0_simple:.4f}")

    # --- Trajectory-averaged ρ²(τ) for fixed-20 lags ---
    rho_sq_traj = {}
    n_pairs_used = {}

    for target_lag_steps in fixed20_lags:
        target_idx_offset = target_lag_steps // 20
        pairs = []
        for i in range(n_times):
            j = i + target_idx_offset
            if j < n_times:
                actual_lag = fixed20_with_zero[j] - fixed20_with_zero[i]
                if actual_lag == target_lag_steps:
                    pairs.append((i, j))

        if len(pairs) == 0:
            continue

        cross_sum = 0.0
        for (i, j) in pairs:
            cross_sum += (z_traj_c[:, i, :] * z_traj_c[:, j, :]).sum()

        cross_avg = cross_sum / (N * len(pairs))
        rho = cross_avg / C0_traj
        rho_sq_traj[target_lag_steps] = float(rho ** 2)
        n_pairs_used[target_lag_steps] = len(pairs)

    # --- Simple-pair ρ²(τ) for log-binary lags ---
    rho_sq_simple = {}
    for lag in log_lags:
        if lag in z_clip_snapshots:
            z_tau_c = z_clip_snapshots[lag] - mu_simple
            cross = (z0_c * z_tau_c).sum() / N
            rho = cross / C0_simple
            rho_sq_simple[lag] = float(rho ** 2)

    # --- Combined ρ² for all lags ---
    all_lags = sorted(set(fixed20_lags + log_lags))
    rho_sq_combined = []
    estimator_type = []
    for lag in all_lags:
        if lag in rho_sq_traj:
            rho_sq_combined.append(rho_sq_traj[lag])
            estimator_type.append('traj')
        elif lag in rho_sq_simple:
            rho_sq_combined.append(rho_sq_simple[lag])
            estimator_type.append('simple')

    rho_sq_arr = np.array(rho_sq_combined)
    phys_lags = np.array(all_lags, dtype=float) * dt

    # --- Free-conv floor truncation ---
    sigma_f_sq = C0_traj / D
    sigma_f = np.sqrt(sigma_f_sq)

    tau_star = None
    floor_sq = []
    for lag in all_lags:
        tau_phys = lag * dt
        try:
            _, lp = fc_edges(tau_phys, sigma=sigma_f, gamma=gamma)
            rho_floor = (lp / C0_traj) ** 2 if C0_traj > 0 else 0
            floor_sq.append(rho_floor)
        except Exception:
            floor_sq.append(0)
    floor_sq = np.array(floor_sq)

    mask_above = rho_sq_arr > floor_sq
    if not mask_above.all():
        first_below = np.where(~mask_above)[0][0]
        tau_star_idx = max(0, first_below - 1)
        tau_star = all_lags[tau_star_idx]
    else:
        tau_star = all_lags[-1]

    # SSA: integrate ρ² up to tau_star
    if tau_star is not None:
        mask_ssa = np.array(all_lags) <= tau_star
        ssa_traj = float(aggregate_ssa(rho_sq_arr[mask_ssa], lags=phys_lags[mask_ssa]))
    else:
        ssa_traj = float(aggregate_ssa(rho_sq_arr, lags=phys_lags))

    ssa_traj_all = float(aggregate_ssa(rho_sq_arr, lags=phys_lags))

    # --- Simple-pair SSA for comparison ---
    rho_sq_simple_all = []
    simple_lags = []
    for lag in all_lags:
        if lag in z_clip_snapshots:
            z_tau_c = z_clip_snapshots[lag] - mu_simple
            cross = (z0_c * z_tau_c).sum() / N
            rho = cross / C0_simple
            rho_sq_simple_all.append(float(rho ** 2))
            simple_lags.append(lag)

    rho_sq_simple_arr = np.array(rho_sq_simple_all)
    phys_lags_simple = np.array(simple_lags, dtype=float) * dt

    if tau_star is not None:
        mask_simple = np.array(simple_lags) <= tau_star
        ssa_simple = float(aggregate_ssa(rho_sq_simple_arr[mask_simple],
                                         lags=phys_lags_simple[mask_simple]))
    else:
        ssa_simple = float(aggregate_ssa(rho_sq_simple_arr, lags=phys_lags_simple))

    return {
        "ssa_traj_averaged": ssa_traj,
        "ssa_traj_averaged_all": ssa_traj_all,
        "ssa_simple_pair": ssa_simple,
        "tau_star_steps": int(tau_star) if tau_star else None,
        "tau_star_phys": float(tau_star * dt) if tau_star else None,
        "C0_traj": float(C0_traj),
        "C0_simple": float(C0_simple),
        "sigma_f": float(sigma_f),
        "gamma": float(gamma),
        "embed_dim": int(D),
        "n_traj": int(N),
        "rho_sq_traj": {str(k): v for k, v in rho_sq_traj.items()},
        "rho_sq_simple": {str(k): v for k, v in zip(simple_lags, rho_sq_simple_all)},
        "rho_sq_combined": rho_sq_combined,
        "all_lags": all_lags,
        "n_pairs_per_lag": {str(k): v for k, v in n_pairs_used.items()},
        "estimator_type": estimator_type,
        "n_fixed20_lags": len(fixed20_lags),
        "n_log_lags": len(log_lags),
    }


# ============================================================
# Stage 2: Bootstrap confidence intervals
# ============================================================

def compute_traj_ssa_one_sample(z_traj, fixed20_lags, fixed20_with_zero,
                                 log_lags, log_data, dt, T, D, gamma):
    """Compute traj-averaged SSA for one (possibly resampled) set of trajectories.

    z_traj: (N, n_times, D) array
    Returns: ssa_traj (float)
    """
    N = z_traj.shape[0]
    n_times = z_traj.shape[1]

    # Trajectory-averaged ρ²(τ) with per-starting-time centering
    rho_sq_traj = {}
    for target_lag in fixed20_lags:
        target_idx_offset = target_lag // 20
        rho_list = []
        for i in range(n_times):
            j = i + target_idx_offset
            if j < n_times:
                if fixed20_with_zero[j] - fixed20_with_zero[i] == target_lag:
                    mu_s = z_traj[:, i, :].mean(axis=0)
                    z_s_c = z_traj[:, i, :] - mu_s
                    z_t_c = z_traj[:, j, :] - mu_s
                    cross = float((z_s_c * z_t_c).sum()) / N
                    var_s = float((z_s_c ** 2).sum()) / N
                    if var_s > 1e-12:
                        rho_list.append(cross / var_s)
        if rho_list:
            rho_sq_traj[target_lag] = float(np.mean(np.array(rho_list) ** 2))

    # Simple-pair ρ²(τ) for log-binary lags
    mu0 = z_traj[:, 0, :].mean(axis=0)
    z0_c = z_traj[:, 0, :] - mu0
    C0 = float((z0_c ** 2).sum()) / N
    rho_sq_simple = {}
    if C0 > 1e-12:
        for lag, z_tau in log_data.items():
            z_tau_c = z_tau - mu0
            cross = float((z0_c * z_tau_c).sum()) / N
            rho = cross / C0
            rho_sq_simple[lag] = float(rho ** 2)

    # Combine
    all_lags = sorted(set(list(rho_sq_traj.keys()) + list(rho_sq_simple.keys())))
    rho_sq_combined = {}
    for lag in all_lags:
        if lag in rho_sq_traj:
            rho_sq_combined[lag] = rho_sq_traj[lag]
        elif lag in rho_sq_simple:
            rho_sq_combined[lag] = rho_sq_simple[lag]

    # Free-conv floor truncation
    sigma_f_sq = C0 / D if C0 > 0 else 1e-6
    sigma_f = np.sqrt(sigma_f_sq)
    tau_star = T

    for lag in sorted(all_lags):
        tau_phys = lag * dt
        try:
            _, lp = fc_edges(tau_phys, sigma=sigma_f, gamma=gamma)
            rho_floor = (lp / C0) ** 2 if C0 > 0 else 0
            if lag in rho_sq_combined and rho_sq_combined[lag] < rho_floor:
                tau_star = lag
                break
        except Exception:
            pass

    # Integrate
    valid_lags = sorted([l for l in rho_sq_combined if l <= tau_star])
    if len(valid_lags) > 1:
        phys = np.array([l * dt for l in valid_lags])
        vals = np.array([rho_sq_combined[l] for l in valid_lags])
        return float(np.trapz(vals, phys))
    return 0.0


def bootstrap_one_cell(npz_path, json_path, B=2000, seed=42):
    """Run B bootstrap reps for one cell.

    Returns: (ssa_orig, se, ci_lo, ci_hi, ssa_boots)
    """
    data = np.load(npz_path)
    z0 = data['z0_clip']
    snapshot_lags = data['snapshot_lags']
    N, D = z0.shape
    gamma = D / N

    jdata = json.load(open(json_path))
    dt = jdata.get('dt', 0.005)
    T = jdata.get('T', 2000)

    # Build trajectory array
    fixed20_lags = sorted([int(l) for l in snapshot_lags if l % 20 == 0 and l > 0])
    log_lags = sorted([int(l) for l in snapshot_lags if l % 20 != 0 and l > 0])
    fixed20_with_zero = [0] + fixed20_lags
    n_times = len(fixed20_with_zero)

    z_traj = np.zeros((N, n_times, D), dtype=np.float32)
    z_traj[:, 0, :] = z0
    for i, lag in enumerate(fixed20_lags):
        z_traj[:, i + 1, :] = data[f'z_clip_{lag}']

    log_data_full = {}
    for lag in log_lags:
        key = f'z_clip_{lag}'
        if key in data:
            log_data_full[lag] = data[key]

    # Original estimate
    ssa_orig = compute_traj_ssa_one_sample(
        z_traj, fixed20_lags, fixed20_with_zero,
        log_lags, log_data_full, dt, T, D, gamma)

    # Bootstrap
    rng = np.random.default_rng(seed)
    ssa_boots = np.zeros(B)
    for b in range(B):
        idx = rng.choice(N, size=N, replace=True)
        z_traj_b = z_traj[idx]
        log_data_b = {lag: arr[idx] for lag, arr in log_data_full.items()}
        ssa_boots[b] = compute_traj_ssa_one_sample(
            z_traj_b, fixed20_lags, fixed20_with_zero,
            log_lags, log_data_b, dt, T, D, gamma)

    se = float(ssa_boots.std())
    ci_lo = float(np.percentile(ssa_boots, 2.5))
    ci_hi = float(np.percentile(ssa_boots, 97.5))

    return ssa_orig, se, ci_lo, ci_hi, ssa_boots


# ============================================================
# Stage 3: DA significance pipeline
# ============================================================

def load_clip_trajectories(traj_path):
    """Load saved CLIP trajectories."""
    data = np.load(traj_path, allow_pickle=True)
    z0_clip = data["z0_clip"]
    snapshot_lags = data["snapshot_lags"].tolist()

    z_snapshots = {}
    for lag in snapshot_lags:
        key = f"z_clip_{lag}"
        if key in data:
            z_snapshots[lag] = data[key]

    return z0_clip, z_snapshots, snapshot_lags


def compute_trajectory_averaged_C_sym(z0_clip, z_snapshots, snapshot_lags, target_lag):
    """Compute symmetrized trajectory-averaged autocovariance at target_lag.

    Uses all (s, s+target_lag) pairs from the fixed-20 grid.

    Returns:
        C_sym: (D, D) symmetrized autocovariance matrix.
        n_pairs: number of (s, s+τ) pairs used.
    """
    from scipy.linalg import eigh  # noqa: F811

    fixed20_lags = sorted([l for l in snapshot_lags if l % 20 == 0 and l > 0])
    fixed20_with_zero = [0] + fixed20_lags

    N, D = z0_clip.shape
    is_fixed20 = (target_lag % 20 == 0) and target_lag > 0

    if is_fixed20 and target_lag <= fixed20_lags[-1]:
        idx_offset = target_lag // 20

        all_z = [z0_clip]
        for lag in fixed20_lags:
            all_z.append(z_snapshots.get(lag))

        available = [z for z in all_z if z is not None]
        mu_traj = np.mean(np.stack(available, axis=1), axis=1, keepdims=True)

        C_sum = np.zeros((D, D), dtype=np.float64)
        n_pairs = 0

        for i in range(len(fixed20_with_zero)):
            j = i + idx_offset
            if j >= len(fixed20_with_zero):
                break
            if all_z[i] is None or all_z[j] is None:
                continue
            actual_lag = fixed20_with_zero[j] - fixed20_with_zero[i]
            if actual_lag != target_lag:
                continue

            z_i = all_z[i] - mu_traj.squeeze(1)
            z_j = all_z[j] - mu_traj.squeeze(1)
            C_sum += (z_j.T @ z_i) / N
            n_pairs += 1

        if n_pairs > 0:
            C = C_sum / n_pairs
        else:
            mu = z0_clip.mean(axis=0)
            z0_c = z0_clip - mu
            z_tau_c = z_snapshots[target_lag] - mu
            C = (z_tau_c.T @ z0_c) / N
            n_pairs = 1
    else:
        mu = z0_clip.mean(axis=0)
        z0_c = z0_clip - mu
        z_tau = z_snapshots.get(target_lag)
        if z_tau is None:
            raise ValueError(f"No snapshot at lag {target_lag}")
        z_tau_c = z_tau - mu
        C = (z_tau_c.T @ z0_c).astype(np.float64) / N
        n_pairs = 1

    C_sym = 0.5 * (C + C.T)
    return C_sym, n_pairs


def spike_counting_pipeline(z0_clip, z_snapshots, snapshot_lags, dt):
    """Count eigenvalue spikes above free-conv edge at each lag.

    Returns dict with taus, counts, eigenvalues_top, tau_star, plateau_count.
    """
    from scipy.linalg import eigh

    N, D = z0_clip.shape
    gamma = D / N

    fixed20_lags = sorted([l for l in snapshot_lags if l % 20 == 0 and l > 0])
    fixed20_with_zero = [0] + fixed20_lags
    available_z = [z0_clip] + [z_snapshots[l] for l in fixed20_lags if l in z_snapshots]
    z_stack = np.stack(available_z, axis=1)
    mu_traj = z_stack.mean(axis=1, keepdims=True)
    z_centered = z_stack - mu_traj
    sigma_f_sq = (z_centered ** 2).sum() / (N * z_stack.shape[1] * D)
    sigma_f = np.sqrt(sigma_f_sq)

    print(f"  sigma_f = {sigma_f:.6f}, gamma = {gamma:.4f}, D = {D}, N = {N}")

    eval_lags = [l for l in fixed20_lags if l in z_snapshots]

    taus_phys = []
    counts = []
    eigenvalues_top = {}

    for lag in eval_lags:
        tau_phys = lag * dt

        C_sym, n_pairs = compute_trajectory_averaged_C_sym(
            z0_clip, z_snapshots, snapshot_lags, lag)

        eigvals = eigh(C_sym, eigvals_only=True)[::-1].copy()

        try:
            _, lp = fc_edges(tau_phys, sigma=sigma_f, gamma=gamma)
        except Exception:
            lp = 0.0

        count = int(np.sum(eigvals > lp))
        taus_phys.append(tau_phys)
        counts.append(count)
        eigenvalues_top[lag] = eigvals[:20].tolist()

    taus_phys = np.array(taus_phys)
    counts = np.array(counts)

    # τ* selection: plateau-max criterion
    plateau_count = int(counts.max()) if len(counts) > 0 else 0
    if plateau_count > 0:
        on_plateau = counts == plateau_count
        plateau_indices = np.where(on_plateau)[0]
        tau_star_idx = plateau_indices[-1]
        tau_star_lag = eval_lags[tau_star_idx]
    else:
        tau_star_lag = eval_lags[-1] if eval_lags else None

    print(f"  Plateau count m̂ = {plateau_count}")
    if tau_star_lag:
        print(f"  τ* = {tau_star_lag} steps ({tau_star_lag * dt:.4f}s)")

    return {
        "taus_phys": taus_phys.tolist(),
        "counts": counts.tolist(),
        "eval_lags_steps": eval_lags,
        "eigenvalues_top": eigenvalues_top,
        "plateau_count": plateau_count,
        "tau_star_steps": tau_star_lag,
        "tau_star_phys": float(tau_star_lag * dt) if tau_star_lag else None,
        "sigma_f": float(sigma_f),
        "gamma": float(gamma),
    }


def da_directions_at_tau_star(z0_clip, z_snapshots, snapshot_lags, tau_star_steps, n_dirs=5):
    """Extract DA directions at τ*.

    Returns: (eigvals, eigvecs[:, :n_dirs], C_sym)
    """
    from scipy.linalg import eigh

    C_sym, n_pairs = compute_trajectory_averaged_C_sym(
        z0_clip, z_snapshots, snapshot_lags, tau_star_steps)

    eigvals, eigvecs = eigh(C_sym)
    eigvals = eigvals[::-1].copy()
    eigvecs = eigvecs[:, ::-1].copy()

    print(f"  Top eigenvalues at τ*: {eigvals[:5]}")
    if eigvals[1] > 0:
        print(f"  Gap λ₁/λ₂ = {eigvals[0]/eigvals[1]:.2f}")

    return eigvals, eigvecs[:, :n_dirs], C_sym


def silverman_significance(z0_clip, eigvecs, m_hat, alpha=0.05, B=1000, seed=42):
    """Silverman modality test on DA projections with Bonferroni correction.

    Returns list of dicts per direction with p_value, significant, h_crit.
    """
    from silverman_test import silverman_test

    mu = z0_clip.mean(axis=0)
    z0_c = z0_clip - mu
    alpha_bonf = alpha / m_hat

    results = []
    for j in range(m_hat):
        proj = z0_c @ eigvecs[:, j]
        test_result = silverman_test(proj, n_modes=2, B=B, seed=seed + j)
        test_result["significant"] = test_result["p_value"] <= alpha_bonf
        test_result["alpha_bonferroni"] = alpha_bonf
        test_result["direction_index"] = j
        results.append(test_result)
        print(f"  DA{j+1}: h_crit={test_result['h_crit']:.4f}, "
              f"p={test_result['p_value']:.4f}, "
              f"{'PASS (bimodal)' if test_result['significant'] else 'FAIL (unimodal)'}")

    return results


def text_probe_analysis(z0_clip, eigvecs, mu, device="cuda"):
    """Interpret DA directions via text probes."""
    from sdxl_components import TextProbeAnalyzer
    from scipy.stats import spearmanr  # noqa: F401

    analyzer = TextProbeAnalyzer(device=device)
    analyzer.define_probes()

    z0_c = z0_clip - mu
    results = {}

    for j in range(eigvecs.shape[1]):
        direction = eigvecs[:, j]
        cosines = analyzer.score_direction(direction)
        proj_scores = z0_c @ direction
        correlations = analyzer.correlate_with_ranking(z0_clip, proj_scores)

        results[f"DA{j+1}"] = {
            "cosines": cosines,
            "correlations": correlations,
            "proj_std": float(proj_scores.std()),
        }

    return results


def compute_pc1(z0_clip):
    """Compute PC1 for cos(DA1, PC1) comparison."""
    mu = z0_clip.mean(axis=0)
    z0_c = z0_clip - mu
    U, S, Vt = np.linalg.svd(z0_c, full_matrices=False)
    return Vt[0]


# ============================================================
# CLI entry points
# ============================================================

def run_recompute(args):
    """Stage 1: Recompute trajectory-averaged SSA from checkpoint."""
    if args.checkpoint:
        x0_np, snapshots, snapshot_lags, var_f = load_checkpoint(args.checkpoint)
    else:
        raise ValueError("Provide --checkpoint")

    print(f"Loaded: N={x0_np.shape[0]}, D_latent={x0_np.shape[1]}, "
          f"{len(snapshot_lags)} lags, var_f={var_f:.6f}")

    z0_clip, z_clip_snapshots, embed_time = embed_trajectories(
        x0_np, snapshots, snapshot_lags, args)

    print("\nComputing trajectory-averaged SSA...")
    results = compute_trajectory_averaged_ssa(
        z0_clip, z_clip_snapshots, snapshot_lags, args.dt, args.T)

    print(f"\n  SSA (trajectory-averaged): {results['ssa_traj_averaged']:.6f}")
    print(f"  SSA (simple-pair):         {results['ssa_simple_pair']:.6f}")
    print(f"  τ* = {results['tau_star_steps']} steps "
          f"({results['tau_star_phys']:.4f}s)")

    # Save JSON
    out_json = {
        "name": args.name,
        "dt": args.dt,
        "T": args.T,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "embed_time_s": embed_time,
        **results,
    }
    json_path = os.path.join(args.output_dir, f"traj_avg_{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # Save CLIP trajectories for bootstrap/DA pipeline
    if args.save_clip_traj:
        traj_path = os.path.join(args.output_dir, f"z_clip_traj_{args.name}.npz")
        save_data = {"z0_clip": z0_clip, "snapshot_lags": np.array(snapshot_lags)}
        for lag, z in z_clip_snapshots.items():
            save_data[f"z_clip_{lag}"] = z
        np.savez_compressed(traj_path, **save_data)
        print(f"Saved CLIP trajectories: {traj_path}")


def run_bootstrap(args):
    """Stage 2: Bootstrap CI for trajectory-averaged SSA."""
    os.chdir(SCRIPT_DIR)
    npz_files = sorted(glob.glob(os.path.join(args.output_dir, "z_clip_traj_*.npz")))
    if args.clip_traj:
        npz_files = [args.clip_traj]
    print(f"Found {len(npz_files)} CLIP trajectory files, B={args.B}")

    results = {}
    for npz_path in npz_files:
        name = os.path.basename(npz_path).replace("z_clip_traj_", "").replace(".npz", "")
        if args.cell and name != args.cell:
            continue
        json_path = os.path.join(os.path.dirname(npz_path), f"traj_avg_{name}.json")
        if not os.path.exists(json_path):
            print(f"  SKIP {name}: no JSON")
            continue

        t0 = time.time()
        ssa_orig, se, ci_lo, ci_hi, _ = bootstrap_one_cell(
            npz_path, json_path, B=args.B)
        elapsed = time.time() - t0
        results[name] = {
            'ssa_traj': ssa_orig, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'B': args.B,
        }
        print(f"  {name}: Ŝ={ssa_orig:.3f} ± {se:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] ({elapsed:.0f}s)")

    # Save
    out_path = os.path.join(args.output_dir, args.bootstrap_output)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Summary table
    print(f"\n{'Name':<45} {'Ŝ':>6} {'SE':>6} {'95% CI':>15}")
    print("-" * 75)
    for name in sorted(results, key=lambda k: -results[k]['ssa_traj']):
        r = results[name]
        print(f"{name:<45} {r['ssa_traj']:6.3f} {r['se']:6.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


def run_da_significance(args):
    """Stage 3: DA significance pipeline."""
    print(f"Loading CLIP trajectories: {args.clip_traj}")
    z0_clip, z_snapshots, snapshot_lags = load_clip_trajectories(args.clip_traj)
    N, D = z0_clip.shape
    print(f"  N={N}, D={D}, {len(snapshot_lags)} lags")

    # Stage 1: Spike counting
    print("\n=== Stage 1: Spike Counting ===")
    spike_results = spike_counting_pipeline(z0_clip, z_snapshots, snapshot_lags, args.dt)

    tau_star = spike_results["tau_star_steps"]
    m_hat = spike_results["plateau_count"]

    if m_hat == 0:
        print("\n  No spikes detected above noise floor. Exiting.")
        out = {"name": args.name, "spike_results": spike_results,
               "conclusion": "no_significant_directions"}
        json_path = os.path.join(args.output_dir, f"da_pipeline_{args.name}.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # Stage 2: DA directions at τ*
    print(f"\n=== Stage 2: DA Directions at τ*={tau_star} ===")
    eigvals, eigvecs, C_sym = da_directions_at_tau_star(
        z0_clip, z_snapshots, snapshot_lags, tau_star, n_dirs=min(m_hat + 2, 10))

    pc1 = compute_pc1(z0_clip)
    cos_da1_pc1 = float(np.abs(eigvecs[:, 0] @ pc1))
    print(f"  |cos(DA1, PC1)| = {cos_da1_pc1:.4f}")

    # Stage 3: Silverman modality test
    print(f"\n=== Stage 3: Silverman Test (m̂={m_hat}, α={args.alpha}) ===")
    silverman_results = silverman_significance(
        z0_clip, eigvecs, m_hat, alpha=args.alpha, B=args.B_silverman, seed=args.seed)

    n_significant = sum(1 for r in silverman_results if r["significant"])
    print(f"\n  {n_significant}/{m_hat} directions pass Silverman test")

    # Stage 4: Text probe interpretation (optional)
    probe_results = None
    if not args.no_text_probes:
        print("\n=== Stage 4: Text Probe Interpretation ===")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu = z0_clip.mean(axis=0)
            probe_results = text_probe_analysis(z0_clip, eigvecs[:, :min(m_hat, 5)], mu, device)
        except Exception as e:
            print(f"  Text probes failed: {e}")

    # Stage 5: MC null comparison (optional)
    null_results = None
    if args.null_traj:
        from scipy.linalg import eigh
        print("\n=== Stage 5: MC Null Comparison ===")
        z0_null, z_null_snaps, null_lags = load_clip_trajectories(args.null_traj)
        available = [l for l in null_lags if l in z_null_snaps]
        closest = min(available, key=lambda l: abs(l - tau_star))
        C_null, _ = compute_trajectory_averaged_C_sym(
            z0_null, z_null_snaps, null_lags, closest)
        null_eigvals = eigh(C_null, eigvals_only=True)[::-1].copy()
        null_results = {
            "null_lambda1": float(null_eigvals[0]),
            "null_lambda2": float(null_eigvals[1]),
            "null_lag_used": closest,
            "observed_lambda1": float(eigvals[0]),
            "ratio_observed_null": float(eigvals[0] / null_eigvals[0])
                if null_eigvals[0] > 0 else None,
        }
        print(f"  Null λ₁ = {null_eigvals[0]:.6e}, Observed λ₁ = {eigvals[0]:.6e}")
        if null_results["ratio_observed_null"]:
            print(f"  Ratio = {null_results['ratio_observed_null']:.2f}×")

    # Assemble output
    output = {
        "name": args.name,
        "dt": args.dt,
        "T": args.T,
        "N": N,
        "D": D,
        "spike_results": spike_results,
        "tau_star_steps": tau_star,
        "tau_star_phys": float(tau_star * args.dt),
        "plateau_count_m_hat": m_hat,
        "eigenvalues_at_tau_star": eigvals[:20].tolist(),
        "gap_lambda1_lambda2": float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else None,
        "cos_da1_pc1": cos_da1_pc1,
        "silverman_results": silverman_results,
        "n_significant_directions": n_significant,
        "alpha": args.alpha,
        "B_silverman": args.B_silverman,
    }
    if probe_results:
        output["text_probes"] = probe_results
    if null_results:
        output["mc_null"] = null_results

    json_path = os.path.join(args.output_dir, f"da_pipeline_{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Save eigenvectors
    eigvec_path = os.path.join(args.output_dir, f"da_eigvecs_{args.name}.npz")
    np.savez_compressed(eigvec_path,
                        eigvals=eigvals,
                        eigvecs=eigvecs,
                        C_sym_tau_star=C_sym,
                        tau_star=tau_star,
                        pc1=pc1)
    print(f"Saved eigenvectors: {eigvec_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory-averaged SSA + bootstrap + DA significance pipeline")

    parser.add_argument("--mode", choices=["recompute", "bootstrap", "da_significance"],
                        default="recompute",
                        help="Pipeline stage to run (default: recompute)")

    # Common args
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Recompute mode
    parser.add_argument("--checkpoint", type=str, help="Langevin checkpoint .npz")
    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--save_clip_traj", action="store_true", default=True)

    # Bootstrap mode
    parser.add_argument("--clip_traj", type=str, help="z_clip_traj_*.npz path")
    parser.add_argument("--B", type=int, default=2000, help="Bootstrap replicates")
    parser.add_argument("--cell", type=str, default=None,
                        help="Run only this cell (name)")
    parser.add_argument("--bootstrap_output", type=str, default="bootstrap_results.json")

    # DA significance mode
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Family-wise error rate for Bonferroni")
    parser.add_argument("--B_silverman", type=int, default=1000,
                        help="Bootstrap replicates for Silverman test")
    parser.add_argument("--null_traj", type=str, default=None,
                        help="Null CLIP trajectories for MC comparison")
    parser.add_argument("--no_text_probes", action="store_true",
                        help="Skip text probe analysis")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "recompute":
        run_recompute(args)
    elif args.mode == "bootstrap":
        run_bootstrap(args)
    elif args.mode == "da_significance":
        run_da_significance(args)


if __name__ == "__main__":
    main()
