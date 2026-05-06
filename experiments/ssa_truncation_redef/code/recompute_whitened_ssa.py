#!/usr/bin/env python3
"""
Recompute SSA in PCA-whitened CLIP space from saved Langevin checkpoints.

For each checkpoint:
1. Load latent snapshots (200 x 65536)
2. VAE decode + CLIP encode each snapshot → 200 x 768
3. PCA-whiten using z0 (lag=0) embeddings: z_w = (z - mu) @ V @ Lambda^{-1/2}
4. Compute ρ²(τ) = [tr(C_w(τ)) / d_eff]² where d_eff = number of PCs used
5. Compute Ŝ = delta * sum(ρ²) truncated at noise floor

In whitened space, an isotropic Gaussian gives Ŝ = 1.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.join(CODEDIR, '..')
REPODIR = os.path.join(BASEDIR, '..', '..')
sys.path.insert(0, REPODIR)
sys.path.insert(0, os.path.join(REPODIR, 'experiments', 'sdxl', 'scripts'))

from sdxl_components import RealCLIPEmbedder


def whiten_clip(z0, z_snapshots, alpha=0.95):
    """
    PCA-whiten CLIP embeddings.

    Args:
        z0: (N, 768) CLIP embeddings at lag 0
        z_snapshots: dict lag -> (N, 768) CLIP embeddings
        alpha: fraction of variance to keep (determines number of PCs)

    Returns:
        z0_w: (N, k) whitened embeddings
        z_snaps_w: dict lag -> (N, k)
        k: number of PCs used
    """
    mu = z0.mean(axis=0)
    z0_c = z0 - mu

    # SVD on centered data
    U, S, Vt = np.linalg.svd(z0_c, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(explained)

    # Keep k PCs capturing alpha of variance
    k = int(np.searchsorted(cumvar, alpha) + 1)
    k = min(k, len(S))

    print(f"  PCA whitening: {k} PCs capture {cumvar[k-1]*100:.1f}% of variance (alpha={alpha})")
    print(f"  Top 5 explained: {explained[:5]}")

    # Whitening transform: W = V_k @ diag(1/s_k) (scaled so each PC has unit variance)
    V_k = Vt[:k].T  # (768, k)
    s_k = S[:k]
    # Project and normalize: z_w = (z - mu) @ V_k / s_k * sqrt(N-1)
    # This makes Cov(z_w) = I_k
    scale = np.sqrt(len(z0) - 1) / s_k  # (k,)

    z0_w = z0_c @ V_k * scale  # (N, k)

    # Verify whitening
    cov_w = np.cov(z0_w.T)
    print(f"  Whitened cov diagonal: mean={np.diag(cov_w).mean():.4f}, std={np.diag(cov_w).std():.4f}")

    z_snaps_w = {}
    for lag, z in z_snapshots.items():
        z_snaps_w[lag] = (z - mu) @ V_k * scale

    return z0_w, z_snaps_w, k


def compute_rho_sq_whitened(z0_w, z_snaps_w, snapshot_lags):
    """Compute trace-normalized ρ²(τ) in whitened space."""
    N, k = z0_w.shape
    tr_cov0 = k  # whitened → tr(Σ) = k (each PC has variance 1)

    rho_sq = []
    for lag in snapshot_lags:
        if lag not in z_snaps_w:
            rho_sq.append(0.0)
            continue
        z_tau = z_snaps_w[lag]
        # C(τ) = (1/N) z0_w^T @ z_tau  (both centered by same mu)
        cross = (z0_w * z_tau).sum(axis=0).mean()  # mean over dimensions of per-dim cross-corr
        # Actually: tr(C(τ)) = sum_j (1/N) sum_i z0_w[i,j] * z_tau[i,j]
        tr_cross = (z0_w * z_tau).mean(axis=0).sum()  # sum of per-dim means
        rho = tr_cross / tr_cov0  # = tr_cross / k
        rho_sq.append(float(rho ** 2))

    return np.array(rho_sq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to langevin checkpoint .npz")
    parser.add_argument("--name", required=True, help="Run name for output")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dt", type=float, default=0.0005, help="dt used in original run")
    parser.add_argument("--alpha", type=float, default=0.95, help="Fraction of variance for PCA")
    parser.add_argument("--clip_batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results_whitened")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"\n=== Loading checkpoint: {args.checkpoint} ===")
    t0 = time.time()
    ckpt = np.load(args.checkpoint, allow_pickle=True)
    x0_np = ckpt["x0_flat"]
    snapshot_lags = sorted([int(k.split("_")[1]) for k in ckpt.keys() if k.startswith("snapshot_") and k != "snapshot_lags"])
    snapshots_np = {lag: ckpt[f"snapshot_{lag}"] for lag in snapshot_lags}
    var_f_latent = float(ckpt["var_f"])
    N = x0_np.shape[0]
    d_latent = x0_np.shape[1]
    print(f"  Loaded in {time.time()-t0:.1f}s: N={N}, d_latent={d_latent}, {len(snapshot_lags)} snapshots")
    print(f"  var_f (latent) = {var_f_latent:.6f}")

    # Infer latent shape
    if d_latent == 65536:
        latent_shape = (4, 128, 128)  # 1024x1024
    elif d_latent == 16384:
        latent_shape = (4, 64, 64)    # 512x512
    else:
        raise ValueError(f"Unknown latent dim {d_latent}")
    print(f"  Inferred latent_shape = {latent_shape}")

    # Load VAE + CLIP
    print(f"\n=== Loading VAE + CLIP ===")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet
    del pipe
    torch.cuda.empty_cache()

    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    # CLIP embed x0
    print(f"\n=== CLIP embedding x0 ({N} samples) ===")
    t0 = time.time()
    z0_clip = clip_embedder.embed(x0_np, latent_shape=latent_shape)
    print(f"  z0: shape={z0_clip.shape}, time={time.time()-t0:.1f}s")

    # CLIP embed all snapshots
    z_snapshots = {}
    print(f"\n=== CLIP embedding {len(snapshot_lags)} snapshots ===")
    t0 = time.time()
    for i, lag in enumerate(snapshot_lags):
        z_snap = clip_embedder.embed(snapshots_np[lag], latent_shape=latent_shape)
        z_snapshots[lag] = z_snap
        if i % 20 == 0:
            print(f"  lag={lag}: done ({i+1}/{len(snapshot_lags)})")
    embed_time = time.time() - t0
    print(f"  All embedded in {embed_time:.1f}s")

    # Free GPU
    del clip_embedder, vae
    torch.cuda.empty_cache()

    # Compute standard (unwhitened) ρ² for comparison
    print(f"\n=== Standard ρ² (unwhitened) ===")
    var_f_clip = np.var(z0_clip, axis=0).mean()
    print(f"  var_f (CLIP) = {var_f_clip:.6f}")
    rho_sq_std = compute_rho_sq_whitened(
        z0_clip - z0_clip.mean(axis=0),
        {lag: z - z0_clip.mean(axis=0) for lag, z in z_snapshots.items()},
        snapshot_lags
    )
    # Fix: use proper trace normalization for unwhitened
    d_clip = z0_clip.shape[1]
    tr_cov = np.trace(np.cov(z0_clip.T))
    rho_sq_std_proper = []
    mu_clip = z0_clip.mean(axis=0)
    for lag in snapshot_lags:
        z_tau = z_snapshots[lag]
        tr_cross = ((z0_clip - mu_clip) * (z_tau - mu_clip)).mean(axis=0).sum()
        rho = tr_cross / tr_cov
        rho_sq_std_proper.append(float(rho ** 2))
    rho_sq_std_proper = np.array(rho_sq_std_proper)

    # PCA-whiten and compute ρ²
    print(f"\n=== PCA whitening (alpha={args.alpha}) ===")
    z0_w, z_snaps_w, k = whiten_clip(z0_clip, z_snapshots, alpha=args.alpha)

    print(f"\n=== Whitened ρ² ===")
    rho_sq_w = compute_rho_sq_whitened(z0_w, z_snaps_w, snapshot_lags)

    # Compute Ŝ using fixed20 lags
    lags_arr = np.array(snapshot_lags)
    fixed20_mask = (lags_arr % 20 == 0) & (lags_arr > 0)
    fixed20_lags = lags_arr[fixed20_mask]
    delta = 20 * args.dt

    nf = 1.0 / N

    for label, rho_sq in [("standard", rho_sq_std_proper), ("whitened", rho_sq_w)]:
        rho_f20 = rho_sq[fixed20_mask]
        below = rho_f20 < nf
        if below.any():
            idx = np.argmax(below)
            s_hat = float(delta * rho_f20[:idx].sum())
            tau_star = float(fixed20_lags[idx] * args.dt)
            hit = True
        else:
            s_hat = float(delta * rho_f20.sum())
            tau_star = float(fixed20_lags[-1] * args.dt)
            hit = False
        print(f"  {label:10s}: Ŝ={s_hat:.4f}, τ*={tau_star:.4f}, hit_floor={hit}, rho2_last={rho_sq[-1]:.6f}")

    # Save results
    results = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "dt": args.dt,
        "alpha": args.alpha,
        "n_pcs": k,
        "N": N,
        "d_latent": d_latent,
        "d_clip": d_clip,
        "var_f_latent": var_f_latent,
        "var_f_clip": float(var_f_clip),
        "snapshot_lags": snapshot_lags,
        "rho_sq_standard": rho_sq_std_proper.tolist(),
        "rho_sq_whitened": rho_sq_w.tolist(),
    }

    out_path = os.path.join(args.output_dir, f"whitened_ssa_{args.name}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
