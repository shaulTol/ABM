"""
generate_and_entropy.py — Generate SDXL images, CLIP-encode, compute entropy suite.

Standalone Phase 1 script: no Langevin needed. Saves x0 latents, CLIP embeddings,
and a comprehensive entropy table for later analysis.

Usage:
    python scripts/generate_and_entropy.py \
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \
        --prompt "a superrealistic professional photograph of a person" \
        --guidance_scale 7.5 --gen_steps 50 \
        --n_samples 2000 --height 1024 --width 1024 \
        --name gs7p5 --output_dir results_guidance_sweep
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from compute_sdxl_ssa_v2 import generate_latents, compute_knn_entropy
from sdxl_components import RealCLIPEmbedder


def pca_project(X, k):
    """PCA-project (N, D) data to (N, k). Returns projected data and eigenvalues."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1].copy()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return Xc @ eigvecs[:, :k], eigvals[:k]


def gaussian_entropy(eigvals):
    """H_gauss = 0.5 * sum(log(2*pi*e * lambda_i))."""
    eigvals = np.maximum(eigvals, 1e-30)
    return 0.5 * np.sum(np.log(2 * np.pi * np.e * eigvals))


def kde_entropy(X_proj, n_eval=5000, seed=42):
    """KDE-based entropy via Monte Carlo on KDE samples."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(X_proj.T)
    rng = np.random.RandomState(seed)
    samples = kde.resample(n_eval, seed=rng)
    log_pdf = kde.logpdf(samples)
    return float(-np.mean(log_pdf))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate SDXL images + CLIP encode + entropy suite"
    )
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--prompt", type=str, default="a superrealistic professional photograph of a person")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--gen_steps", type=int, default=50)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--gen_batch_size", type=int, default=1)
    p.add_argument("--clip_batch_size", type=int, default=4)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, args.height // 8, args.width // 8)
    d_latent = int(np.prod(latent_shape))

    print(f"=== generate_and_entropy.py ===")
    print(f"Model: {args.model_id}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"N samples: {args.n_samples}")
    print(f"Latent shape: {latent_shape} ({d_latent}D)")
    print(f"Device: {device}")

    # =====================================================================
    # Stage 1: Generate latents
    # =====================================================================
    print(f"\n=== Loading SDXL pipeline ===")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to(device)

    print(f"\n=== Generating {args.n_samples} latents ===")
    t0 = time.time()
    latents = generate_latents(
        pipe, args.prompt, args.n_samples,
        guidance_scale=args.guidance_scale, gen_steps=args.gen_steps,
        height=args.height, width=args.width,
        batch_size=args.gen_batch_size, seed=args.seed,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s ({gen_time/3600:.1f}h)")

    latents_flat = latents.view(latents.shape[0], -1).float()
    var_f = latents_flat.var(dim=0).mean().item()
    print(f"  var_f = {var_f:.6f}")
    print(f"  Latent range: [{latents.min():.3f}, {latents.max():.3f}]")

    # Save x0 latents
    x0_path = os.path.join(args.output_dir, f"gen_{args.name}_x0.npz")
    np.savez_compressed(x0_path, x0=latents_flat.numpy())
    print(f"  Saved x0 latents: {x0_path} ({os.path.getsize(x0_path)/1e6:.1f} MB)")

    # =====================================================================
    # Stage 2: Free UNet, load CLIP, encode all images
    # =====================================================================
    print("\n=== Freeing UNet, loading CLIP ===")
    vae = pipe.vae.to(device).float().eval()
    pipe.unet = pipe.unet.to("cpu")
    del pipe
    torch.cuda.empty_cache()

    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    print(f"\n=== CLIP-encoding {args.n_samples} images ===")
    t0 = time.time()
    z0_clip = clip_embedder.embed(latents_flat.numpy(), latent_shape=latent_shape)
    clip_time = time.time() - t0
    print(f"  Encoded in {clip_time:.1f}s, shape={z0_clip.shape}")

    # Save CLIP embeddings
    clip_path = os.path.join(args.output_dir, f"gen_{args.name}_clip.npz")
    np.savez_compressed(clip_path, z0=z0_clip)
    print(f"  Saved CLIP embeddings: {clip_path} ({os.path.getsize(clip_path)/1e6:.1f} MB)")

    # =====================================================================
    # Stage 3: Compute entropy suite
    # =====================================================================
    print("\n=== Computing entropy suite ===")
    entropy_results = {}

    # PCA dimensions for different estimators
    k_pca_all = [2, 5, 10, 15, 20, 30, 50]
    k_pca_kde = [2, 5]
    knn_k_values = [3, 5, 10]

    # Pre-compute PCA for max k
    max_k = max(k_pca_all)
    _, all_eigvals = pca_project(z0_clip, max_k)

    # Gaussian entropy
    print("\n  --- Gaussian entropy ---")
    gauss_results = {}
    for k in k_pca_all:
        h = gaussian_entropy(all_eigvals[:k])
        gauss_results[str(k)] = float(h)
        print(f"    H_gauss(d={k}): {h:.4f}")
    entropy_results["gaussian"] = gauss_results

    # kNN entropy at multiple k values
    print("\n  --- kNN entropy ---")
    knn_results = {}
    for knn_k in knn_k_values:
        knn_k_results = {}
        for k_pca in k_pca_all:
            X_proj, _ = pca_project(z0_clip, k_pca)
            h = compute_knn_entropy(torch.from_numpy(X_proj), k=knn_k)
            knn_k_results[str(k_pca)] = float(h)
            print(f"    H_knn(k={knn_k}, d={k_pca}): {h:.4f}")
        knn_results[f"k{knn_k}"] = knn_k_results
    entropy_results["knn"] = knn_results

    # KDE entropy (only low d)
    print("\n  --- KDE entropy ---")
    kde_results = {}
    for k_pca in k_pca_kde:
        X_proj, _ = pca_project(z0_clip, k_pca)
        h = kde_entropy(X_proj)
        kde_results[str(k_pca)] = float(h)
        print(f"    H_kde(d={k_pca}): {h:.4f}")
    entropy_results["kde"] = kde_results

    # Save entropy JSON
    entropy_out = {
        "name": args.name,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "n_samples": args.n_samples,
        "height": args.height,
        "width": args.width,
        "var_f": var_f,
        "gen_time_s": gen_time,
        "clip_time_s": clip_time,
        "clip_model": args.clip_model,
        "entropy": entropy_results,
    }
    entropy_path = os.path.join(args.output_dir, f"gen_{args.name}_entropy.json")
    with open(entropy_path, "w") as f:
        json.dump(entropy_out, f, indent=2)
    print(f"\n  Saved entropy: {entropy_path}")

    # =====================================================================
    # Summary
    # =====================================================================
    total_time = gen_time + clip_time
    print(f"\n{'='*60}")
    print(f"Summary: {args.name}")
    print(f"  Model: {args.model_id}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  N samples: {args.n_samples}")
    print(f"  var_f: {var_f:.6f}")
    print(f"  CLIP embed dim: {z0_clip.shape[1]}")
    print(f"  H_gauss(d=5): {gauss_results['5']:.4f}")
    print(f"  H_knn(k=3, d=5): {knn_results['k3']['5']:.4f}")
    print(f"  H_kde(d=2): {kde_results['2']:.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"  Files:")
    print(f"    {x0_path}")
    print(f"    {clip_path}")
    print(f"    {entropy_path}")


if __name__ == "__main__":
    main()
