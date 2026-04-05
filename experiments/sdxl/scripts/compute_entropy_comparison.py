"""
compute_entropy_comparison.py — Post-hoc differential entropy in CLIP PCA space.

Computes three entropy estimators at various PCA dimensions:
  1. Gaussian approximation: H_gauss(k) = 0.5 * sum(log(2*pi*e * lambda_i))
  2. kNN (Kozachenko-Leonenko): reuses compute_knn_entropy from compute_sdxl_ssa_v2
  3. KDE (scipy.stats.gaussian_kde): feasible up to ~10D with N=200

Usage:
    python scripts/compute_entropy_comparison.py \
        --latents_npz results_clip_da/clip_da_latents_dps_face_dt001.npz \
        --model_id stabilityai/sdxl-turbo \
        --output_path results_clip_da/entropy_dps_face_dt001.json
"""

import os
import sys
import json
import argparse
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def pca_project(X, k):
    """PCA-project (N, D) data to (N, k). Returns projected data and eigenvalues."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return Xc @ eigvecs[:, :k], eigvals[:k]


def gaussian_entropy(eigvals):
    """H_gauss = 0.5 * sum(log(2*pi*e * lambda_i)) for eigenvalues lambda_i."""
    eigvals = np.maximum(eigvals, 1e-30)
    return 0.5 * np.sum(np.log(2 * np.pi * np.e * eigvals))


def kde_entropy(X_proj, n_eval=5000, seed=42):
    """KDE-based entropy estimate using scipy.stats.gaussian_kde.

    Estimates H = -E[log p(x)] via Monte Carlo on KDE samples.
    """
    from scipy.stats import gaussian_kde
    N, d = X_proj.shape
    kde = gaussian_kde(X_proj.T)  # expects (d, N)
    rng = np.random.RandomState(seed)
    samples = kde.resample(n_eval, seed=rng)  # (d, n_eval)
    log_pdf = kde.logpdf(samples)
    return float(-np.mean(log_pdf))


def main():
    p = argparse.ArgumentParser(description="Post-hoc entropy comparison in CLIP PCA space")
    p.add_argument("--latents_npz", type=str, required=True,
                   help="Path to clip_da_latents_*.npz (contains x0)")
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=16)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--knn_k", type=int, default=10, help="k for kNN entropy")
    args = p.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load latents
    print(f"Loading latents from {args.latents_npz}")
    data = np.load(args.latents_npz)
    x0 = data["x0"]  # (N, d_latent)
    N = x0.shape[0]
    print(f"  x0 shape: {x0.shape}, N={N}")

    # Embed in CLIP
    print(f"\n=== Embedding {N} samples via real CLIP ===")
    from sdxl_components import RealCLIPEmbedder

    # Load VAE
    if "turbo" in args.model_id.lower():
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae",
            torch_dtype=torch.float16,
        ).to(device).eval()
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id, torch_dtype=torch.float16, variant="fp16",
        )
        vae = pipe.vae.to(device).float().eval()
        del pipe
        torch.cuda.empty_cache()

    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    t0 = time.time()
    z0_clip = clip_embedder.embed(x0)  # (N, 768)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s, shape={z0_clip.shape}")

    # Free GPU
    del clip_embedder, vae
    torch.cuda.empty_cache()

    # PCA dimensions to evaluate
    k_pca_gauss = [5, 10, 20, 50]
    k_pca_knn = [5, 10, 20, 50]
    k_pca_kde = [2, 5]

    results = {
        "latents_npz": args.latents_npz,
        "n_samples": N,
        "clip_dim": int(z0_clip.shape[1]),
        "embed_time_s": embed_time,
    }

    # 1. Gaussian entropy
    print("\n=== Gaussian entropy ===")
    # Compute PCA once at max k
    max_k = max(k_pca_gauss)
    _, all_eigvals = pca_project(z0_clip, max_k)

    gauss_results = {}
    for k in k_pca_gauss:
        h = gaussian_entropy(all_eigvals[:k])
        gauss_results[str(k)] = float(h)
        print(f"  k={k:3d}: H_gauss = {h:.4f}")
    results["gaussian_entropy"] = gauss_results

    # 2. kNN entropy
    print("\n=== kNN entropy (Kozachenko-Leonenko) ===")
    from compute_sdxl_ssa_v2 import compute_knn_entropy

    knn_results = {}
    for k in k_pca_knn:
        X_proj, _ = pca_project(z0_clip, k)
        h = compute_knn_entropy(torch.from_numpy(X_proj), k=args.knn_k)
        knn_results[str(k)] = float(h)
        print(f"  k_pca={k:3d}: H_knn = {h:.4f}")
    results["knn_entropy"] = knn_results

    # 3. KDE entropy (only low-dim)
    print("\n=== KDE entropy ===")
    kde_results = {}
    for k in k_pca_kde:
        X_proj, _ = pca_project(z0_clip, k)
        h = kde_entropy(X_proj)
        kde_results[str(k)] = float(h)
        print(f"  k_pca={k:3d}: H_kde = {h:.4f}")
    results["kde_entropy"] = kde_results

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
