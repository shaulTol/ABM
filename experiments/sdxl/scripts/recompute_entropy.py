"""Recompute kNN entropy with k=2 from saved latents (VAE decode + CLIP encode)."""
import os
import sys
import json
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import RealCLIPEmbedder
from compute_sdxl_ssa_v2 import compute_knn_entropy


def pca_project(X, k):
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    return Xc @ eigvecs[:, :k], eigvals[:k]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", required=True, help="Path to latents npz with x0 key")
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--clip_batch_size", type=int, default=1)
    p.add_argument("--output_json", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, args.height // 8, args.width // 8)

    # Load VAE
    print("=== Loading SDXL VAE ===")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    # Load x0 and CLIP-encode
    print(f"=== Loading latents from {args.latents_npz} ===")
    x0 = np.load(args.latents_npz)["x0"]
    print(f"  x0 shape: {x0.shape}")

    print("=== CLIP encoding ===")
    embedder = RealCLIPEmbedder(vae, device=device, clip_batch_size=args.clip_batch_size)
    z0 = embedder.embed(x0, latent_shape=latent_shape)
    print(f"  z0 shape: {z0.shape}")

    # Compute entropy with k=2
    k_nn = 2
    k_pcas = [2, 5, 10]
    results = {"k_nn": k_nn, "N": z0.shape[0], "D": z0.shape[1]}

    print(f"\n=== kNN entropy (k={k_nn}) ===")
    for k_pca in k_pcas:
        X_proj, eigvals = pca_project(z0, k_pca)
        h = compute_knn_entropy(torch.from_numpy(X_proj), k=k_nn)
        results[f"knn_k{k_nn}_pca{k_pca}"] = float(h)
        print(f"  k_pca={k_pca:>2}: H_knn = {h:.4f}")

        # Also Gaussian for reference
        h_gauss = 0.5 * np.sum(np.log(2 * np.pi * np.e * np.maximum(eigvals, 1e-30)))
        results[f"gaussian_pca{k_pca}"] = float(h_gauss)
        print(f"  k_pca={k_pca:>2}: H_gauss = {h_gauss:.4f}")

    # KDE for low dims
    from scipy.stats import gaussian_kde
    for k_pca in [2, 5]:
        X_proj, _ = pca_project(z0, k_pca)
        kde = gaussian_kde(X_proj.T)
        rng = np.random.RandomState(42)
        samples = kde.resample(5000, seed=rng)
        log_pdf = kde.logpdf(samples)
        h_kde = float(-np.mean(log_pdf))
        results[f"kde_pca{k_pca}"] = h_kde
        print(f"  k_pca={k_pca:>2}: H_kde = {h_kde:.4f}")

    # Save
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
