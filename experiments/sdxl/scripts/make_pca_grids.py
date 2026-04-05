"""Generate PC1/PC2-sorted top/bottom grids for autoresearch experiments.

Loads latents, embeds in CLIP, computes PCA, makes grids.
Also makes DA1/DA2 grids if DA npz exists (for consistency).
"""

import os
import sys
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import (
    RealCLIPEmbedder, make_topbottom_grid,
)


def pca_top_k(X, k=2):
    N = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    return eigvals[idx[:k]], eigvecs[:, idx[:k]], mu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", type=str, required=True)
    p.add_argument("--da_npz", type=str, default=None)
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lat = dict(np.load(args.latents_npz))
    x0 = lat["x0"]
    N = x0.shape[0]
    print(f"Loaded {N} latents from {args.latents_npz}")

    out_dir = args.output_dir or os.path.dirname(args.latents_npz)
    basename = os.path.basename(args.latents_npz).replace("clip_da_latents_", "").replace(".npz", "")

    # Load VAE for decoding
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()

    # Embed in CLIP
    print("Embedding in CLIP...")
    clip_embedder = RealCLIPEmbedder(vae, device=device)
    z_clip = clip_embedder.embed(x0)
    print(f"  CLIP embeddings: {z_clip.shape}")

    # PCA
    print("Computing PCA...")
    eigvals, eigvecs, mu = pca_top_k(z_clip, k=2)
    Xc = z_clip - mu
    pc1_scores = Xc @ eigvecs[:, 0]
    pc2_scores = Xc @ eigvecs[:, 1]

    # PC1 grid
    pc1_path = os.path.join(out_dir, f"{basename}_pc1_grid.png")
    print(f"Making PC1 grid...")
    make_topbottom_grid(vae, x0, pc1_scores, n=8, path=pc1_path, device=device)

    # PC2 grid
    pc2_path = os.path.join(out_dir, f"{basename}_pc2_grid.png")
    print(f"Making PC2 grid...")
    make_topbottom_grid(vae, x0, pc2_scores, n=8, path=pc2_path, device=device)

    # DA grids if DA npz provided
    if args.da_npz and os.path.exists(args.da_npz):
        da = dict(np.load(args.da_npz))
        if "da1_eigvec" in da and "mu_clip" in da:
            da1_scores = (z_clip - da["mu_clip"]) @ da["da1_eigvec"]
            da1_path = os.path.join(out_dir, f"{basename}_da1_grid.png")
            print(f"Making DA1 grid...")
            make_topbottom_grid(vae, x0, da1_scores, n=8, path=da1_path, device=device)

        if "da2_eigvec" in da and "mu_clip" in da:
            da2_scores = (z_clip - da["mu_clip"]) @ da["da2_eigvec"]
            da2_path = os.path.join(out_dir, f"{basename}_da2_grid.png")
            print(f"Making DA2 grid...")
            make_topbottom_grid(vae, x0, da2_scores, n=8, path=da2_path, device=device)

    print("Done!")


if __name__ == "__main__":
    main()
