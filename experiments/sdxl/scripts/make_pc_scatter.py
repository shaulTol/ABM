"""Plot images at their PC1 x PC2 coordinates as a thumbnail scatter."""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import RealCLIPEmbedder, decode_latents_to_pil


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
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--thumb_size", type=int, default=72)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lat = dict(np.load(args.latents_npz))
    x0 = lat["x0"]
    N = x0.shape[0]
    print(f"Loaded {N} latents from {args.latents_npz}")

    # Load VAE
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()

    # Decode to thumbnails
    print("Decoding latents to images...")
    images = decode_latents_to_pil(vae, x0, batch_size=8, device=device)

    # Embed in CLIP
    print("Embedding in CLIP...")
    clip_embedder = RealCLIPEmbedder(vae, device=device)
    z_clip = clip_embedder.embed(x0)
    print(f"  CLIP embeddings: {z_clip.shape}")

    # PCA
    eigvals, eigvecs, mu = pca_top_k(z_clip, k=2)
    Xc = z_clip - mu
    pc1 = Xc @ eigvecs[:, 0]
    pc2 = Xc @ eigvecs[:, 1]

    var_ratio1 = eigvals[0] / eigvals.sum() if eigvals.sum() > 0 else 0
    var_ratio2 = eigvals[1] / eigvals.sum() if eigvals.sum() > 0 else 0

    # Plot
    print("Making scatter plot...")
    fig, ax = plt.subplots(figsize=(16, 14))

    for i in range(N):
        thumb = images[i].resize((args.thumb_size, args.thumb_size))
        im = OffsetImage(np.array(thumb), zoom=1.0)
        ab = AnnotationBbox(im, (pc1[i], pc2[i]), frameon=False, pad=0)
        ax.add_artist(ab)

    margin = 0.05
    x_range = pc1.max() - pc1.min()
    y_range = pc2.max() - pc2.min()
    ax.set_xlim(pc1.min() - margin * x_range, pc1.max() + margin * x_range)
    ax.set_ylim(pc2.min() - margin * y_range, pc2.max() + margin * y_range)
    ax.set_xlabel(f"PC1 ({var_ratio1:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio2:.1%} variance)", fontsize=12)
    ax.set_title("Images in CLIP PC1 × PC2 space", fontsize=14, fontweight="bold")

    out = args.output_path
    if out is None:
        base = os.path.basename(args.latents_npz).replace("clip_da_latents_", "").replace(".npz", "")
        out = os.path.join(os.path.dirname(args.latents_npz), f"pc_scatter_{base}.png")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
