"""
Compute "square direction" in CLIP space:
  direction = mean(square_embeddings) - mean(nobox_embeddings)
Then project Square images onto this direction and plot histogram.
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from glob import glob

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--square_latents", type=str, required=True,
                   help="Path to clip_da_latents_autoresearch_square.npz")
    p.add_argument("--nobox_images_dir", type=str, required=True,
                   help="Directory with nobox img_*.png files")
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--output_dir", type=str, default="results_autoresearch")
    args = p.parse_args()
    return args


def vae_decode_latents(x0_flat, vae, device, batch_size=4):
    """Decode flat latents (N, C*H*W) to PIL images."""
    N = x0_flat.shape[0]
    # Infer shape: 16384 = 4*64*64 (512px) or 65536 = 4*128*128 (1024px)
    D = x0_flat.shape[1]
    C = 4
    H = W = int((D // C) ** 0.5)
    scale = 0.13025  # SDXL VAE scale

    images = []
    for i in range(0, N, batch_size):
        batch = torch.tensor(x0_flat[i:i+batch_size]).reshape(-1, C, H, W).to(device)
        batch = batch / scale
        with torch.no_grad():
            decoded = vae.decode(batch.half()).sample.float()
        decoded = (decoded.clamp(-1, 1) + 1) / 2
        for j in range(decoded.shape[0]):
            img = decoded[j].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img))
        if (i + batch_size) % 40 == 0:
            print(f"  Decoded {min(i+batch_size, N)}/{N}", flush=True)
    return images


def clip_embed_images(images, model, processor, device, batch_size=16):
    """Embed PIL images with CLIP."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load VAE for decoding Square latents ---
    print("=== Loading VAE ===", flush=True)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float16).to(device)

    # --- Decode Square latents ---
    print("=== Decoding Square latents ===", flush=True)
    sq_data = np.load(args.square_latents)
    sq_x0 = sq_data["x0"]  # (200, 16384)
    sq_images = vae_decode_latents(sq_x0, vae, device)
    print(f"  Decoded {len(sq_images)} Square images")

    # --- Load nobox images ---
    print("=== Loading nobox images ===", flush=True)
    nobox_paths = sorted(glob(os.path.join(args.nobox_images_dir, "sample_*.png")))
    if not nobox_paths:
        nobox_paths = sorted(glob(os.path.join(args.nobox_images_dir, "img_*.png")))
    nobox_images = [Image.open(p).convert("RGB") for p in nobox_paths]
    print(f"  Loaded {len(nobox_images)} nobox images")

    # --- Free VAE, load CLIP ---
    del vae
    torch.cuda.empty_cache()

    print("=== Loading CLIP ===", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    # --- Embed both sets ---
    print("=== Embedding Square images ===", flush=True)
    sq_embs = clip_embed_images(sq_images, clip_model, clip_processor, device)
    print(f"  Square embeddings: {sq_embs.shape}")

    print("=== Embedding nobox images ===", flush=True)
    nb_embs = clip_embed_images(nobox_images, clip_model, clip_processor, device)
    print(f"  Nobox embeddings: {nb_embs.shape}")

    # --- Compute square direction ---
    sq_mean = sq_embs.mean(axis=0)
    nb_mean = nb_embs.mean(axis=0)
    square_dir = sq_mean - nb_mean
    square_dir = square_dir / np.linalg.norm(square_dir)
    print(f"  ||sq_mean - nb_mean|| = {np.linalg.norm(sq_mean - nb_mean):.4f}")

    # --- Project Square embeddings onto square direction ---
    sq_proj = sq_embs @ square_dir
    nb_proj = nb_embs @ square_dir

    # --- Save ---
    out_npz = os.path.join(args.output_dir, "square_direction_analysis.npz")
    np.savez(out_npz,
             sq_embs=sq_embs, nb_embs=nb_embs,
             square_dir=square_dir,
             sq_proj=sq_proj, nb_proj=nb_proj,
             sq_mean=sq_mean, nb_mean=nb_mean)
    print(f"  Saved: {out_npz}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of Square projections
    ax = axes[0]
    kde_sq = gaussian_kde(sq_proj)
    xs = np.linspace(min(sq_proj.min(), nb_proj.min()) - 0.3,
                     max(sq_proj.max(), nb_proj.max()) + 0.3, 300)
    ax.hist(sq_proj, bins=30, density=True, alpha=0.5, color='steelblue',
            edgecolor='k', linewidth=0.5, label='Square (with box)')
    ax.hist(nb_proj, bins=30, density=True, alpha=0.5, color='orange',
            edgecolor='k', linewidth=0.5, label='Nobox (no box)')
    ax.plot(xs, kde_sq(xs), 'b-', lw=2)
    kde_nb = gaussian_kde(nb_proj)
    ax.plot(xs, kde_nb(xs), 'r-', lw=2)
    ax.set_xlabel('Projection onto "square direction"')
    ax.set_ylabel('Density')
    ax.set_title('Square vs Nobox: projection onto mean-difference direction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PCA scatter colored by square direction
    ax2 = axes[1]
    all_embs = np.concatenate([sq_embs, nb_embs], axis=0)
    mu = all_embs.mean(axis=0)
    Xc = all_embs - mu
    cov = (Xc.T @ Xc) / len(all_embs)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1].copy()
    pc1 = eigvecs[:, idx[0]]
    pc2 = eigvecs[:, idx[1]]
    proj1 = Xc @ pc1
    proj2 = Xc @ pc2
    N_sq = len(sq_embs)
    ax2.scatter(proj1[:N_sq], proj2[:N_sq], c='steelblue', alpha=0.5, s=15, label='Square')
    ax2.scatter(proj1[N_sq:], proj2[N_sq:], c='orange', alpha=0.5, s=15, label='Nobox')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA of combined embeddings')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('"Square direction" analysis: face_boxed vs face_nobox scribble', fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "square_direction_analysis.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
