#!/usr/bin/env python3
"""Paper figure: Elderly DA1 vs PC1 — top 5 and bottom 5 images for each direction."""

import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

VAE_SCALE = 0.13025


def decode_latents(vae, latents_flat, latent_shape, device, batch_size=8):
    images = []
    N = latents_flat.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(latents_flat[start:end]).float().to(device)
        z = batch.view(-1, *latent_shape) / VAE_SCALE
        with torch.no_grad():
            decoded = vae.decode(z, return_dict=False)[0]
        pixels = ((decoded + 1.0) / 2.0).clamp(0, 1)
        pixel_bytes = (pixels * 255).byte().cpu()
        for img_t in pixel_bytes:
            images.append(Image.fromarray(img_t.permute(1, 2, 0).numpy()))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_da_pcavda")
    parser.add_argument("--output", default="results_da_pcavda/fig_elderly_da_vs_pca.png")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--k", type=int, default=5, help="Number of top/bottom images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, 128, 128)  # 1024x1024

    # Load VAE
    print("Loading SDXL VAE...")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    # Load DA results
    js = json.load(open(os.path.join(args.results_dir, "clip_da_elderly_gs7p5_n200.json")))
    da_npz = np.load(os.path.join(args.results_dir, "clip_da_elderly_gs7p5_n200.npz"))
    da1 = da_npz["da1_eigvec"]

    # Load and decode images
    x0_path = "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz"
    x0_flat = np.load(x0_path)["x0"]
    print(f"Decoding {x0_flat.shape[0]} images...")
    images = decode_latents(vae, x0_flat, latent_shape, device)

    # CLIP embed for PCA
    from sdxl_components import RealCLIPEmbedder
    clip_embedder = RealCLIPEmbedder(vae, device=device, clip_batch_size=16)
    print("CLIP embedding...")
    z0_clip = clip_embedder.embed(x0_flat, latent_shape=latent_shape)

    # PCA
    z0c = z0_clip - z0_clip.mean(axis=0)
    cov = (z0c.T @ z0c) / z0_clip.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1].copy()
    pc1 = eigvecs[:, idx[0]]
    pc1_var = eigvals[idx[0]] / eigvals.sum()

    cos_da1_pc1 = abs(float(np.dot(da1, pc1)))

    # Project
    da1_scores = z0c @ da1
    pc1_scores = z0c @ pc1

    # Get top/bottom k
    k = args.k
    da1_order = np.argsort(da1_scores)
    pc1_order = np.argsort(pc1_scores)

    da1_low = da1_order[:k]
    da1_high = da1_order[-k:][::-1]
    pc1_low = pc1_order[:k]
    pc1_high = pc1_order[-k:][::-1]

    # DA1 correlations
    da1_corr = js.get("da1_correlations", {})
    da1_top3 = sorted(da1_corr.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)[:3]

    # PC1 correlations
    from scipy.stats import spearmanr
    from transformers import CLIPModel, CLIPTokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    probe_pairs = {
        "gender": ("a photograph of a man", "a photograph of a woman"),
        "age": ("a photograph of a young person", "a photograph of an old person"),
        "realism": ("a realistic photograph", "an artistic painting"),
    }
    pc1_corrs = {}
    for name, (pos, neg) in probe_pairs.items():
        with torch.no_grad():
            tok_p = tokenizer(pos, return_tensors="pt", padding=True).to(device)
            tok_n = tokenizer(neg, return_tensors="pt", padding=True).to(device)
            emb_p = clip_model.get_text_features(**tok_p)
            emb_n = clip_model.get_text_features(**tok_n)
            emb_p = emb_p / emb_p.norm(dim=-1, keepdim=True)
            emb_n = emb_n / emb_n.norm(dim=-1, keepdim=True)
            axis = (emb_p - emb_n).squeeze().cpu().numpy()
            axis = axis / np.linalg.norm(axis)
        proj = z0c @ axis
        rho, _ = spearmanr(pc1_scores, proj)
        pc1_corrs[name] = rho
    pc1_top3 = sorted(pc1_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    # --- Build figure ---
    thumb = 180
    fig_w = k * thumb / 100 + 1.5  # inches
    fig_h = 5.5

    fig = plt.figure(figsize=(fig_w + 1, fig_h))
    gs = GridSpec(4, k + 1, figure=fig, hspace=0.15, wspace=0.05,
                  left=0.01, right=0.99, top=0.98, bottom=0.02,
                  width_ratios=[0.6] + [1] * k)

    def place_row(row, indices, label_text):
        # Label
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.9, 0.5, label_text, ha="right", va="center",
                      fontsize=9, fontweight="bold", transform=ax_label.transAxes)
        ax_label.axis("off")
        # Images
        for col, img_idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(images[img_idx])
            ax.axis("off")

    place_row(0, da1_high, "DA1\nhigh")
    place_row(1, da1_low, "DA1\nlow")
    place_row(2, pc1_high, "PC1\nhigh")
    place_row(3, pc1_low, "PC1\nlow")

    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")

    # Also save high-res version
    hires = args.output.replace(".png", "_hires.png")
    fig.savefig(hires, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {hires}")
    plt.close(fig)


if __name__ == "__main__":
    main()
