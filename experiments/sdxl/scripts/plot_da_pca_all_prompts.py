#!/usr/bin/env python3
"""Plot DA1 vs PC1 histograms + sorted image grids for all 8 prompts in a single PDF."""

import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

VAE_SCALE = 0.13025

PROMPTS = [
    ("cat", "a photograph of a cat sitting on a windowsill"),
    ("mountains", "a landscape photograph of mountains at sunset"),
    ("flowers", "a still life painting of flowers in a vase"),
    ("street", "a street photograph of a busy city intersection"),
    ("food", "a photograph of a plate of food at a restaurant"),
    ("elderly", "a portrait of an elderly person"),
    ("car", "a photograph of a car parked on a street"),
    ("person", "a superrealistic professional photograph of a person"),
]


def decode_latents(vae, latents_flat, latent_shape, device, batch_size=8):
    """Decode flat latents to PIL images."""
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


def make_image_grid(images, indices, ncols=10, thumb_size=128):
    """Create a grid of thumbnails in the given order."""
    n = len(indices)
    nrows = (n + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * thumb_size, nrows * thumb_size), (255, 255, 255))
    for i, idx in enumerate(indices):
        row, col = divmod(i, ncols)
        img = images[idx].resize((thumb_size, thumb_size), Image.LANCZOS)
        grid.paste(img, (col * thumb_size, row * thumb_size))
    return grid


def plot_prompt(pdf, label, prompt_text, images, da1_scores, pc1_scores, cos_da1_pc1,
                da1_corr, ssa, da1_gap, da1_tau):
    """Add 4 pages per prompt: DA1 hist, DA1 grid, PC1 hist, PC1 grid."""
    N = len(images)

    # DA1 top correlations
    da1_top3 = sorted(da1_corr.items(), key=lambda x: abs(x[1]['rho']), reverse=True)[:3]
    da1_top_str = ", ".join(f"{n}({v['rho']:+.2f})" for n, v in da1_top3)

    for direction_name, scores in [("DA1", da1_scores), ("PC1", pc1_scores)]:
        # Page 1: Histogram
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(scores, bins=30, density=True, alpha=0.6, color="steelblue" if direction_name == "DA1" else "coral",
                edgecolor="k", linewidth=0.5)
        ax.set_xlabel(f"{direction_name} projection")
        ax.set_ylabel("Density")
        title = f"{label}: {direction_name} histogram"
        if direction_name == "DA1":
            title += f"  |  SSA={ssa:.3f}, gap={da1_gap:.2f}x, τ*={da1_tau}"
            title += f"\nDA1 correlates: {da1_top_str}  |  |cos(DA1,PC1)|={cos_da1_pc1:.3f}"
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Sorted image grid
        order = np.argsort(scores)  # low to high
        grid = make_image_grid(images, order, ncols=20, thumb_size=80)

        fig, ax = plt.subplots(figsize=(16, 0.5 + N / 20 * 0.8))
        ax.imshow(grid)
        ax.set_title(f"{label}: images sorted by {direction_name} (left=low, right=high)", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_da_pcavda")
    parser.add_argument("--output", default="results_da_pcavda/da_vs_pca_all_prompts.pdf")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_shape = (4, args.height // 8, args.width // 8)

    # Load VAE
    print("Loading SDXL VAE...")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()
    print("VAE loaded.")

    # Load CLIP model for text probes (to compute PC1 correlations)
    from transformers import CLIPModel, CLIPTokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    probe_pairs = {
        "gender": ("a photograph of a man", "a photograph of a woman"),
        "age": ("a photograph of a young person", "a photograph of an old person"),
        "realism": ("a realistic photograph", "an artistic painting"),
        "brightness": ("a bright photograph", "a dark photograph"),
        "framing": ("a close-up portrait", "a wide-angle photograph"),
        "complexity": ("a simple photograph", "a complex detailed photograph"),
        "expression": ("a person with a neutral expression", "a person smiling"),
        "glasses": ("a person without glasses", "a person wearing glasses"),
    }

    # Compute text probe axes
    text_axes = {}
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
            text_axes[name] = axis

    # Also need RealCLIPEmbedder for image CLIP embeddings
    from sdxl_components import RealCLIPEmbedder
    clip_embedder = RealCLIPEmbedder(vae, device=device, clip_batch_size=16)

    # x0 paths
    x0_paths = {
        "cat": "results_multiprompt_sweep/gen_mp_p0_cat_gs7p5_quick_x0.npz",
        "mountains": "results_multiprompt_sweep/gen_mp_p1_mountains_gs7p5_quick_x0.npz",
        "flowers": "results_multiprompt_sweep/gen_mp_p2_flowers_gs7p5_quick_x0.npz",
        "street": "results_multiprompt_sweep/gen_mp_p3_street_gs7p5_quick_x0.npz",
        "food": "results_multiprompt_sweep/gen_mp_p4_food_gs7p5_quick_x0.npz",
        "elderly": "results_multiprompt_sweep/gen_mp_p5_elderly_gs7p5_quick_x0.npz",
        "car": "results_multiprompt_sweep/gen_mp_p6_car_gs7p5_quick_x0.npz",
        "person": "results_guidance_sweep/gen_sweep_gs7p5_quick_x0.npz",
    }

    from scipy.stats import spearmanr

    with PdfPages(args.output) as pdf:
        for label, prompt_text in PROMPTS:
            print(f"\n=== {label} ===")
            name = f"{label}_gs7p5_n200"

            # Load DA results
            js_path = os.path.join(args.results_dir, f"clip_da_{name}.json")
            npz_path = os.path.join(args.results_dir, f"clip_da_{name}.npz")
            if not os.path.exists(js_path):
                print(f"  Skipping {label}: no JSON")
                continue

            js = json.load(open(js_path))
            da_npz = np.load(npz_path)
            da1 = da_npz["da1_eigvec"]
            mu = da_npz["mu_clip"]

            # Load x0 latents and decode
            x0_path = x0_paths[label]
            x0_data = np.load(x0_path)
            x0_flat = x0_data["x0"] if "x0" in x0_data else x0_data["x0_flat"]
            N = x0_flat.shape[0]
            print(f"  Decoding {N} images...")
            images = decode_latents(vae, x0_flat, latent_shape, device)

            # CLIP embed for PCA
            print(f"  CLIP embedding {N} images...")
            z0_clip = clip_embedder.embed(x0_flat, latent_shape=latent_shape)
            print(f"  z0_clip shape: {z0_clip.shape}")

            # PCA
            z0c = z0_clip - z0_clip.mean(axis=0)
            cov = (z0c.T @ z0c) / N
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = eigvals.argsort()[::-1].copy()
            pc1 = eigvecs[:, idx[0]]

            cos_da1_pc1 = abs(float(np.dot(da1, pc1)))

            # Project onto DA1 and PC1
            da1_scores = z0c @ da1
            pc1_scores = z0c @ pc1

            # PC1 correlations with text probes
            pc1_corr = {}
            for pname, axis in text_axes.items():
                proj = z0c @ axis
                rho, pval = spearmanr(pc1_scores, proj)
                pc1_corr[pname] = {"rho": float(rho), "pvalue": float(pval)}

            ssa = js.get("ssa_fixed20", js.get("ssa_clip", 0))
            da1_gap = js.get("da1_gap", 0)
            da1_tau = js.get("da1_optimal_lag", 0)
            da1_corr = js.get("da1_correlations", {})

            print(f"  |cos(DA1,PC1)| = {cos_da1_pc1:.4f}")
            print(f"  DA1 top: {max(da1_corr.items(), key=lambda x: abs(x[1]['rho']))}")
            print(f"  PC1 top: {max(pc1_corr.items(), key=lambda x: abs(x[1]['rho']))}")

            plot_prompt(pdf, label, prompt_text, images, da1_scores, pc1_scores,
                       cos_da1_pc1, da1_corr, ssa, da1_gap, da1_tau)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
