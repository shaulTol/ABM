"""
plot_dps_scatter.py — Scatter plots of DPS scribble samples in CLIP space.

For each experiment: PC1 vs PC2, and PC1 vs gender direction.
Points colored by gender score.

Usage:
    python scripts/plot_dps_scatter.py \
        --results_dir results_clip_da \
        --model_id stabilityai/sdxl-turbo
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(EXP_DIR, "data")
sys.path.insert(0, SCRIPT_DIR)

matplotlib.rcParams.update({"font.size": 9})

EXPERIMENTS = [
    ("dps_face_dt001", "DPS Face (zeta5)", os.path.join(DATA_DIR, "dps_face_scribble.png")),
    ("dps_first_dt001", "DPS First (seed 1)", os.path.join(DATA_DIR, "dps_first_scribble.jpeg")),
    ("dps_second_dt001", "DPS Second (seed 2)", os.path.join(DATA_DIR, "dps_second_scribble.jpeg")),
]


def embed_latents(latents_npz_path, model_id, clip_model, device, clip_batch_size=16):
    """Load latents, VAE decode, CLIP embed → (N, 768)."""
    from sdxl_components import RealCLIPEmbedder

    data = np.load(latents_npz_path)
    x0 = data["x0"]

    # Load VAE
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16,
    ).to(device).eval()

    embedder = RealCLIPEmbedder(
        vae, clip_model_name=clip_model, device=device,
        clip_batch_size=clip_batch_size,
    )
    z0_clip = embedder.embed(x0)
    del embedder, vae
    torch.cuda.empty_cache()
    return z0_clip, x0


def get_gender_axis(clip_model_name, device):
    """Compute normalized gender direction in CLIP space."""
    from sdxl_components import TextProbeAnalyzer
    analyzer = TextProbeAnalyzer(clip_model_name=clip_model_name, device=device)
    analyzer.define_probes()
    return analyzer.probe_axes["gender"], analyzer


def pca_project(z, k=2):
    """PCA on (N, D). Returns scores (N, k), eigvecs (D, k), mu."""
    mu = z.mean(axis=0)
    zc = z - mu
    cov = (zc.T @ zc) / z.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx[:k]]
    eigvecs = eigvecs[:, idx[:k]]
    scores = zc @ eigvecs
    return scores, eigvecs, mu, eigvals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str,
                   default=os.path.join(EXP_DIR, "results_clip_da"))
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--output_path", type=str, default=None)
    args = p.parse_args()

    rd = args.results_dir
    out_path = args.output_path or os.path.join(rd, "dps_scatter.pdf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get gender axis
    print("=== Loading gender axis ===")
    gender_axis, analyzer = get_gender_axis(args.clip_model, device)

    # Embed all experiments
    embeddings = {}
    for name, label, scribble in EXPERIMENTS:
        latents_path = os.path.join(rd, f"clip_da_latents_{name}.npz")
        if not os.path.exists(latents_path):
            print(f"  SKIP {name}: {latents_path} not found")
            continue
        print(f"\n=== Embedding {name} ===")
        z0, x0 = embed_latents(latents_path, args.model_id, args.clip_model,
                                device, clip_batch_size=16)
        embeddings[name] = z0
        print(f"  Embedded: {z0.shape}")

    # Also load DA1 eigenvectors from NPZ
    da_info = {}
    for name, label, scribble in EXPERIMENTS:
        npz_path = os.path.join(rd, f"clip_da_{name}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            da_info[name] = {
                "da1_eigvec": d["da1_eigvec"],
                "da2_eigvec": d["da2_eigvec"],
                "mu": d["mu_clip"],
            }

    # Load JSON for SSA values
    json_info = {}
    for name, label, scribble in EXPERIMENTS:
        json_path = os.path.join(rd, f"clip_da_{name}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                json_info[name] = json.load(f)

    print(f"\n=== Generating scatter plots ===")
    from PIL import Image

    with PdfPages(out_path) as pdf:
        # ====== Page 1: PC1 vs PC2 (3 experiments side by side) ======
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("CLIP Space: PC1 vs PC2 (colored by gender score)",
                      fontsize=13, fontweight="bold")

        for i, (name, label, scribble) in enumerate(EXPERIMENTS):
            ax = axes[i]
            if name not in embeddings:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(label)
                continue

            z0 = embeddings[name]
            scores_pca, eigvecs, mu, eigvals = pca_project(z0, k=2)
            gender_scores = z0 @ gender_axis

            sc = ax.scatter(scores_pca[:, 0], scores_pca[:, 1],
                           c=gender_scores, cmap="RdBu_r", s=15, alpha=0.7,
                           edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="gender score", shrink=0.8)

            # Annotate with SSA
            ssa = json_info.get(name, {}).get("ssa_clip", "?")
            ssa_g = json_info.get(name, {}).get("ssa_gender_sliced", "?")
            ax.set_xlabel(f"PC1 ({eigvals[0]:.2f})")
            ax.set_ylabel(f"PC2 ({eigvals[1]:.2f})")
            ax.set_title(f"{label}\nSSA={ssa:.4f}, sliced={ssa_g:.4f}", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 1: PC1 vs PC2")

        # ====== Page 2: PC1 vs Gender (3 experiments side by side) ======
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("CLIP Space: PC1 vs Gender Direction (colored by gender score)",
                      fontsize=13, fontweight="bold")

        for i, (name, label, scribble) in enumerate(EXPERIMENTS):
            ax = axes[i]
            if name not in embeddings:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(label)
                continue

            z0 = embeddings[name]
            scores_pca, eigvecs, mu, eigvals = pca_project(z0, k=2)
            gender_scores = z0 @ gender_axis

            sc = ax.scatter(scores_pca[:, 0], gender_scores,
                           c=gender_scores, cmap="RdBu_r", s=15, alpha=0.7,
                           edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="gender score", shrink=0.8)

            # Correlation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(scores_pca[:, 0], gender_scores)

            ssa = json_info.get(name, {}).get("ssa_clip", "?")
            ax.set_xlabel(f"PC1 ({eigvals[0]:.2f})")
            ax.set_ylabel("Gender direction")
            ax.set_title(f"{label}\nPC1-gender ρ={rho:.3f}", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 2: PC1 vs Gender")

        # ====== Page 3: DA1 vs Gender (3 experiments side by side) ======
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("CLIP Space: DA1 vs Gender Direction",
                      fontsize=13, fontweight="bold")

        for i, (name, label, scribble) in enumerate(EXPERIMENTS):
            ax = axes[i]
            if name not in embeddings or name not in da_info:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(label)
                continue

            z0 = embeddings[name]
            da = da_info[name]
            da1_scores = (z0 - da["mu"]) @ da["da1_eigvec"]
            gender_scores = z0 @ gender_axis

            sc = ax.scatter(da1_scores, gender_scores,
                           c=gender_scores, cmap="RdBu_r", s=15, alpha=0.7,
                           edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, label="gender score", shrink=0.8)

            rho, pval = spearmanr(da1_scores, gender_scores)
            ax.set_xlabel("DA1 score")
            ax.set_ylabel("Gender direction")
            ax.set_title(f"{label}\nDA1-gender ρ={rho:.3f}", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 3: DA1 vs Gender")

        # ====== Page 4: Scribble images + DA1 grids ======
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle("Scribble Inputs & DA1-Sorted Grids",
                      fontsize=13, fontweight="bold")

        for row, (name, label, scribble) in enumerate(EXPERIMENTS):
            # Scribble
            ax = axes[row, 0]
            if os.path.exists(scribble):
                img = Image.open(scribble)
                ax.imshow(img)
            ax.set_title(f"{label}\nScribble", fontsize=9)
            ax.axis("off")

            # DA1 grid
            ax = axes[row, 1]
            grid_path = os.path.join(rd, f"clip_da_{name}_da1_grid.png")
            if os.path.exists(grid_path):
                img = Image.open(grid_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Grid not found", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title("DA1-sorted grid", fontsize=9)
            ax.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        print("  Page 4: Scribbles & DA1 grids")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
