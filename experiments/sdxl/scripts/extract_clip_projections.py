"""
Extract CLIP embeddings from Langevin checkpoints, compute PC1,
project onto DA1 and PC1, save small projection files for local analysis.

Usage:
    python scripts/extract_clip_projections.py \
        --results_dir results_da_pcavda \
        --pattern "*_gs7p5_n200"
"""
import os
import sys
import argparse
import glob
import json
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import RealCLIPEmbedder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--pattern", type=str, default="*_gs7p5_n200",
                   help="Glob pattern for matching result names")
    p.add_argument("--model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--clip_model", type=str,
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--clip_batch_size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all NPZ result files matching pattern (exclude latents files)
    npz_pattern = os.path.join(args.results_dir, f"clip_da_{args.pattern}.npz")
    npz_files = sorted(f for f in glob.glob(npz_pattern)
                        if "latents_" not in os.path.basename(f))
    if not npz_files:
        print(f"No files matching {npz_pattern}")
        return

    print(f"Found {len(npz_files)} result files")

    # Load VAE for decode
    print("\n=== Loading SDXL VAE ===")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    vae = pipe.vae.to(device).float().eval()
    del pipe.unet, pipe
    torch.cuda.empty_cache()

    # Load CLIP embedder
    print("=== Loading CLIP embedder ===")
    clip_embedder = RealCLIPEmbedder(
        vae, clip_model_name=args.clip_model, device=device,
        clip_batch_size=args.clip_batch_size,
    )

    # Load text probe axes for correlation
    from sdxl_components import TextProbeAnalyzer
    analyzer = TextProbeAnalyzer(clip_model_name=args.clip_model, device=device)
    analyzer.define_probes()

    for npz_file in npz_files:
        name = os.path.basename(npz_file).replace("clip_da_", "").replace(".npz", "")
        print(f"\n{'='*60}")
        print(f"Processing: {name}")

        # Load DA results
        da_data = np.load(npz_file, allow_pickle=True)
        da1_eigvec = da_data["da1_eigvec"]
        da2_eigvec = da_data["da2_eigvec"]
        mu_clip = da_data["mu_clip"]

        # Load latents
        latents_file = os.path.join(args.results_dir, f"clip_da_latents_{name}.npz")
        if not os.path.exists(latents_file):
            print(f"  Latents not found: {latents_file}, skipping")
            continue
        latents = np.load(latents_file)
        x0 = latents["x0"]
        print(f"  x0 shape: {x0.shape}")

        # Embed through CLIP
        print("  Embedding through CLIP...")
        z0_clip = clip_embedder.embed(x0)
        print(f"  z0_clip shape: {z0_clip.shape}")

        # PCA
        z0_c = z0_clip - z0_clip.mean(axis=0)
        cov = (z0_c.T @ z0_c) / z0_clip.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1].copy()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        pc1 = eigvecs[:, 0]
        pc2 = eigvecs[:, 1]

        # Projections
        proj_da1 = z0_c @ da1_eigvec
        proj_da2 = z0_c @ da2_eigvec
        proj_pc1 = z0_c @ pc1
        proj_pc2 = z0_c @ pc2

        # Cosine between DA1 and PC1
        cos_da1_pc1 = float(np.dot(da1_eigvec, pc1))
        cos_da1_pc2 = float(np.dot(da1_eigvec, pc2))
        cos_da2_pc1 = float(np.dot(da2_eigvec, pc1))
        print(f"  |cos(DA1, PC1)| = {abs(cos_da1_pc1):.4f}")
        print(f"  |cos(DA1, PC2)| = {abs(cos_da1_pc2):.4f}")
        print(f"  |cos(DA2, PC1)| = {abs(cos_da2_pc1):.4f}")
        print(f"  PC1 var explained: {eigvals[0]/eigvals.sum():.4f}")
        print(f"  PC2 var explained: {eigvals[1]/eigvals.sum():.4f}")

        # Text probe correlations for PC1
        pc1_cosines = analyzer.score_direction(pc1)
        pc1_scores = z0_c @ pc1
        pc1_corr = analyzer.correlate_with_ranking(z0_clip, pc1_scores)

        print(f"  PC1 cosines:  ", "  ".join(
            f"{k}={v:+.3f}" for k, v in sorted(
                pc1_cosines.items(), key=lambda x: -abs(x[1]))))
        print(f"  PC1 corr:     ", "  ".join(
            f"{k}={v['rho']:+.3f}" for k, v in sorted(
                pc1_corr.items(), key=lambda x: -abs(x[1]['rho']))))

        # Save projections
        out_file = os.path.join(args.results_dir, f"projections_{name}.npz")
        np.savez_compressed(out_file,
            z0_clip=z0_clip,
            proj_da1=proj_da1, proj_da2=proj_da2,
            proj_pc1=proj_pc1, proj_pc2=proj_pc2,
            da1_eigvec=da1_eigvec, da2_eigvec=da2_eigvec,
            pc1=pc1, pc2=pc2,
            eigvals_pca=eigvals[:20],
            cos_da1_pc1=np.array(cos_da1_pc1),
        )
        print(f"  Saved: {out_file}")

        # Save summary JSON
        summary = {
            "name": name,
            "n_clip": int(z0_clip.shape[0]),
            "cos_da1_pc1": cos_da1_pc1,
            "cos_da1_pc2": cos_da1_pc2,
            "cos_da2_pc1": cos_da2_pc1,
            "pc1_var_explained": float(eigvals[0] / eigvals.sum()),
            "pc2_var_explained": float(eigvals[1] / eigvals.sum()),
            "pc1_cosines": pc1_cosines,
            "pc1_correlations": {k: v for k, v in pc1_corr.items()},
        }
        json_file = os.path.join(args.results_dir, f"projections_{name}.json")
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {json_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
