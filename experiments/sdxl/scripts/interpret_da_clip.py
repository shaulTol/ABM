"""
interpret_da_clip.py — Post-hoc interpretation of existing LC-DA directions.

Takes existing LC-DA results (512D eigenvectors), projects x0 latents onto DA1,
decodes top/bottom extremes to pixels, encodes via real CLIP, and runs text probes
to determine what DA1 captures semantically.

Usage:
    python scripts/interpret_da_clip.py \
        --results_dir results_controlnet \
        --name sdxl_ssa_cn_dps_face \
        --model_id stabilityai/sdxl-turbo
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import (
    LatentCLIPEmbedder, RealCLIPEmbedder, TextProbeAnalyzer,
    decode_latents_to_pil, make_topbottom_grid, DEFAULT_PROBE_PAIRS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc interpretation of LC-DA directions via CLIP text probes"
    )
    p.add_argument("--results_dir", type=str, required=True,
                   help="Directory containing sdxl_da_*.npz and sdxl_latents_*.npz")
    p.add_argument("--name", type=str, required=True,
                   help="Run name (matches filenames sdxl_da_{name}.npz etc.)")
    p.add_argument("--model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="Model ID for VAE loading")
    p.add_argument("--clip_model", type=str,
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--n_extremes", type=int, default=50,
                   help="Number of top/bottom samples for correlation analysis")
    p.add_argument("--n_grid", type=int, default=8,
                   help="Number of images in top/bottom grid")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (defaults to results_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # --- Load existing results ---
    da_path = os.path.join(args.results_dir, f"sdxl_da_{args.name}.npz")
    lat_path = os.path.join(args.results_dir, f"sdxl_latents_{args.name}.npz")

    print(f"=== interpret_da_clip.py ===")
    print(f"  DA results: {da_path}")
    print(f"  Latents: {lat_path}")

    da_data = np.load(da_path)
    lat_data = np.load(lat_path)

    x0 = lat_data["x0"]  # (N, 16384)
    N = x0.shape[0]
    print(f"  N = {N} latents")

    da1_eigvec = da_data["da1_eigvec"]  # (512,)
    mu_clip = da_data["mu_clip"]  # (512,)
    da1_gap = float(da_data.get("gap12", [0])[np.argmax(da_data.get("gap12", [0]))])

    if "da2_eigvec" in da_data:
        da2_eigvec = da_data["da2_eigvec"]
    else:
        da2_eigvec = None

    # --- Load Latent-CLIP and project onto DA1 ---
    print("\n=== Loading Latent-CLIP ===")
    lc_embedder = LatentCLIPEmbedder(device)
    z0_lc = lc_embedder.embed(x0)  # (N, 512)
    z0_lc_centered = z0_lc - mu_clip

    da1_scores = z0_lc_centered @ da1_eigvec  # (N,)
    da1_rank = np.argsort(np.argsort(da1_scores))  # ranks

    print(f"  DA1 projection range: [{da1_scores.min():.4f}, {da1_scores.max():.4f}]")

    # Select top + bottom extremes for detailed analysis
    sorted_idx = np.argsort(da1_scores)
    n_ext = min(args.n_extremes, N // 2)
    extreme_idx = np.concatenate([sorted_idx[:n_ext], sorted_idx[-n_ext:]])
    x0_extremes = x0[extreme_idx]
    da1_rank_extremes = da1_rank[extreme_idx]

    # --- Load VAE + real CLIP ---
    print("\n=== Loading VAE + CLIP ===")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()

    clip_embedder = RealCLIPEmbedder(vae, clip_model_name=args.clip_model,
                                      device=device)

    # --- Encode extremes in real CLIP ---
    print(f"\n=== Encoding {len(extreme_idx)} extremes via real CLIP ===")
    t0 = time.time()
    clip_embs = clip_embedder.embed(x0_extremes)  # (2*n_ext, 768)
    print(f"  Encoded in {time.time() - t0:.1f}s")

    # --- Text probe analysis ---
    print("\n=== Text probe analysis ===")
    analyzer = TextProbeAnalyzer(clip_model_name=args.clip_model, device=device)
    analyzer.define_probes()

    # Correlate DA1 rank with CLIP text probes
    correlations = analyzer.correlate_with_ranking(clip_embs, da1_rank_extremes)

    print("\n  DA1 vs CLIP text probes (extremes):")
    print(f"  {'Probe':<15} {'Spearman rho':>12} {'p-value':>10}")
    print(f"  {'-'*37}")
    for name in analyzer.probe_names:
        r = correlations[name]
        print(f"  {name:<15} {r['rho']:>12.4f} {r['pvalue']:>10.2e}")

    # Also do full-N analysis (project all samples, correlate with all DA1 scores)
    print(f"\n=== Full-N CLIP encoding ({N} samples) ===")
    t0 = time.time()
    clip_all = clip_embedder.embed(x0)  # (N, 768)
    print(f"  Encoded in {time.time() - t0:.1f}s")

    correlations_full = analyzer.correlate_with_ranking(clip_all, da1_scores)

    print("\n  DA1 vs CLIP text probes (full N):")
    print(f"  {'Probe':<15} {'Spearman rho':>12} {'p-value':>10}")
    print(f"  {'-'*37}")
    for name in analyzer.probe_names:
        r = correlations_full[name]
        print(f"  {name:<15} {r['rho']:>12.4f} {r['pvalue']:>10.2e}")

    # --- DA2 analysis (if available) ---
    da2_results = {}
    if da2_eigvec is not None:
        da2_scores = z0_lc_centered @ da2_eigvec
        correlations_da2 = analyzer.correlate_with_ranking(clip_all, da2_scores)
        da2_results = {name: correlations_da2[name] for name in analyzer.probe_names}

        print("\n  DA2 vs CLIP text probes (full N):")
        print(f"  {'Probe':<15} {'Spearman rho':>12} {'p-value':>10}")
        print(f"  {'-'*37}")
        for name in analyzer.probe_names:
            r = da2_results[name]
            print(f"  {name:<15} {r['rho']:>12.4f} {r['pvalue']:>10.2e}")

    # --- Visualization: top/bottom grid ---
    print("\n=== Generating grid PNG ===")
    grid_path = os.path.join(output_dir, f"interpret_da_{args.name}_grid.png")
    make_topbottom_grid(vae, torch.from_numpy(x0).float(), da1_scores,
                        n=args.n_grid, path=grid_path, device=device)

    if da2_eigvec is not None:
        grid2_path = os.path.join(output_dir, f"interpret_da2_{args.name}_grid.png")
        make_topbottom_grid(vae, torch.from_numpy(x0).float(),
                            z0_lc_centered @ da2_eigvec,
                            n=args.n_grid, path=grid2_path, device=device)

    # --- Save results ---
    results = {
        "name": args.name,
        "model_id": args.model_id,
        "clip_model": args.clip_model,
        "N": N,
        "n_extremes": n_ext,
        "da1_gap": da1_gap,
        "da1_correlations_extremes": correlations,
        "da1_correlations_full": correlations_full,
        "da2_correlations_full": da2_results,
        "time_s": time.time() - t_start,
    }

    json_path = os.path.join(output_dir, f"interpret_da_{args.name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
