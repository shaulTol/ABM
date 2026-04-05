"""
full_correlation_table.py — Full correlation matrix: DA1, DA2, PC1, PC2, text probes.

For each space (LC 512D and CLIP 768D) separately:
  1. Embed all N samples
  2. Compute PCA → PC1, PC2 scores
  3. Load DA1, DA2 eigenvectors → DA1, DA2 scores
  4. Compute text probe scores (gender, age, realism, etc.)
  5. Correlate everything with everything → table

Usage:
    python scripts/full_correlation_table.py \
        --latents_npz results_clip_da/clip_da_latents_clip_cn_dps_face.npz \
        --clip_da_npz results_clip_da/clip_da_clip_cn_dps_face.npz \
        --lc_da_npz results_controlnet/sdxl_da_sdxl_ssa_cn_dps_face.npz \
        --model_id stabilityai/sdxl-turbo \
        --label "DPS Face dt=0.0001" \
        --output_dir results_clip_da
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from sdxl_components import (
    LatentCLIPEmbedder, RealCLIPEmbedder, TextProbeAnalyzer,
    DEFAULT_PROBE_PAIRS,
)


def pca_top_k(X, k=2):
    """PCA on (N, D) centered data. Returns eigvals, eigvecs (D, k)."""
    N = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    return eigvals[idx[:k]], eigvecs[:, idx[:k]], mu


def compute_all_scores(embeddings, da_eigvecs, da_mu, probe_axes, label_prefix):
    """Project embeddings onto DA, PCA, and probe directions. Returns dict of (N,) scores."""
    N, D = embeddings.shape

    scores = {}

    # PCA
    eigvals, eigvecs, pca_mu = pca_top_k(embeddings, k=2)
    Xc_pca = embeddings - pca_mu
    scores[f"{label_prefix}_PC1"] = Xc_pca @ eigvecs[:, 0]
    scores[f"{label_prefix}_PC2"] = Xc_pca @ eigvecs[:, 1]

    # DA
    Xc_da = embeddings - da_mu
    if "da1_eigvec" in da_eigvecs:
        scores[f"{label_prefix}_DA1"] = Xc_da @ da_eigvecs["da1_eigvec"]
    if "da2_eigvec" in da_eigvecs:
        scores[f"{label_prefix}_DA2"] = Xc_da @ da_eigvecs["da2_eigvec"]

    # Text probes
    for name, axis in probe_axes.items():
        if axis.shape[0] == D:
            scores[f"{label_prefix}_{name}"] = embeddings @ axis

    return scores


def correlation_matrix(scores_dict):
    """Spearman correlation matrix from dict of (N,) score arrays."""
    names = list(scores_dict.keys())
    N_dirs = len(names)
    mat = np.zeros((N_dirs, N_dirs))
    for i in range(N_dirs):
        for j in range(N_dirs):
            rho, _ = spearmanr(scores_dict[names[i]], scores_dict[names[j]])
            mat[i, j] = rho
    return mat, names


def orthogonalize_probes(scores_dict, probe_names):
    """For each probe, regress out all other probes and keep residuals.

    Args:
        scores_dict: dict of (N,) score arrays (DA, PC, probes)
        probe_names: list of keys in scores_dict that are text probes

    Returns:
        dict with same keys as scores_dict but probe scores replaced by
        orthogonalized residuals. DA/PC scores are unchanged.
    """
    out = dict(scores_dict)  # shallow copy
    probe_scores = {k: scores_dict[k] for k in probe_names if k in scores_dict}
    if len(probe_scores) < 2:
        return out

    for target_name in probe_names:
        if target_name not in scores_dict:
            continue
        # Regressors: all other probes
        other_names = [k for k in probe_names if k != target_name and k in scores_dict]
        if not other_names:
            continue
        y = scores_dict[target_name]
        X = np.column_stack([scores_dict[k] for k in other_names])
        # Add intercept
        X_int = np.column_stack([X, np.ones(len(y))])
        # Least-squares
        beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        residual = y - X_int @ beta
        out[target_name] = residual

    return out


def print_table(mat, names, title=""):
    """Pretty-print correlation matrix."""
    # Strip prefix for display
    short = [n.split("_", 1)[-1] if "_" in n else n for n in names]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    # Header
    header = f"{'':>12}"
    for s in short:
        header += f" {s:>8}"
    print(header)

    for i, row_name in enumerate(short):
        row = f"{row_name:>12}"
        for j in range(len(short)):
            val = mat[i, j]
            row += f" {val:>8.3f}"
        print(row)


def parse_args():
    p = argparse.ArgumentParser(
        description="Full correlation table: DA, PCA, text probes in LC and CLIP spaces"
    )
    p.add_argument("--latents_npz", type=str, required=True,
                   help="Path to clip_da_latents_*.npz (contains x0)")
    p.add_argument("--clip_da_npz", type=str, required=True,
                   help="Path to clip_da_*.npz (CLIP-space DA eigvecs)")
    p.add_argument("--lc_da_npz", type=str, default=None,
                   help="Path to sdxl_da_*.npz (LC-space DA eigvecs, optional)")
    p.add_argument("--model_id", type=str, default="stabilityai/sdxl-turbo")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(args.latents_npz)
    os.makedirs(output_dir, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load latents
    lat = dict(np.load(args.latents_npz))
    x0 = lat["x0"]  # (N, 16384)
    N = x0.shape[0]
    print(f"=== Full Correlation Table: {args.label} ===")
    print(f"  N = {N}, latents shape = {x0.shape}")

    # Load CLIP-space DA
    clip_da = dict(np.load(args.clip_da_npz))

    # Load LC-space DA (optional)
    lc_da = None
    if args.lc_da_npz and os.path.exists(args.lc_da_npz):
        lc_da = dict(np.load(args.lc_da_npz))

    # --- Embed in LC ---
    print("\n=== Embedding in Latent-CLIP (512D) ===")
    lc_embedder = LatentCLIPEmbedder(device)
    z_lc = lc_embedder.embed(x0)  # (N, 512)

    # --- Embed in CLIP ---
    print("\n=== Embedding in CLIP (768D) ===")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae",
                                         torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    clip_embedder = RealCLIPEmbedder(vae, clip_model_name=args.clip_model, device=device)
    z_clip = clip_embedder.embed(x0)  # (N, 768)

    # --- Text probes (CLIP space only — LC probes are broken) ---
    print("\n=== Computing text probes ===")
    analyzer = TextProbeAnalyzer(clip_model_name=args.clip_model, device=device)
    analyzer.define_probes()
    clip_probe_axes = dict(analyzer.probe_axes)  # {name: (768,)}

    # =====================================================================
    # CLIP space correlation table
    # =====================================================================
    print("\n=== CLIP space (768D) ===")
    clip_scores = compute_all_scores(
        z_clip,
        da_eigvecs=clip_da,
        da_mu=clip_da["mu_clip"],
        probe_axes=clip_probe_axes,
        label_prefix="CLIP",
    )
    clip_mat, clip_names = correlation_matrix(clip_scores)
    print_table(clip_mat, clip_names, title=f"CLIP Space — {args.label}")

    # =====================================================================
    # LC space correlation table
    # =====================================================================
    print("\n=== LC space (512D) ===")
    lc_da_vecs = {}
    lc_mu = z_lc.mean(axis=0)
    if lc_da is not None:
        lc_da_vecs = {"da1_eigvec": lc_da["da1_eigvec"], "da2_eigvec": lc_da["da2_eigvec"]}
        lc_mu = lc_da["mu_clip"]

    # No text probes for LC (broken), so empty
    lc_scores = compute_all_scores(
        z_lc,
        da_eigvecs=lc_da_vecs,
        da_mu=lc_mu,
        probe_axes={},  # LC text probes broken
        label_prefix="LC",
    )
    lc_mat, lc_names = correlation_matrix(lc_scores)
    print_table(lc_mat, lc_names, title=f"LC Space — {args.label}")

    # =====================================================================
    # Orthogonalized CLIP table: each probe residualized against all others
    # =====================================================================
    print("\n=== CLIP space — orthogonalized probes ===")
    clip_probe_keys = [k for k in clip_scores if any(
        k.endswith(f"_{pn}") for pn in clip_probe_axes
    )]
    clip_scores_orth = orthogonalize_probes(clip_scores, clip_probe_keys)
    clip_orth_mat, clip_orth_names = correlation_matrix(clip_scores_orth)
    print_table(clip_orth_mat, clip_orth_names,
                title=f"CLIP Space (orthogonalized probes) — {args.label}")

    # =====================================================================
    # Cross-space table: correlate LC directions with CLIP directions
    # =====================================================================
    print("\n=== Cross-space (LC scores vs CLIP scores) ===")
    all_scores = {}
    all_scores.update(clip_scores)
    all_scores.update(lc_scores)
    cross_mat, cross_names = correlation_matrix(all_scores)
    print_table(cross_mat, cross_names, title=f"Cross-Space — {args.label}")

    # =====================================================================
    # Save
    # =====================================================================
    basename = os.path.basename(args.latents_npz).replace("clip_da_latents_", "").replace(".npz", "")
    out = {
        "label": args.label,
        "N": N,
        "clip_names": clip_names,
        "clip_corr": clip_mat.tolist(),
        "clip_orth_names": clip_orth_names,
        "clip_orth_corr": clip_orth_mat.tolist(),
        "lc_names": lc_names,
        "lc_corr": lc_mat.tolist(),
        "cross_names": cross_names,
        "cross_corr": cross_mat.tolist(),
    }
    json_path = os.path.join(output_dir, f"corr_table_{basename}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
