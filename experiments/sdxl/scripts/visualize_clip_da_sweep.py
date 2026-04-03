"""
visualize_clip_da_sweep.py — Compare SSA/DA across guidance scales in CLIP vs LC space.

Reads clip_da_*.json (CLIP-mode) and sdxl_ssa_*.json (LC-mode) results,
plots SSA curves, entropy, DA1 interpretation, and cross-space comparison.

Usage:
    python scripts/visualize_clip_da_sweep.py \
        --clip_dir results_clip_da \
        --lc_dir results_guidance_sweep \
        --output_dir figures
"""

import os
import sys
import json
import argparse
import glob

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def load_results(directory, pattern, key_field="guidance_scale"):
    """Load all JSON results matching pattern, indexed by guidance_scale."""
    results = {}
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path) as f:
            data = json.load(f)
        w = data.get(key_field, None)
        if w is not None:
            results[w] = data
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare CLIP-mode vs LC-mode SSA/DA across guidance sweep"
    )
    p.add_argument("--clip_dir", type=str, required=True,
                   help="Directory with clip_da_*.json files")
    p.add_argument("--lc_dir", type=str, default=None,
                   help="Directory with sdxl_ssa_*.json files (LC-mode, optional)")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(EXP_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load CLIP-mode results
    clip_results = load_results(args.clip_dir, "clip_da_*.json")
    print(f"Loaded {len(clip_results)} CLIP-mode results: w={sorted(clip_results.keys())}")

    # Load LC-mode results (optional)
    lc_results = {}
    if args.lc_dir:
        lc_results = load_results(args.lc_dir, "sdxl_ssa_*.json")
        print(f"Loaded {len(lc_results)} LC-mode results: w={sorted(lc_results.keys())}")

    if not clip_results:
        print("No CLIP results found. Exiting.")
        return

    # Sort by guidance scale
    clip_ws = sorted(clip_results.keys())
    clip_ssa = [clip_results[w]["ssa_clip"] for w in clip_ws]
    clip_entropy = [clip_results[w].get("entropy_clip") for w in clip_ws]
    clip_da1_gap = [clip_results[w]["da1_gap"] for w in clip_ws]

    lc_ws = sorted(lc_results.keys()) if lc_results else []
    lc_ssa = [lc_results[w]["ssa"] for w in lc_ws]
    lc_entropy = [lc_results[w].get("entropy") for w in lc_ws]

    # =====================================================================
    # Plot 1: SSA_CLIP vs SSA_LC vs guidance
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(clip_ws, clip_ssa, "o-", color="tab:blue", label="SSA (CLIP 768D)", linewidth=2)
    if lc_ws:
        ax.plot(lc_ws, lc_ssa, "s--", color="tab:orange", label="SSA (LC 512D)", linewidth=2)
    ax.set_xlabel("Guidance scale w")
    ax.set_ylabel("SSA")
    ax.set_title("SSA vs Guidance Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_ssa_clip_vs_lc.png"), dpi=150)
    plt.close(fig)
    print("  Saved: sweep_ssa_clip_vs_lc.png")

    # =====================================================================
    # Plot 2: Entropy vs guidance
    # =====================================================================
    has_clip_ent = any(e is not None for e in clip_entropy)
    has_lc_ent = any(e is not None for e in lc_entropy)
    if has_clip_ent or has_lc_ent:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if has_clip_ent:
            ws_e = [w for w, e in zip(clip_ws, clip_entropy) if e is not None]
            es_e = [e for e in clip_entropy if e is not None]
            ax.plot(ws_e, es_e, "o-", color="tab:blue", label="Entropy (CLIP)")
        if has_lc_ent:
            ws_e = [w for w, e in zip(lc_ws, lc_entropy) if e is not None]
            es_e = [e for e in lc_entropy if e is not None]
            ax.plot(ws_e, es_e, "s--", color="tab:orange", label="Entropy (LC)")
        ax.set_xlabel("Guidance scale w")
        ax.set_ylabel("kNN Entropy")
        ax.set_title("Entropy vs Guidance Scale")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "sweep_entropy.png"), dpi=150)
        plt.close(fig)
        print("  Saved: sweep_entropy.png")

    # =====================================================================
    # Plot 3: DA1 cosine with gender probe vs guidance
    # =====================================================================
    probe_name = "gender"
    if all("da1_cosines" in clip_results[w] for w in clip_ws):
        gender_cos = [abs(clip_results[w]["da1_cosines"].get(probe_name, 0))
                      for w in clip_ws]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(clip_ws, gender_cos, "o-", color="tab:red", linewidth=2)
        ax.set_xlabel("Guidance scale w")
        ax.set_ylabel("|cos(DA1, gender)|")
        ax.set_title("DA1 Alignment with Gender Probe vs Guidance")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "sweep_da1_gender_cos.png"), dpi=150)
        plt.close(fig)
        print("  Saved: sweep_da1_gender_cos.png")

    # =====================================================================
    # Plot 4: DA1 gap vs guidance
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(clip_ws, clip_da1_gap, "o-", color="tab:green", linewidth=2)
    ax.set_xlabel("Guidance scale w")
    ax.set_ylabel("DA1 gap (lambda1/lambda2)")
    ax.set_title("DA1 Spectral Gap vs Guidance Scale")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweep_da1_gap.png"), dpi=150)
    plt.close(fig)
    print("  Saved: sweep_da1_gap.png")

    # =====================================================================
    # Plot 5: All probe cosines as heatmap
    # =====================================================================
    if all("da1_cosines" in clip_results[w] for w in clip_ws):
        probe_names = list(clip_results[clip_ws[0]]["da1_cosines"].keys())
        heatmap = np.array([
            [clip_results[w]["da1_cosines"].get(p, 0) for p in probe_names]
            for w in clip_ws
        ])

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        im = ax.imshow(heatmap.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(clip_ws)))
        ax.set_xticklabels([f"w={w}" for w in clip_ws], rotation=45, ha="right")
        ax.set_yticks(range(len(probe_names)))
        ax.set_yticklabels(probe_names)
        ax.set_title("DA1 Cosine with Text Probes")
        fig.colorbar(im, ax=ax, label="Cosine similarity")
        # Add values
        for i in range(len(clip_ws)):
            for j in range(len(probe_names)):
                ax.text(i, j, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="k" if abs(heatmap[i, j]) < 0.5 else "w")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "sweep_da1_probes_heatmap.png"), dpi=150)
        plt.close(fig)
        print("  Saved: sweep_da1_probes_heatmap.png")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'w':>6}  {'SSA_CLIP':>10}  {'SSA_LC':>10}  {'DA1 gap':>8}  {'|cos(gender)|':>14}")
    print(f"{'---':>6}  {'---':>10}  {'---':>10}  {'---':>8}  {'---':>14}")
    for w in clip_ws:
        ssa_c = clip_results[w]["ssa_clip"]
        ssa_l = lc_results.get(w, {}).get("ssa", "—")
        da1g = clip_results[w]["da1_gap"]
        gcos = abs(clip_results[w].get("da1_cosines", {}).get("gender", 0))
        ssa_l_str = f"{ssa_l:.6f}" if isinstance(ssa_l, float) else ssa_l
        print(f"{w:>6.1f}  {ssa_c:>10.6f}  {ssa_l_str:>10}  {da1g:>8.2f}  {gcos:>14.4f}")

    print(f"\nDone! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
