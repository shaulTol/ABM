"""
plot_ssa_comparison.py — Generate comparison PDFs for SSA experiments.

Creates multi-page PDFs comparing two conditions side-by-side:
  Page 1: Summary table (SSA, entropy, gender/glasses sliced SSA)
  Page 2: Rho decay curves (overlaid)
  Page 3: Gender histograms (side-by-side)
  Page 4: Glasses histograms (side-by-side)
  Page 5+: Image grids (one per condition)

Usage:
    python scripts/plot_ssa_comparison.py \
        --json_a results_clip_da/clip_da_guidance_gs1p0_v2.json \
        --json_b results_clip_da/clip_da_guidance_gs7p5_v2.json \
        --label_a "gs=1.0" --label_b "gs=7.5" \
        --title "Guidance Scale Comparison (v2)" \
        --grid_a results_clip_da/grid_guidance_gs1p0.png \
        --grid_b results_clip_da/grid_guidance_gs7p5.png \
        --gender_hist_a results_clip_da/clip_da_guidance_gs1p0_v2_gender_hist.png \
        --gender_hist_b results_clip_da/clip_da_guidance_gs7p5_v2_gender_hist.png \
        --glasses_hist_a results_clip_da/clip_da_guidance_gs1p0_v2_glasses_hist.png \
        --glasses_hist_b results_clip_da/clip_da_guidance_gs7p5_v2_glasses_hist.png \
        --curves_a results_clip_da/clip_da_guidance_gs1p0_v2_curves.png \
        --curves_b results_clip_da/clip_da_guidance_gs7p5_v2_curves.png \
        --output results_clip_da/comparison_guidance_v2.pdf
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_summary_page(fig, ax, data_a, data_b, label_a, label_b, title):
    """Summary table with key metrics."""
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    rows = []
    row_labels = []

    def add_row(label, key_or_fn, fmt=".4f"):
        row_labels.append(label)
        if callable(key_or_fn):
            va, vb = key_or_fn(data_a), key_or_fn(data_b)
        else:
            va = data_a.get(key_or_fn)
            vb = data_b.get(key_or_fn)
        sa = f"{va:{fmt}}" if va is not None else "—"
        sb = f"{vb:{fmt}}" if vb is not None else "—"
        rows.append([sa, sb])

    add_row("Model", "model_id", "s")
    add_row("Guidance Scale", "guidance_scale", ".1f")
    add_row("Resolution", lambda d: f"{d['height']}×{d['width']}", "s")
    add_row("dt", "dt", ".4f")
    add_row("T", "T", "d")
    add_row("N samples", "n_samples", "d")
    add_row("", lambda d: "", "s")  # spacer
    add_row("SSA_CLIP (filtered)", "ssa_clip", ".4f")
    add_row("SSA_CLIP (all lags)", "ssa_clip_all", ".4f")
    add_row("ssa_min_lag", "ssa_min_lag", "d")
    add_row("", lambda d: "", "s")
    add_row("SSA_gender_sliced", "ssa_gender_sliced", ".4f")
    add_row("SSA_glasses_sliced", "ssa_glasses_sliced", ".4f")
    add_row("Gender/Full ratio",
            lambda d: d.get("ssa_gender_sliced", 0) / max(d.get("ssa_clip", 1e-12), 1e-12),
            ".3f")
    add_row("", lambda d: "", "s")
    add_row("H_gender_1d (KDE)", "gender_entropy_1d", ".3f")
    add_row("H_knn (full 768D)", "entropy_clip", ".1f")

    # Entropy suite
    es_a = data_a.get("entropy_suite", {})
    es_b = data_b.get("entropy_suite", {})
    if es_a and es_b:
        add_row("", lambda d: "", "s")
        for method in ["gaussian", "knn", "kde"]:
            if method in es_a:
                for k in sorted(es_a[method].keys(), key=int):
                    row_labels.append(f"H_{method}(k={k})")
                    rows.append([f"{es_a[method][k]:.3f}", f"{es_b[method][k]:.3f}"])

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[label_a, label_b],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    # Style header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[r-1][0] == "" and rows[r-1][1] == "":
            cell.set_facecolor("#f0f0f0")
            cell.set_edgecolor("#f0f0f0")


def make_rho_overlay_page(fig, ax, data_a, data_b, label_a, label_b):
    """Overlay rho^2 decay curves."""
    rho_a = data_a.get("rho_sq_gender")
    rho_b = data_b.get("rho_sq_gender")

    # Compute snapshot lags from T
    T_a, T_b = data_a["T"], data_b["T"]
    lags_a = sorted(set(int(round(x)) for x in np.linspace(10, T_a, 20)))
    if T_a not in lags_a:
        lags_a.append(T_a)
    lags_b = sorted(set(int(round(x)) for x in np.linspace(10, T_b, 20)))
    if T_b not in lags_b:
        lags_b.append(T_b)

    # Plot gender rho^2
    if rho_a and rho_b:
        ax.plot(lags_a[:len(rho_a)], rho_a, "o-", color="#4472C4", label=f"{label_a} gender", ms=4)
        ax.plot(lags_b[:len(rho_b)], rho_b, "s-", color="#ED7D31", label=f"{label_b} gender", ms=4)

    ssa_min = data_a.get("ssa_min_lag", 0)
    if ssa_min > 0:
        ax.axvline(ssa_min, color="gray", ls="--", alpha=0.5, label=f"ssa_min_lag={ssa_min}")

    ax.set_xlabel("Lag (Langevin steps)")
    ax.set_ylabel("ρ(τ)² (gender-sliced)")
    ax.set_title("Gender Autocorrelation Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add SSA annotations
    ssa_a = data_a.get("ssa_gender_sliced")
    ssa_b = data_b.get("ssa_gender_sliced")
    if ssa_a is not None and ssa_b is not None:
        ax.text(0.98, 0.95, f"SSA_gender: {label_a}={ssa_a:.4f}, {label_b}={ssa_b:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))


def add_image_page(pdf, img_path, title):
    """Add a full-page image."""
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path)
        w, h = img.size
        aspect = h / w
        fig_w = 11
        fig_h = min(fig_w * aspect + 0.8, 16)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def add_side_by_side_images(pdf, img_a_path, img_b_path, label_a, label_b, suptitle):
    """Two images side-by-side."""
    if not (img_a_path and os.path.exists(img_a_path) and
            img_b_path and os.path.exists(img_b_path)):
        return
    img_a = Image.open(img_a_path)
    img_b = Image.open(img_b_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(img_a)
    ax1.axis("off")
    ax1.set_title(label_a, fontsize=12)
    ax2.imshow(img_b)
    ax2.axis("off")
    ax2.set_title(label_b, fontsize=12)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_a", required=True)
    p.add_argument("--json_b", required=True)
    p.add_argument("--label_a", required=True)
    p.add_argument("--label_b", required=True)
    p.add_argument("--title", default="SSA Comparison")
    p.add_argument("--grid_a", default=None)
    p.add_argument("--grid_b", default=None)
    p.add_argument("--gender_hist_a", default=None)
    p.add_argument("--gender_hist_b", default=None)
    p.add_argument("--glasses_hist_a", default=None)
    p.add_argument("--glasses_hist_b", default=None)
    p.add_argument("--curves_a", default=None)
    p.add_argument("--curves_b", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data_a = load_json(args.json_a)
    data_b = load_json(args.json_b)

    with PdfPages(args.output) as pdf:
        # Page 1: Summary table
        fig, ax = plt.subplots(figsize=(11, 14))
        make_summary_page(fig, ax, data_a, data_b, args.label_a, args.label_b, args.title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Rho decay overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        make_rho_overlay_page(fig, ax, data_a, data_b, args.label_a, args.label_b)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Individual rho curves side-by-side
        add_side_by_side_images(pdf, args.curves_a, args.curves_b,
                                args.label_a, args.label_b, "Autocorrelation Decay Curves")

        # Page 4: Gender histograms side-by-side
        add_side_by_side_images(pdf, args.gender_hist_a, args.gender_hist_b,
                                args.label_a, args.label_b, "Gender Score Distribution")

        # Page 5: Glasses histograms side-by-side
        add_side_by_side_images(pdf, args.glasses_hist_a, args.glasses_hist_b,
                                args.label_a, args.label_b, "Glasses Score Distribution")

        # Page 6+: Image grids
        add_image_page(pdf, args.grid_a, f"All Images: {args.label_a}")
        add_image_page(pdf, args.grid_b, f"All Images: {args.label_b}")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
