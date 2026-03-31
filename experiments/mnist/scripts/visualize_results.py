"""
visualize_results.py — Generate all paper figures for MNIST DDPM SSA/DA experiment.

Figures:
1. SSA comparison bar chart: 10 digits, DDPM-784D vs KDE-PCA-20
2. SSA vs noise level t: Line plot for all digits (from sweep)
3. DA1 pixel heatmaps: 28x28 for each digit
4. DA1 overlay on samples: Top-20 pixels highlighted on real digits
5. DA convergence: Eigenvalue gap vs lag
6. Ranking correlation: Spearman rho between DDPM-SSA and KDE-PCA20

Usage:
    python scripts/visualize_results.py --ssa_dir results --da_dir results_da
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIGURES_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Baseline PCA-20 KDE results from sliced_ssa_mnist experiment
BASELINE_DIR = os.path.join(EXP_DIR, "..", "sliced_ssa_mnist", "results_pca20_noslice")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize MNIST DDPM SSA/DA results")
    p.add_argument("--ssa_dir", type=str, default=os.path.join(EXP_DIR, "results"),
                   help="Directory with ssa_digit_*.json files")
    p.add_argument("--da_dir", type=str, default=os.path.join(EXP_DIR, "results_da"),
                   help="Directory with da_digit_*.npz files")
    p.add_argument("--noise_level", type=int, default=None,
                   help="Which noise level to use for bar chart (default: first available)")
    p.add_argument("--data_dir", type=str, default="data",
                   help="MNIST data directory (for overlay figure)")
    return p.parse_args()


def load_ssa_results(ssa_dir):
    """Load SSA results for all digits."""
    results = {}
    for d in range(10):
        path = os.path.join(ssa_dir, f"ssa_digit_{d}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[d] = json.load(f)
    return results


def load_da_results(da_dir):
    """Load DA results for all digits."""
    results = {}
    for d in range(10):
        json_path = os.path.join(da_dir, f"da_digit_{d}.json")
        npz_path = os.path.join(da_dir, f"da_digit_{d}.npz")
        if os.path.exists(json_path) and os.path.exists(npz_path):
            with open(json_path) as f:
                meta = json.load(f)
            data = np.load(npz_path)
            results[d] = {"meta": meta, "data": data}
    return results


def load_baseline_ssa():
    """Load PCA-20 KDE baseline SSA from sliced_ssa_mnist."""
    csv_path = os.path.join(BASELINE_DIR, "sliced_ssa_results.csv")
    if not os.path.exists(csv_path):
        print(f"  Baseline not found: {csv_path}")
        return None
    import csv
    baseline = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            di = int(row["digit_i"])
            dj = int(row["digit_j"])
            if di == dj:  # Single-digit SSA
                baseline[di] = float(row["sliced_mssa"])
    return baseline


# ---------------------------------------------------------------------------
# Figure 1: SSA bar chart comparison
# ---------------------------------------------------------------------------

def fig_ssa_bar_chart(ssa_results, baseline, noise_level):
    """10-digit bar chart: DDPM-784D vs KDE-PCA-20."""
    digits = sorted(ssa_results.keys())
    if not digits:
        print("  No SSA results to plot")
        return

    # Pick noise level
    t_key = str(noise_level)
    ddpm_ssa = [ssa_results[d]["noise_levels"][t_key]["ssa"] for d in digits]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(digits))
    width = 0.35

    ax.bar(x - width/2, ddpm_ssa, width, label=f"DDPM 784D (t={noise_level})",
           color="C0", alpha=0.8)

    if baseline:
        baseline_ssa = [baseline.get(d, 0) for d in digits]
        ax.bar(x + width/2, baseline_ssa, width, label="KDE PCA-20",
               color="C1", alpha=0.8)

    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("SSA", fontsize=12)
    ax.set_title("SSA per digit: DDPM score (784D) vs KDE (PCA-20)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_ssa_bar_chart.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: SSA vs noise level
# ---------------------------------------------------------------------------

def fig_ssa_vs_noise(ssa_results):
    """Line plot of SSA vs noise level for all digits."""
    digits = sorted(ssa_results.keys())
    if not digits:
        return

    # Get all noise levels from first digit
    noise_levels = sorted([int(t) for t in ssa_results[digits[0]]["noise_levels"].keys()])
    if len(noise_levels) < 2:
        print("  Only one noise level — skipping noise sweep figure")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for d in digits:
        ssa_vals = [ssa_results[d]["noise_levels"][str(t)]["ssa"] for t in noise_levels]
        ax.plot(noise_levels, ssa_vals, "o-", label=f"Digit {d}", markersize=4)

    ax.set_xlabel("Noise level t", fontsize=12)
    ax.set_ylabel("SSA", fontsize=12)
    ax.set_title("SSA vs DDPM noise level per digit", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_ssa_vs_noise.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: DA1 pixel heatmaps for all digits
# ---------------------------------------------------------------------------

def fig_da_heatmaps(da_results):
    """DA1 and PCA1 heatmaps for all 10 digits."""
    digits = sorted(da_results.keys())
    if not digits:
        print("  No DA results to plot")
        return

    n_digits = len(digits)
    fig, axes = plt.subplots(2, n_digits, figsize=(n_digits * 1.5, 3.5))
    if n_digits == 1:
        axes = axes.reshape(2, 1)

    for col, d in enumerate(digits):
        data = da_results[d]["data"]
        da_evecs = data["da_eigenvectors"]
        pca_evecs = data["pca_eigenvectors"]

        da1_img = da_evecs[:, 0].reshape(28, 28)
        pca1_img = pca_evecs[:, 0].reshape(28, 28)

        vmax = max(np.abs(da1_img).max(), np.abs(pca1_img).max())

        axes[0, col].imshow(da1_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].set_title(str(d), fontsize=9, fontweight="bold")

        axes[1, col].imshow(pca1_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("DA1", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("PCA1", fontsize=10, fontweight="bold")
    fig.suptitle("DA1 vs PCA1 directions per digit (28x28 pixel space)", fontsize=11, y=1.02)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_da_heatmaps.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: DA1 overlay on digit samples
# ---------------------------------------------------------------------------

def fig_da_overlay(da_results, data_dir):
    """Top-20 DA1 pixels highlighted on real digit images."""
    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST(root=data_dir, train=True, download=True,
                    transform=transforms.ToTensor())

    # Focus on digits with interesting DA: digit 1 (highest SSA)
    focus_digits = [1, 0, 3, 7]
    focus_digits = [d for d in focus_digits if d in da_results]
    if not focus_digits:
        print("  No DA results for focus digits")
        return

    n_samples = 5
    np.random.seed(42)

    fig, axes = plt.subplots(len(focus_digits), n_samples,
                             figsize=(n_samples * 1.5, len(focus_digits) * 1.5))
    if len(focus_digits) == 1:
        axes = axes.reshape(1, -1)

    for row, digit in enumerate(focus_digits):
        data = da_results[digit]["data"]
        da1 = data["da_eigenvectors"][:, 0]
        da1_abs = np.abs(da1)
        top_k = 20
        top_idx = np.argsort(da1_abs)[-top_k:][::-1]

        # Get samples of this digit
        digit_imgs = []
        for img, label in dataset:
            if label == digit:
                digit_imgs.append(img.squeeze().numpy())
            if len(digit_imgs) >= 200:
                break

        sample_ids = np.random.choice(len(digit_imgs), n_samples, replace=False)

        for col, sid in enumerate(sample_ids):
            img = digit_imgs[sid]
            rgb = np.stack([img] * 3, axis=-1)
            for flat_idx in top_idx:
                r, c = divmod(flat_idx, 28)
                rgb[r, c] = [1.0, 0.0, 0.0]

            axes[row, col].imshow(rgb, interpolation="nearest")
            axes[row, col].axis("off")

        meta = da_results[digit]["meta"]
        axes[row, 0].set_ylabel(f"Digit {digit}\n(gap={meta['optimal_gap']:.1f}x)",
                                fontsize=8, fontweight="bold")

    fig.suptitle("Top-20 DA1 pixels (red) on sample images", fontsize=11, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_da_overlay.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: DA convergence profiles
# ---------------------------------------------------------------------------

def fig_da_convergence(da_results):
    """Eigenvalue gap vs lag for select digits."""
    digits = sorted(da_results.keys())
    if not digits:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for d in digits:
        data = da_results[d]["data"]
        meta = da_results[d]["meta"]
        cps = data["checkpoints"]
        phys_time = cps * meta["dt"]
        ax1.plot(phys_time, data["gap_profile"], "o-", label=f"Digit {d}",
                 markersize=3, alpha=0.8)
        ax2.plot(phys_time, data["angles_deg"], "o-", label=f"Digit {d}",
                 markersize=3, alpha=0.8)

    ax1.set_xlabel("Physical time (lag x dt)")
    ax1.set_ylabel("Eigenvalue gap lambda_1/lambda_2")
    ax1.set_title("DA convergence: eigenvalue gap")
    ax1.set_xscale("log")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Physical time (lag x dt)")
    ax2.set_ylabel("Angle(DA1, PCA1) [degrees]")
    ax2.set_title("DA1 rotation from PCA1")
    ax2.set_xscale("log")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_da_convergence.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Ranking correlation
# ---------------------------------------------------------------------------

def fig_ranking_correlation(ssa_results, baseline, noise_level):
    """Scatter plot + Spearman rho between DDPM and KDE rankings."""
    if baseline is None:
        print("  No baseline for correlation")
        return

    digits = sorted(set(ssa_results.keys()) & set(baseline.keys()))
    if len(digits) < 3:
        print("  Too few digits for correlation")
        return

    t_key = str(noise_level)
    ddpm_ssa = np.array([ssa_results[d]["noise_levels"][t_key]["ssa"] for d in digits])
    kde_ssa = np.array([baseline[d] for d in digits])

    rho, p_val = spearmanr(ddpm_ssa, kde_ssa)
    r, _ = pearsonr(ddpm_ssa, kde_ssa)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(kde_ssa, ddpm_ssa, s=50, zorder=5)
    for i, d in enumerate(digits):
        ax.annotate(str(d), (kde_ssa[i], ddpm_ssa[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Fit line
    z = np.polyfit(kde_ssa, ddpm_ssa, 1)
    x_line = np.linspace(kde_ssa.min(), kde_ssa.max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5)

    ax.set_xlabel("KDE SSA (PCA-20)", fontsize=11)
    ax.set_ylabel(f"DDPM SSA (784D, t={noise_level})", fontsize=11)
    ax.set_title(f"Spearman rho={rho:.3f} (p={p_val:.3f}), Pearson r={r:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGURES_DIR, f"fig_ranking_correlation.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()

    print(f"  Spearman rho = {rho:.3f} (p = {p_val:.4f})")
    print(f"  Pearson r = {r:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=== Loading results ===")
    ssa_results = load_ssa_results(args.ssa_dir)
    da_results = load_da_results(args.da_dir)
    baseline = load_baseline_ssa()

    print(f"  SSA results: digits {sorted(ssa_results.keys())}")
    print(f"  DA results:  digits {sorted(da_results.keys())}")
    print(f"  Baseline:    {'loaded' if baseline else 'not found'}")

    # Determine noise level
    if args.noise_level is not None:
        noise_level = args.noise_level
    elif ssa_results:
        first_digit = list(ssa_results.values())[0]
        noise_level = int(list(first_digit["noise_levels"].keys())[0])
    else:
        noise_level = 10

    print(f"  Using noise level: t={noise_level}")

    print("\n=== Figure 1: SSA bar chart ===")
    fig_ssa_bar_chart(ssa_results, baseline, noise_level)

    print("\n=== Figure 2: SSA vs noise level ===")
    fig_ssa_vs_noise(ssa_results)

    print("\n=== Figure 3: DA1 heatmaps ===")
    fig_da_heatmaps(da_results)

    print("\n=== Figure 4: DA1 overlay ===")
    fig_da_overlay(da_results, args.data_dir)

    print("\n=== Figure 5: DA convergence ===")
    fig_da_convergence(da_results)

    print("\n=== Figure 6: Ranking correlation ===")
    fig_ranking_correlation(ssa_results, baseline, noise_level)

    print("\nDone! Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
